use std::{panic::AssertUnwindSafe, sync::Arc};

use anyhow::{Context, Result, anyhow, ensure};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, sys,
};
use secp256k1::{All, Secp256k1, SecretKey};
use zeroize::{Zeroize, Zeroizing};

use super::{Address, AddressBackend, cpu, table};

const WINDOW_BITS: u8 = 16;
const INCREMENT_STRIDE: u32 = super::DEFAULT_INCREMENT_STRIDE;
const BLOCK_SIZE: u32 = 128;
const INFLIGHT: usize = 2;

struct GpuSlot {
    stream: Arc<CudaStream>,
    event: CudaEvent,
    keys: CudaSlice<u8>,
    addresses: CudaSlice<u8>,
    host_keys: Zeroizing<Vec<u8>>,
    submitted: bool,
}

pub(crate) struct CudaBackend {
    kernel: CudaFunction,
    table: CudaSlice<u8>,
    slots: Vec<GpuSlot>,
    collect_at: usize,
    pending: usize,
    capacity: usize,
    sample_index: usize,
    verifier: Secp256k1<All>,
    device_name: String,
}

impl CudaBackend {
    pub(crate) fn new(capacity: usize) -> Result<Option<Self>> {
        if cfg!(target_os = "macos") {
            return Ok(None);
        }
        ensure!(
            (1..=super::MAX_GPU_BATCH_SIZE as usize).contains(&capacity),
            "invalid CUDA batch capacity"
        );
        let Some(ctx) = pick_device()? else {
            return Ok(None);
        };
        create_backend(ctx, capacity).and_then(|mut backend| {
            backend
                .self_test()
                .context("CUDA startup self-test failed")?;
            Ok(Some(backend))
        })
    }

    pub(crate) fn device_name(&self) -> String {
        self.device_name.clone()
    }

    fn self_test(&mut self) -> Result<()> {
        for keys in super::gpu_self_test_batches(INCREMENT_STRIDE as usize, self.capacity)? {
            let mut addresses = vec![[0; 20]; keys.len()];
            self.derive_batch(&keys, &mut addresses)?;
            for (key, address) in keys.iter().zip(&addresses) {
                cpu::verify_address(key, address, &self.verifier)?;
            }
        }
        self.sample_index = 0;
        Ok(())
    }

    fn write_keys(host: &mut Zeroizing<Vec<u8>>, keys: &[SecretKey]) {
        let bytes = keys.len() * 32;
        debug_assert!(bytes <= host.len());
        for (slot, key) in host[..bytes].chunks_exact_mut(32).zip(keys) {
            let secret = Zeroizing::new(key.secret_bytes());
            slot.copy_from_slice(secret.as_ref());
        }
    }

    fn wait_slot(&mut self, index: usize) -> Result<()> {
        let slot = &self.slots[index];
        if !slot.submitted {
            return Ok(());
        }
        slot.event
            .synchronize()
            .map_err(driver_err("CUDA event wait failed"))?;
        Ok(())
    }
}

impl AddressBackend for CudaBackend {
    fn inflight_capacity(&self) -> usize {
        self.slots.len()
    }

    fn increment_stride(&self) -> usize {
        INCREMENT_STRIDE as usize
    }

    fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        self.begin_batch(keys)?;
        self.end_batch(keys, addresses)
    }

    fn begin_batch(&mut self, keys: &[SecretKey]) -> Result<()> {
        ensure!(keys.len() <= self.capacity, "batch exceeds CUDA capacity");
        ensure!(
            self.pending < self.slots.len(),
            "CUDA in-flight slots exhausted"
        );
        if keys.is_empty() {
            return Ok(());
        }
        let submit_at = (self.collect_at + self.pending) % self.slots.len();
        ensure!(
            !self.slots[submit_at].submitted,
            "CUDA slot still holds an in-flight command"
        );
        Self::write_keys(&mut self.slots[submit_at].host_keys, keys);
        let key_bytes = keys.len() * 32;
        let count = keys.len() as u32;
        let threads = count.div_ceil(INCREMENT_STRIDE);
        let groups = threads.div_ceil(BLOCK_SIZE);
        let cfg = LaunchConfig {
            grid_dim: (groups, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let kernel = self.kernel.clone();
        let CudaBackend { table, slots, .. } = self;
        let slot = &mut slots[submit_at];
        {
            let mut key_view = slot
                .keys
                .try_slice_mut(0..key_bytes)
                .ok_or_else(|| anyhow!("CUDA key slice is out of range"))?;
            slot.stream
                .memcpy_htod(&slot.host_keys[..key_bytes], &mut key_view)
                .map_err(driver_err("CUDA key upload failed"))?;
        }
        unsafe {
            slot.stream
                .launch_builder(&kernel)
                .arg(&slot.keys)
                .arg(table)
                .arg(&mut slot.addresses)
                .arg(&count)
                .launch(cfg)
        }
        .map_err(driver_err("CUDA kernel launch failed"))?;
        slot.event
            .record(&slot.stream)
            .map_err(driver_err("CUDA event record failed"))?;
        slot.submitted = true;
        self.pending += 1;
        Ok(())
    }

    fn end_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        ensure!(
            keys.len() == addresses.len(),
            "batch input/output lengths differ"
        );
        if keys.is_empty() {
            return Ok(());
        }
        ensure!(self.pending > 0, "no in-flight CUDA batch to collect");
        let collect_at = self.collect_at;
        self.wait_slot(collect_at)?;
        let address_bytes = keys.len() * 20;
        let slot = &mut self.slots[collect_at];
        let mut host = vec![0u8; address_bytes];
        {
            let view = slot
                .addresses
                .try_slice(0..address_bytes)
                .ok_or_else(|| anyhow!("CUDA address slice is out of range"))?;
            slot.stream
                .memcpy_dtoh(&view, &mut host)
                .map_err(driver_err("CUDA address download failed"))?;
        }
        slot.stream
            .synchronize()
            .map_err(driver_err("CUDA address download wait failed"))?;
        for (index, address) in addresses.iter_mut().enumerate() {
            address.copy_from_slice(&host[index * 20..index * 20 + 20]);
        }
        let sample = self.sample_index % keys.len();
        cpu::verify_address(&keys[sample], &addresses[sample], &self.verifier)?;
        self.sample_index = self.sample_index.wrapping_add(1);
        slot.stream
            .memset_zeros(&mut slot.keys)
            .map_err(driver_err("CUDA key wipe failed"))?;
        slot.stream
            .synchronize()
            .map_err(driver_err("CUDA key wipe wait failed"))?;
        slot.host_keys.zeroize();
        slot.submitted = false;
        self.collect_at = (collect_at + 1) % self.slots.len();
        self.pending -= 1;
        Ok(())
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if slot.submitted {
                let _ = slot.event.synchronize();
                slot.submitted = false;
            }
            let _ = slot.stream.memset_zeros(&mut slot.keys);
            let _ = slot.stream.synchronize();
            slot.host_keys.zeroize();
        }
    }
}

fn driver_err(label: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> anyhow::Error {
    move |error| anyhow!("{label}: {error:?}")
}

fn driver_device_count() -> Option<i32> {
    // cudarc 在找不到 libcuda 时 panic，而不是返回 Err。
    std::panic::catch_unwind(AssertUnwindSafe(CudaContext::device_count))
        .ok()
        .and_then(Result::ok)
}

fn pick_device() -> Result<Option<Arc<CudaContext>>> {
    let Some(count) = driver_device_count() else {
        return Ok(None);
    };
    if count <= 0 {
        return Ok(None);
    }
    let mut best: Option<(u8, (i32, i32), Arc<CudaContext>)> = None;
    for ordinal in 0..count as usize {
        let Ok(ctx) = CudaContext::new(ordinal) else {
            continue;
        };
        let Ok((major, minor)) = ctx.compute_capability() else {
            continue;
        };
        if major < 6 {
            continue;
        }
        let discrete = match ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
        {
            Ok(0) => 1,
            Ok(_) => 0,
            Err(_) => continue,
        };
        match &best {
            Some((best_discrete, best_cc, _))
                if (*best_discrete, *best_cc) >= (discrete, (major, minor)) => {}
            _ => best = Some((discrete, (major, minor), ctx)),
        }
    }
    Ok(best.map(|(_, _, ctx)| ctx))
}

fn create_backend(ctx: Arc<CudaContext>, capacity: usize) -> Result<CudaBackend> {
    ctx.set_blocking_synchronize()
        .map_err(driver_err("CUDA blocking sync failed"))?;
    // 表只读、每槽密钥/地址互不共享；自行用 event 同步，避免 cudarc 把双 stream 串起来。
    unsafe { ctx.disable_event_tracking() };

    let ptx = cudarc::nvrtc::Ptx::from_src(include_str!("shader.ptx"));
    let module = ctx
        .load_module(ptx)
        .map_err(driver_err("CUDA PTX load failed"))?;
    let kernel = module
        .load_function("chunk_derive_addresses")
        .map_err(driver_err("CUDA kernel load failed"))?;

    let verifier = Secp256k1::new();
    let table_bytes = table::build_table(&verifier, WINDOW_BITS)?;
    let upload = ctx.default_stream();
    let table = upload
        .clone_htod(table_bytes.as_slice())
        .map_err(driver_err("CUDA table upload failed"))?;
    upload
        .synchronize()
        .map_err(driver_err("CUDA table upload wait failed"))?;

    let device_name = ctx
        .name()
        .map_err(driver_err("CUDA device name query failed"))?;
    let mut backend = CudaBackend {
        kernel,
        table,
        slots: Vec::with_capacity(INFLIGHT),
        collect_at: 0,
        pending: 0,
        capacity,
        sample_index: 0,
        verifier,
        device_name,
    };
    let key_bytes = capacity * 32;
    let address_bytes = capacity * 20;
    for _ in 0..INFLIGHT {
        let stream = ctx
            .new_stream()
            .map_err(driver_err("CUDA stream creation failed"))?;
        let event = ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC))
            .map_err(driver_err("CUDA event creation failed"))?;
        let keys = stream
            .alloc_zeros::<u8>(key_bytes)
            .map_err(driver_err("CUDA key buffer allocation failed"))?;
        let addresses = stream
            .alloc_zeros::<u8>(address_bytes)
            .map_err(driver_err("CUDA address buffer allocation failed"))?;
        backend.slots.push(GpuSlot {
            stream,
            event,
            keys,
            addresses,
            host_keys: Zeroizing::new(vec![0u8; key_bytes]),
            submitted: false,
        });
    }
    Ok(backend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_is_unavailable_on_macos() {
        #[cfg(target_os = "macos")]
        {
            assert!(CudaBackend::new(1024).unwrap().is_none());
        }
    }

    #[test]
    #[ignore = "requires a real CUDA compute device; absence is a failure"]
    fn cuda_differential() -> Result<()> {
        let mut backend = CudaBackend::new(super::super::MAX_GPU_BATCH_SIZE as usize)?
            .context("GPU required for hardware acceptance")?;
        let keys = super::super::sequential_test_keys(backend.capacity)?;
        for chunk in keys.chunks(backend.capacity) {
            let mut addresses = vec![[0; 20]; chunk.len()];
            backend.derive_batch(chunk, &mut addresses)?;
            for (key, address) in chunk.iter().zip(&addresses) {
                cpu::verify_address(key, address, &backend.verifier)?;
            }
        }
        for count in [1usize, 7, 8, 9, 33, 65] {
            let batch: Vec<_> = keys.iter().copied().take(count).collect();
            let mut addresses = vec![[0; 20]; batch.len()];
            backend.derive_batch(&batch, &mut addresses)?;
            for (key, address) in batch.iter().zip(&addresses) {
                cpu::verify_address(key, address, &backend.verifier)?;
            }
        }
        inflight_overlap_differential(&mut backend, &keys[..66.min(keys.len())])?;
        input_is_cleared_after_batch(&mut backend)?;
        Ok(())
    }

    fn inflight_overlap_differential(backend: &mut CudaBackend, keys: &[SecretKey]) -> Result<()> {
        ensure!(keys.len() >= 2, "overlap test needs two keys");
        let mid = keys.len() / 2;
        let first = &keys[..mid];
        let second = &keys[mid..];
        ensure!(
            backend.inflight_capacity() >= 2,
            "expected two in-flight slots"
        );
        backend.begin_batch(first)?;
        backend.begin_batch(second)?;
        let mut first_out = vec![[0; 20]; first.len()];
        let mut second_out = vec![[0; 20]; second.len()];
        backend.end_batch(first, &mut first_out)?;
        backend.end_batch(second, &mut second_out)?;
        for (key, address) in first.iter().zip(&first_out) {
            cpu::verify_address(key, address, &backend.verifier)?;
        }
        for (key, address) in second.iter().zip(&second_out) {
            cpu::verify_address(key, address, &backend.verifier)?;
        }
        Ok(())
    }

    fn input_is_cleared_after_batch(backend: &mut CudaBackend) -> Result<()> {
        let cleared = (backend.collect_at + backend.slots.len() - 1) % backend.slots.len();
        let slot = &backend.slots[cleared];
        ensure!(
            slot.host_keys.iter().all(|&byte| byte == 0),
            "CUDA host key input was not wiped"
        );
        let device = slot
            .stream
            .clone_dtoh(&slot.keys)
            .map_err(driver_err("CUDA key readback failed"))?;
        ensure!(
            device.iter().all(|&byte| byte == 0),
            "CUDA device key input was not wiped"
        );
        Ok(())
    }
}
