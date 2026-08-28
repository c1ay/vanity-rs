use std::{ffi::c_void, ptr::NonNull, slice};

use anyhow::{Context, Result, anyhow, ensure};
use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLPipelineOption, MTLResourceOptions,
    MTLSize,
};
use secp256k1::{All, PublicKey, Secp256k1, SecretKey};
use zeroize::{Zeroize, Zeroizing};

use super::{Address, AddressBackend, cpu};
use crate::timing::{Noop, Observer, Stage};

// Experiments remain available only to the test harness. The buffer, threadgroup
// and arithmetic candidates did not demonstrate the required stable gain.
#[derive(Clone, Copy, Default)]
pub(crate) struct MetalConfig {
    pub(crate) bulk: bool,
    pub(crate) group: Option<usize>,
    pub(crate) square: bool,
    pub(crate) fast_add: bool,
}

#[cfg(test)]
impl MetalConfig {
    pub(crate) fn from_env() -> Result<Self> {
        let mut config = Self::default();
        for (name, setting) in [
            ("VANITY_BENCH_BULK", &mut config.bulk),
            ("VANITY_BENCH_SQUARE", &mut config.square),
            ("VANITY_BENCH_ADD", &mut config.fast_add),
        ] {
            if let Ok(value) = std::env::var(name) {
                *setting = match value.as_str() {
                    "0" => false,
                    "1" => true,
                    _ => anyhow::bail!("{name} must be 0 or 1"),
                };
            }
        }
        if let Ok(value) = std::env::var("VANITY_BENCH_GROUP") {
            config.group = if value == "auto" {
                None
            } else {
                Some(value.parse().context("invalid benchmark group")?)
            };
        }
        Ok(config)
    }
}

// Required by MTLCreateSystemDefaultDevice in command-line programs.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

type Object<T> = Retained<ProtocolObject<T>>;

struct SharedBuffer {
    object: Object<dyn MTLBuffer>,
    sensitive: bool,
}

// SAFETY: this owner never exposes/clones its MTLBuffer handle or CPU pointers.
// dispatch borrows it until GPU completion, so ownership can only move while
// idle. Metal buffers have no thread affinity. It deliberately is NOT Sync:
// CPU access and GPU submission remain serialized by the owning backend.
unsafe impl Send for SharedBuffer {}

impl SharedBuffer {
    fn new(device: &ProtocolObject<dyn MTLDevice>, bytes: usize, sensitive: bool) -> Result<Self> {
        let object = device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .context("Metal shared buffer allocation failed")?;
        Ok(Self { object, sensitive })
    }

    /// Only called before submission or after CommandCompletion has waited.
    fn write(&mut self, offset: usize, bytes: &[u8]) {
        assert!(offset <= self.object.length() && bytes.len() <= self.object.length() - offset);
        // SAFETY: shared allocation is live, CPU/GPU access is serialized, bounds
        // were checked, and no other CPU slice/reference to this buffer exists.
        unsafe {
            let destination = self.object.contents().cast::<u8>().as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination, bytes.len());
        }
    }

    fn read(&self, offset: usize, bytes: &mut [u8]) {
        assert!(offset <= self.object.length() && bytes.len() <= self.object.length() - offset);
        // SAFETY: every caller waits for completion before reading. This copies
        // initialized byte output, never aliases the allocation with mutable CPU data.
        unsafe {
            let source = self.object.contents().cast::<u8>().as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(source, bytes.as_mut_ptr(), bytes.len());
        }
    }

    fn clear(&mut self) {
        // SAFETY: the command guard completes GPU work before this buffer can be
        // reused/dropped. The slice is local and covers exactly the owned allocation.
        unsafe {
            slice::from_raw_parts_mut(
                self.object.contents().cast::<u8>().as_ptr(),
                self.object.length(),
            )
            .zeroize();
        }
    }

    /// The mapping cannot escape this exclusive CPU access. Only called before
    /// submission; its borrow ends before the command is committed.
    fn with_bytes_mut(&mut self, length: usize, write: impl FnOnce(&mut [u8])) {
        assert!(length <= self.object.length());
        // SAFETY: shared storage is retained, bounds are checked once, and the
        // caller owns the buffer exclusively with no in-flight GPU access.
        let bytes = unsafe {
            slice::from_raw_parts_mut(self.object.contents().cast::<u8>().as_ptr(), length)
        };
        write(bytes);
    }
}

impl Drop for SharedBuffer {
    fn drop(&mut self) {
        if self.sensitive {
            self.clear();
        }
    }
}

struct SecretUpload<'a>(&'a mut SharedBuffer);

impl Drop for SecretUpload<'_> {
    fn drop(&mut self) {
        self.0.clear();
    }
}

struct CommandCompletion {
    command: Object<dyn MTLCommandBuffer>,
    submitted: bool,
}

impl CommandCompletion {
    fn complete<O: Observer>(mut self, observer: &O, encoded: O::Stamp) -> Result<()> {
        self.command.commit();
        self.submitted = true;
        observer.finish(Stage::EncodeSubmit, encoded);
        let waiting = observer.start();
        self.command.waitUntilCompleted();
        self.submitted = false;
        observer.finish(Stage::Wait, waiting);
        // Noop ignores these values, but avoid even Objective-C timestamp calls
        // in normal execution through the observer's static profiling flag.
        if O::ENABLED {
            observer.gpu_seconds(self.command.GPUStartTime(), self.command.GPUEndTime());
        }
        ensure!(
            self.command.status() == MTLCommandBufferStatus::Completed,
            "Metal command failed: {}",
            self.command.error().map_or_else(
                || "unknown GPU error".to_string(),
                |error| error.to_string()
            )
        );
        Ok(())
    }
}

impl Drop for CommandCompletion {
    fn drop(&mut self) {
        if self.submitted {
            self.command.waitUntilCompleted();
        }
    }
}

pub(crate) struct MetalBackend {
    device: Object<dyn MTLDevice>,
    queue: Object<dyn MTLCommandQueue>,
    #[cfg(test)]
    library: Object<dyn MTLLibrary>,
    pipeline: Object<dyn MTLComputePipelineState>,
    input: SharedBuffer,
    output: SharedBuffer,
    table: SharedBuffer,
    capacity: usize,
    sample_index: usize,
    verifier: Secp256k1<All>,
    config: MetalConfig,
}

impl MetalBackend {
    pub(crate) fn new(capacity: usize) -> Result<Option<Self>> {
        Self::with_config(capacity, MetalConfig::default())
    }

    pub(crate) fn with_config(capacity: usize, config: MetalConfig) -> Result<Option<Self>> {
        ensure!(
            (1..=super::MAX_GPU_BATCH_SIZE as usize).contains(&capacity),
            "invalid Metal batch capacity"
        );
        autoreleasepool(|_| {
            let Some(device) = MTLCreateSystemDefaultDevice() else {
                return Ok(None);
            };
            let mut source = format!(
                "#define OPT_SQUARE {}\n#define OPT_ADD {}\n{}",
                u8::from(config.square),
                u8::from(config.fast_add),
                include_str!("shader.metal")
            );
            // Diagnostic entry points are not included in production binaries.
            if cfg!(test) {
                source.push_str(include_str!("shader_tests.metal"));
            }
            let library = device
                .newLibraryWithSource_options_error(&NSString::from_str(&source), None)
                .map_err(|error| anyhow!("Metal shader compilation failed: {error}"))?;
            let pipeline =
                pipeline_with_group(&device, &library, "derive_addresses", config.group)?;
            let queue = device
                .newCommandQueue()
                .context("Metal command queue unavailable")?;
            let verifier = Secp256k1::new();
            let mut table = SharedBuffer::new(&device, 64 * 16 * 64, false)?;
            populate_table(&mut table, &verifier)?;
            let input = SharedBuffer::new(&device, capacity * 32, true)?;
            let output = SharedBuffer::new(&device, capacity * 20, false)?;
            let mut backend = Self {
                device,
                queue,
                #[cfg(test)]
                library,
                pipeline,
                input,
                output,
                table,
                capacity,
                sample_index: 0,
                verifier,
                config,
            };
            backend
                .self_test()
                .context("Metal startup self-test failed")?;
            Ok(Some(backend))
        })
    }

    pub(crate) fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    fn self_test(&mut self) -> Result<()> {
        let mut one = [0; 32];
        one[31] = 1;
        let mut two = one;
        two[31] = 2;
        let mut last = secp256k1::constants::CURVE_ORDER;
        last[31] -= 1;
        let keys: Vec<_> = [one, two, last]
            .into_iter()
            .map(SecretKey::from_byte_array)
            .collect::<std::result::Result<_, _>>()?;
        for chunk in keys.chunks(self.capacity) {
            let mut addresses = vec![[0; 20]; chunk.len()];
            self.derive_batch(chunk, &mut addresses)?;
            for (key, address) in chunk.iter().zip(&addresses) {
                cpu::verify_address(key, address, &self.verifier)?;
            }
        }
        self.sample_index = 0;
        Ok(())
    }
}

#[cfg(test)]
fn pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Object<dyn MTLComputePipelineState>> {
    pipeline_with_group(device, library, name, None)
}

fn pipeline_with_group(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
    group: Option<usize>,
) -> Result<Object<dyn MTLComputePipelineState>> {
    let function = library
        .newFunctionWithName(&NSString::from_str(name))
        .with_context(|| format!("Metal kernel {name} missing"))?;
    let result = if let Some(group) = group {
        ensure!(
            [32, 64, 128, 256].contains(&group),
            "unsupported tuning group"
        );
        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));
        descriptor.setMaxTotalThreadsPerThreadgroup(group);
        // Leave the unsafe full-SIMD-group promise unset: dispatchThreads also
        // accepts empty/small/nonuniform tail batches.
        device.newComputePipelineStateWithDescriptor_options_reflection_error(
            &descriptor,
            MTLPipelineOption::None,
            None,
        )
    } else {
        device.newComputePipelineStateWithFunction_error(&function)
    }
    .map_err(|error| anyhow!("Metal pipeline {name} failed: {error}"))?;
    if let Some(group) = group {
        let width = result.threadExecutionWidth();
        ensure!(
            width > 0 && group % width == 0 && group <= result.maxTotalThreadsPerThreadgroup(),
            "invalid tuned threadgroup size"
        );
    }
    Ok(result)
}

fn populate_table(table: &mut SharedBuffer, secp: &Secp256k1<All>) -> Result<()> {
    table.clear();
    for window in 0..64 {
        for digit in 1..16 {
            // Public constants, not wallet keys: digit * 16^window * G.
            let mut scalar = [0; 32];
            scalar[31 - window / 2] = (digit as u8) << ((window % 2) * 4);
            let key = SecretKey::from_byte_array(scalar)?;
            let public = PublicKey::from_secret_key(secp, &key).serialize_uncompressed();
            let offset = (window * 16 + digit) * 64;
            for coordinate in 0..2 {
                for limb in 0..8 {
                    let start = 1 + coordinate * 32 + (7 - limb) * 4;
                    let word = u32::from_be_bytes(public[start..start + 4].try_into().unwrap());
                    table.write(offset + coordinate * 32 + limb * 4, &word.to_le_bytes());
                }
            }
        }
    }
    Ok(())
}

fn upload_keys<'a>(input: &'a mut SharedBuffer, keys: &[SecretKey]) -> SecretUpload<'a> {
    let upload = SecretUpload(input);
    for (index, key) in keys.iter().enumerate() {
        let bytes = Zeroizing::new(key.secret_bytes());
        upload.0.write(index * 32, bytes.as_ref());
    }
    upload
}

fn upload_keys_bulk<'a>(input: &'a mut SharedBuffer, keys: &[SecretKey]) -> SecretUpload<'a> {
    let upload = SecretUpload(input);
    let bytes = keys
        .len()
        .checked_mul(32)
        .expect("key upload length overflow");
    upload.0.with_bytes_mut(bytes, |destination| {
        for (slot, key) in destination.chunks_exact_mut(32).zip(keys) {
            let bytes = Zeroizing::new(key.secret_bytes());
            slot.copy_from_slice(bytes.as_ref());
        }
    });
    upload
}

/// Buffer layouts are defined together with shader.metal: raw 32-byte big-endian
/// scalars, LE u32 table limbs, packed 20-byte addresses, and a copied u32 count.
fn dispatch<O: Observer>(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: (&SharedBuffer, &SharedBuffer, &SharedBuffer),
    count: usize,
    group: Option<usize>,
    observer: &O,
) -> Result<()> {
    let (input, table, output) = buffers;
    let encoded = observer.start();
    let command = queue
        .commandBuffer()
        .context("Metal command buffer unavailable")?;
    let completion = CommandCompletion {
        command,
        submitted: false,
    };
    let encoder = completion
        .command
        .computeCommandEncoder()
        .context("Metal compute encoder unavailable")?;
    encoder.setComputePipelineState(pipeline);
    let count = u32::try_from(count)?;
    // SAFETY: every bound buffer is retained and lives through complete(). No CPU
    // reads/writes overlap GPU work. Offsets are zero, slots/types agree with MSL,
    // and callers allocate at least count elements. setBytes copies the u32 now.
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(&input.object), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&table.object), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&output.object), 0, 2);
        encoder.setBytes_length_atIndex(
            NonNull::from(&count).cast::<c_void>(),
            size_of::<u32>(),
            3,
        );
    }
    let width = pipeline.threadExecutionWidth();
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    ensure!(
        width > 0 && max_threads >= width,
        "invalid Metal pipeline thread limits"
    );
    let group = group.unwrap_or_else(|| (128usize.max(width).min(max_threads) / width) * width);
    ensure!(
        group > 0 && group <= max_threads && group % width == 0,
        "invalid dispatch group"
    );
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: count as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: group,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    completion.complete(observer, encoded)
}

impl AddressBackend for MetalBackend {
    fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        self.derive_observed(keys, addresses, &Noop)
    }
}

impl MetalBackend {
    pub(crate) fn derive_observed<O: Observer>(
        &mut self,
        keys: &[SecretKey],
        addresses: &mut [Address],
        observer: &O,
    ) -> Result<()> {
        ensure!(
            keys.len() == addresses.len(),
            "batch input/output lengths differ"
        );
        ensure!(keys.len() <= self.capacity, "batch exceeds Metal capacity");
        if keys.is_empty() {
            return Ok(());
        }
        autoreleasepool(|_| {
            let uploaded = observer.start();
            let upload = if self.config.bulk {
                upload_keys_bulk(&mut self.input, keys)
            } else {
                upload_keys(&mut self.input, keys)
            };
            observer.finish(Stage::Upload, uploaded);
            dispatch(
                &self.queue,
                &self.pipeline,
                (upload.0, &self.table, &self.output),
                keys.len(),
                self.config.group,
                observer,
            )?;
            let read = observer.start();
            if self.config.bulk {
                self.output.read(0, addresses.as_flattened_mut());
            } else {
                for (index, address) in addresses.iter_mut().enumerate() {
                    self.output.read(index * 20, address);
                }
            }
            observer.finish(Stage::ReadbackCleanup, read);
            let verified = observer.start();
            let sample = self.sample_index % keys.len();
            cpu::verify_address(&keys[sample], &addresses[sample], &self.verifier)?;
            self.sample_index = self.sample_index.wrapping_add(1);
            observer.finish(Stage::SampleVerify, verified);
            let cleared = observer.start();
            drop(upload);
            observer.finish(Stage::ReadbackCleanup, cleared);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigUint;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn diagnostic_public_keys(
        backend: &mut MetalBackend,
        keys: &[SecretKey],
    ) -> Result<Vec<[u8; 64]>> {
        ensure!(
            keys.len() <= backend.capacity,
            "diagnostic batch exceeds capacity"
        );
        let pipeline = pipeline_with_group(
            &backend.device,
            &backend.library,
            "derive_public_keys",
            backend.config.group,
        )?;
        let output = SharedBuffer::new(&backend.device, keys.len().max(1) * 64, false)?;
        let mut public = vec![[0; 64]; keys.len()];
        if !keys.is_empty() {
            let upload = upload_keys(&mut backend.input, keys);
            dispatch(
                &backend.queue,
                &pipeline,
                (upload.0, &backend.table, &output),
                keys.len(),
                backend.config.group,
                &Noop,
            )?;
            for (index, value) in public.iter_mut().enumerate() {
                output.read(index * 64, value);
            }
        }
        Ok(public)
    }

    fn field_differential(backend: &MetalBackend, rng: &mut ChaCha20Rng) -> Result<()> {
        let p = (BigUint::from(1u8) << 256) - (BigUint::from(1u8) << 32) - BigUint::from(977u32);
        let mut values = vec![BigUint::from(0u8), BigUint::from(1u8), &p - 1u8, &p - 2u8];
        for bit in [31usize, 32, 63, 64, 127, 128, 223, 224, 255] {
            values.push((BigUint::from(1u8) << bit) - 1u8);
            values.push(BigUint::from(1u8) << bit);
        }
        for limb in 1..=7 {
            let boundary = BigUint::from(1u8) << (limb * 32);
            values.push(&p - &boundary);
            values.push(&boundary - 1u8);
        }
        for delta in 3u32..20 {
            values.push(&p - delta);
        }
        let boundary_count = values.len();
        for _ in 0..128 {
            let mut bytes = [0; 32];
            rng.fill_bytes(&mut bytes);
            values.push(BigUint::from_bytes_le(&bytes) % &p);
        }
        let pairs: Vec<_> = values
            .iter()
            .flat_map(|a| values.iter().take(boundary_count).map(move |b| (a, b)))
            .collect();
        let mut input = SharedBuffer::new(&backend.device, pairs.len() * 64, false)?;
        let output = SharedBuffer::new(&backend.device, pairs.len() * 160, false)?;
        for (index, (a, b)) in pairs.iter().enumerate() {
            for (coordinate, value) in [a, b].iter().enumerate() {
                let mut bytes = value.to_bytes_le();
                bytes.resize(32, 0);
                input.write(index * 64 + coordinate * 32, &bytes);
            }
        }
        let pipeline = pipeline(&backend.device, &backend.library, "field_operations")?;
        dispatch(
            &backend.queue,
            &pipeline,
            (&input, &backend.table, &output),
            pairs.len(),
            None,
            &Noop,
        )?;
        for (index, (a, b)) in pairs.iter().enumerate() {
            let expected = [
                (*a + *b) % &p,
                (*a + &p - *b) % &p,
                (*a * *b) % &p,
                (*a * *a) % &p,
                a.modpow(&(&p - 2u8), &p),
            ];
            for (operation, expected) in expected.iter().enumerate() {
                let mut actual = [0; 32];
                output.read(index * 160 + operation * 32, &mut actual);
                assert_eq!(
                    &BigUint::from_bytes_le(&actual),
                    expected,
                    "field case {index}, operation {operation}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires actual Apple Silicon GPU access; absence is a failure"]
    fn metal_differential() -> Result<()> {
        autoreleasepool(|_| {
            let mut backend = MetalBackend::with_config(
                super::super::MAX_GPU_BATCH_SIZE as usize,
                MetalConfig::from_env()?,
            )?
            .context("GPU required for hardware acceptance")?;
            let mut rng = ChaCha20Rng::from_seed([93; 32]);
            field_differential(&backend, &mut rng)?;
            let mut keys = Vec::with_capacity(backend.capacity);
            let mut last = secp256k1::constants::CURVE_ORDER;
            last[31] -= 1;
            keys.push(SecretKey::from_byte_array(last)?);
            for bit in 0..256 {
                let mut bytes = [0; 32];
                bytes[31 - bit / 8] = 1 << (bit % 8);
                keys.push(SecretKey::from_byte_array(bytes)?);
            }
            while keys.len() < backend.capacity {
                keys.push(crate::search::generate_secret_key(&mut rng));
            }
            let counts: &[usize] =
                if std::env::var("VANITY_DIFFERENTIAL_QUICK").as_deref() == Ok("1") {
                    &[0, 1, 31, 32, 33, 127, 128, 129, 10000]
                } else {
                    &[
                        0, 1, 31, 32, 33, 127, 128, 129, 4095, 4096, 10000, 65536, 65537, 131072,
                        131073, 262143, 262144,
                    ]
                };
            for &count in counts {
                let mut addresses = vec![[0; 20]; count];
                backend.derive_batch(&keys[..count], &mut addresses)?;
                let public = diagnostic_public_keys(&mut backend, &keys[..count])?;
                for index in 0..count {
                    let expected_public =
                        PublicKey::from_secret_key(&backend.verifier, &keys[index])
                            .serialize_uncompressed();
                    assert_eq!(
                        public[index],
                        expected_public[1..],
                        "public key case {index}, batch {count}"
                    );
                    assert_eq!(
                        addresses[index],
                        cpu::derive_address(&keys[index], &backend.verifier),
                        "address case {index}, batch {count}"
                    );
                }
                let mut cleared = vec![1; backend.input.object.length()];
                backend.input.read(0, &mut cleared);
                assert!(
                    cleared.iter().all(|&byte| byte == 0),
                    "GPU secret buffer not cleared"
                );
                eprintln!("GPU differential batch {count}: passed");
            }
            assert!(backend.derive_batch(&keys[..1], &mut []).is_err());
            keys.push(keys[0]);
            assert!(
                backend
                    .derive_batch(&keys, &mut vec![[0; 20]; keys.len()])
                    .is_err()
            );
            // Check whole-buffer access and guard cleanup on both error and unwind.
            for bulk in [false, true] {
                let failure: Result<()> = (|| {
                    let upload = if bulk {
                        upload_keys_bulk(&mut backend.input, &keys[..33])
                    } else {
                        upload_keys(&mut backend.input, &keys[..33])
                    };
                    let mut actual = Zeroizing::new(vec![0; 33 * 32]);
                    upload.0.read(0, &mut actual);
                    for (slot, key) in actual.chunks_exact(32).zip(&keys[..33]) {
                        assert!(slot == key.secret_bytes());
                    }
                    anyhow::bail!("injected failure after upload");
                })();
                assert!(failure.is_err());
                let mut cleared = vec![1; backend.input.object.length()];
                backend.input.read(0, &mut cleared);
                assert!(cleared.iter().all(|&byte| byte == 0));
            }
            let mut entered = false;
            let too_large = backend.input.object.length() + 1;
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.input.with_bytes_mut(too_large, |_| entered = true);
            }));
            assert!(unwind.is_err() && !entered);
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _upload = upload_keys_bulk(&mut backend.input, &keys[..33]);
                panic!("injected upload unwind");
            }));
            assert!(unwind.is_err());
            let mut cleared = vec![1; backend.input.object.length()];
            backend.input.read(0, &mut cleared);
            assert!(cleared.iter().all(|&byte| byte == 0));
            Ok(())
        })
    }
}
