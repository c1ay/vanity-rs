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
use secp256k1::{All, Secp256k1, SecretKey};
use zeroize::{Zeroize, Zeroizing};

use super::{Address, AddressBackend, cpu, table};
use crate::timing::{Noop, Observer, Stage};

// Experiments remain available only to the test harness. Arithmetic and
// threadgroup candidates from earlier rounds stay off; threadgroup Montgomery
// invert was measured and rejected. Defaults that passed the retention gate:
// two in-flight GPU commands, 16-bit fixed-base windows, per-thread
// chunked Montgomery inversion (chunk = 8), and a fused jacobian+invert
// kernel. Bit-interleaved Keccak stayed within noise and remains off.
#[derive(Clone, Copy)]
pub(crate) struct MetalConfig {
    pub(crate) bulk: bool,
    pub(crate) group: Option<usize>,
    pub(crate) square: bool,
    pub(crate) fast_add: bool,
    pub(crate) invert: bool,
    pub(crate) window_bits: u8,
    pub(crate) inflight: u8,
    pub(crate) chunk: u8,
    pub(crate) keccak: bool,
    pub(crate) fuse: bool,
    pub(crate) stride: u8,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            bulk: false,
            group: None,
            square: false,
            fast_add: false,
            invert: false,
            window_bits: 16,
            inflight: 2,
            chunk: 8,
            keccak: false,
            fuse: true,
            stride: super::DEFAULT_INCREMENT_STRIDE as u8,
        }
    }
}

impl MetalConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            [4, 8, 16].contains(&self.window_bits),
            "VANITY_BENCH_WINDOW must be 4, 8 or 16"
        );
        ensure!(
            self.inflight == 1 || self.inflight == 2,
            "VANITY_BENCH_INFLIGHT must be 1 or 2"
        );
        ensure!(
            [0, 4, 8, 16, 32].contains(&self.chunk),
            "VANITY_BENCH_CHUNK must be 0, 4, 8, 16 or 32"
        );
        ensure!(
            self.chunk == 0 || !self.invert,
            "chunked and threadgroup inversion are mutually exclusive"
        );
        ensure!(
            !self.fuse || self.chunk > 0,
            "VANITY_BENCH_FUSE requires chunked inversion"
        );
        ensure!(
            !self.fuse || !self.invert,
            "fused and threadgroup inversion are mutually exclusive"
        );
        ensure!(
            self.stride == 1
                || ([8, 16, 32, 64].contains(&self.stride)
                    && self.fuse
                    && self.chunk > 0
                    && self.stride % self.chunk == 0),
            "VANITY_BENCH_STRIDE must be 1, or 8/16/32/64 with fused chunk inversion"
        );
        Ok(())
    }
}

#[cfg(test)]
impl MetalConfig {
    pub(crate) fn from_env() -> Result<Self> {
        let mut config = Self::default();
        for (name, setting) in [
            ("VANITY_BENCH_BULK", &mut config.bulk),
            ("VANITY_BENCH_SQUARE", &mut config.square),
            ("VANITY_BENCH_ADD", &mut config.fast_add),
            ("VANITY_BENCH_INVERT", &mut config.invert),
            ("VANITY_BENCH_KECCAK", &mut config.keccak),
            ("VANITY_BENCH_FUSE", &mut config.fuse),
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
        if let Ok(value) = std::env::var("VANITY_BENCH_WINDOW") {
            config.window_bits = value.parse().context("invalid benchmark window")?;
        }
        if let Ok(value) = std::env::var("VANITY_BENCH_INFLIGHT") {
            config.inflight = value.parse().context("invalid benchmark inflight")?;
        }
        if let Ok(value) = std::env::var("VANITY_BENCH_CHUNK") {
            config.chunk = value.parse().context("invalid benchmark chunk")?;
        } else if config.invert {
            // Threadgroup inversion excludes the default chunked inversion.
            config.chunk = 0;
        }
        if config.invert && std::env::var("VANITY_BENCH_FUSE").is_err() {
            config.fuse = false;
        }
        if let Ok(value) = std::env::var("VANITY_BENCH_STRIDE") {
            config.stride = value.parse().context("invalid benchmark stride")?;
        } else if !config.fuse || config.chunk == 0 {
            config.stride = 1;
        }
        config.validate()?;
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

#[cfg(test)]
struct SecretUpload<'a>(&'a mut SharedBuffer);

#[cfg(test)]
impl Drop for SecretUpload<'_> {
    fn drop(&mut self) {
        self.0.clear();
    }
}

struct CommandCompletion {
    command: Object<dyn MTLCommandBuffer>,
    submitted: bool,
}

// SAFETY: the backend never shares a live command buffer. CPU waits or Drop
// complete GPU work before the slot is reused or moved to another thread.
unsafe impl Send for CommandCompletion {}

impl CommandCompletion {
    fn commit<O: Observer>(&mut self, observer: &O, encoded: O::Stamp) {
        self.command.commit();
        self.submitted = true;
        observer.finish(Stage::EncodeSubmit, encoded);
    }

    fn wait<O: Observer>(mut self, observer: &O) -> Result<()> {
        let waiting = observer.start();
        self.command.waitUntilCompleted();
        self.submitted = false;
        observer.finish(Stage::Wait, waiting);
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

    #[cfg(test)]
    fn complete<O: Observer>(mut self, observer: &O, encoded: O::Stamp) -> Result<()> {
        self.commit(observer, encoded);
        self.wait(observer)
    }
}

impl Drop for CommandCompletion {
    fn drop(&mut self) {
        if self.submitted {
            self.command.waitUntilCompleted();
        }
    }
}

struct GpuSlot {
    input: SharedBuffer,
    output: SharedBuffer,
    xyz: Option<SharedBuffer>,
    command: Option<CommandCompletion>,
}

pub(crate) struct MetalBackend {
    device: Object<dyn MTLDevice>,
    queue: Object<dyn MTLCommandQueue>,
    #[cfg(test)]
    library: Object<dyn MTLLibrary>,
    pipeline: Object<dyn MTLComputePipelineState>,
    jacobian: Option<Object<dyn MTLComputePipelineState>>,
    invert_pipeline: Option<Object<dyn MTLComputePipelineState>>,
    chunk_pipeline: Option<Object<dyn MTLComputePipelineState>>,
    slots: Vec<GpuSlot>,
    collect_at: usize,
    pending: usize,
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
        config.validate()?;
        autoreleasepool(|_| {
            let Some(device) = MTLCreateSystemDefaultDevice() else {
                return Ok(None);
            };
            let mut source = format!(
                "#define OPT_SQUARE {}\n#define OPT_ADD {}\n#define OPT_INVERT {}\n#define WINDOW_BITS {}\n#define CHUNK_SIZE {}\n#define OPT_KECCAK {}\n#define INCREMENT_STRIDE {}\n{}",
                u8::from(config.square),
                u8::from(config.fast_add),
                u8::from(config.invert),
                config.window_bits,
                config.chunk,
                u8::from(config.keccak),
                config.stride,
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
            let split = config.invert || (config.chunk > 0 && !config.fuse);
            let jacobian = split
                .then(|| pipeline_with_group(&device, &library, "jacobian_points", config.group))
                .transpose()?;
            let invert_pipeline = config
                .invert
                .then(|| {
                    pipeline_with_group(&device, &library, "invert_affine_keccak", config.group)
                })
                .transpose()?;
            let chunk_pipeline = (config.chunk > 0)
                .then(|| {
                    pipeline_with_group(
                        &device,
                        &library,
                        if config.fuse {
                            "chunk_derive_addresses"
                        } else {
                            "chunk_invert_affine_keccak"
                        },
                        config.group,
                    )
                })
                .transpose()?;
            let queue = device
                .newCommandQueue()
                .context("Metal command queue unavailable")?;
            let verifier = Secp256k1::new();
            let mut table =
                SharedBuffer::new(&device, table::table_bytes(config.window_bits), false)?;
            populate_table(&mut table, &verifier, config.window_bits)?;
            let mut slots = Vec::with_capacity(config.inflight as usize);
            for _ in 0..config.inflight {
                slots.push(GpuSlot {
                    input: SharedBuffer::new(&device, capacity * 32, true)?,
                    output: SharedBuffer::new(&device, capacity * 20, false)?,
                    xyz: split
                        .then(|| SharedBuffer::new(&device, capacity * 96, false))
                        .transpose()?,
                    command: None,
                });
            }
            let mut backend = Self {
                device,
                queue,
                #[cfg(test)]
                library,
                pipeline,
                jacobian,
                invert_pipeline,
                chunk_pipeline,
                slots,
                collect_at: 0,
                pending: 0,
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
        for keys in super::gpu_self_test_batches(self.config.stride as usize, self.capacity)? {
            let mut addresses = vec![[0; 20]; keys.len()];
            self.derive_batch(&keys, &mut addresses)?;
            for (key, address) in keys.iter().zip(&addresses) {
                cpu::verify_address(key, address, &self.verifier)?;
            }
        }
        self.sample_index = 0;
        Ok(())
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            slot.command.take();
            slot.input.clear();
        }
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

fn populate_table(table: &mut SharedBuffer, secp: &Secp256k1<All>, window_bits: u8) -> Result<()> {
    let bytes = table::build_table(secp, window_bits)?;
    table.clear();
    table.write(0, &bytes);
    Ok(())
}

fn write_keys(input: &mut SharedBuffer, keys: &[SecretKey], bulk: bool) {
    if bulk {
        let bytes = keys
            .len()
            .checked_mul(32)
            .expect("key upload length overflow");
        input.with_bytes_mut(bytes, |destination| {
            for (slot, key) in destination.chunks_exact_mut(32).zip(keys) {
                let bytes = Zeroizing::new(key.secret_bytes());
                slot.copy_from_slice(bytes.as_ref());
            }
        });
    } else {
        for (index, key) in keys.iter().enumerate() {
            let bytes = Zeroizing::new(key.secret_bytes());
            input.write(index * 32, bytes.as_ref());
        }
    }
}

#[cfg(test)]
fn upload_keys<'a>(input: &'a mut SharedBuffer, keys: &[SecretKey]) -> SecretUpload<'a> {
    let upload = SecretUpload(input);
    for (index, key) in keys.iter().enumerate() {
        let bytes = Zeroizing::new(key.secret_bytes());
        upload.0.write(index * 32, bytes.as_ref());
    }
    upload
}

#[cfg(test)]
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
/// `threads` is the dispatch width; chunked kernels use fewer threads than items.
fn encode_compute(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: (&SharedBuffer, &SharedBuffer, &SharedBuffer),
    count: usize,
    threads: usize,
    group: Option<usize>,
) -> Result<()> {
    ensure!(
        threads > 0 && threads <= count,
        "invalid Metal dispatch width"
    );
    let (input, table, output) = buffers;
    let encoder = command
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
        group > 0 && group <= max_threads && group <= 256 && group % width == 0,
        "invalid dispatch group"
    );
    encoder.dispatchThreads_threadsPerThreadgroup(
        MTLSize {
            width: threads,
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
    Ok(())
}

#[cfg(test)]
fn dispatch<O: Observer>(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    buffers: (&SharedBuffer, &SharedBuffer, &SharedBuffer),
    count: usize,
    group: Option<usize>,
    observer: &O,
) -> Result<()> {
    let encoded = observer.start();
    let command = queue
        .commandBuffer()
        .context("Metal command buffer unavailable")?;
    let completion = CommandCompletion {
        command,
        submitted: false,
    };
    encode_compute(&completion.command, pipeline, buffers, count, count, group)?;
    completion.complete(observer, encoded)
}

impl AddressBackend for MetalBackend {
    fn inflight_capacity(&self) -> usize {
        self.slots.len()
    }

    fn increment_stride(&self) -> usize {
        if self.config.fuse && self.config.stride > 1 {
            self.config.stride as usize
        } else {
            1
        }
    }

    fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        self.derive_observed(keys, addresses, &Noop)
    }

    fn begin_batch(&mut self, keys: &[SecretKey]) -> Result<()> {
        self.begin_observed(keys, &Noop)
    }

    fn end_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        self.end_observed(keys, addresses, &Noop)
    }
}

impl MetalBackend {
    pub(crate) fn derive_observed<O: Observer>(
        &mut self,
        keys: &[SecretKey],
        addresses: &mut [Address],
        observer: &O,
    ) -> Result<()> {
        self.begin_observed(keys, observer)?;
        self.end_observed(keys, addresses, observer)
    }

    pub(crate) fn begin_observed<O: Observer>(
        &mut self,
        keys: &[SecretKey],
        observer: &O,
    ) -> Result<()> {
        ensure!(keys.len() <= self.capacity, "batch exceeds Metal capacity");
        ensure!(
            self.pending < self.slots.len(),
            "Metal in-flight slots exhausted"
        );
        if keys.is_empty() {
            return Ok(());
        }
        autoreleasepool(|_| {
            let submit_at = (self.collect_at + self.pending) % self.slots.len();
            ensure!(
                self.slots[submit_at].command.is_none(),
                "Metal slot still holds an in-flight command"
            );
            let invert = self.config.invert;
            let bulk = self.config.bulk;
            let group = self.config.group;
            let uploaded = observer.start();
            write_keys(&mut self.slots[submit_at].input, keys, bulk);
            observer.finish(Stage::Upload, uploaded);
            let encoded = observer.start();
            let command = self
                .queue
                .commandBuffer()
                .context("Metal command buffer unavailable")?;
            let mut completion = CommandCompletion {
                command,
                submitted: false,
            };
            let slot = &self.slots[submit_at];
            let chunk = self.config.chunk as usize;
            if self.config.fuse {
                let pipeline = self
                    .chunk_pipeline
                    .as_ref()
                    .context("fused path missing chunk pipeline")?;
                let step = if self.config.stride > 1 {
                    self.config.stride as usize
                } else {
                    chunk
                };
                encode_compute(
                    &completion.command,
                    pipeline,
                    (&slot.input, &self.table, &slot.output),
                    keys.len(),
                    keys.len().div_ceil(step.max(1)),
                    group,
                )?;
            } else if invert || chunk > 0 {
                let jacobian = self
                    .jacobian
                    .as_ref()
                    .context("split path missing jacobian pipeline")?;
                let xyz = slot.xyz.as_ref().context("split path missing XYZ buffer")?;
                encode_compute(
                    &completion.command,
                    jacobian,
                    (&slot.input, &self.table, xyz),
                    keys.len(),
                    keys.len(),
                    group,
                )?;
                let (second, threads) = if invert {
                    let pipeline = self
                        .invert_pipeline
                        .as_ref()
                        .context("invert path missing invert pipeline")?;
                    (pipeline, keys.len())
                } else {
                    let pipeline = self
                        .chunk_pipeline
                        .as_ref()
                        .context("chunk path missing chunk pipeline")?;
                    (pipeline, keys.len().div_ceil(chunk))
                };
                encode_compute(
                    &completion.command,
                    second,
                    (xyz, &self.table, &slot.output),
                    keys.len(),
                    threads,
                    group,
                )?;
            } else {
                encode_compute(
                    &completion.command,
                    &self.pipeline,
                    (&slot.input, &self.table, &slot.output),
                    keys.len(),
                    keys.len(),
                    group,
                )?;
            }
            completion.commit(observer, encoded);
            self.slots[submit_at].command = Some(completion);
            self.pending += 1;
            Ok(())
        })
    }

    pub(crate) fn end_observed<O: Observer>(
        &mut self,
        keys: &[SecretKey],
        addresses: &mut [Address],
        observer: &O,
    ) -> Result<()> {
        ensure!(
            keys.len() == addresses.len(),
            "batch input/output lengths differ"
        );
        if keys.is_empty() {
            return Ok(());
        }
        ensure!(self.pending > 0, "no in-flight Metal batch to collect");
        autoreleasepool(|_| {
            let collect_at = self.collect_at;
            let command = self.slots[collect_at]
                .command
                .take()
                .context("Metal slot missing in-flight command")?;
            command.wait(observer)?;
            let slot = &mut self.slots[collect_at];
            let read = observer.start();
            if self.config.bulk {
                slot.output.read(0, addresses.as_flattened_mut());
            } else {
                for (index, address) in addresses.iter_mut().enumerate() {
                    slot.output.read(index * 20, address);
                }
            }
            observer.finish(Stage::ReadbackCleanup, read);
            let verified = observer.start();
            let sample = self.sample_index % keys.len();
            cpu::verify_address(&keys[sample], &addresses[sample], &self.verifier)?;
            self.sample_index = self.sample_index.wrapping_add(1);
            observer.finish(Stage::SampleVerify, verified);
            let cleared = observer.start();
            slot.input.clear();
            observer.finish(Stage::ReadbackCleanup, cleared);
            self.collect_at = (collect_at + 1) % self.slots.len();
            self.pending -= 1;
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
    use secp256k1::PublicKey;

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
            let upload = upload_keys(&mut backend.slots[0].input, keys);
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

    fn threadgroup_invert_differential(backend: &MetalBackend) -> Result<()> {
        let p = (BigUint::from(1u8) << 256) - (BigUint::from(1u8) << 32) - BigUint::from(977u32);
        let one = BigUint::from(1u8);
        let pipeline = pipeline(&backend.device, &backend.library, "threadgroup_invert")?;
        for &count in &[1usize, 129] {
            let mut values = vec![one.clone(); count];
            values[0] = &p - 1u8;
            if count > 1 {
                values[1] = one.clone();
                for (index, slot) in values.iter_mut().enumerate().skip(2) {
                    *slot = BigUint::from((index as u32) + 3) % &p;
                    if slot == &0u8.into() {
                        *slot = one.clone();
                    }
                }
            }
            let mut input = SharedBuffer::new(&backend.device, count * 32, false)?;
            let output = SharedBuffer::new(&backend.device, count.max(1) * 32, false)?;
            for (index, value) in values.iter().enumerate() {
                let mut bytes = value.to_bytes_le();
                bytes.resize(32, 0);
                input.write(index * 32, &bytes);
            }
            dispatch(
                &backend.queue,
                &pipeline,
                (&input, &backend.table, &output),
                count,
                None,
                &Noop,
            )?;
            for (index, value) in values.iter().enumerate() {
                let mut actual = [0; 32];
                output.read(index * 32, &mut actual);
                assert_eq!(
                    BigUint::from_bytes_le(&actual),
                    value.modpow(&(&p - 2u8), &p),
                    "threadgroup invert case {index}, count {count}"
                );
            }
        }
        Ok(())
    }

    fn inflight_overlap_differential(backend: &mut MetalBackend, keys: &[SecretKey]) -> Result<()> {
        ensure!(
            backend.inflight_capacity() >= 2 && keys.len() >= 66,
            "inflight overlap requires two slots"
        );
        let first = &keys[..33];
        let second = &keys[33..66];
        let mut out_first = vec![[0; 20]; first.len()];
        let mut out_second = vec![[0; 20]; second.len()];
        backend.begin_batch(first)?;
        backend.begin_batch(second)?;
        backend.end_batch(first, &mut out_first)?;
        backend.end_batch(second, &mut out_second)?;
        for (key, address) in first.iter().zip(&out_first) {
            cpu::verify_address(key, address, &backend.verifier)?;
        }
        for (key, address) in second.iter().zip(&out_second) {
            cpu::verify_address(key, address, &backend.verifier)?;
        }
        Ok(())
    }

    #[test]
    fn metal_config_rejects_invalid_window_inflight_and_chunk() {
        let mut invalid = MetalConfig {
            window_bits: 5,
            ..MetalConfig::default()
        };
        assert!(invalid.validate().is_err());
        invalid.window_bits = 16;
        assert!(invalid.validate().is_ok());
        invalid.inflight = 3;
        assert!(invalid.validate().is_err());
        invalid.inflight = 1;
        invalid.chunk = 3;
        assert!(invalid.validate().is_err());
        invalid.chunk = 8;
        invalid.invert = true;
        assert!(invalid.validate().is_err());
        invalid.invert = false;
        assert!(invalid.validate().is_ok());
        invalid.fuse = true;
        invalid.chunk = 0;
        assert!(invalid.validate().is_err());
        invalid.chunk = 8;
        assert!(invalid.validate().is_ok());
        invalid.chunk = 16;
        assert!(invalid.validate().is_ok());
        invalid.chunk = 32;
        assert!(invalid.validate().is_ok());
        invalid.chunk = 64;
        assert!(invalid.validate().is_err());
        invalid.chunk = 8;
        invalid.invert = true;
        assert!(invalid.validate().is_err());
        invalid.invert = false;
        assert!(invalid.validate().is_ok());
        invalid.stride = 3;
        assert!(invalid.validate().is_err());
        invalid.stride = 32;
        invalid.fuse = false;
        assert!(invalid.validate().is_err());
        invalid.fuse = true;
        invalid.stride = 1;
        assert!(invalid.validate().is_ok());
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
            threadgroup_invert_differential(&backend)?;
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
            let seq_keys = {
                let mut bytes = [0; 32];
                bytes[31] = 2;
                let mut key = SecretKey::from_byte_array(bytes)?;
                let mut chain = Vec::with_capacity(backend.capacity);
                for index in 0..backend.capacity {
                    if index > 0 {
                        key = super::super::increment_secret_key(&key)
                            .context("sequential test key overflow")?;
                    }
                    chain.push(key);
                }
                chain
            };
            let address_keys = if backend.increment_stride() > 1 {
                &seq_keys
            } else {
                &keys
            };
            {
                let mut dual = MetalConfig::from_env()?;
                dual.inflight = 2;
                if let Some(mut dual) = MetalBackend::with_config(66, dual)? {
                    let overlap = if dual.increment_stride() > 1 {
                        &seq_keys[..66]
                    } else {
                        &keys[..66]
                    };
                    inflight_overlap_differential(&mut dual, overlap)?;
                }
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
                backend.derive_batch(&address_keys[..count], &mut addresses)?;
                let public = diagnostic_public_keys(&mut backend, &address_keys[..count])?;
                for index in 0..count {
                    let expected_public =
                        PublicKey::from_secret_key(&backend.verifier, &address_keys[index])
                            .serialize_uncompressed();
                    assert_eq!(
                        public[index],
                        expected_public[1..],
                        "public key case {index}, batch {count}"
                    );
                    assert_eq!(
                        addresses[index],
                        cpu::derive_address(&address_keys[index], &backend.verifier),
                        "address case {index}, batch {count}"
                    );
                }
                let mut cleared = vec![1; backend.slots[0].input.object.length()];
                backend.slots[0].input.read(0, &mut cleared);
                assert!(
                    cleared.iter().all(|&byte| byte == 0),
                    "GPU secret buffer not cleared"
                );
                eprintln!("GPU differential batch {count}: passed");
            }
            // Structural variants around the defaults: unchunked path, chunk
            // sizes 4/8/16/32, window widths, interleaved Keccak, and increment
            // strides. Tails are covered by counts that are not multiples of 4/8.
            let candidates = [
                MetalConfig {
                    chunk: 0,
                    window_bits: 8,
                    fuse: false,
                    stride: 1,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    chunk: 4,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    chunk: 8,
                    window_bits: 8,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    chunk: 8,
                    window_bits: 4,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    keccak: true,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    fuse: false,
                    stride: 1,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    stride: 1,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    stride: 8,
                    chunk: 8,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    stride: 64,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    chunk: 16,
                    ..MetalConfig::default()
                },
                MetalConfig {
                    chunk: 32,
                    ..MetalConfig::default()
                },
            ];
            for (which, config) in candidates.into_iter().enumerate() {
                let batch_keys = if config.stride > 1 { &seq_keys } else { &keys };
                let mut candidate = MetalBackend::with_config(4099, config)?
                    .context("GPU required for hardware acceptance")?;
                for &count in &[1usize, 33, 129, 4095, 4099] {
                    let mut addresses = vec![[0; 20]; count];
                    candidate.derive_batch(&batch_keys[..count], &mut addresses)?;
                    for index in 0..count {
                        assert_eq!(
                            addresses[index],
                            cpu::derive_address(&batch_keys[index], &candidate.verifier),
                            "candidate {which} address case {index}, batch {count}"
                        );
                    }
                }
                for (slot_index, slot) in candidate.slots.iter().enumerate() {
                    let mut cleared = vec![1; slot.input.object.length()];
                    slot.input.read(0, &mut cleared);
                    assert!(
                        cleared.iter().all(|&byte| byte == 0),
                        "candidate {which} slot {slot_index} secret buffer not cleared"
                    );
                }
                eprintln!("GPU candidate config {which}: passed");
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
                        upload_keys_bulk(&mut backend.slots[0].input, &keys[..33])
                    } else {
                        upload_keys(&mut backend.slots[0].input, &keys[..33])
                    };
                    let mut actual = Zeroizing::new(vec![0; 33 * 32]);
                    upload.0.read(0, &mut actual);
                    for (slot, key) in actual.chunks_exact(32).zip(&keys[..33]) {
                        assert!(slot == key.secret_bytes());
                    }
                    anyhow::bail!("injected failure after upload");
                })();
                assert!(failure.is_err());
                let mut cleared = vec![1; backend.slots[0].input.object.length()];
                backend.slots[0].input.read(0, &mut cleared);
                assert!(cleared.iter().all(|&byte| byte == 0));
            }
            let mut entered = false;
            let too_large = backend.slots[0].input.object.length() + 1;
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.slots[0]
                    .input
                    .with_bytes_mut(too_large, |_| entered = true);
            }));
            assert!(unwind.is_err() && !entered);
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _upload = upload_keys_bulk(&mut backend.slots[0].input, &keys[..33]);
                panic!("injected upload unwind");
            }));
            assert!(unwind.is_err());
            let mut cleared = vec![1; backend.slots[0].input.object.length()];
            backend.slots[0].input.read(0, &mut cleared);
            assert!(cleared.iter().all(|&byte| byte == 0));
            Ok(())
        })
    }
}
