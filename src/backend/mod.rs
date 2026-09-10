use anyhow::{Result, bail, ensure};
use clap::ValueEnum;
use secp256k1::{All, Secp256k1, SecretKey};

pub(crate) mod cpu;
pub(crate) mod cuda;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod metal;
pub(crate) mod table;
pub(crate) mod vulkan;

pub(crate) type Address = [u8; 20];

pub(crate) const MAX_GPU_BATCH_SIZE: u32 = 262_144;
// Smallest measured batch within 2% of peak M4 Pro throughput; no startup tuning.
pub(crate) const DEFAULT_GPU_BATCH_SIZE: u32 = 262_144;
// Fused kernel: one scalar mul, then this many - 1 generator additions.
// Occupancy vs amortization is measured; 32 kept the M4 Pro thread grid full enough.
pub(crate) const DEFAULT_INCREMENT_STRIDE: u32 = 32;

/// Next curve scalar, or `None` if adding 1 leaves the secp256k1 range.
pub(crate) fn increment_secret_key(key: &SecretKey) -> Option<SecretKey> {
    offset_secret_key(key, 1)
}

/// `start + offset` as a curve scalar, or `None` if it leaves the range.
/// Big-endian byte addition; `offset` is at most a chain stride (< 2^32).
pub(crate) fn offset_secret_key(start: &SecretKey, offset: usize) -> Option<SecretKey> {
    let mut bytes = start.secret_bytes();
    let mut carry = offset as u64;
    for byte in bytes.iter_mut().rev() {
        if carry == 0 {
            break;
        }
        let sum = u64::from(*byte) + (carry & 0xff);
        *byte = sum as u8;
        carry = (carry >> 8) + (sum >> 8);
    }
    if carry != 0 {
        return None;
    }
    SecretKey::from_byte_array(bytes).ok()
}

/// Chain starts needed for `count` addresses when every chain covers `stride`
/// consecutive scalars (the last chain may be shorter).
pub(crate) fn chain_count(count: usize, stride: usize) -> usize {
    count.div_ceil(stride.max(1))
}

/// Scalar of address `index`: chain `index / stride` start plus `index % stride`.
pub(crate) fn chain_key(starts: &[SecretKey], stride: usize, index: usize) -> Result<SecretKey> {
    let stride = stride.max(1);
    let start = starts
        .get(index / stride)
        .ok_or_else(|| anyhow::anyhow!("address index outside the key batch"))?;
    offset_secret_key(start, index % stride)
        .ok_or_else(|| anyhow::anyhow!("chain leaves scalar range"))
}

/// `keys` must hold exactly one start per chain for `count` addresses.
pub(crate) fn check_chain_batch(keys: &[SecretKey], count: usize, stride: usize) -> Result<()> {
    ensure!(
        keys.len() == chain_count(count, stride),
        "batch key count does not match address count"
    );
    Ok(())
}

/// Host-side rule for a chain `k, k+1, …, k+len-1`. Kernels add `i·G` (or `G`
/// repeatedly) with incomplete affine/mixed formulas, whose exceptions are
/// `P = ±Q`: doubling when `k = i` for some `0 < i < len`, infinity when
/// `k + i = n`. Requiring `k >= len` and `k + len - 1 <= n - 1` excludes both.
/// Chains of length one add nothing and accept any valid scalar.
pub(crate) fn chain_start_accepted(start: &SecretKey, len: usize) -> bool {
    if len <= 1 {
        return true;
    }
    let bytes = start.secret_bytes();
    let small = bytes[..24].iter().all(|&byte| byte == 0)
        && u64::from_be_bytes(bytes[24..].try_into().unwrap()) < len as u64;
    !small && offset_secret_key(start, len - 1).is_some()
}

/// Startup vectors: `1`, `n-1`, and one full chain. Returns the complete
/// scalar list per batch; the chain start is `max(stride, 2)` so it passes
/// `chain_start_accepted` for every kernel formula.
fn gpu_self_test_batches(stride: usize, capacity: usize) -> Result<Vec<Vec<SecretKey>>> {
    let mut one = [0; 32];
    one[31] = 1;
    let mut last = secp256k1::constants::CURVE_ORDER;
    last[31] -= 1;
    let chain_len = stride.max(3).min(capacity);
    let mut first = [0; 32];
    first[31] = stride.max(2) as u8;
    let mut chain = Vec::with_capacity(chain_len);
    let mut key = SecretKey::from_byte_array(first)?;
    chain.push(key);
    for _ in 1..chain_len {
        key = increment_secret_key(&key)
            .ok_or_else(|| anyhow::anyhow!("self-test chain overflow"))?;
        chain.push(key);
    }
    Ok(vec![
        vec![SecretKey::from_byte_array(one)?],
        vec![SecretKey::from_byte_array(last)?],
        chain,
    ])
}

/// Every start of consecutive `stride`-long chains in a full scalar list.
pub(crate) fn chain_starts(keys: &[SecretKey], stride: usize) -> Vec<SecretKey> {
    keys.iter().copied().step_by(stride.max(1)).collect()
}

/// Writes one big-endian 32-byte scalar per address (`count * 32` bytes) from
/// chain starts, for kernels that still read a full per-address key layout.
/// Plain byte increments: starts were accepted by the host, so no scalar can
/// leave the range; a 256-bit wrap is still reported.
pub(crate) fn expand_chain_keys(
    starts: &[SecretKey],
    stride: usize,
    count: usize,
    destination: &mut [u8],
) -> Result<()> {
    let stride = stride.max(1);
    check_chain_batch(starts, count, stride)?;
    ensure!(
        destination.len() >= count * 32,
        "expanded key buffer too small"
    );
    for (chain, start) in starts.iter().enumerate() {
        let first = chain * stride;
        let len = (count - first).min(stride);
        let mut current = zeroize::Zeroizing::new(start.secret_bytes());
        for offset in 0..len {
            if offset > 0 {
                let mut carried = true;
                for byte in current.iter_mut().rev() {
                    let (next, overflow) = byte.overflowing_add(1);
                    *byte = next;
                    if !overflow {
                        carried = false;
                        break;
                    }
                }
                ensure!(!carried, "chain wrapped past 2^256");
            }
            destination[(first + offset) * 32..][..32].copy_from_slice(current.as_ref());
        }
    }
    Ok(())
}

/// Runs the startup vectors through `backend` and checks every address on CPU.
pub(crate) fn run_self_test<B: AddressBackend>(
    backend: &mut B,
    verifier: &Secp256k1<All>,
    capacity: usize,
) -> Result<()> {
    let stride = backend.increment_stride().max(1);
    for keys in gpu_self_test_batches(stride, capacity)? {
        let mut addresses = vec![[0; 20]; keys.len()];
        backend.derive_batch(&chain_starts(&keys, stride), &mut addresses)?;
        for (key, address) in keys.iter().zip(&addresses) {
            cpu::verify_address(key, address, verifier)?;
        }
    }
    Ok(())
}

/// Consecutive scalars from 64 upwards: accepted as chain starts for every
/// supported stride (≤ 64), so tests can slice any prefix into chains.
#[cfg(test)]
pub(crate) fn sequential_test_keys(count: usize) -> Result<Vec<SecretKey>> {
    let mut bytes = [0; 32];
    bytes[31] = 64;
    let mut key = SecretKey::from_byte_array(bytes)?;
    let mut keys = Vec::with_capacity(count);
    for index in 0..count {
        if index > 0 {
            key = increment_secret_key(&key)
                .ok_or_else(|| anyhow::anyhow!("sequential test key overflow"))?;
        }
        keys.push(key);
    }
    Ok(keys)
}

/// A batch is either fully derived or rejected. No output may be used on error.
/// Implementations own their compute resources, never search state or file I/O.
///
/// `keys` are chain starts, not one scalar per address: address `j` belongs to
/// `keys[j / stride] + j % stride` (see [`chain_key`]), and
/// `keys.len() == chain_count(addresses.len(), stride)`. With stride 1 the two
/// views coincide. Hosts only generate, hold, upload, and wipe the starts.
pub(crate) trait AddressBackend {
    /// Only the CPU reference implementation opts out; new accelerators must
    /// keep independent CPU verification before publishing search candidates.
    const VERIFY_CANDIDATES: bool = true;

    fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()>;

    fn inflight_capacity(&self) -> usize {
        1
    }

    /// Consecutive scalars per chain start. `1` keeps independent CSPRNG keys
    /// and one scalar multiplication per address.
    fn increment_stride(&self) -> usize {
        1
    }

    /// Submits `count` addresses from the chain starts in `keys` without waiting.
    fn begin_batch(&mut self, keys: &[SecretKey], count: usize) -> Result<()> {
        let _ = (keys, count);
        bail!("begin_batch is GPU-only")
    }

    fn end_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        self.derive_batch(keys, addresses)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum BackendChoice {
    Auto,
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

pub(crate) enum GpuBackend {
    Metal(metal::MetalBackend),
    Cuda(Box<cuda::CudaBackend>),
    Vulkan(Box<vulkan::VulkanBackend>),
}

impl GpuBackend {
    pub(crate) fn kind_name(&self) -> &'static str {
        match self {
            Self::Metal(_) => "metal",
            Self::Cuda(_) => "cuda",
            Self::Vulkan(_) => "vulkan",
        }
    }

    pub(crate) fn device_name(&self) -> String {
        match self {
            Self::Metal(backend) => backend.device_name(),
            Self::Cuda(backend) => backend.device_name(),
            Self::Vulkan(backend) => backend.device_name(),
        }
    }
}

impl AddressBackend for GpuBackend {
    fn inflight_capacity(&self) -> usize {
        match self {
            Self::Metal(backend) => backend.inflight_capacity(),
            Self::Cuda(backend) => backend.inflight_capacity(),
            Self::Vulkan(backend) => backend.inflight_capacity(),
        }
    }

    fn increment_stride(&self) -> usize {
        match self {
            Self::Metal(backend) => backend.increment_stride(),
            Self::Cuda(backend) => backend.increment_stride(),
            Self::Vulkan(backend) => backend.increment_stride(),
        }
    }

    fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        match self {
            Self::Metal(backend) => backend.derive_batch(keys, addresses),
            Self::Cuda(backend) => backend.derive_batch(keys, addresses),
            Self::Vulkan(backend) => backend.derive_batch(keys, addresses),
        }
    }

    fn begin_batch(&mut self, keys: &[SecretKey], count: usize) -> Result<()> {
        match self {
            Self::Metal(backend) => backend.begin_batch(keys, count),
            Self::Cuda(backend) => backend.begin_batch(keys, count),
            Self::Vulkan(backend) => backend.begin_batch(keys, count),
        }
    }

    fn end_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        match self {
            Self::Metal(backend) => backend.end_batch(keys, addresses),
            Self::Cuda(backend) => backend.end_batch(keys, addresses),
            Self::Vulkan(backend) => backend.end_batch(keys, addresses),
        }
    }
}

pub(crate) enum Selection {
    Cpu { fallback: bool },
    Gpu(Box<GpuBackend>),
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resolved {
    Cpu { fallback: bool },
    Metal,
    Cuda,
    Vulkan,
}

/// Only absence is recoverable. Compilation, self-test and runtime failures are fatal.
#[cfg(test)]
fn resolve(choice: BackendChoice, metal: bool, cuda: bool, vulkan: bool) -> Result<Resolved> {
    match choice {
        BackendChoice::Cpu => Ok(Resolved::Cpu { fallback: false }),
        BackendChoice::Metal if metal => Ok(Resolved::Metal),
        BackendChoice::Metal => {
            bail!("Metal unavailable: no accessible GPU or unsupported platform")
        }
        BackendChoice::Cuda if cuda => Ok(Resolved::Cuda),
        BackendChoice::Cuda => {
            bail!("CUDA unavailable: no accessible GPU or unsupported platform")
        }
        BackendChoice::Vulkan if vulkan => Ok(Resolved::Vulkan),
        BackendChoice::Vulkan => {
            bail!("Vulkan unavailable: no accessible GPU or unsupported platform")
        }
        BackendChoice::Auto if metal => Ok(Resolved::Metal),
        BackendChoice::Auto if cuda => Ok(Resolved::Cuda),
        BackendChoice::Auto if vulkan => Ok(Resolved::Vulkan),
        BackendChoice::Auto => Ok(Resolved::Cpu { fallback: true }),
    }
}

pub(crate) fn select(choice: BackendChoice, capacity: usize) -> Result<Selection> {
    match choice {
        BackendChoice::Cpu => Ok(Selection::Cpu { fallback: false }),
        BackendChoice::Metal => match metal::MetalBackend::new(capacity)? {
            Some(backend) => Ok(Selection::Gpu(Box::new(GpuBackend::Metal(backend)))),
            None => bail!("Metal unavailable: no accessible GPU or unsupported platform"),
        },
        BackendChoice::Cuda => match cuda::CudaBackend::new(capacity)? {
            Some(backend) => Ok(Selection::Gpu(Box::new(GpuBackend::Cuda(Box::new(
                backend,
            ))))),
            None => bail!("CUDA unavailable: no accessible GPU or unsupported platform"),
        },
        BackendChoice::Vulkan => match vulkan::VulkanBackend::new(capacity)? {
            Some(backend) => Ok(Selection::Gpu(Box::new(GpuBackend::Vulkan(Box::new(
                backend,
            ))))),
            None => bail!("Vulkan unavailable: no accessible GPU or unsupported platform"),
        },
        BackendChoice::Auto => {
            if let Some(backend) = metal::MetalBackend::new(capacity)? {
                return Ok(Selection::Gpu(Box::new(GpuBackend::Metal(backend))));
            }
            if let Some(backend) = cuda::CudaBackend::new(capacity)? {
                return Ok(Selection::Gpu(Box::new(GpuBackend::Cuda(Box::new(
                    backend,
                )))));
            }
            if let Some(backend) = vulkan::VulkanBackend::new(capacity)? {
                return Ok(Selection::Gpu(Box::new(GpuBackend::Vulkan(Box::new(
                    backend,
                )))));
            }
            Ok(Selection::Cpu { fallback: true })
        }
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
pub(crate) mod metal {
    use super::*;

    pub(crate) struct MetalBackend;

    impl MetalBackend {
        pub(crate) fn new(_capacity: usize) -> Result<Option<Self>> {
            Ok(None)
        }

        pub(crate) fn device_name(&self) -> String {
            unreachable!("Metal is unavailable on this platform")
        }
    }

    impl AddressBackend for MetalBackend {
        fn derive_batch(&mut self, _: &[SecretKey], _: &mut [Address]) -> Result<()> {
            bail!("Metal is unavailable on this platform")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn increment_secret_key_walks_valid_scalars() {
        let mut one = [0; 32];
        one[31] = 1;
        let start = SecretKey::from_byte_array(one).unwrap();
        let two = increment_secret_key(&start).unwrap();
        assert_eq!(two.secret_bytes()[31], 2);
        let mut last = secp256k1::constants::CURVE_ORDER;
        last[31] -= 1;
        let last = SecretKey::from_byte_array(last).unwrap();
        assert!(increment_secret_key(&last).is_none());
    }

    #[test]
    fn offset_secret_key_carries_across_limbs_and_stops_at_the_order() {
        let mut bytes = [0; 32];
        bytes[30] = 0x01;
        bytes[31] = 0xff;
        let start = SecretKey::from_byte_array(bytes).unwrap();
        let moved = offset_secret_key(&start, 3).unwrap().secret_bytes();
        assert_eq!((moved[30], moved[31]), (0x02, 0x02));
        assert_eq!(offset_secret_key(&start, 0).unwrap(), start);
        let mut near = secp256k1::constants::CURVE_ORDER;
        near[31] -= 5;
        let near = SecretKey::from_byte_array(near).unwrap();
        assert!(offset_secret_key(&near, 4).is_some());
        assert!(offset_secret_key(&near, 5).is_none());
        let mut all_ff = [0xff; 32];
        all_ff[0] = 0x7f;
        let big = SecretKey::from_byte_array(all_ff).unwrap();
        assert!(offset_secret_key(&big, 1).is_some());
    }

    #[test]
    fn chain_helpers_index_and_validate_starts() {
        let keys = sequential_test_keys(70).unwrap();
        let starts = chain_starts(&keys, 32);
        assert_eq!(starts.len(), 3);
        assert_eq!(chain_count(70, 32), 3);
        assert_eq!(chain_count(64, 32), 2);
        assert_eq!(chain_count(0, 32), 0);
        assert_eq!(chain_count(5, 1), 5);
        for (index, key) in keys.iter().enumerate() {
            assert_eq!(chain_key(&starts, 32, index).unwrap(), *key);
        }
        assert!(chain_key(&starts, 32, 96).is_err());
        assert!(check_chain_batch(&starts, 70, 32).is_ok());
        assert!(check_chain_batch(&starts, 64, 32).is_err());
        assert!(check_chain_batch(&[], 0, 32).is_ok());

        let scalar = |value: u8| {
            let mut bytes = [0; 32];
            bytes[31] = value;
            SecretKey::from_byte_array(bytes).unwrap()
        };
        // Single-element chains never add, so 1 and n-1 stay acceptable.
        assert!(chain_start_accepted(&scalar(1), 1));
        assert!(!chain_start_accepted(&scalar(1), 2));
        assert!(!chain_start_accepted(&scalar(31), 32));
        assert!(chain_start_accepted(&scalar(32), 32));
        let mut last = secp256k1::constants::CURVE_ORDER;
        last[31] -= 1;
        let last = SecretKey::from_byte_array(last).unwrap();
        assert!(chain_start_accepted(&last, 1));
        assert!(!chain_start_accepted(&last, 2));
        let mut fits = secp256k1::constants::CURVE_ORDER;
        fits[31] -= 32;
        assert!(chain_start_accepted(
            &SecretKey::from_byte_array(fits).unwrap(),
            32
        ));
        let mut too_short = secp256k1::constants::CURVE_ORDER;
        too_short[31] -= 31;
        assert!(!chain_start_accepted(
            &SecretKey::from_byte_array(too_short).unwrap(),
            32
        ));
    }

    #[test]
    fn expand_chain_keys_matches_per_address_scalars() {
        let keys = sequential_test_keys(70).unwrap();
        let starts = chain_starts(&keys, 32);
        let mut bytes = vec![0xaa; 70 * 32 + 7];
        expand_chain_keys(&starts, 32, 70, &mut bytes).unwrap();
        for (index, key) in keys.iter().enumerate() {
            assert_eq!(&bytes[index * 32..][..32], &key.secret_bytes()[..]);
        }
        assert!(bytes[70 * 32..].iter().all(|&byte| byte == 0xaa));
        assert!(expand_chain_keys(&starts, 32, 64, &mut bytes).is_err());
        assert!(expand_chain_keys(&starts, 32, 70, &mut bytes[..100]).is_err());
        let mut carry = [0xff; 32];
        carry[0] = 0x7f;
        carry[30] = 0xff;
        carry[31] = 0xfe;
        let start = SecretKey::from_byte_array(carry).unwrap();
        let mut out = vec![0; 3 * 32];
        expand_chain_keys(&[start], 3, 3, &mut out).unwrap();
        assert_eq!(&out[64..][29..], &[0x00, 0x00, 0x00]);
        assert_eq!(out[64 + 28], 0x00);
        assert_eq!(out[64], 0x80);
        let mut single = vec![0; 32];
        expand_chain_keys(&[], 32, 0, &mut single).unwrap();
    }

    #[test]
    fn selection_only_falls_back_when_device_is_absent() {
        assert_eq!(
            resolve(BackendChoice::Cpu, true, true, true).unwrap(),
            Resolved::Cpu { fallback: false }
        );
        assert_eq!(
            resolve(BackendChoice::Auto, true, true, true).unwrap(),
            Resolved::Metal
        );
        assert_eq!(
            resolve(BackendChoice::Auto, false, true, true).unwrap(),
            Resolved::Cuda
        );
        assert_eq!(
            resolve(BackendChoice::Auto, false, false, true).unwrap(),
            Resolved::Vulkan
        );
        assert_eq!(
            resolve(BackendChoice::Auto, false, false, false).unwrap(),
            Resolved::Cpu { fallback: true }
        );
        assert_eq!(
            resolve(BackendChoice::Metal, true, false, false).unwrap(),
            Resolved::Metal
        );
        assert_eq!(
            resolve(BackendChoice::Cuda, false, true, false).unwrap(),
            Resolved::Cuda
        );
        assert_eq!(
            resolve(BackendChoice::Vulkan, false, false, true).unwrap(),
            Resolved::Vulkan
        );
        assert_eq!(
            resolve(BackendChoice::Metal, false, true, true)
                .unwrap_err()
                .to_string(),
            "Metal unavailable: no accessible GPU or unsupported platform"
        );
        assert_eq!(
            resolve(BackendChoice::Cuda, true, false, true)
                .unwrap_err()
                .to_string(),
            "CUDA unavailable: no accessible GPU or unsupported platform"
        );
        assert_eq!(
            resolve(BackendChoice::Vulkan, true, true, false)
                .unwrap_err()
                .to_string(),
            "Vulkan unavailable: no accessible GPU or unsupported platform"
        );
    }

    #[test]
    fn explicit_cuda_does_not_fall_back_when_unavailable() {
        if cuda::CudaBackend::new(1024).unwrap().is_none() {
            let error = select(BackendChoice::Cuda, 1024)
                .err()
                .expect("explicit cuda must fail when no device is present");
            assert_eq!(
                error.to_string(),
                "CUDA unavailable: no accessible GPU or unsupported platform"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn explicit_vulkan_does_not_fall_back_when_unavailable() {
        let error = select(BackendChoice::Vulkan, 1024)
            .err()
            .expect("explicit vulkan must fail when no device is present");
        assert_eq!(
            error.to_string(),
            "Vulkan unavailable: no accessible GPU or unsupported platform"
        );
    }
}
