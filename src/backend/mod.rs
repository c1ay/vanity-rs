use anyhow::{Result, bail};
use clap::ValueEnum;
use secp256k1::SecretKey;

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

/// A batch is either fully derived or rejected. No output may be used on error.
/// Implementations own their compute resources, never search state or file I/O.
pub(crate) trait AddressBackend {
    /// Only the CPU reference implementation opts out; new accelerators must
    /// keep independent CPU verification before publishing search candidates.
    const VERIFY_CANDIDATES: bool = true;

    fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()>;

    fn inflight_capacity(&self) -> usize {
        1
    }

    fn begin_batch(&mut self, keys: &[SecretKey]) -> Result<()> {
        let _ = keys;
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

    fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        match self {
            Self::Metal(backend) => backend.derive_batch(keys, addresses),
            Self::Cuda(backend) => backend.derive_batch(keys, addresses),
            Self::Vulkan(backend) => backend.derive_batch(keys, addresses),
        }
    }

    fn begin_batch(&mut self, keys: &[SecretKey]) -> Result<()> {
        match self {
            Self::Metal(backend) => backend.begin_batch(keys),
            Self::Cuda(backend) => backend.begin_batch(keys),
            Self::Vulkan(backend) => backend.begin_batch(keys),
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
