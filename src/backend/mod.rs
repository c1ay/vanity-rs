use anyhow::{Result, bail};
use clap::ValueEnum;
use secp256k1::SecretKey;

pub(crate) mod cpu;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) mod metal;

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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub(crate) enum BackendChoice {
    Auto,
    Cpu,
    Metal,
}

pub(crate) enum Selection<T> {
    Cpu { fallback: bool },
    Metal(T),
}

/// Only absence is recoverable. Compilation, self-test and runtime failures are fatal.
pub(crate) fn select<T>(
    choice: BackendChoice,
    initialize: impl FnOnce() -> Result<Option<T>>,
) -> Result<Selection<T>> {
    if choice == BackendChoice::Cpu {
        return Ok(Selection::Cpu { fallback: false });
    }
    match initialize()? {
        Some(backend) => Ok(Selection::Metal(backend)),
        None if choice == BackendChoice::Auto => Ok(Selection::Cpu { fallback: true }),
        None => bail!("Metal unavailable: no accessible GPU or unsupported platform"),
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
        assert!(matches!(
            select::<()>(BackendChoice::Cpu, || panic!("must not initialize")),
            Ok(Selection::Cpu { fallback: false })
        ));
        assert!(matches!(
            select(BackendChoice::Auto, || Ok(Some(42))),
            Ok(Selection::Metal(42))
        ));
        assert!(matches!(
            select::<()>(BackendChoice::Auto, || Ok(None)),
            Ok(Selection::Cpu { fallback: true })
        ));
        assert!(select::<()>(BackendChoice::Metal, || Ok(None)).is_err());
        for choice in [BackendChoice::Auto, BackendChoice::Metal] {
            let result = select::<()>(choice, || bail!("self-test failed"));
            assert_eq!(result.err().unwrap().to_string(), "self-test failed");
        }
    }
}
