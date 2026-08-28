use std::sync::Arc;

use anyhow::{Result, ensure};
use secp256k1::{All, PublicKey, Secp256k1, SecretKey};
use tiny_keccak::{Hasher, Keccak};

use super::{Address, AddressBackend};

pub(crate) struct CpuBackend {
    secp: Arc<Secp256k1<All>>,
}

impl CpuBackend {
    pub(crate) fn new(secp: Arc<Secp256k1<All>>) -> Self {
        Self { secp }
    }
}

impl AddressBackend for CpuBackend {
    const VERIFY_CANDIDATES: bool = false;

    #[inline]
    fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
        ensure!(
            keys.len() == addresses.len(),
            "batch input/output lengths differ"
        );
        for (key, address) in keys.iter().zip(addresses) {
            *address = derive_address(key, &self.secp);
        }
        Ok(())
    }
}

#[inline]
pub(crate) fn derive_address(key: &SecretKey, secp: &Secp256k1<All>) -> Address {
    let public = PublicKey::from_secret_key(secp, key).serialize_uncompressed();
    address_from_public_key(&public[1..])
}

pub(crate) fn address_from_public_key(public: &[u8]) -> Address {
    let mut keccak = Keccak::v256();
    keccak.update(public);
    let mut hash = [0; 32];
    keccak.finalize(&mut hash);
    hash[12..].try_into().unwrap()
}

pub(crate) fn verify_address(
    key: &SecretKey,
    address: &Address,
    secp: &Secp256k1<All>,
) -> Result<()> {
    // Never include key bytes in diagnostics, including failed GPU self-tests.
    ensure!(
        derive_address(key, secp) == *address,
        "GPU/CPU address verification failed"
    );
    Ok(())
}
