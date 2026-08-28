use anyhow::{Result, anyhow};
use secp256k1::{All, PublicKey, Secp256k1, SecretKey};

pub(crate) fn table_bytes(window_bits: u8) -> usize {
    let windows = 256 / window_bits as usize;
    let radix = 1usize << window_bits;
    windows * radix * 64
}

/// Uncompressed affine X||Y as little-endian u32 limbs, matching load order in
/// the GPU shaders. These are public fixed-base constants, not wallet keys.
pub(crate) fn write_table_entry(slot: &mut [u8], public: &PublicKey) {
    let public = public.serialize_uncompressed();
    for coordinate in 0..2 {
        for limb in 0..8 {
            let start = 1 + coordinate * 32 + (7 - limb) * 4;
            let word = u32::from_be_bytes(public[start..start + 4].try_into().unwrap());
            slot[coordinate * 32 + limb * 4..][..4].copy_from_slice(&word.to_le_bytes());
        }
    }
}

/// Table entry `(window, digit)` holds `digit * 2^(window_bits*window) * G`;
/// digit 0 stays zero. Entries are built incrementally (one point addition per
/// digit) instead of one full scalar multiplication each: at 16-bit windows the
/// 1M-entry table would otherwise dominate startup. Windows are independent
/// and fill in parallel.
pub(crate) fn build_table(secp: &Secp256k1<All>, window_bits: u8) -> Result<Vec<u8>> {
    let windows = 256 / window_bits as usize;
    let radix = 1usize << window_bits;
    let mut bytes = vec![0u8; table_bytes(window_bits)];
    std::thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::with_capacity(windows);
        for (window, slice) in bytes.chunks_mut(radix * 64).enumerate() {
            workers.push(scope.spawn(move || -> Result<()> {
                let bit = window_bits as usize * window;
                let mut scalar = [0; 32];
                scalar[31 - bit / 8] = 1 << (bit % 8);
                let base = PublicKey::from_secret_key(secp, &SecretKey::from_byte_array(scalar)?);
                let mut entry = base;
                for digit in 1..radix {
                    write_table_entry(&mut slice[digit * 64..(digit + 1) * 64], &entry);
                    if digit + 1 < radix {
                        // digit*B + B < n*B: the sum can never hit infinity.
                        entry = entry
                            .combine(&base)
                            .map_err(|_| anyhow!("table point addition failed"))?;
                    }
                }
                Ok(())
            }));
        }
        for worker in workers {
            worker
                .join()
                .map_err(|_| anyhow!("table builder panicked"))??;
        }
        Ok(())
    })?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn table_byte_counts_match_window_geometry() {
        assert_eq!(table_bytes(4), 64 * 16 * 64);
        assert_eq!(table_bytes(8), 32 * 256 * 64);
        assert_eq!(table_bytes(16), 16 * 65536 * 64);
    }

    #[test]
    fn incremental_table_matches_scalar_multiplication() -> Result<()> {
        let secp = Secp256k1::new();
        let mut rng = ChaCha20Rng::from_seed([71; 32]);
        for window_bits in [4u8, 8, 16] {
            let table = build_table(&secp, window_bits)?;
            assert_eq!(table.len(), table_bytes(window_bits));
            let windows = 256 / window_bits as usize;
            let radix = 1usize << window_bits;
            for window in 0..windows {
                assert!(
                    table[window * radix * 64..window * radix * 64 + 64]
                        .iter()
                        .all(|&byte| byte == 0),
                    "window {window} digit 0 must stay zero"
                );
                // Small radices verify every digit; 16-bit spot-checks borders
                // plus random digits (a full check would be 1M scalar mults).
                let mut digits = vec![1, 2, radix / 2, radix - 2, radix - 1];
                if window_bits == 16 {
                    for _ in 0..6 {
                        digits.push(1 + (rng.next_u32() as usize) % (radix - 1));
                    }
                } else {
                    digits.extend(3..radix - 2);
                }
                for digit in digits {
                    let value = num_bigint::BigUint::from(digit) << (window_bits as usize * window);
                    let mut scalar = [0; 32];
                    let big_endian = value.to_bytes_be();
                    scalar[32 - big_endian.len()..].copy_from_slice(&big_endian);
                    let public =
                        PublicKey::from_secret_key(&secp, &SecretKey::from_byte_array(scalar)?);
                    let mut expected = [0; 64];
                    write_table_entry(&mut expected, &public);
                    let offset = (window * radix + digit) * 64;
                    assert_eq!(
                        &table[offset..offset + 64],
                        &expected[..],
                        "window {window} digit {digit} ({window_bits}-bit)"
                    );
                }
            }
        }
        Ok(())
    }
}
