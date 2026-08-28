# Security

`vanity-rs` generates Ethereum private keys on the local machine. Treat every output file as a wallet backup.

## What this tool does and does not do

- Keys are produced with `OsRng` → ChaCha20 and rejection sampling of valid secp256k1 scalars. There is no sequential key scan and no production fixed-seed switch.
- Hits are written to disk with Unix mode `0600`. Default logs omit private keys; `--stdout` prints them.
- The Metal and Vulkan backends are custom finite-field / secp256k1 / Keccak implementations. Start-up self-tests and per-batch CPU checks compare GPU results to libsecp256k1 and tiny-keccak. **Those checks are not a substitute for an independent cryptographic or side-channel audit.**

Do not share `found_wallet.jsonl`, `found_wallet-closest.json`, or terminal output that includes `--stdout`.

## Reporting a vulnerability

Please open a [GitHub security advisory](https://github.com/c1ay/vanity-rs/security/advisories/new) rather than a public issue when the report involves key leakage, incorrect address derivation, or a way to weaken the CSPRNG. Include the affected version or commit and steps to reproduce.
