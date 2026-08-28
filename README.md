# vanity-rs

EVM vanity address generator: hex prefix/suffix filters, CPU parallelism, and Apple Silicon Metal GPU search.

[中文说明](README.zh.md)

## Usage

```sh
cargo run --release -- --prefix dead --suffix beef
cargo run --release -- --backend metal --gpu-batch-size 65536 --prefix abc
cargo run --release -- --backend cpu --workers 14 --suffix abc
```

`--backend` defaults to `auto`: Metal when a device is available, otherwise CPU. Explicit `--backend metal` does not fall back. Shader compile, self-test, and runtime verification failures always abort; they are not silently replaced by CPU. `auto` chooses by device availability, not by speed.

Metal is enabled on macOS ARM64 and has been measured on M4 Pro. Other platforms keep the CPU backend. MSL is embedded and compiled at startup; no offline `metal` compiler is required. First launch includes compile time, a fixed base-point table, and a known-vector self-test.

`--workers` applies only to CPU; GPU mode warns and ignores it. `--gpu-batch-size` defaults to **262144** (range **1–262144**). On M4 Pro, sustained search at 262144 is about **6.01 million addresses/s** with 8-bit fixed-base windows and two in-flight GPU commands. There is no startup auto-tune.

Batches of 65536 or more overlap the next random-key batch on one CPU thread with GPU compute. Smaller batches stay synchronous. There are two host key batches and two Metal in/out buffer pairs (at most two in-flight GPU commands). CPU and GPU address search are not mixed. Threadgroups stay at 128 on M4 Pro; dedicated square, fast modular add, bulk-map, and threadgroup Montgomery invert experiments did not show a stable gain and are off.

For faster stop response, use `--gpu-batch-size 65536` (observed stop-tail median ~19 ms vs ~50 ms at 262144 with two in-flight commands; observations, not worst-case bounds). Pairing results: [M4 Pro report](docs/performance-m4-pro.md#结构优化2026-08-28第三轮实施).

Hits append to `found_wallet.jsonl` by default; the best candidate is `found_wallet-closest.json`. Use `--format txt` and `--append` for text files. Logs omit private keys unless you pass `--stdout`.

Requires Rust 1.85+ (`edition = "2024"`).

## Backend boundary

`backend::AddressBackend` takes a slice of valid `SecretKey`s and fills a same-length slice of 20-byte addresses. Success means the whole batch completed; on error the output must not be used. Backends do not match, draw progress, or write files.

- `backend::cpu` uses libsecp256k1 and tiny-keccak. CPU workers dispatch one-address batches.
- `backend::metal` owns the device, runtime compile, shared buffers, and sync. MSL does exact integer field arithmetic, fixed-window base-point multiplication, and Ethereum Keccak-256.
- `search` owns RNG, matching, ranking, counters, and cancel. GPU uses one dispatch thread plus an optional key-prep thread.
- `main` owns the CLI, UI, and files. Closest-candidate snapshots are replaced atomically via a temp file with Unix mode `0600`.

`tries` is the hit index in that compute stream. CPU `worker_id` is the thread index; GPU uses 0. `Total tries` counts completed addresses. A GPU batch of 4096 that hits on the first item can record `tries = 1` while completed work is 4096.

## Keys and correctness

Each worker seeds ChaCha20 from OsRng and rejection-samples valid scalars. There is no sequential private-key scan and no production fixed-seed switch.

GPU startup runs a known-vector self-test. Each batch spot-checks one item on CPU. Every hit or published best candidate is recomputed independently. Matching always uses the same Rust logic. A verification failure stops the search and does not publish that batch.

GPU secret input buffers are wiped after work, including error paths. Host key storage is best-effort wiped. That does not prove compiler copies, registers, or caches are gone, and constant-time-looking source is not a side-channel proof. **This custom GPU crypto has not had an independent security audit. Differential tests are not an audit.**

See [SECURITY.md](SECURITY.md).

## Tests

CPU-only (CI):

```sh
cargo build
cargo test
cargo test --release
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
```

Hardware tests fail if no Metal device is present; they do not skip as pass:

```sh
cargo test --release metal_differential -- --ignored --nocapture
cargo test --release --test cli metal_cli_compatibility_and_persistence -- --ignored --nocapture
```

Sustained benches use the production search loop (RNG, match, CPU verify, candidate snapshot; no terminal draw). Warmup 3 s; default three rounds of 30 s; output under `target/gpu-verification/`:

```sh
VANITY_BENCH_BACKEND=cpu VANITY_BENCH_WORKERS=14 \
  cargo test --release --bin vanity-rs benchmark_backends -- --ignored --nocapture
VANITY_BENCH_BACKEND=metal VANITY_BENCH_BATCH=262144 \
  cargo test --release --bin vanity-rs benchmark_backends -- --ignored --nocapture
```

Those variables are test-only. Bench files store counts and times, not keys. `VANITY_BENCH_PROFILE=1` enables diagnostic timing; `VANITY_BENCH_PIPELINE=0|1` compares sync vs pipeline. Other experiment switches (`VANITY_BENCH_BULK`, `VANITY_BENCH_ADD`, `VANITY_BENCH_SQUARE`, `VANITY_BENCH_GROUP`, `VANITY_BENCH_INVERT`, `VANITY_BENCH_WINDOW`, `VANITY_BENCH_INFLIGHT`) are unused by the normal binary.

Implementation bounds: [GPU design notes](docs/gpu-optimization-design.md). Measurements: [M4 Pro report](docs/performance-m4-pro.md).

## License

MIT. See [LICENSE](LICENSE).
