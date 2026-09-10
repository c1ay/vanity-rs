# vanity-rs

EVM vanity address generator: hex prefix/suffix filters, CPU parallelism, Apple Silicon Metal, CUDA, and Vulkan GPU search.

[中文说明](README.zh.md)

## Usage

```sh
cargo run --release -- --prefix dead --suffix beef
cargo run --release -- --backend metal --gpu-batch-size 65536 --prefix abc
cargo run --release -- --backend cuda --gpu-batch-size 65536 --prefix abc
cargo run --release -- --backend vulkan --gpu-batch-size 65536 --prefix abc
cargo run --release -- --backend cpu --workers 14 --suffix abc
```

`--backend` defaults to `auto`: Metal on macOS ARM when a device is available, otherwise CUDA when an NVIDIA driver is available, otherwise Vulkan on Linux/Windows when a suitable compute device is available, otherwise CPU. Explicit `--backend metal`, `--backend cuda`, or `--backend vulkan` does not fall back. Shader compile, self-test, and runtime verification failures always abort; they are not silently replaced by CPU. `auto` chooses by device availability, not by speed.

Metal is enabled on macOS ARM64 and has been measured on M4 Pro. CUDA loads the NVIDIA driver at runtime (`libcuda.so.1` / `nvcuda.dll`); it is skipped on macOS so it never competes with Metal, and it does not require the CUDA Toolkit or `nvcc` to build or run. Vulkan is compiled in on all platforms, loads the system Vulkan loader at runtime (`libvulkan.so.1` / `vulkan-1.dll`), and is skipped on macOS so it never competes with Metal. Other platforms without a usable GPU keep the CPU backend. MSL is embedded and compiled at startup; CUDA ships precompiled PTX (`src/backend/shader.ptx`, rebuilt with `nvcc -ptx -arch=compute_60 -o src/backend/shader.ptx src/backend/shader.cu`); Vulkan ships a precompiled SPIR-V module (`src/backend/shader.spv`, rebuilt with `glslangValidator -V --target-env vulkan1.1 -o src/backend/shader.spv src/backend/shader.comp`). First launch includes table setup and a known-vector self-test.

`--workers` applies only to CPU; GPU mode warns and ignores it. `--gpu-batch-size` defaults to **262144** (range **1–262144**). On M4 Pro the Metal path uses 16-bit fixed-base windows, two in-flight GPU commands, and increment chains of 32 with affine batched addition (`k·G` once, then `P0 + i·G` for `i < 32` from the fixed-base table). After the first batch the GPU keeps affine chain starts and steps by `32·G`; Keccak runs as a second kernel at one thread per address. A 12-second measurement of this kernel was about **79 million addresses/s** (see the [M4 Pro report](docs/performance-m4-pro.md)). There is no startup auto-tune.

Hosts only generate, upload, and wipe one **chain start** per 32 addresses; Metal reuses those starts across batches (the GPU stores affine `k·G` and adds `32·G`). The CPU recomputes `start + offset` only for hits and candidates. CUDA/Vulkan still expand starts on the host each batch. Batches of 65536 or more overlap the next start batch on one CPU thread with GPU compute. Smaller batches stay synchronous. There are two host key batches and two Metal in/out buffer pairs (at most two in-flight GPU commands). CPU and GPU address search are not mixed. Threadgroups stay at 128 on M4 Pro; dedicated square, fast modular add, bulk-map, and threadgroup Montgomery invert experiments did not show a stable gain and are off.

For faster stop response, use `--gpu-batch-size 65536` (observed stop-tail median ~12 ms vs ~40 ms at 262144 with two in-flight fused commands; observations, not worst-case bounds). Pairing results: [M4 Pro report](docs/performance-m4-pro.md#窗口位宽与拆核融合2026-08-28第四轮测量).

Hits append to `found_wallet.jsonl` by default; the best candidate is `found_wallet-closest.json`. Use `--format txt` and `--append` for text files. Logs omit private keys unless you pass `--stdout`. The summary line shows elapsed time, overall search rate, progress versus the geometric mean, and ETA (mean / 50% / 95%); worker lines show a recent sample rate.

Requires Rust 1.85+ (`edition = "2024"`).

## Backend boundary

`backend::AddressBackend` takes a slice of valid chain-start `SecretKey`s (one per `increment_stride()` addresses; address `j` belongs to `keys[j / stride] + j % stride`) and fills a slice of 20-byte addresses. Success means the whole batch completed; on error the output must not be used. Backends do not match, draw progress, or write files.

- `backend::cpu` uses libsecp256k1 and tiny-keccak. CPU workers dispatch one-address batches.
- `backend::metal` owns the device, runtime compile, shared buffers, and sync. MSL does exact integer field arithmetic, fixed-window base-point multiplication, Ethereum Keccak-256, and persistent affine chain points across batches.
- `backend::cuda` owns the CUDA context, precompiled PTX, dual streams, and events. The CUDA kernel is the previous Metal path (16-bit windows, fused `P += G` chains with chunk-8 invert, block size 128) and still reads one scalar per address, so the host expands chain starts before upload. Requires an NVIDIA driver; compute capability 6.0 or newer.
- `backend::vulkan` owns the instance, device, precompiled SPIR-V, host-visible slots, and fences. The GLSL kernel matches the CUDA path (16-bit windows, fused chunk-8 invert, host-expanded keys). NVIDIA/Intel Vulkan devices may work; AMD is the validation target.
- `search` owns RNG, matching, ranking, counters, and cancel. GPU uses one dispatch thread plus an optional key-prep thread.
- `main` owns the CLI, UI, and files. Closest-candidate snapshots are replaced atomically via a temp file with Unix mode `0600`.

`tries` is the hit index in that compute stream. CPU `worker_id` is the thread index; GPU uses 0. `Total tries` counts completed addresses. A GPU batch of 4096 that hits on the first item can record `tries = 1` while completed work is 4096.

## Keys and correctness

Each worker seeds ChaCha20 from OsRng and rejection-samples valid scalars. GPU batches then walk a short increment chain from each CSPRNG start (`k, k+1, …, k+31` by default) so the kernel can add `i·G` instead of repeating a full scalar multiplication. Metal then keeps those chain points and continues `k+32, k+33, …` on later batches. Starts are rejected when `k < chain length` or `k + chain length - 1 > n - 1`, which are exactly the doubling/infinity exceptions of the incomplete addition formulas (probability about 2^-250 per draw). Persistent Metal chains also require room for `+stride·G` and about 2^20 further batches. There is no scan from a low-entropy origin and no production fixed-seed switch.

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

Hardware tests fail if no matching GPU is present; they do not skip as pass. CUDA hardware acceptance needs Linux/Windows with an NVIDIA driver.

```sh
cargo test --release metal_differential -- --ignored --nocapture
cargo test --release cuda_differential -- --ignored --nocapture
cargo test --release vulkan_differential -- --ignored --nocapture
cargo test --release --test cli metal_cli_compatibility_and_persistence -- --ignored --nocapture
cargo test --release --test cli cuda_cli_compatibility_and_persistence -- --ignored --nocapture
cargo test --release --test cli vulkan_cli_compatibility_and_persistence -- --ignored --nocapture
```

Sustained benches use the production search loop (RNG, match, CPU verify, candidate snapshot; no terminal draw). Warmup 3 s; default three rounds of 30 s; output under `target/gpu-verification/`:

```sh
VANITY_BENCH_BACKEND=cpu VANITY_BENCH_WORKERS=14 \
  cargo test --release --bin vanity-rs benchmark_backends -- --ignored --nocapture
VANITY_BENCH_BACKEND=metal VANITY_BENCH_BATCH=262144 \
  cargo test --release --bin vanity-rs benchmark_backends -- --ignored --nocapture
VANITY_BENCH_BACKEND=cuda VANITY_BENCH_BATCH=262144 \
  cargo test --release --bin vanity-rs benchmark_backends -- --ignored --nocapture
VANITY_BENCH_BACKEND=vulkan VANITY_BENCH_BATCH=262144 \
  cargo test --release --bin vanity-rs benchmark_backends -- --ignored --nocapture
```

Those variables are test-only. Bench files store counts and times, not keys. `VANITY_BENCH_PROFILE=1` enables diagnostic timing; `VANITY_BENCH_PIPELINE=0|1` compares sync vs pipeline. Other experiment switches (`VANITY_BENCH_BULK`, `VANITY_BENCH_ADD`, `VANITY_BENCH_SQUARE`, `VANITY_BENCH_GROUP`, `VANITY_BENCH_INVERT`, `VANITY_BENCH_WINDOW`, `VANITY_BENCH_INFLIGHT`, `VANITY_BENCH_CHUNK`, `VANITY_BENCH_KECCAK`, `VANITY_BENCH_FUSE`, `VANITY_BENCH_STRIDE`, `VANITY_BENCH_AFFINE`, `VANITY_BENCH_KSPLIT`, `VANITY_BENCH_PERSIST`, `VANITY_BENCH_SIMD`) are unused by the normal binary.

Implementation bounds: [GPU design notes](docs/gpu-optimization-design.md). Measurements: [M4 Pro report](docs/performance-m4-pro.md).

## License

MIT. See [LICENSE](LICENSE).
