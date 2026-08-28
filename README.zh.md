# vanity-rs

EVM 靓号地址生成器，支持十六进制前缀/后缀、CPU 并行搜索和 Apple Silicon Metal GPU 计算。

命令行帮助与错误信息为英文。English README: [README.md](README.md)。

## 使用

```sh
cargo run --release -- --prefix dead --suffix beef
cargo run --release -- --backend metal --gpu-batch-size 65536 --prefix abc
cargo run --release -- --backend cpu --workers 14 --suffix abc
```

`--backend` 默认 `auto`：有可用 Metal 设备时使用 GPU；平台不支持或无法访问设备时提示并使用 CPU。显式指定 `metal` 时不回退。内核编译、自检和运行时校验错误始终报错退出，不用 CPU 静默掩盖错误。`auto` 按设备可用性选择，不会根据速度选择后端。

Metal 目前在 macOS ARM64 上启用，已在 M4 Pro 验证；其他平台保留 CPU 实现。MSL 源码嵌入程序并在启动时编译，无需单独的离线 `metal` 编译器。首次启动时间包括编译、固定基点表生成及已知向量自检。

`--workers` 只控制 CPU 线程数；GPU 模式传入该参数会提示忽略。`--gpu-batch-size` 默认 **262144**，支持 **1–262144**。M4 Pro 持续测试中，262144 约为 **1400 万地址/秒**（16-bit 固定基窗口 + 融合分块求逆 + 两个在途 GPU 命令）。没有启动时自动调优。

批次达到 65536 时，程序用一个 CPU 准备线程生成下一批随机私钥，与 GPU 计算重叠；较小批次保持同步路径。有两份主机批次和两套 Metal 输入输出缓冲区（最多两个在途 GPU 命令），不进行 CPU/GPU 混合地址搜索。线程组在 M4 Pro 上继续使用 128；专用平方、快速模加、批量映射和线程组 Montgomery 求逆实验未证明稳定收益，未启用。

更重视停止响应时可显式使用 `--gpu-batch-size 65536`。双在途融合配置下，65536 停止收尾中位数约为 12ms，262144 约为 40ms；这些是观测值，不是最坏情况保证。完整配对结果见 [性能报告](docs/performance-m4-pro.md#窗口位宽与拆核融合2026-08-28第四轮测量)。

默认结果写入 `found_wallet.jsonl`，最佳候选写入 `found_wallet-closest.json`。JSON 输出按行追加；TXT 使用 `--format txt`，可配合 `--append`。普通日志不打印私钥；只有显式 `--stdout` 会打印命中私钥。摘要行会显示已用时间、整体搜索速度、相对几何期望的进度，以及预计等待时间（均值 / 50% / 95%）；工作线程行显示近期采样速率。

需要 Rust 1.85+（`edition = "2024"`）。

## 后端边界

`backend::AddressBackend` 接收有效 `SecretKey` 切片，填写同长度的 20 字节地址切片。调用成功表示整批完成；失败时输出不可使用。后端不处理匹配条件、进度条或文件。

- `backend::cpu` 使用 libsecp256k1 与 tiny-keccak；CPU 工作线程以单元素批次静态分派，保留流式计算方式。
- `backend::metal` 管理设备、运行时编译、共享缓冲区及同步。MSL 执行精确整数有限域运算、固定窗口基点乘法和 Ethereum Keccak-256。
- `search` 共用随机数生成、匹配、候选排名、计数及取消逻辑。GPU 仅用一个调度线程；大批次另用一个线程准备私钥，不与 CPU 地址搜索混跑。
- `main` 负责 CLI、界面和文件输出。候选快照由主线程通过临时文件原子替换，Unix 权限为 `0600`。

输出字段保持兼容。`tries` 是命中地址在对应计算流中的序号；CPU 的 `worker_id` 是线程编号，GPU 的编号为 0。汇总 `Total tries` 统计实际完成的地址数。例如 GPU 批大小 4096、批内第一项命中时，记录的 `tries` 可以是 1，而完成数是 4096。进度报告按跨越阈值触发。

## 私钥与正确性

各工作线程使用 OsRng 播种 ChaCha20Rng，并通过拒绝采样生成独立有效私钥。没有连续私钥搜索、低熵种子或生产环境固定种子开关。

GPU 启动时执行已知向量自检，每批轮换抽样一项由 CPU 复算；每个准备发布的命中或最佳候选还要独立复算地址。匹配始终使用同一套 Rust 逻辑。任何校验失败都停止搜索，不发布该批候选。计算和持久化错误会停止新批次提交，等待在途任务结束后返回错误。

GPU 私钥输入缓冲区在任务完成后清零，包括错误返回路径。新增主机私钥存储也做尽力清理；这不保证清除编译器临时副本、寄存器或硬件缓存。源代码中的固定循环与掩码选择也不等于经过编译后的侧信道安全证明。**该自定义 GPU 密码实现尚未经过独立安全审计，差分测试不能替代安全审计。**

详见 [SECURITY.md](SECURITY.md)。

## 验证

普通测试不要求 GPU：

```sh
cargo build
cargo test
cargo test --release
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings
```

硬件验收必须实际访问 Metal 设备；不存在设备时测试失败，而非视为通过：

```sh
cargo test --release metal_differential -- --ignored --nocapture
cargo test --release --test cli metal_cli_compatibility_and_persistence -- --ignored --nocapture
```

差分测试覆盖有限域边界运算、已知/单比特/随机私钥、公钥及地址逐项比较、非线程组整数倍批次、最大批次和私钥输入缓冲区清理。CLI 测试验证自动选择、输出格式、权限及写盘失败时的退出。

持续基准使用生产搜索循环，包含随机数生成、匹配、CPU 复核和候选快照，排除终端绘制。每组预热 3 秒，默认运行三轮、每轮 30 秒，结果在 `target/gpu-verification/`：

```sh
VANITY_BENCH_BACKEND=cpu VANITY_BENCH_WORKERS=14 \
  cargo test --release --bin vanity-rs benchmark_backends -- --ignored --nocapture
VANITY_BENCH_BACKEND=metal VANITY_BENCH_BATCH=262144 \
  cargo test --release --bin vanity-rs benchmark_backends -- --ignored --nocapture
```

这些环境变量只供测试程序使用。基准文件仅记录计数和时间，不保留生成的私钥。诊断计时可用 `VANITY_BENCH_PROFILE=1`；`VANITY_BENCH_PIPELINE=0|1` 可做同步/流水线对照。其余实验开关为 `VANITY_BENCH_BULK=0|1`、`VANITY_BENCH_ADD=0|1`、`VANITY_BENCH_SQUARE=0|1`、`VANITY_BENCH_GROUP=auto|32|64|128|256`、`VANITY_BENCH_INVERT=0|1`、`VANITY_BENCH_WINDOW=4|8|16`、`VANITY_BENCH_INFLIGHT=1|2`、`VANITY_BENCH_CHUNK=0|4|8`、`VANITY_BENCH_KECCAK=0|1` 和 `VANITY_BENCH_FUSE=0|1`，普通程序不读取这些开关。正式吞吐比较应关闭诊断计时。

所有实验的冻结源码、二进制、配置、哈希、日志和原始数据保存在独立的 `target/gpu-optimization/20260828-m4-stages/`，没有覆盖旧实验。设计和整数上界证明见 [实现说明](docs/gpu-optimization-design.md)。

具体测量与限制见 [M4 Pro 性能报告](docs/performance-m4-pro.md)。

## 许可证

MIT，见 [LICENSE](LICENSE)。
