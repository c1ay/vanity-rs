# GPU 优化实验的实现边界

本轮保留 `AddressBackend::derive_batch` 同步接口。GPU 优化配置只在测试构建中通过环境变量选择；普通 CLI 不提供线程组、算术或流水线调优开关。最终配置依据持续基准决定，见 [性能报告](performance-m4-pro.md)。

## 分段计时
`timing::Observer` 使用泛型静态分派。正式执行使用 `Noop`，时间戳类型为 `()`；逐地址 CPU 路径不增加计时调用或统计锁。测试构建的 `Recorder` 分别累加准备、上传、编码提交、CPU 等待、回读清理、抽样复核、匹配复核和队列等待时间。

GPU 时间来自命令完成后的 `GPUEndTime - GPUStartTime`。零值、非有限值或逆序时间戳记录为不可用，不用 CPU 等待时间替代。流水线中准备和队列等待可能与 GPU 执行重叠，各阶段时间不能相加作为墙钟时间。

## Metal 缓冲区与线程组

批量映射实验在访问前检查整段长度，只取得一次输入指针；每个私钥仍通过 `secret_bytes()` 写入 32 字节槽，不依赖 `SecretKey` 内存布局，也不创建整批明文中间副本。输出一次复制至连续地址切片。所有指针和映射均留在后端内部。

输入清理守卫比 GPU 完成守卫活得更久：命令完成或错误返回前不会读写共享数据；成功、错误及 Rust unwind 后均尽力清理输入。边界错误在创建切片前检查。清理不保证消除编译器副本、寄存器或硬件缓存。

线程组实验检查执行宽度和 pipeline 上限，创建 pipeline 时设置相同的最大线程组容量提示。没有启用要求整 SIMD 组的优化承诺，非整组尾批仍由内核边界检查保护。

## 有限域算术边界

设 `B = 2^32`、`c = B + 977`、`p = B^8 - c`，输入均规范化至 `[0, p−1]`。

快速模加法先求八 limb 和及最高进位 `h`。因为 `a+b ≤ 2p−2 < 2B^8`，`h` 只能是 0 或 1。若 `h=1`，低八 limb 的值 `l ≤ B^8−2c−2`，折叠后 `l+c ≤ p−2`，不会再次溢出 256 位。若 `h=0`，和小于 `B^8 < 2p`，至多减一次 `p` 即可规范化。每个 limb 的加法和进位均容纳于 `u64`；最终用掩码选择。

专用平方逐列处理对角乘积和交叉乘积。每列最多八个有序的 32×32 位乘积，加上上一列小于 `8B` 的进位，总量小于 `8B²+8B < 2^68`。累加器由三个 `u32` 组成，容量为 96 位。交叉乘积只计算一次，再分别加两次，避免在 `u64` 中直接翻倍溢出。逐列输出低 limb 并移位保持精确整数；因为 `a² < B^16`，最后剩余进位可放入第 16 个 limb。最终仍调用原来的模约减。

循环与分支只依赖公开的列/limb 下标，未改变通用乘法、求逆加法链、点运算公式、固定窗口表选择或 Keccak。专用平方额外评估了显式循环展开版本。正确性通过不代表性能更好，也不等于侧信道安全证明。

## 有界准备流水线

一个准备线程与一个 GPU 调度线程共用两份通过所有权转移的主机私钥批次。准备队列容量 1，回收队列容量 2。私钥批次不实现 `Clone` 或 `Debug`。

准备线程使用独立的 `OsRng → ChaCha20Rng` 和合法标量拒绝采样，每生成 1024 项检查取消；队列发送和接收最多等待 10ms 就检查取消。批次序号按 FIFO 验证。搜索在 GPU 完成并匹配/复核后才回收该份存储；双在途时 `begin_batch` 之后立即回收准备队列中的那份，因为搜索循环持有私钥副本直到 `end_batch`。

只统计成功推导的完整批次；提前准备但未提交的私钥不计数。停止期间完成的在途批次仍计数。准备失败、GPU 错误、校验失败、写盘失败或线程异常都会取消搜索。消费者暂存命中，等待准备线程加入后才返回；准备线程的错误或 panic 优先于命中。消费者 unwind 的取消守卫先于 scope 隐式 join 执行，避免后台线程滞留。

普通测试覆盖 FIFO 和两份存储复用、队列取消、未提交批次计数、空匹配/奇数长度/重叠条件、生成失败、生产者及消费者 panic、计算及复核错误，以及命中与随后错误并发的优先级。实际 GPU 测试覆盖运算、整批读写、尾批、容量上限和清理路径；实际 CLI 验证持久化错误与输出兼容。

## 线程组 Montgomery 求逆

`public_jacobian` 写出每点 96 字节的 `X||Y||Z`。`invert_affine_keccak` 在 threadgroup 内做 Montgomery 乘积求逆（零 Z 先换成 1，再把逆元清零），然后仿射化并做 Keccak。尾组使用 `threads_per_threadgroup`，静态数组上限 256。`derive_batch` 仍同步：两个 compute encoder 落在同一 command buffer 上一次等待。该路径通过差分测试，但持续吞吐低于每地址一次 `fe_inverse`，默认关闭（`VANITY_BENCH_INVERT`）。

## 8/16-bit 固定基窗口与增量建表

表大小为 `windows * radix * 64` 字节。4-bit 仍扫描 16 项，地址只依赖公开循环下标。8-bit 按 digit 索引，`offset = (window * 256 + digit) * 16`；16-bit 同理，digit 由两个大端字节组成（`key[31-2w]` 为低字节），表为 16 窗 × 65536 项 = 64 MB。digit 进入地址是性能选择，不是常量时间扫描。

表项为公开常量 `digit * 2^(window_bits*window) * G`，digit 0 保持零。主机建表改为增量：每窗口先算基点 `B_w`，随后 `entry[d] = entry[d-1] + B_w` 逐项点加（`PublicKey::combine`，和不可能到无穷点），窗口间并行。单元测试将增量结果与逐项标量乘对照（4/8-bit 全量、16-bit 抽查边界及随机 digit）。默认 `WINDOW_BITS=16`。

## 线程内分块 Montgomery 求逆

`chunk_invert_affine_keccak` 让每个线程独立处理 `CHUNK_SIZE` 个连续点：正向累积前缀积（线程私有数组，无 barrier、无 threadgroup 内存），一次 `fe_inverse` 后反向展开，把每地址约 270 次域乘的求逆摊薄为 ~270/C + 3 次乘法。零 Z 沿用掩码防御（乘积中换 1，逆元清零）；尾部不足 C 项的线程按 `index < count` 跳过，填充项贡献 1、跳过 inv 更新是精确的。被否决的 threadgroup 版本（串行压在 lane 0）保留为 `VANITY_BENCH_INVERT`，与 chunk 互斥。默认 `CHUNK_SIZE=8`。在 stride=32 的增量路径上，chunk 16/32（整链一次求逆）15 秒初筛与 chunk 8 相差不到 0.3%，未达 3% 保留门槛，默认不改。

## 融合 Jacobian + 分块求逆

默认不再把 `jacobian_points` 的 96 字节/点写到 device 再读回。`chunk_derive_addresses` 在同一线程里算出 Jacobian、做分块求逆并 Keccak，dispatch 宽度为 `ceil(count / C)`，不分配 xyz 缓冲。拆核路径仍可通过 `VANITY_BENCH_FUSE=0` 对照。持续基准中融合在 batch 262144 上约 +20%、65536 上约 +4.6%，超过 3% 保留门槛，默认开启。

## 位交错 Keccak（实验，未启用）

`OPT_KECCAK=1` 把每个 64 位 lane 拆为偶位/奇位两个 32 位字，θ/χ/ι 逐半字操作，ρ 的 64 位旋转变为 32 位旋转（奇数旋转量交换两半）。RC 常量预先交错，吸收/挤出时做位压缩/展开转换。差分正确，但短测中单独约 +1%、叠加在 chunk+16-bit 窗口上约 ±1%，未达 3% 保留门槛，默认关闭（`VANITY_BENCH_KECCAK`）。

## 双在途 GPU 命令

`AddressBackend::derive_batch` 仍是 `begin_batch` + `end_batch`。每套槽有独立 input/output（仅拆核/threadgroup 求逆启用时另有 xyz），只读表共享。`begin` 上传并 commit，不等待；`end` 等待最旧命令、回读、抽样复核、清零该槽 input。Drop 等待所有在途命令。搜索在 `inflight_capacity() > 1` 时先 begin 再在槽满时 end+匹配；私钥副本活到 end。默认两个槽。停止收尾可能包含最多两批 GPU 时间。

## 增量点加（stride=32，`VANITY_BENCH_AFFINE=0` 对照路径）

融合 kernel 对每条链只做一次 `public_jacobian`，随后 `INCREMENT_STRIDE-1` 次 `P += G`（G 取自窗口 0 digit 1，走不完整 `add_mixed`，不再做无穷点/零 digit 选择），再按 chunk 分块求逆。`VANITY_BENCH_STRIDE=1` 回到每地址一次标量乘。Dispatch 宽度为 `ceil(count / stride)`。CPU 后端仍逐钥 `from_secret_key`，不走增量。位交错 Keccak 在增量路径上重测仍约 ±1%，保持关闭。

## 链起点契约（2026-09-10）

`AddressBackend` 的 `keys` 改为**链起点**：地址 `j` 属于 `keys[j / stride] + j % stride`，`keys.len() == ceil(count / stride)`；`begin_batch` 额外携带地址数，`end_batch` 校验它与在途命令一致。主机 `fill_secret_keys` 每链只抽一个 CSPRNG 标量，命中、候选与抽样复核用 `chain_key` 现场重算 `起点 + 偏移`（≤ 31 次字节加法）。Metal 输入缓冲缩为 `ceil(capacity / stride) * 32` 字节，kernel 读 `keys + gid*32`。CUDA/Vulkan 的预编译内核仍按每地址一个标量读取，`expand_chain_keys` 在主机展开后上传，行为不变。

起点接受规则 `chain_start_accepted(k, len)`：`len <= 1` 任意有效标量；否则要求 `k >= len` 且 `k + len - 1 <= n - 1`。这恰好排除两类不完整公式的例外：`P0 = i·G`（`k = i`，倍点）与 `P0 = -i·G`（`k + i = n`，无穷点），对 `P += G` 路径同样充分（`k = 1` 与 `k + i = n - 1` 都包含在内）。自检与差分测试的顺序标量从 64 起，任何前缀都能切成合法链。

动机：调度线程每批（262144）此前约 9.3 ms 主机工作——`write_keys` 2.1、`clone_key_batch` 2.6、8 MB 输入 zeroize 2.45、`KeyBatch` 擦除 0.55、匹配 1.4、回读 0.2——与约 9 ms 的 GPU 批时间相当；生产进程的调度线程 `ps -M` 显示 99% CPU。改起点后前四项各缩小 32 倍，匹配改为按字节早退（`prefix_match_len_bytes`，不再展开 40 个 nibble），回读改为一次整块拷贝。

## 仿射批量加（默认，`OPT_AFFINE=1`）

`chain_affine_addresses`：每线程一条链，`P0 = k·G` 走窗口标量乘得到 Jacobian `(X, Y, Z)`；`i·G`（`1 <= i < stride`）取表 row 0 的 digit `i`（要求 radix > stride，故 4-bit 窗口不支持 stride 32）。所有 `i·G` 是仿射常量，`dx_i = gx_i - x0` 只依赖 `P0`，于是整链可与 `Z` 一起做一次 Montgomery 求逆：

- `zz = Z²`，`zzz = Z³`；`d_i = gx_i·zz - X`，`e_i = gy_i·zzz - Y`；则 `λ_i = e_i / (Z·d_i) = e_i · d_i⁻¹ · Z⁻¹`。
- 前缀积 `prefix[i] = d_1⋯d_i`（填充项贡献 1），`inv = (prefix[S-1]·Z)⁻¹`，`Z⁻¹ = inv·prefix[S-1]`，`inv ← inv·Z`。
- 反向展开：`d_i⁻¹ = inv·prefix[i-1]`，`inv ← inv·d_i`；`x_i = λ² - x0 - gx_i`，`y_i = λ(x0 - x_i) - y0`。`d_i`、`e_i` 反向时重算（各 1 次乘），换取只存 `prefix[S]`（256 uint，与 chunk-8 路径的 `pts[8]+prefix[8]` 相当）。

每地址约 11 次域乘 + `fe_inverse/32`，对照 `P += G` 路径的 11（mixed add）+ ~34（chunk-8 求逆分摊）+ 6（仿射化）。整链域乘约 1790 → 约 800。例外由主机起点规则排除，kernel 不再做零 Z / 零 d 掩码；错误地址只会浪费工作，命中与候选仍经 CPU 复核。差分测试覆盖 stride 8/32/64、8/16-bit 窗口与全部批次尾部，并与 `affine=false` 的旧路径同批对照。

## 链点驻留（默认，`OPT_PERSIST=1`）

第一批仍从密钥做 `public_jacobian`。kernel 把 `d_S` 一并纳入 Montgomery 乘积，反向展开时把仿射 `(k+S)·G` 写入每槽 64 字节的 state 缓冲（不是密钥）。后续 `begin_resumed(..., resume=true)` 不再上传标量，从 state 读 Z=1 的 P0，再走同一套 `P0 + i·G`。主机 pipeline 把同一套链起点按 stride 进位；进位失败则重抽并重新 init。`derive_batch` 始终 `resume=false`，测试与自检不受驻留影响。接受规则加严为 `chain_start_accepted(k, stride+1)`（排除 `k=S` 的倍点与 `k+S=n` 的无穷点），并要求约 `2^20` 批的余量。CUDA/Vulkan 忽略 resume，每批仍从当前起点展开。

## simdgroup 求逆（默认，`OPT_SIMD=1`）

每条链仍累积自己的 `Z·∏d_i`。32 宽 simdgroup 做前后缀积，只在 lane 31 调用一次 `fe_inverse`，再 shuffle 回各 lane。Dispatch 宽度上取整到 32，空 lane 贡献 1。这与已否决的 threadgroup 串行求逆不同：后者把整组压在 lane 0，这里是并行 scan。`VANITY_BENCH_SIMD=0` 回到每链一次求逆。

## 拆核 Keccak（默认，`VANITY_BENCH_KSPLIT=1`）

`chain_affine_points` 只写仿射坐标（64 字节/地址），`keccak_points` 每地址一个线程做哈希，避免 Keccak 状态与 `prefix[S]` 挤在同一线程的寄存器里。12 秒交错测量约 +31%（60.3M → 79.3M 地址/秒）。`VANITY_BENCH_KSPLIT=0` 回到融合路径。位交错 Keccak 叠在拆核上约 +3%，未达稳定保留门槛，默认仍关。
