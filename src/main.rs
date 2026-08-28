mod backend;
mod search;
mod timing;

use anyhow::{Context, anyhow};
use backend::{BackendChoice, Selection, cpu::CpuBackend, metal::MetalBackend};
use clap::{ArgAction, Parser, ValueEnum};
use console::{Color, Style};
use crossbeam::channel;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, info, warn};
use search::{
    BestCandidate, BestState, HitRecord, Targets, WorkerContext, address_nibbles, nibbles_to_hex,
};
use secp256k1::Secp256k1;
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use tempfile::NamedTempFile;
use time::OffsetDateTime;

const SPINNER_FRAMES: [&str; 4] = ["-", "\\", "|", "/"];

const SOLARIZED_ACCENTS: [Color; 8] = [
    Color::Color256(102), // softened yellow
    Color::Color256(108), // softened orange
    Color::Color256(131), // softened red
    Color::Color256(139), // softened magenta
    Color::Color256(60),  // muted violet
    Color::Color256(66),  // muted blue
    Color::Color256(72),  // muted cyan
    Color::Color256(70),  // muted green
];

const SOLARIZED_BASE1: Color = Color::Color256(242);

fn colorize_summary(msg: impl Into<String>) -> String {
    Style::new()
        .fg(SOLARIZED_BASE1)
        .apply_to(msg.into())
        .to_string()
}

fn colorize_worker(worker_id: usize, msg: impl Into<String>) -> String {
    let color = SOLARIZED_ACCENTS[worker_id % SOLARIZED_ACCENTS.len()];
    Style::new()
        .fg(color)
        .dim()
        .apply_to(msg.into())
        .to_string()
}

fn colorize_note(msg: impl Into<String>) -> String {
    Style::new()
        .fg(Color::Color256(244))
        .apply_to(msg.into())
        .to_string()
}

fn colorize_best(msg: impl Into<String>) -> String {
    Style::new()
        .fg(Color::Color256(243))
        .italic()
        .apply_to(msg.into())
        .to_string()
}

/// 将十六进制字符串转为 nibble 序列（0~15）
fn hex_to_nibbles(s: &str) -> Option<Vec<u8>> {
    let mut v = Vec::with_capacity(s.len());
    for b in s.bytes() {
        let n = match b {
            b'0'..=b'9' => b - b'0',
            b'a'..=b'f' => b - b'a' + 10,
            b'A'..=b'F' => b - b'A' + 10,
            _ => return None,
        };
        v.push(n);
    }
    Some(v)
}

fn is_hex(s: &str) -> bool {
    s.bytes().all(|b| b.is_ascii_hexdigit())
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum OutFmt {
    Json,
    Txt,
}

#[derive(Parser, Debug)]
#[command(
    name = "vanity-rs",
    version,
    about = "EVM vanity address generator with prefix/suffix filters, CPU parallelism, and Apple Silicon Metal GPU support"
)]
struct Args {
    /// Hex prefix without 0x (optional)
    #[arg(long, default_value = "")]
    prefix: String,

    /// Hex suffix without 0x (optional)
    #[arg(long, default_value = "")]
    suffix: String,

    /// Print a progress sample every N attempts per worker (0 disables)
    #[arg(long, default_value_t = 200_000)]
    report_every: u64,

    /// Compute backend: auto prefers Metal and falls back to CPU if unavailable
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
    backend: BackendChoice,

    /// Metal addresses per batch (1..=262144)
    #[arg(long, default_value_t = backend::DEFAULT_GPU_BATCH_SIZE, value_parser = clap::value_parser!(u32).range(1..=backend::MAX_GPU_BATCH_SIZE as i64))]
    gpu_batch_size: u32,

    /// CPU worker threads (defaults to CPU count; ignored on GPU)
    #[arg(long)]
    workers: Option<usize>,

    /// Output file path (JSON Lines by default)
    #[arg(long, default_value = "found_wallet.jsonl")]
    out: String,

    /// Output format: json or txt
    #[arg(long, value_enum, default_value_t = OutFmt::Json)]
    format: OutFmt,

    /// Append to the output file (JSON Lines already appends by default)
    #[arg(long, action = ArgAction::SetTrue)]
    append: bool,

    /// Print the hit, including the private key, to the terminal
    #[arg(long, action = ArgAction::SetTrue)]
    stdout: bool,
}

#[derive(Serialize)]
struct ClosestRecord<'a> {
    address: &'a str,
    private_key: &'a str,
    tries: u64,
    prefix_match: usize,
    suffix_match: usize,
    score: u32,
    updated_utc: String,
}

fn validate_targets(prefix: &str, suffix: &str) -> anyhow::Result<()> {
    if !is_hex(prefix) || !is_hex(suffix) {
        anyhow::bail!("prefix and suffix may only contain hex characters 0-9a-f (no 0x)");
    }
    if prefix.len() > 40 || suffix.len() > 40 {
        anyhow::bail!("prefix and suffix must be at most 40 hex digits");
    }
    // 重叠一致性
    let overlap = prefix.len() as i32 + suffix.len() as i32 - 40;
    if overlap > 0 {
        let pl = prefix.len();
        let p_tail = &prefix[(pl - overlap as usize)..];
        let s_head = &suffix[..overlap as usize];
        if !p_tail.eq_ignore_ascii_case(s_head) {
            anyhow::bail!(
                "impossible filter: prefix tail {} and suffix head {} disagree on {} overlapping digits",
                p_tail,
                s_head,
                overlap
            );
        }
    }
    Ok(())
}

/// 受约束的十六进制位数（0~40），用于概率估计
fn effective_fixed(prefix: &str, suffix: &str) -> usize {
    (prefix.len() + suffix.len()).min(40)
}

fn percent_str(p: f64) -> String {
    if (p - 100.0).abs() < f64::EPSILON {
        return "100%".into();
    }
    if p >= 1e-6 {
        let s = format!("{:.12}", p)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        format!("{}%", s)
    } else {
        format!("{:.6e}%", p)
    }
}

fn closest_output_path(out: &str) -> PathBuf {
    let mut path = PathBuf::from(out);
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("closest");
    let closest_name = format!("{}-closest.json", stem);
    path.set_file_name(closest_name);
    path
}

fn write_closest_candidate(path: &Path, best: Option<&BestCandidate>) -> anyhow::Result<()> {
    if let Some(best) = best {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        // A same-directory temporary file makes replacement atomic. Tempfile creates
        // it with mode 0600 on Unix, so no partial or world-readable key is exposed.
        let mut file = NamedTempFile::new_in(parent)?;
        let record = ClosestRecord {
            address: &best.address,
            private_key: &best.private_key,
            tries: best.tries,
            prefix_match: best.prefix_match,
            suffix_match: best.suffix_match,
            score: best.score,
            updated_utc: OffsetDateTime::now_utc()
                .format(&time::format_description::well_known::Rfc3339)
                .unwrap_or_else(|_| "NA".into()),
        };
        {
            let mut writer = BufWriter::new(file.as_file_mut());
            serde_json::to_writer_pretty(&mut writer, &record)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        file.as_file().sync_all()?;
        file.persist(path).map_err(|error| error.error)?;
        info!("Closest candidate saved to {}", path.display());
    } else {
        match fs::remove_file(path) {
            Ok(()) => info!("Removed stale closest candidate at {}", path.display()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn open_output(path: &str, append: bool) -> std::io::Result<BufWriter<File>> {
    let p = Path::new(path);
    if let Some(dir) = p.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    let mut opts = OpenOptions::new();
    opts.create(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    if append {
        opts.append(true);
    } else {
        opts.truncate(true);
    }
    let f = opts.open(p)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok(BufWriter::new(f))
}

fn receive_hit(
    hits: &channel::Receiver<anyhow::Result<HitRecord>>,
    updates: &channel::Receiver<()>,
    mut save_closest: impl FnMut() -> anyhow::Result<()>,
) -> anyhow::Result<Option<HitRecord>> {
    loop {
        channel::select! {
            recv(hits) -> hit => return hit.ok().transpose(),
            recv(updates) -> update => {
                if update.is_err() {
                    return hits.recv().ok().transpose();
                }
                save_closest()?;
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .try_init();

    let args = Args::parse();

    validate_targets(&args.prefix, &args.suffix)?;

    let initialization_start = Instant::now();
    let selected = backend::select(args.backend, || {
        MetalBackend::new(args.gpu_batch_size as usize)
    })?;
    let (workers, mut metal_backend) = match selected {
        Selection::Cpu { fallback } => {
            if fallback {
                warn!(
                    "Metal unavailable (no accessible GPU or unsupported platform); falling back to CPU"
                );
            }
            info!("Backend: cpu");
            (args.workers.unwrap_or_else(num_cpus::get).max(1), None)
        }
        Selection::Metal(metal) => {
            info!(
                "Backend: metal | Device: {} | Batch: {}",
                metal.device_name(),
                args.gpu_batch_size
            );
            if args.workers.is_some() {
                warn!("--workers only applies to CPU and is ignored with Metal");
            }
            (1, Some(metal))
        }
    };
    info!(
        "Backend initialization: {:.3}s",
        initialization_start.elapsed().as_secs_f64()
    );

    // 预处理为 nibble 序列（大小写对 nibble 无影响）
    let want_prefix = hex_to_nibbles(&args.prefix).unwrap();
    let want_suffix = hex_to_nibbles(&args.suffix).unwrap();

    // 概率估计
    let k = effective_fixed(&args.prefix, &args.suffix);
    let one_in = if k == 0 { 1.0 } else { 16f64.powi(k as i32) };
    let p_pct = 100.0 / one_in;

    info!("=== EVM vanity search ===");
    info!(
        "Prefix     : {}",
        if args.prefix.is_empty() {
            "(none)"
        } else {
            &args.prefix
        }
    );
    info!(
        "Suffix     : {}",
        if args.suffix.is_empty() {
            "(none)"
        } else {
            &args.suffix
        }
    );
    info!("Workers    : {}", workers);
    info!(
        "Report every {} attempts/worker (0 disables)",
        args.report_every
    );
    info!("Fixed hex digits: {} / 40", k);
    info!(
        "Hit chance : {}  (about 1 / {})",
        percent_str(p_pct),
        format_one_in(one_in)
    );
    if k == 0 {
        info!("Expected tries: unconstrained (any address matches)");
    } else {
        // 几何分布：期望 one_in，中位数 ln(2)·one_in，95% 分位 -ln(0.05)·one_in
        info!(
            "Expected tries: mean 1 in {} | median ~{} | 95% ~{}",
            format_one_in(one_in),
            format_one_in(one_in * std::f64::consts::LN_2),
            format_one_in(one_in * (-0.05_f64.ln()))
        );
        info!("Search rate and ETA appear on the summary line after the first samples");
    }
    info!(
        "Output     : {}  (format: {:?}, {})",
        args.out,
        args.format,
        if args.append {
            "append"
        } else {
            "overwrite/as needed"
        }
    );

    let closest_path = closest_output_path(&args.out);
    info!("Closest snapshot: {}", closest_path.display());

    let multi = Arc::new(MultiProgress::new());
    let summary_pb = multi.add(ProgressBar::new_spinner());
    summary_pb.set_style(
        ProgressStyle::with_template("{spinner:.dim} [summary] {msg}")
            .unwrap()
            .tick_strings(&SPINNER_FRAMES),
    );
    summary_pb.enable_steady_tick(Duration::from_millis(120));
    let summary_note = Arc::new(Mutex::new(String::new()));
    {
        let mut note_guard = summary_note.lock().unwrap();
        *note_guard = colorize_note("status: waiting for workers");
        summary_pb.set_message(colorize_summary(format_live_status(
            0.0,
            0,
            one_in,
            note_guard.as_str(),
        )));
    }

    // 输出文件准备
    let mut writer = open_output(
        &args.out,
        args.append || matches!(args.format, OutFmt::Json),
    )
    .with_context(|| format!("cannot open output file: {}", args.out))?;

    // 全局停止标志与结果通道
    let stop = Arc::new(AtomicBool::new(false));
    let (tx, rx) = channel::unbounded::<anyhow::Result<HitRecord>>();
    let secp = Arc::new(Secp256k1::new());

    // Workers batch counters and only notify the main thread when the best improves.
    let global_tries = Arc::new(AtomicU64::new(0));
    let best_candidate = Arc::new(BestState::default());
    let (closest_tx, closest_rx) = channel::bounded::<()>(1);
    let save_closest = || {
        let snapshot = best_candidate.snapshot();
        if let Some(best) = &snapshot {
            *summary_note.lock().unwrap() = colorize_best(format!(
                "closest {} (prefix {}/{}, suffix {}/{})",
                best.address,
                best.prefix_match,
                want_prefix.len(),
                best.suffix_match,
                want_suffix.len()
            ));
        }
        write_closest_candidate(&closest_path, snapshot.as_ref()).with_context(|| {
            format!(
                "cannot save closest-candidate snapshot: {}",
                closest_path.display()
            )
        })
    };

    // 启动工作线程
    let summary_running = Arc::new(AtomicBool::new(true));
    let start_all = Instant::now();
    let summary_updater = {
        let summary_pb = summary_pb.clone();
        let summary_note = Arc::clone(&summary_note);
        let tries = Arc::clone(&global_tries);
        let running = Arc::clone(&summary_running);
        let start_snapshot = start_all;
        let summary_one_in = one_in;
        thread::spawn(move || {
            loop {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                let elapsed = start_snapshot.elapsed().as_secs_f64();
                let total = tries.load(Ordering::Relaxed);
                let note = summary_note.lock().unwrap();
                summary_pb.set_message(colorize_summary(format_live_status(
                    elapsed,
                    total,
                    summary_one_in,
                    &note,
                )));
                thread::sleep(Duration::from_millis(200));
            }
        })
    };
    let scope_result = crossbeam::scope(|scope| -> anyhow::Result<Option<HitRecord>> {
        // Also cancel already-started workers if orchestration or thread creation
        // panics before receive_hit can set the normal shutdown flag.
        let _stop_on_exit = search::StopOnExit(&stop);
        for wid in 0..workers {
            let stop = Arc::clone(&stop);
            let tx = tx.clone();
            let secp = Arc::clone(&secp);
            let targets = Targets {
                prefix: want_prefix.clone(),
                suffix: want_suffix.clone(),
            };
            let global_tries = Arc::clone(&global_tries);
            let best_candidate = Arc::clone(&best_candidate);
            let closest_tx = closest_tx.clone();
            let report_every = args.report_every;
            let batch_size = args.gpu_batch_size as usize;
            let metal = metal_backend.take();
            let label = if metal.is_some() {
                "GPU".to_string()
            } else {
                format!("W{wid}")
            };
            let worker_pb = multi.add(ProgressBar::new_spinner());
            worker_pb.set_style(
                ProgressStyle::with_template("{spinner:.dim} {msg}")
                    .unwrap()
                    .tick_strings(&SPINNER_FRAMES),
            );
            worker_pb.enable_steady_tick(Duration::from_millis(120));
            worker_pb.set_message(colorize_worker(wid, format!("{label} waiting")));
            scope.spawn(move |_| {
                let _stop_on_exit = search::StopOnExit(&stop);
                let context = WorkerContext {
                    worker_id: wid,
                    stop: &stop,
                    total: &global_tries,
                    best: &best_candidate,
                    updates: &closest_tx,
                    targets: &targets,
                    report_every,
                    verifier: &secp,
                };
                let progress = |p: search::Progress| {
                    let address = nibbles_to_hex(&address_nibbles(&p.sample));
                    worker_pb.set_position(p.completed);
                    worker_pb.set_message(colorize_worker(
                        wid,
                        format!(
                            "[{label}] total={} rate={}/s sample={address}",
                            format_compact(p.total as f64),
                            format_compact(p.rate),
                        ),
                    ));
                    debug!(
                        "[{label}] tries={} ({}/s) sample={address}",
                        p.completed,
                        format_compact(p.rate)
                    );
                };
                let result = match metal {
                    Some(mut backend) => {
                        search::run_gpu_worker(&mut backend, batch_size, context, progress)
                    }
                    None => search::run_worker(
                        &mut CpuBackend::new(Arc::clone(&secp)),
                        1,
                        context,
                        progress,
                    ),
                };
                match result {
                    Ok(Some(hit)) => {
                        worker_pb.finish_with_message(colorize_worker(
                            wid,
                            format!("{label} HIT tries={}", hit.tries),
                        ));
                        let _ = tx.send(Ok(hit));
                    }
                    Ok(None) => worker_pb
                        .finish_with_message(colorize_worker(wid, format!("{label} stopped"))),
                    Err(error) => {
                        worker_pb
                            .finish_with_message(colorize_worker(wid, format!("{label} failed")));
                        let _ = tx.send(Err(error.context(format!("{label} search failed"))));
                    }
                }
            });
        }

        drop(tx); // 关闭发送端副本

        // A terminal event stops all producers before the scope joins them. The
        // channel is unbounded but carries at most one event per worker, so an
        // error concurrent with a hit cannot deadlock a worker during shutdown.
        let received = receive_hit(&rx, &closest_rx, &save_closest);
        stop.store(true, Ordering::Relaxed);
        received
    })
    .map_err(|_| anyhow!("worker thread panicked"));

    summary_running.store(false, Ordering::Relaxed);
    let summary_result = summary_updater.join();
    let closest_result = save_closest();
    let received = scope_result??;
    // Do not hide a worker error that arrived after the first terminal event.
    for remaining in rx.try_iter() {
        remaining?;
    }
    summary_result.map_err(|_| anyhow!("summary thread panicked"))?;
    closest_result?;
    let summary_handle = &summary_pb;
    let summary_start = start_all;
    if let Some(rec) = received {
        let elapsed = summary_start.elapsed().as_secs_f64();
        let total = global_tries.load(Ordering::Relaxed);
        summary_handle.finish_with_message(colorize_summary(format!(
            "elapsed {} | match found by worker {} after {} tries ({}/s)",
            format_duration(elapsed),
            rec.worker_id,
            rec.tries,
            format_compact(if elapsed > 0.0 {
                total as f64 / elapsed
            } else {
                0.0
            }),
        )));
        // 写入文件
        match args.format {
            OutFmt::Json => {
                serde_json::to_writer(&mut writer, &rec)?;
                writer.write_all(b"\n")?;
            }
            OutFmt::Txt => {
                writeln!(
                    &mut writer,
                    "=== MATCH FOUND ===\nTime(UTC): {}\nAddress  : {}\nPrivate  : {}\nWorker   : {}\nTries    : {}\nElapsed  : {:.1}s\n===================\n",
                    rec.ts_utc,
                    rec.address,
                    rec.private_key,
                    rec.worker_id,
                    rec.tries,
                    rec.elapsed_sec
                )?;
            }
        }
        writer.flush()?;
        info!("Wrote hit to {}", &args.out);

        if args.stdout {
            info!(
                "=== MATCH FOUND ===\nWorker   : {}\nAddress  : {}\nPrivate  : {}\nTries    : {}  (this worker)\nElapsed  : {:.1}s (this worker)\n===================",
                rec.worker_id, rec.address, rec.private_key, rec.tries, rec.elapsed_sec
            );
        } else {
            info!(
                "Address: {} | Tries: {} | Time(UTC): {}",
                rec.address, rec.tries, rec.ts_utc
            );
        }
    } else {
        let elapsed = summary_start.elapsed().as_secs_f64();
        let total = global_tries.load(Ordering::Relaxed);
        summary_handle.finish_with_message(colorize_summary(format!(
            "elapsed {} | no match found after {} tries ({}/s)",
            format_duration(elapsed),
            total,
            format_compact(if elapsed > 0.0 {
                total as f64 / elapsed
            } else {
                0.0
            }),
        )));
        warn!("No hit (search interrupted).");
    }
    if !summary_pb.is_finished() {
        summary_pb.finish_and_clear();
    }

    let wall = start_all.elapsed().as_secs_f64();
    let total_attempts = global_tries.load(Ordering::Relaxed);
    info!(
        "WallTime: {} | Total tries: {} | Average rate: {}/s",
        format_duration(wall),
        total_attempts,
        format_compact(if wall > 0.0 {
            total_attempts as f64 / wall
        } else {
            0.0
        })
    );

    Ok(())
}

fn format_one_in(v: f64) -> String {
    // 对超大值使用科学计数法，避免巨长字符串
    if v.is_infinite() || v.is_nan() {
        "∞".into()
    } else if v >= 1e9 {
        format!("{:.3e}", v)
    } else if v >= 1.0 {
        format!("{}", v as u128)
    } else {
        format!("{v:.3}")
    }
}

/// 将尝试次数或速率格式为紧凑英文单位
fn format_compact(n: f64) -> String {
    if !n.is_finite() || n < 0.0 {
        return "n/a".into();
    }
    if n >= 1e12 {
        format!("{:.2}T", n / 1e12)
    } else if n >= 1e9 {
        format!("{:.2}B", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.2}M", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.1}k", n / 1e3)
    } else if n >= 10.0 {
        format!("{:.0}", n)
    } else if n > 0.0 {
        format!("{:.1}", n)
    } else {
        "0".into()
    }
}

/// 将秒数格式为短人类可读时长
fn format_duration(secs: f64) -> String {
    if !secs.is_finite() || secs < 0.0 {
        return "n/a".into();
    }
    const YEAR: f64 = 365.25 * 86_400.0;
    if secs >= 100.0 * YEAR {
        return ">100y".into();
    }
    if secs >= YEAR {
        return format!("{:.1}y", secs / YEAR);
    }
    if secs >= 86_400.0 {
        let days = (secs / 86_400.0).floor() as u64;
        let hours = ((secs % 86_400.0) / 3_600.0).floor() as u64;
        return format!("{days}d{hours}h");
    }
    if secs < 60.0 {
        return format!("{:.1}s", secs);
    }
    let total = secs.round() as u64;
    let hours = total / 3_600;
    let minutes = (total % 3_600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m")
    } else {
        format!("{minutes}m{seconds:02}s")
    }
}

/// 几何分布剩余等待：无记忆，期望尝试次数恒为 one_in
fn format_eta(one_in: f64, rate: f64) -> String {
    if !rate.is_finite() || rate <= 0.0 || !one_in.is_finite() {
        return "ETA n/a".into();
    }
    if one_in <= 1.0 {
        return "ETA <1s".into();
    }
    let mean = one_in / rate;
    let median = std::f64::consts::LN_2 * mean;
    let p95 = (-0.05_f64.ln()) * mean;
    format!(
        "ETA ~{} (50% ~{} · 95% ~{})",
        format_duration(mean),
        format_duration(median),
        format_duration(p95)
    )
}

/// 摘要行：已用时间、整体速率、相对期望进度、几何 ETA
fn format_live_status(elapsed: f64, tries: u64, one_in: f64, note: &str) -> String {
    let rate = if elapsed > 0.0 {
        tries as f64 / elapsed
    } else {
        0.0
    };
    let rate_part = if tries == 0 || rate <= 0.0 {
        "measuring rate".to_string()
    } else {
        format!("{}/s", format_compact(rate))
    };
    let progress = if one_in > 0.0 && one_in.is_finite() {
        let ratio = tries as f64 / one_in;
        if ratio >= 10.0 {
            format!("{}x mean", format_compact(ratio))
        } else {
            format!("{:.1}% of mean", ratio * 100.0)
        }
    } else {
        "n/a".into()
    };
    let eta = if tries == 0 || rate <= 0.0 {
        "ETA pending".to_string()
    } else {
        format_eta(one_in, rate)
    };
    let extra = if note.is_empty() {
        String::new()
    } else {
        format!(" | {note}")
    };
    format!(
        "elapsed {} | {} | {} tries | {} | {}{extra}",
        format_duration(elapsed),
        rate_part,
        format_compact(tries as f64),
        progress,
        eta
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{CryptoRng, RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use search::{
        GLOBAL_TRY_FLUSH, StopOnExit, flush_global_tries, generate_secret_key, prefix_match_len,
        sk_to_hex, suffix_match_len,
    };
    use secp256k1::SecretKey;
    use std::io::Read;
    use tiny_keccak::{Hasher, Keccak};

    fn eth_address_nibbles_from_secret(
        key: &SecretKey,
        secp: &Secp256k1<secp256k1::All>,
    ) -> [u8; 40] {
        address_nibbles(&backend::cpu::derive_address(key, secp))
    }

    fn test_key(value: u8) -> SecretKey {
        let mut bytes = [0u8; 32];
        bytes[31] = value;
        SecretKey::from_byte_array(bytes).unwrap()
    }

    #[test]
    fn live_status_formats_rate_progress_and_geometric_eta() {
        assert_eq!(format_compact(6_010_000.0), "6.01M");
        assert_eq!(format_compact(1_200.0), "1.2k");
        assert_eq!(format_duration(12.4), "12.4s");
        assert_eq!(format_duration(754.0), "12m34s");
        assert_eq!(format_duration(3_661.0), "1h01m");
        assert_eq!(format_eta(1.0, 1e6), "ETA <1s");
        let status = format_live_status(10.0, 60_000_000, 16f64.powi(8), "closest 0xab");
        assert!(status.contains("elapsed 10.0s"), "{status}");
        assert!(status.contains("6.00M/s"), "{status}");
        assert!(status.contains("60.00M tries"), "{status}");
        assert!(status.contains("% of mean"), "{status}");
        assert!(status.contains("ETA ~"), "{status}");
        assert!(status.contains("50% ~"), "{status}");
        assert!(status.contains("95% ~"), "{status}");
        assert!(status.contains("closest 0xab"), "{status}");
        let pending = format_live_status(0.2, 0, 65536.0, "");
        assert!(pending.contains("measuring rate"), "{pending}");
        assert!(pending.contains("ETA pending"), "{pending}");
    }

    #[test]
    fn backend_arguments_validate_batch_size_and_preserve_existing_options() {
        let defaults = Args::try_parse_from(["vanity-rs"]).unwrap();
        assert_eq!(defaults.backend, BackendChoice::Auto);
        assert_eq!(defaults.gpu_batch_size, backend::DEFAULT_GPU_BATCH_SIZE);
        let args = Args::try_parse_from([
            "vanity-rs",
            "--backend",
            "metal",
            "--gpu-batch-size",
            "33",
            "--prefix",
            "abc",
            "--workers",
            "10",
        ])
        .unwrap();
        assert_eq!(args.backend, BackendChoice::Metal);
        assert_eq!(args.gpu_batch_size, 33);
        assert_eq!(args.workers, Some(10));
        for valid in ["1", "65536", "131072", "262144"] {
            let parsed = Args::try_parse_from(["vanity-rs", "--gpu-batch-size", valid]).unwrap();
            assert_eq!(parsed.gpu_batch_size.to_string(), valid);
        }
        for invalid in ["0", "262145", "-1", "abc"] {
            assert!(Args::try_parse_from(["vanity-rs", "--gpu-batch-size", invalid]).is_err());
        }
        assert!(Args::try_parse_from(["vanity-rs", "--backend", "cuda"]).is_err());
    }

    #[test]
    fn receiving_a_compute_error_preserves_the_error() {
        let (hit_tx, hit_rx) = channel::unbounded();
        let (_updates, update_rx) = channel::bounded(1);
        hit_tx.send(Err(anyhow!("GPU execution failed"))).unwrap();
        let result = receive_hit(&hit_rx, &update_rx, || Ok(()));
        assert_eq!(result.err().unwrap().to_string(), "GPU execution failed");
    }

    fn candidate() -> BestCandidate {
        BestCandidate {
            score: 3,
            prefix_match: 2,
            suffix_match: 1,
            tries: 42,
            address: format!("0x{}", "ab".repeat(20)),
            private_key: sk_to_hex(&test_key(1)),
        }
    }

    fn hit_record() -> HitRecord {
        let best = candidate();
        HitRecord {
            address: best.address,
            private_key: best.private_key,
            tries: best.tries,
            elapsed_sec: 0.1,
            worker_id: 0,
            ts_utc: "2026-08-27T00:00:00Z".into(),
        }
    }

    #[test]
    fn hex_conversion_handles_empty_case_and_odd_lengths() {
        assert_eq!(hex_to_nibbles(""), Some(vec![]));
        assert_eq!(hex_to_nibbles("0aF19"), Some(vec![0, 10, 15, 1, 9]));
        for invalid in ["0x12", "gg", " a", "a\n", "你好"] {
            assert_eq!(hex_to_nibbles(invalid), None);
        }
    }

    #[test]
    fn target_validation_accepts_consistent_overlaps() {
        let full = "a".repeat(40);
        for (prefix, suffix, fixed) in [
            ("", "", 0),
            ("a", "B", 2),
            (full.as_str(), "", 40),
            ("", full.as_str(), 40),
            (full.as_str(), "AAA", 40),
            (full.as_str(), full.as_str(), 40),
        ] {
            validate_targets(prefix, suffix).unwrap();
            assert_eq!(effective_fixed(prefix, suffix), fixed);
        }
        let prefix = format!("{}ab", "0".repeat(20));
        let suffix = format!("AB{}", "f".repeat(18));
        validate_targets(&prefix, &suffix).unwrap();
    }

    #[test]
    fn target_validation_rejects_invalid_and_conflicting_patterns() {
        let long = "a".repeat(41);
        let full = "a".repeat(40);
        for (prefix, suffix) in [
            ("0x12", ""),
            ("", "g"),
            ("你好", ""),
            (long.as_str(), ""),
            ("", long.as_str()),
            (full.as_str(), "b"),
        ] {
            assert!(validate_targets(prefix, suffix).is_err());
        }
    }

    #[test]
    fn matches_count_from_the_correct_address_edges() {
        let address: [u8; 40] = hex_to_nibbles("0123456789012345678901234567890123456789")
            .unwrap()
            .try_into()
            .unwrap();
        assert_eq!(prefix_match_len(&address, &[]), 0);
        assert_eq!(suffix_match_len(&address, &[]), 0);
        assert_eq!(prefix_match_len(&address, &[0, 1, 2, 15]), 3);
        assert_eq!(suffix_match_len(&address, &[15, 6, 7, 8, 9]), 4);
        assert_eq!(prefix_match_len(&address, &[15]), 0);
        assert_eq!(suffix_match_len(&address, &[0]), 0);
        assert_eq!(prefix_match_len(&address, &address), 40);
        assert_eq!(suffix_match_len(&address, &address), 40);
    }

    #[test]
    fn known_ethereum_address_vectors() {
        let secp = Secp256k1::new();
        for (key, expected) in [
            (1, "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf"),
            (2, "0x2b5ad5c4795c026514f8317c7a215e218dccd6cf"),
        ] {
            let address = eth_address_nibbles_from_secret(&test_key(key), &secp);
            assert_eq!(nibbles_to_hex(&address), expected);
        }
    }

    #[test]
    fn address_stream_matches_the_pre_upgrade_checksum() {
        // Golden digest from secp256k1 0.28.2 over 4096 addresses, not usable wallets.
        let mut rng = ChaCha20Rng::from_seed([91; 32]);
        let secp = Secp256k1::new();
        let mut checksum = Keccak::v256();
        for _ in 0..4096 {
            let key = generate_secret_key(&mut rng);
            checksum.update(&eth_address_nibbles_from_secret(&key, &secp));
        }
        let mut digest = [0u8; 32];
        checksum.finalize(&mut digest);
        assert_eq!(
            digest,
            [
                179, 211, 234, 56, 193, 181, 240, 38, 155, 133, 204, 142, 22, 70, 166, 242, 77,
                138, 108, 127, 206, 47, 68, 153, 14, 220, 102, 45, 249, 41, 35, 104
            ]
        );
    }

    #[test]
    fn secret_generation_rejects_invalid_scalars() {
        // Deliberately non-random input, restricted to this rejection-sampling test.
        struct CandidateRng(std::vec::IntoIter<[u8; 32]>);
        impl RngCore for CandidateRng {
            fn next_u32(&mut self) -> u32 {
                unreachable!("expected a full scalar request")
            }
            fn next_u64(&mut self) -> u64 {
                unreachable!("expected a full scalar request")
            }
            fn fill_bytes(&mut self, dest: &mut [u8]) {
                dest.copy_from_slice(&self.0.next().unwrap());
            }
            fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
                self.fill_bytes(dest);
                Ok(())
            }
        }
        impl CryptoRng for CandidateRng {}
        let valid = test_key(1).secret_bytes();
        let mut rng = CandidateRng(
            vec![[0; 32], secp256k1::constants::CURVE_ORDER, [255; 32], valid].into_iter(),
        );
        assert_eq!(generate_secret_key(&mut rng).secret_bytes(), valid);
        assert!(rng.0.next().is_none());
    }

    #[test]
    fn best_candidate_preserves_ranking_and_tie_breaks() {
        let best = BestState::default();
        let key = test_key(1);
        let address = [0; 40];
        assert!(!best.consider(&address, &key, 1, 0, 0));
        assert!(best.snapshot().is_none());
        assert!(best.consider(&address, &key, 10, 1, 0));
        assert!(!best.consider(&address, &key, 1, 0, 1));
        assert!(!best.consider(&address, &key, 10, 1, 0));
        assert!(best.consider(&address, &key, 9, 1, 0));
        assert!(best.consider(&address, &key, 100, 1, 1));
        assert!(best.consider(&address, &key, 200, 2, 0));
        assert!(!best.consider(&address, &key, 201, 2, 0));
        assert!(!best.consider(&address, &key, 1, 1, 0));
        let saved = best.snapshot().unwrap();
        assert_eq!(
            (saved.score, saved.prefix_match, saved.suffix_match),
            (2, 2, 0)
        );
        assert_eq!(saved.tries, 200);
        assert_eq!(best.cached_score(), saved.score);
    }

    #[test]
    fn concurrent_best_updates_keep_the_score_monotonic() {
        let best = BestState::default();
        let barrier = std::sync::Barrier::new(8);
        thread::scope(|scope| {
            for worker in 0..8 {
                let best = &best;
                let barrier = &barrier;
                scope.spawn(move || {
                    let key = test_key(1);
                    let mut previous = 0;
                    barrier.wait();
                    for iteration in 0..64 {
                        let matches = (iteration + worker) % 8 + 1;
                        best.consider(&[0; 40], &key, iteration as u64 + 10, matches, 0);
                        let observed = best.cached_score();
                        assert!(observed >= previous);
                        previous = observed;
                    }
                    best.consider(&[0; 40], &key, worker as u64, 8, 0);
                });
            }
        });
        let saved = best.snapshot().unwrap();
        assert_eq!((saved.score, saved.prefix_match, saved.tries), (8, 8, 0));
        assert_eq!(best.cached_score(), saved.score);
    }

    #[test]
    fn batched_counters_include_every_workers_final_remainder() {
        let counter = AtomicU64::new(0);
        thread::scope(|scope| {
            for _ in 0..4 {
                let counter = &counter;
                scope.spawn(move || {
                    let mut pending = 0;
                    for _ in 0..GLOBAL_TRY_FLUSH + 17 {
                        pending += 1;
                        if pending >= GLOBAL_TRY_FLUSH {
                            flush_global_tries(counter, &mut pending);
                        }
                    }
                    flush_global_tries(counter, &mut pending);
                    assert_eq!(pending, 0);
                });
            }
        });
        let expected = 4 * (GLOBAL_TRY_FLUSH + 17);
        assert_eq!(counter.load(Ordering::Relaxed), expected);
        assert_eq!(flush_global_tries(&counter, &mut 0), expected);
    }

    #[test]
    fn closest_snapshot_replaces_the_file_and_preserves_the_old_handle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/closest.json");
        let mut best = candidate();
        write_closest_candidate(&path, Some(&best)).unwrap();
        let mut old_file = File::open(&path).unwrap();
        best.tries = 99;
        write_closest_candidate(&path, Some(&best)).unwrap();
        let current: serde_json::Value =
            serde_json::from_reader(File::open(&path).unwrap()).unwrap();
        assert_eq!(current["tries"], 99);
        assert_eq!(current["private_key"], best.private_key);
        assert_eq!(current["score"], best.score);
        let mut old_contents = String::new();
        old_file.read_to_string(&mut old_contents).unwrap();
        let old: serde_json::Value = serde_json::from_str(&old_contents).unwrap();
        assert_eq!(old["tries"], 42);
        assert_eq!(fs::read_dir(path.parent().unwrap()).unwrap().count(), 1);
    }

    #[test]
    fn closest_write_failure_is_reported_and_cleans_up_the_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let blocked = dir.path().join("closest.json");
        fs::create_dir(&blocked).unwrap();
        fs::write(blocked.join("keep"), "existing data").unwrap();
        assert!(write_closest_candidate(&blocked, Some(&candidate())).is_err());
        assert_eq!(
            fs::read_to_string(blocked.join("keep")).unwrap(),
            "existing data"
        );
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn missing_candidate_removes_a_stale_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("closest.json");
        write_closest_candidate(&path, Some(&candidate())).unwrap();
        write_closest_candidate(&path, None).unwrap();
        assert!(!path.exists());
        write_closest_candidate(&path, None).unwrap();
    }

    #[test]
    fn output_preserves_append_and_overwrite_behavior() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/output.txt");
        for (append, text) in [(false, b"first"), (true, b"later")] {
            let mut writer = open_output(path.to_str().unwrap(), append).unwrap();
            writer.write_all(text).unwrap();
            writer.flush().unwrap();
        }
        assert_eq!(fs::read_to_string(&path).unwrap(), "firstlater");
        let mut writer = open_output(path.to_str().unwrap(), false).unwrap();
        writer.write_all(b"replacement").unwrap();
        writer.flush().unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "replacement");
    }

    #[cfg(unix)]
    #[test]
    fn private_key_files_are_restricted_even_when_replacing_existing_files() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        for closest in [false, true] {
            let path = dir
                .path()
                .join(if closest { "closest.json" } else { "hit.jsonl" });
            for existing in [false, true] {
                if existing {
                    fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
                }
                if closest {
                    write_closest_candidate(&path, Some(&candidate())).unwrap();
                } else {
                    open_output(path.to_str().unwrap(), true).unwrap();
                }
                assert_eq!(
                    fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
        }
    }

    #[test]
    fn receiving_a_hit_services_pending_snapshot_notifications() {
        let (hit_tx, hit_rx) = channel::bounded(1);
        let (update_tx, update_rx) = channel::bounded(1);
        update_tx.send(()).unwrap();
        let mut saves = 0;
        let hit = receive_hit(&hit_rx, &update_rx, || {
            saves += 1;
            hit_tx.send(Ok(hit_record()))?;
            Ok(())
        })
        .unwrap()
        .unwrap();
        assert_eq!(saves, 1);
        assert_eq!(hit.address, hit_record().address);
    }

    #[test]
    fn receiving_a_hit_propagates_snapshot_errors() {
        let (_hit_tx, hit_rx) = channel::bounded(1);
        let (update_tx, update_rx) = channel::bounded(1);
        update_tx.send(()).unwrap();
        let result = receive_hit(&hit_rx, &update_rx, || anyhow::bail!("write failed"));
        assert_eq!(result.err().unwrap().to_string(), "write failed");
    }

    #[test]
    fn closed_notification_channel_does_not_lose_a_hit() {
        let (hit_tx, hit_rx) = channel::bounded(1);
        let (update_tx, update_rx) = channel::bounded(1);
        drop(update_tx);
        hit_tx.send(Ok(hit_record())).unwrap();
        assert!(
            receive_hit(&hit_rx, &update_rx, || unreachable!())
                .unwrap()
                .is_some()
        );
        drop(hit_tx);
        assert!(
            receive_hit(&hit_rx, &update_rx, || unreachable!())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn worker_unwind_signals_other_workers_to_stop() {
        let stop = AtomicBool::new(false);
        let result = std::panic::catch_unwind(|| {
            let _stop_on_exit = StopOnExit(&stop);
            panic!("simulated worker failure");
        });
        assert!(result.is_err());
        assert!(stop.load(Ordering::Relaxed));
    }
}
