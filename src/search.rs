use crate::backend::{Address, AddressBackend, cpu};
use crate::timing::{Noop, Observer, Stage};
use anyhow::{Context, Result, ensure};
use crossbeam::channel;
use rand::{CryptoRng, RngCore, SeedableRng, rngs::OsRng};
use rand_chacha::ChaCha20Rng;
use secp256k1::{All, Secp256k1, SecretKey};
use serde::Serialize;
use std::{
    cmp::Reverse,
    collections::VecDeque,
    sync::{
        Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    },
    time::Instant,
};
use time::OffsetDateTime;

mod pipeline;

// Three sustained paired rounds passed the retention gate on M4 Pro.
// Smaller batches keep synchronous preparation to avoid queue overhead.
const PIPELINE_ENABLED: bool = true;
const MIN_PIPELINED_BATCH: usize = 65_536;

pub(crate) fn address_nibbles(address: &Address) -> [u8; 40] {
    let mut nibbles = [0; 40];
    for (byte, pair) in address.iter().zip(nibbles.chunks_exact_mut(2)) {
        pair[0] = byte >> 4;
        pair[1] = byte & 15;
    }
    nibbles
}

pub(crate) struct StopOnExit<'a>(pub(crate) &'a AtomicBool);

impl Drop for StopOnExit<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

#[derive(Default)]
pub(crate) struct Targets {
    pub(crate) prefix: Vec<u8>,
    pub(crate) suffix: Vec<u8>,
}

pub(crate) struct WorkerContext<'a> {
    pub(crate) worker_id: usize,
    pub(crate) stop: &'a AtomicBool,
    pub(crate) total: &'a AtomicU64,
    pub(crate) best: &'a BestState,
    pub(crate) updates: &'a channel::Sender<()>,
    pub(crate) targets: &'a Targets,
    pub(crate) report_every: u64,
    pub(crate) verifier: &'a Secp256k1<All>,
}

pub(crate) struct Progress {
    pub(crate) completed: u64,
    pub(crate) total: u64,
    pub(crate) rate: f64,
    pub(crate) sample: Address,
}

struct TryCounter<'a> {
    global: &'a AtomicU64,
    pending: u64,
}

impl TryCounter<'_> {
    fn flush(&mut self) -> u64 {
        flush_global_tries(self.global, &mut self.pending)
    }
}

impl Drop for TryCounter<'_> {
    fn drop(&mut self) {
        self.flush();
    }
}

// SecretKey does not promise secure erasure. Wipe the storage of our batch on
// every exit (including unwinding), after the backend has finished using it.
struct KeyBatch(Vec<SecretKey>);

impl Drop for KeyBatch {
    fn drop(&mut self) {
        for key in &mut self.0 {
            key.non_secure_erase();
        }
    }
}

pub(crate) fn run_worker<B: AddressBackend>(
    backend: &mut B,
    batch_size: usize,
    context: WorkerContext<'_>,
    progress: impl FnMut(Progress),
) -> Result<Option<HitRecord>> {
    run_worker_observed(backend, batch_size, context, progress, Noop)
}

fn seed_rng() -> Result<ChaCha20Rng> {
    let mut seed = zeroize::Zeroizing::new([0; 32]);
    OsRng
        .try_fill_bytes(seed.as_mut())
        .context("cannot seed search CSPRNG")?;
    Ok(ChaCha20Rng::from_seed(*seed))
}

pub(crate) fn run_gpu_worker<B: AddressBackend>(
    backend: &mut B,
    batch_size: usize,
    context: WorkerContext<'_>,
    progress: impl FnMut(Progress),
) -> Result<Option<HitRecord>> {
    if PIPELINE_ENABLED && batch_size >= MIN_PIPELINED_BATCH {
        pipeline::run(backend, batch_size, context, progress, Noop, seed_rng)
    } else {
        run_worker(backend, batch_size, context, progress)
    }
}

fn run_worker_observed<B: AddressBackend, O: Observer>(
    backend: &mut B,
    batch_size: usize,
    context: WorkerContext<'_>,
    progress: impl FnMut(Progress),
    observer: O,
) -> Result<Option<HitRecord>> {
    let _stop_on_exit = StopOnExit(context.stop);
    let mut rng = seed_rng()?;
    run_with_rng_observed(backend, batch_size, context, progress, &mut rng, observer)
}

#[cfg(test)]
fn run_with_rng<B: AddressBackend>(
    backend: &mut B,
    batch_size: usize,
    context: WorkerContext<'_>,
    progress: impl FnMut(Progress),
    rng: &mut (impl RngCore + CryptoRng),
) -> Result<Option<HitRecord>> {
    run_with_rng_observed(backend, batch_size, context, progress, rng, Noop)
}

trait KeySource {
    fn next(&mut self, stop: &AtomicBool) -> Result<Option<&[SecretKey]>>;
    fn recycle(&mut self, stop: &AtomicBool) -> Result<()>;
}

struct SequentialSource<'a, R, O> {
    rng: &'a mut R,
    keys: KeyBatch,
    size: usize,
    observer: O,
}

impl<R: RngCore + CryptoRng, O: Observer> KeySource for SequentialSource<'_, R, O> {
    #[inline]
    fn next(&mut self, stop: &AtomicBool) -> Result<Option<&[SecretKey]>> {
        let started = self.observer.start();
        for key in &mut self.keys.0 {
            key.non_secure_erase();
            *key = generate_secret_key(self.rng);
        }
        while self.keys.0.len() < self.size {
            self.keys.0.push(generate_secret_key(self.rng));
        }
        self.observer.finish(Stage::Prepare, started);
        Ok((!stop.load(Ordering::Relaxed)).then_some(&self.keys.0))
    }
    #[inline]
    fn recycle(&mut self, _: &AtomicBool) -> Result<()> {
        Ok(())
    }
}

fn run_with_rng_observed<B: AddressBackend, O: Observer>(
    backend: &mut B,
    batch_size: usize,
    context: WorkerContext<'_>,
    progress: impl FnMut(Progress),
    rng: &mut (impl RngCore + CryptoRng),
    observer: O,
) -> Result<Option<HitRecord>> {
    ensure!(
        (1..=crate::backend::MAX_GPU_BATCH_SIZE as usize).contains(&batch_size),
        "invalid search batch size"
    );
    let mut source = SequentialSource {
        rng,
        keys: KeyBatch(Vec::with_capacity(batch_size)),
        size: batch_size,
        observer: observer.clone(),
    };
    run_with_source(
        backend,
        batch_size,
        context,
        progress,
        &mut source,
        &observer,
    )
}

fn clone_key_batch(keys: &[SecretKey]) -> Result<KeyBatch> {
    let mut copied = Vec::with_capacity(keys.len());
    for key in keys {
        copied.push(
            SecretKey::from_byte_array(key.secret_bytes())
                .map_err(|_| anyhow::anyhow!("cannot copy search key"))?,
        );
    }
    Ok(KeyBatch(copied))
}

fn run_with_source<B: AddressBackend, S: KeySource, O: Observer>(
    backend: &mut B,
    batch_size: usize,
    context: WorkerContext<'_>,
    mut progress: impl FnMut(Progress),
    source: &mut S,
    observer: &O,
) -> Result<Option<HitRecord>> {
    let _stop_on_exit = StopOnExit(context.stop);
    ensure!(
        (1..=crate::backend::MAX_GPU_BATCH_SIZE as usize).contains(&batch_size),
        "invalid search batch size"
    );
    if backend.inflight_capacity() > 1 {
        return run_inflight_source(backend, batch_size, context, progress, source, observer);
    }
    let mut addresses = vec![[0; 20]; batch_size];
    let mut counter = TryCounter {
        global: context.total,
        pending: 0,
    };
    let mut completed = 0u64;
    let start = Instant::now();
    let mut last_report = start;
    let mut last_report_count = 0;

    while !context.stop.load(Ordering::Relaxed) {
        let Some(keys) = source.next(context.stop)? else {
            break;
        };
        if context.stop.load(Ordering::Relaxed) {
            break;
        }
        backend.derive_batch(keys, &mut addresses)?;
        let base = completed;
        completed = completed
            .checked_add(batch_size as u64)
            .context("attempt counter overflow")?;
        counter.pending += batch_size as u64;
        if counter.pending >= GLOBAL_TRY_FLUSH {
            counter.flush();
        }
        if context.stop.load(Ordering::Relaxed) {
            break;
        }

        // Evaluate a whole batch before publishing anything. In particular, an
        // invalid proposed hit must not leave a partial snapshot from this batch.
        let matched = observer.start();
        let (best, hit) = evaluate_batch(&addresses, context.targets);
        if let Some(index) = hit
            && B::VERIFY_CANDIDATES
        {
            cpu::verify_address(&keys[index], &addresses[index], context.verifier)?;
        }
        if let Some(candidate) = best {
            let index = candidate.index;
            let nibbles = address_nibbles(&addresses[index]);
            let improved = context.best.consider_checked(
                &nibbles,
                &keys[index],
                base + index as u64 + 1,
                candidate.prefix,
                candidate.suffix,
                || {
                    if B::VERIFY_CANDIDATES {
                        cpu::verify_address(&keys[index], &addresses[index], context.verifier)?;
                    }
                    Ok(())
                },
            )?;
            if improved {
                let _ = context.updates.try_send(());
            }
        }
        observer.finish(Stage::MatchVerify, matched);
        if crossed_report_threshold(last_report_count, completed, context.report_every) {
            let elapsed = last_report.elapsed().as_secs_f64();
            progress(Progress {
                completed,
                total: counter.flush(),
                rate: (completed - last_report_count) as f64 / elapsed.max(1e-6),
                sample: addresses[batch_size - 1],
            });
            last_report = Instant::now();
            last_report_count = completed;
        }

        if let Some(index) = hit {
            if context.stop.swap(true, Ordering::SeqCst) {
                break;
            }
            return Ok(Some(HitRecord {
                address: nibbles_to_hex(&address_nibbles(&addresses[index])),
                private_key: sk_to_hex(&keys[index]),
                tries: base + index as u64 + 1,
                elapsed_sec: start.elapsed().as_secs_f64(),
                worker_id: context.worker_id,
                ts_utc: OffsetDateTime::now_utc()
                    .format(&time::format_description::well_known::Rfc3339)
                    .unwrap_or_else(|_| "NA".into()),
            }));
        }
        source.recycle(context.stop)?;
    }
    Ok(None)
}

fn run_inflight_source<B: AddressBackend, S: KeySource, O: Observer>(
    backend: &mut B,
    batch_size: usize,
    context: WorkerContext<'_>,
    mut progress: impl FnMut(Progress),
    source: &mut S,
    observer: &O,
) -> Result<Option<HitRecord>> {
    let cap = backend.inflight_capacity();
    let mut addresses = vec![[0; 20]; batch_size];
    let mut held = VecDeque::with_capacity(cap);
    let mut counter = TryCounter {
        global: context.total,
        pending: 0,
    };
    let mut completed = 0u64;
    let start = Instant::now();
    let mut last_report = start;
    let mut last_report_count = 0;
    let mut hit = None;
    while hit.is_none() {
        while held.len() < cap && !context.stop.load(Ordering::Relaxed) {
            let Some(keys) = source.next(context.stop)? else {
                break;
            };
            backend.begin_batch(keys)?;
            held.push_back(clone_key_batch(keys)?);
            source.recycle(context.stop)?;
        }
        let Some(keys) = held.pop_front() else {
            break;
        };
        backend.end_batch(&keys.0, &mut addresses)?;
        let base = completed;
        completed = completed
            .checked_add(batch_size as u64)
            .context("attempt counter overflow")?;
        counter.pending += batch_size as u64;
        if counter.pending >= GLOBAL_TRY_FLUSH {
            counter.flush();
        }
        let matched = observer.start();
        let (best, found) = evaluate_batch(&addresses, context.targets);
        if let Some(index) = found
            && B::VERIFY_CANDIDATES
        {
            cpu::verify_address(&keys.0[index], &addresses[index], context.verifier)?;
        }
        if let Some(candidate) = best {
            let index = candidate.index;
            let nibbles = address_nibbles(&addresses[index]);
            let improved = context.best.consider_checked(
                &nibbles,
                &keys.0[index],
                base + index as u64 + 1,
                candidate.prefix,
                candidate.suffix,
                || {
                    if B::VERIFY_CANDIDATES {
                        cpu::verify_address(&keys.0[index], &addresses[index], context.verifier)?;
                    }
                    Ok(())
                },
            )?;
            if improved {
                let _ = context.updates.try_send(());
            }
        }
        observer.finish(Stage::MatchVerify, matched);
        if crossed_report_threshold(last_report_count, completed, context.report_every) {
            let elapsed = last_report.elapsed().as_secs_f64();
            progress(Progress {
                completed,
                total: counter.flush(),
                rate: (completed - last_report_count) as f64 / elapsed.max(1e-6),
                sample: addresses[batch_size - 1],
            });
            last_report = Instant::now();
            last_report_count = completed;
        }
        if let Some(index) = found {
            context.stop.store(true, Ordering::SeqCst);
            hit = Some(HitRecord {
                address: nibbles_to_hex(&address_nibbles(&addresses[index])),
                private_key: sk_to_hex(&keys.0[index]),
                tries: base + index as u64 + 1,
                elapsed_sec: start.elapsed().as_secs_f64(),
                worker_id: context.worker_id,
                ts_utc: OffsetDateTime::now_utc()
                    .format(&time::format_description::well_known::Rfc3339)
                    .unwrap_or_else(|_| "NA".into()),
            });
        }
    }
    while let Some(keys) = held.pop_front() {
        backend.end_batch(&keys.0, &mut addresses)?;
    }
    Ok(hit)
}

fn crossed_report_threshold(previous: u64, completed: u64, every: u64) -> bool {
    every != 0 && previous / every < completed / every
}

#[derive(Clone, Copy)]
struct BatchCandidate {
    index: usize,
    prefix: usize,
    suffix: usize,
}

impl BatchCandidate {
    fn rank(self) -> (usize, usize, usize, Reverse<usize>) {
        (
            self.prefix + self.suffix,
            self.prefix,
            self.suffix,
            Reverse(self.index),
        )
    }
}

fn evaluate_batch(
    addresses: &[Address],
    targets: &Targets,
) -> (Option<BatchCandidate>, Option<usize>) {
    let mut best: Option<BatchCandidate> = None;
    let mut hit = None;
    for (index, address) in addresses.iter().enumerate() {
        let nibbles = address_nibbles(address);
        let candidate = BatchCandidate {
            index,
            prefix: prefix_match_len(&nibbles, &targets.prefix),
            suffix: suffix_match_len(&nibbles, &targets.suffix),
        };
        if candidate.prefix + candidate.suffix > 0
            && best.is_none_or(|current| candidate.rank() > current.rank())
        {
            best = Some(candidate);
        }
        if hit.is_none()
            && candidate.prefix == targets.prefix.len()
            && candidate.suffix == targets.suffix.len()
        {
            hit = Some(index);
        }
    }
    (best, hit)
}

pub(crate) const GLOBAL_TRY_FLUSH: u64 = 8_192;

pub(crate) fn flush_global_tries(counter: &AtomicU64, pending: &mut u64) -> u64 {
    if *pending == 0 {
        counter.load(Ordering::Relaxed)
    } else {
        let total = counter.fetch_add(*pending, Ordering::Relaxed) + *pending;
        *pending = 0;
        total
    }
}

#[derive(Clone)]
pub(crate) struct BestCandidate {
    pub(crate) score: u32,
    pub(crate) prefix_match: usize,
    pub(crate) suffix_match: usize,
    pub(crate) tries: u64,
    pub(crate) address: String,
    pub(crate) private_key: String,
}

#[derive(Default)]
pub(crate) struct BestState {
    score: AtomicU32,
    candidate: Mutex<Option<BestCandidate>>,
}

impl BestState {
    #[cfg(test)]
    pub(crate) fn cached_score(&self) -> u32 {
        self.score.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn consider(
        &self,
        nibbles: &[u8; 40],
        key: &SecretKey,
        tries: u64,
        prefix: usize,
        suffix: usize,
    ) -> bool {
        self.consider_checked(nibbles, key, tries, prefix, suffix, || Ok(()))
            .unwrap()
    }

    fn consider_checked(
        &self,
        nibbles: &[u8; 40],
        secret_key: &SecretKey,
        tries: u64,
        prefix_match: usize,
        suffix_match: usize,
        verify: impl FnOnce() -> anyhow::Result<()>,
    ) -> anyhow::Result<bool> {
        let score = (prefix_match + suffix_match) as u32;
        // This atomic is only a rejection hint; candidate data stays behind the mutex.
        if score == 0 || score < self.score.load(Ordering::Relaxed) {
            return Ok(false);
        }

        let mut guard = self.candidate.lock().unwrap();
        let rank = (score, prefix_match, suffix_match, Reverse(tries));
        if let Some(current) = guard.as_ref()
            && rank
                <= (
                    current.score,
                    current.prefix_match,
                    current.suffix_match,
                    Reverse(current.tries),
                )
        {
            return Ok(false);
        }

        verify()?;
        *guard = Some(BestCandidate {
            score,
            prefix_match,
            suffix_match,
            tries,
            address: nibbles_to_hex(nibbles),
            private_key: sk_to_hex(secret_key),
        });
        // Publish before unlocking so an older update cannot lower the cached score.
        self.score.store(score, Ordering::Relaxed);
        Ok(true)
    }

    pub(crate) fn snapshot(&self) -> Option<BestCandidate> {
        self.candidate.lock().unwrap().clone()
    }
}

pub(crate) fn prefix_match_len(nibbles: &[u8; 40], pattern: &[u8]) -> usize {
    pattern
        .iter()
        .zip(nibbles.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

pub(crate) fn suffix_match_len(nibbles: &[u8; 40], pattern: &[u8]) -> usize {
    pattern
        .iter()
        .rev()
        .zip(nibbles.iter().rev())
        .take_while(|(a, b)| a == b)
        .count()
}

pub(crate) fn generate_secret_key(rng: &mut (impl RngCore + CryptoRng)) -> SecretKey {
    // Keep the existing rand 0.8 / ChaCha20 stream across the secp256k1 API upgrade.
    // Rejection sampling excludes zero and values outside the curve's scalar range.
    let mut bytes = zeroize::Zeroizing::new([0u8; 32]);
    loop {
        rng.fill_bytes(bytes.as_mut());
        if let Ok(key) = SecretKey::from_byte_array(*bytes) {
            return key;
        }
    }
}

/// 将 40 个 nibbles（0..15）转为小写十六进制字符串（仅在写文件/打印时使用）
pub(crate) fn nibbles_to_hex(nibs: &[u8; 40]) -> String {
    // 比 string::from_utf8(hex::encode(...)) 少一步拷贝；这里手工映射更快
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 42]; // 包含 "0x"
    out[0] = b'0';
    out[1] = b'x';
    for i in 0..40 {
        out[2 + i] = HEX[nibs[i] as usize];
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// 将 SecretKey 转为 "0x" + 64位小写十六进制
pub(crate) fn sk_to_hex(sk: &SecretKey) -> String {
    let b = sk.secret_bytes(); // [32]
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 66];
    out[0] = b'0';
    out[1] = b'x';
    for (i, byte) in b.iter().enumerate() {
        out[2 + i * 2] = HEX[(byte >> 4) as usize];
        out[3 + i * 2] = HEX[(byte & 0x0F) as usize];
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[derive(Clone, Serialize)]
pub(crate) struct HitRecord {
    pub(crate) address: String,
    pub(crate) private_key: String,
    pub(crate) tries: u64,
    pub(crate) elapsed_sec: f64,
    pub(crate) worker_id: usize,
    pub(crate) ts_utc: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    enum Behavior<'a> {
        Correct,
        CorruptHit,
        CorruptBest,
        Fail,
        Cancel(&'a AtomicBool),
    }

    struct TestBackend<'a> {
        cpu: cpu::CpuBackend,
        behavior: Behavior<'a>,
        calls: usize,
    }

    impl AddressBackend for TestBackend<'_> {
        fn derive_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
            self.calls += 1;
            if matches!(self.behavior, Behavior::Fail) {
                anyhow::bail!("injected compute failure");
            }
            self.cpu.derive_batch(keys, addresses)?;
            match self.behavior {
                Behavior::CorruptHit => addresses[0] = [0xab; 20],
                Behavior::CorruptBest => {
                    *addresses.last_mut().unwrap() = [0xaa; 20];
                    addresses.last_mut().unwrap()[19] = 0xab;
                }
                Behavior::Cancel(stop) => stop.store(true, Ordering::Relaxed),
                _ => (),
            }
            Ok(())
        }
    }

    #[test]
    fn batch_completion_counts_work_beyond_first_hit_and_reports_crossed_threshold() {
        let secp = Arc::new(Secp256k1::new());
        let mut backend = TestBackend {
            cpu: cpu::CpuBackend::new(Arc::clone(&secp)),
            behavior: Behavior::Correct,
            calls: 0,
        };
        let stop = AtomicBool::new(false);
        let total = AtomicU64::new(0);
        let best = BestState::default();
        let (updates, _) = channel::bounded(1);
        let mut reports = vec![];
        let hit = run_with_rng(
            &mut backend,
            8,
            WorkerContext {
                worker_id: 0,
                stop: &stop,
                total: &total,
                best: &best,
                updates: &updates,
                targets: &Targets::default(),
                report_every: 5,
                verifier: &secp,
            },
            |p| reports.push(p),
            &mut ChaCha20Rng::from_seed([17; 32]),
        )
        .unwrap()
        .unwrap();
        assert_eq!(hit.tries, 1);
        assert_eq!(total.load(Ordering::Relaxed), 8);
        assert_eq!(reports.len(), 1);
        assert_eq!((reports[0].completed, reports[0].total), (8, 8));
        assert!(stop.load(Ordering::Relaxed));
        assert_eq!(backend.calls, 1);
        assert!(!crossed_report_threshold(8, 9, 5));
        assert!(crossed_report_threshold(8, 16, 5));
        assert!(!crossed_report_threshold(0, 16, 0));
    }

    #[test]
    fn compute_and_verification_errors_never_publish_candidates() {
        let secp = Arc::new(Secp256k1::new());
        for behavior in [Behavior::Fail, Behavior::CorruptHit, Behavior::CorruptBest] {
            let targets = if matches!(behavior, Behavior::CorruptBest) {
                Targets {
                    prefix: vec![10; 40],
                    suffix: vec![],
                }
            } else {
                Targets::default()
            };
            let mut backend = TestBackend {
                cpu: cpu::CpuBackend::new(Arc::clone(&secp)),
                behavior,
                calls: 0,
            };
            let stop = AtomicBool::new(false);
            let total = AtomicU64::new(0);
            let best = BestState::default();
            let (updates, receiver) = channel::bounded(1);
            let result = run_with_rng(
                &mut backend,
                8,
                WorkerContext {
                    worker_id: 0,
                    stop: &stop,
                    total: &total,
                    best: &best,
                    updates: &updates,
                    targets: &targets,
                    report_every: 0,
                    verifier: &secp,
                },
                |_| {},
                &mut ChaCha20Rng::from_seed([17; 32]),
            );
            assert!(result.is_err());
            assert!(stop.load(Ordering::Relaxed));
            assert!(best.snapshot().is_none());
            assert!(receiver.try_recv().is_err());
            assert_eq!(backend.calls, 1);
        }
    }

    #[test]
    fn cancellation_before_and_during_batch_does_not_publish_a_hit() {
        let secp = Arc::new(Secp256k1::new());
        for already_cancelled in [true, false] {
            let stop = AtomicBool::new(already_cancelled);
            let mut backend = TestBackend {
                cpu: cpu::CpuBackend::new(Arc::clone(&secp)),
                behavior: Behavior::Cancel(&stop),
                calls: 0,
            };
            let total = AtomicU64::new(0);
            let best = BestState::default();
            let (updates, receiver) = channel::bounded(1);
            let result = run_with_rng(
                &mut backend,
                8,
                WorkerContext {
                    worker_id: 0,
                    stop: &stop,
                    total: &total,
                    best: &best,
                    updates: &updates,
                    targets: &Targets::default(),
                    report_every: 0,
                    verifier: &secp,
                },
                |_| {},
                &mut ChaCha20Rng::from_seed([17; 32]),
            )
            .unwrap();
            assert!(result.is_none());
            assert_eq!(backend.calls, usize::from(!already_cancelled));
            assert_eq!(
                total.load(Ordering::Relaxed),
                if already_cancelled { 0 } else { 8 }
            );
            assert!(best.snapshot().is_none());
            assert!(receiver.try_recv().is_err());
        }
    }

    #[test]
    fn batch_matching_preserves_nibble_order_overlap_and_ties() {
        let mut addresses = [[0; 20]; 4];
        addresses[0][0] = 0xab;
        addresses[1][0] = 0xac;
        addresses[2][0] = 0xab;
        addresses[2][19] = 0xcd;
        addresses[3] = addresses[2];
        let (best, hit) = evaluate_batch(
            &addresses,
            &Targets {
                prefix: vec![10, 11, 0],
                suffix: vec![12, 13],
            },
        );
        assert_eq!(hit, Some(2));
        let best = best.unwrap();
        assert_eq!((best.index, best.prefix, best.suffix), (2, 3, 2));
        let full = address_nibbles(&addresses[2]).to_vec();
        let (best, hit) = evaluate_batch(
            &addresses,
            &Targets {
                prefix: full.clone(),
                suffix: full,
            },
        );
        assert_eq!(hit, Some(2));
        assert_eq!(best.unwrap().rank().0, 80);
        let (_, hit) = evaluate_batch(&[], &Targets::default());
        assert_eq!(hit, None);
    }

    // This harness measures the real search loop (CSPRNG, matching, validation,
    // counters and atomic snapshots), excluding terminal rendering. No private
    // keys are printed or retained outside the automatically deleted tempdir.
    fn timed_search(
        secp: &Arc<Secp256k1<All>>,
        workers: usize,
        metal: Option<&mut crate::backend::metal::MetalBackend>,
        batch_size: usize,
        duration: std::time::Duration,
        profiled: bool,
        pipelined: bool,
    ) -> Result<serde_json::Value> {
        let stop = AtomicBool::new(false);
        let total = AtomicU64::new(0);
        let best = BestState::default();
        let targets = Targets {
            prefix: vec![0; 40],
            suffix: vec![],
        };
        let (updates, receiver) = channel::bounded(1);
        let barrier = std::sync::Barrier::new(workers + 1);
        let directory = tempfile::tempdir()?;
        let snapshot = directory.path().join("closest.json");
        let mut metal = metal;
        let mut started = Instant::now();
        let mut stopped = started;
        let recorder = crate::timing::Recorder::default();
        std::thread::scope(|scope| -> Result<()> {
            let mut handles = vec![];
            for worker_id in 0..workers {
                let context = WorkerContext {
                    worker_id,
                    stop: &stop,
                    total: &total,
                    best: &best,
                    updates: &updates,
                    targets: &targets,
                    report_every: 0,
                    verifier: secp,
                };
                let barrier = &barrier;
                let gpu = metal.take();
                let recorder = recorder.clone();
                handles.push(scope.spawn(move || {
                    let mut cpu = cpu::CpuBackend::new(Arc::clone(secp));
                    barrier.wait();
                    match gpu {
                        Some(backend) => {
                            if profiled {
                                benchmark_gpu(backend, batch_size, context, recorder, pipelined)
                            } else {
                                benchmark_gpu(backend, batch_size, context, Noop, pipelined)
                            }
                        }
                        None => run_worker(&mut cpu, 1, context, |_| {}),
                    }
                }));
            }
            barrier.wait();
            started = Instant::now();
            let _stop_on_error = StopOnExit(&stop);
            loop {
                let remaining = duration.saturating_sub(started.elapsed());
                if remaining.is_zero() || stop.load(Ordering::Relaxed) {
                    break;
                }
                match receiver.recv_timeout(remaining.min(std::time::Duration::from_millis(100))) {
                    Ok(()) => crate::write_closest_candidate(&snapshot, best.snapshot().as_ref())?,
                    Err(channel::RecvTimeoutError::Timeout) => (),
                    Err(channel::RecvTimeoutError::Disconnected) => break,
                }
            }
            stopped = Instant::now();
            stop.store(true, Ordering::Relaxed);
            for handle in handles {
                ensure!(
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("benchmark worker panicked"))??
                        .is_none(),
                    "unexpected benchmark hit"
                );
            }
            crate::write_closest_candidate(&snapshot, best.snapshot().as_ref())?;
            Ok(())
        })?;
        let elapsed = started.elapsed().as_secs_f64();
        let count = total.load(Ordering::Relaxed);
        ensure!(count > 0, "benchmark completed no work");
        Ok(
            serde_json::json!({"tries":count,"elapsed_sec":elapsed,"keys_per_sec":count as f64/elapsed,
            "stop_sec":stopped.elapsed().as_secs_f64(), "profile": profiled.then(|| recorder.snapshot())}),
        )
    }

    fn benchmark_gpu<O: Observer>(
        backend: &mut crate::backend::metal::MetalBackend,
        batch: usize,
        context: WorkerContext<'_>,
        observer: O,
        pipelined: bool,
    ) -> Result<Option<HitRecord>> {
        struct Observed<'a, O> {
            backend: &'a mut crate::backend::metal::MetalBackend,
            observer: O,
        }
        impl<O: Observer> AddressBackend for Observed<'_, O> {
            fn inflight_capacity(&self) -> usize {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    self.backend.inflight_capacity()
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    1
                }
            }

            fn derive_batch(
                &mut self,
                keys: &[SecretKey],
                addresses: &mut [Address],
            ) -> Result<()> {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    self.backend
                        .derive_observed(keys, addresses, &self.observer)
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    let _ = &self.observer;
                    self.backend.derive_batch(keys, addresses)
                }
            }

            fn begin_batch(&mut self, keys: &[SecretKey]) -> Result<()> {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    self.backend.begin_observed(keys, &self.observer)
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    let _ = keys;
                    self.backend.begin_batch(keys)
                }
            }

            fn end_batch(&mut self, keys: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
                #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                {
                    self.backend.end_observed(keys, addresses, &self.observer)
                }
                #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                {
                    self.backend.end_batch(keys, addresses)
                }
            }
        }
        let mut measured = Observed {
            backend,
            observer: observer.clone(),
        };
        if pipelined {
            pipeline::run(&mut measured, batch, context, |_| {}, observer, seed_rng)
        } else {
            run_worker_observed(&mut measured, batch, context, |_| {}, observer)
        }
    }

    #[test]
    #[ignore = "sustained benchmark; requires Metal for VANITY_BENCH_BACKEND=metal"]
    fn benchmark_backends() -> Result<()> {
        let name = std::env::var("VANITY_BENCH_BACKEND").unwrap_or_else(|_| "cpu".into());
        ensure!(
            name == "cpu" || name == "metal",
            "unknown benchmark backend"
        );
        let number = |key: &str, default: u64| -> Result<u64> {
            std::env::var(key).map_or(Ok(default), |value| {
                value.parse().context("invalid benchmark setting")
            })
        };
        let workers = if name == "metal" {
            1
        } else {
            number("VANITY_BENCH_WORKERS", 14)? as usize
        };
        let batch = number("VANITY_BENCH_BATCH", 4096)? as usize;
        let seconds = number("VANITY_BENCH_SECONDS", 30)?;
        let rounds = number("VANITY_BENCH_ROUNDS", 3)?;
        let profiled = number("VANITY_BENCH_PROFILE", 0)? != 0;
        let pipelined = number(
            "VANITY_BENCH_PIPELINE",
            u64::from(PIPELINE_ENABLED && batch >= MIN_PIPELINED_BATCH),
        )? != 0;
        ensure!(
            workers > 0 && rounds > 0 && seconds > 0,
            "invalid benchmark configuration"
        );
        let initialization = Instant::now();
        let secp = Arc::new(Secp256k1::new());
        let mut metal = if name == "metal" {
            Some(benchmark_metal(batch)?.context("GPU required for benchmark")?)
        } else {
            None
        };
        let initialization_sec = initialization.elapsed().as_secs_f64();
        timed_search(
            &secp,
            workers,
            metal.as_mut(),
            batch,
            std::time::Duration::from_secs(3),
            profiled,
            pipelined,
        )?;
        let mut measurements = vec![];
        for round in 0..rounds {
            let value = timed_search(
                &secp,
                workers,
                metal.as_mut(),
                batch,
                std::time::Duration::from_secs(seconds),
                profiled,
                pipelined,
            )?;
            eprintln!("benchmark {name} workers={workers} batch={batch} round={round}: {value}");
            measurements.push(value);
        }
        let report = serde_json::json!({"backend":name,"workers":workers,"batch_size":batch,
            "initialization_sec":initialization_sec,"measurements":measurements,
            "profiled":profiled,"pipelined":pipelined});
        let directory = std::path::Path::new("target/gpu-verification");
        std::fs::create_dir_all(directory)?;
        std::fs::write(
            directory.join(format!("after-{name}-{workers}-{batch}.json")),
            serde_json::to_vec_pretty(&report)?,
        )?;
        Ok(())
    }

    fn benchmark_metal(batch: usize) -> Result<Option<crate::backend::metal::MetalBackend>> {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            crate::backend::metal::MetalBackend::with_config(
                batch,
                crate::backend::metal::MetalConfig::from_env()?,
            )
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            crate::backend::metal::MetalBackend::new(batch)
        }
    }
}
