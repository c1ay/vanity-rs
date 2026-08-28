//! Two owned host batches, one GPU dispatch at a time. The producer knows
//! nothing about Metal, matching, or files. No secret batch is cloned/logged.
use super::*;
use std::time::Duration;

const CANCEL_POLL: Duration = Duration::from_millis(10);

struct PreparedBatch {
    sequence: u64,
    keys: KeyBatch,
}

fn receive<T>(receiver: &channel::Receiver<T>, stop: &AtomicBool) -> Result<Option<T>> {
    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(None);
        }
        match receiver.recv_timeout(CANCEL_POLL) {
            Ok(value) => return Ok(Some(value)),
            Err(channel::RecvTimeoutError::Timeout) => (),
            Err(channel::RecvTimeoutError::Disconnected) => {
                ensure!(
                    stop.load(Ordering::Relaxed),
                    "batch source disconnected unexpectedly"
                );
                return Ok(None);
            }
        }
    }
}

fn send<T>(sender: &channel::Sender<T>, mut value: T, stop: &AtomicBool) -> Result<bool> {
    loop {
        if stop.load(Ordering::Relaxed) {
            return Ok(false);
        }
        match sender.send_timeout(value, CANCEL_POLL) {
            Ok(()) => return Ok(true),
            Err(channel::SendTimeoutError::Timeout(returned)) => value = returned,
            Err(channel::SendTimeoutError::Disconnected(_)) => {
                ensure!(
                    stop.load(Ordering::Relaxed),
                    "batch consumer disconnected unexpectedly"
                );
                return Ok(false);
            }
        }
    }
}

struct PipelineSource<O> {
    ready: channel::Receiver<PreparedBatch>,
    recycle: channel::Sender<PreparedBatch>,
    current: Option<PreparedBatch>,
    expected: u64,
    observer: O,
}

impl<O: Observer> KeySource for PipelineSource<O> {
    fn next(&mut self, stop: &AtomicBool) -> Result<Option<&[SecretKey]>> {
        ensure!(self.current.is_none(), "batch not recycled");
        let waiting = self.observer.start();
        self.current = receive(&self.ready, stop)?;
        self.observer.finish(Stage::QueueWait, waiting);
        let Some(batch) = &self.current else {
            return Ok(None);
        };
        ensure!(
            batch.sequence == self.expected,
            "prepared batch order mismatch"
        );
        self.expected = self
            .expected
            .checked_add(1)
            .context("batch sequence overflow")?;
        Ok(Some(&batch.keys.0))
    }

    fn recycle(&mut self, stop: &AtomicBool) -> Result<()> {
        if let Some(batch) = self.current.take() {
            let waiting = self.observer.start();
            send(&self.recycle, batch, stop)?;
            self.observer.finish(Stage::QueueWait, waiting);
        }
        Ok(())
    }
}

fn produce<R: RngCore + CryptoRng, O: Observer>(
    recycle: channel::Receiver<PreparedBatch>,
    ready: channel::Sender<PreparedBatch>,
    stop: &AtomicBool,
    batch_size: usize,
    observer: O,
    seed: impl FnOnce() -> Result<R>,
) -> Result<()> {
    let _stop_on_exit = StopOnExit(stop);
    let mut rng = seed()?;
    let mut sequence = 0u64;
    loop {
        let waiting = observer.start();
        let next = receive(&recycle, stop)?;
        observer.finish(Stage::QueueWait, waiting);
        let Some(mut batch) = next else {
            break;
        };
        let preparing = observer.start();
        for first in (0..batch_size).step_by(1024) {
            if stop.load(Ordering::Relaxed) {
                return Ok(());
            }
            for index in first..(first + 1024).min(batch_size) {
                if let Some(key) = batch.keys.0.get_mut(index) {
                    key.non_secure_erase();
                    *key = generate_secret_key(&mut rng);
                } else {
                    batch.keys.0.push(generate_secret_key(&mut rng));
                }
            }
        }
        observer.finish(Stage::Prepare, preparing);
        batch.sequence = sequence;
        sequence = sequence.checked_add(1).context("batch sequence overflow")?;
        let waiting = observer.start();
        let sent = send(&ready, batch, stop)?;
        observer.finish(Stage::QueueWait, waiting);
        if !sent {
            break;
        }
    }
    Ok(())
}

pub(super) fn run<B, O, R, F>(
    backend: &mut B,
    batch_size: usize,
    context: WorkerContext<'_>,
    progress: impl FnMut(Progress),
    observer: O,
    seed: F,
) -> Result<Option<HitRecord>>
where
    B: AddressBackend,
    O: Observer,
    R: RngCore + CryptoRng,
    F: FnOnce() -> Result<R> + Send,
{
    let stop = context.stop;
    let _stop_on_exit = StopOnExit(stop);
    ensure!(
        (1..=crate::backend::MAX_GPU_BATCH_SIZE as usize).contains(&batch_size),
        "invalid pipeline batch size"
    );
    let (ready_tx, ready_rx) = channel::bounded(1);
    let (recycle_tx, recycle_rx) = channel::bounded(2);
    for _ in 0..2 {
        recycle_tx.send(PreparedBatch {
            sequence: 0,
            keys: KeyBatch(Vec::with_capacity(batch_size)),
        })?;
    }
    std::thread::scope(|scope| {
        // On consumer unwind this guard cancels before scope's implicit join.
        let _scope_stop = StopOnExit(stop);
        let producer_observer = observer.clone();
        let producer = scope.spawn(move || {
            produce(
                recycle_rx,
                ready_tx,
                stop,
                batch_size,
                producer_observer,
                seed,
            )
        });
        let mut source = PipelineSource {
            ready: ready_rx,
            recycle: recycle_tx,
            current: None,
            expected: 0,
            observer: observer.clone(),
        };
        let result = run_with_source(
            backend,
            batch_size,
            context,
            progress,
            &mut source,
            &observer,
        );
        stop.store(true, Ordering::Relaxed);
        drop(source);
        // Always join before returning a hit; entropy errors/panics must not be
        // hidden by a successful consumer result. Queued keys drop with guards.
        producer
            .join()
            .map_err(|_| anyhow::anyhow!("private-key producer panicked"))??;
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn exercise<B: AddressBackend, R: RngCore + CryptoRng>(
        backend: &mut B,
        seed: impl FnOnce() -> Result<R> + Send,
        targets: &Targets,
        progress: impl FnMut(Progress),
    ) -> (Result<Option<HitRecord>>, u64, BestState) {
        let stop = AtomicBool::new(false);
        let total = AtomicU64::new(0);
        let best = BestState::default();
        let (updates, _) = channel::bounded(1);
        let result = run(
            backend,
            8,
            WorkerContext {
                worker_id: 0,
                stop: &stop,
                total: &total,
                best: &best,
                updates: &updates,
                targets,
                report_every: 1,
                verifier: &Secp256k1::new(),
            },
            progress,
            Noop,
            seed,
        );
        assert!(stop.load(Ordering::Relaxed));
        (result, total.load(Ordering::Relaxed), best)
    }

    #[test]
    fn fifo_reuses_two_batches_without_repeating_keys() -> Result<()> {
        let stop = AtomicBool::new(false);
        let (ready_tx, ready_rx) = channel::bounded(1);
        let (recycle_tx, recycle_rx) = channel::bounded(2);
        for _ in 0..2 {
            recycle_tx.send(PreparedBatch {
                sequence: 0,
                keys: KeyBatch(Vec::with_capacity(33)),
            })?;
        }
        std::thread::scope(|scope| -> Result<()> {
            let guard = StopOnExit(&stop);
            let producer = scope.spawn(|| {
                produce(recycle_rx, ready_tx, &stop, 33, Noop, || {
                    Ok(ChaCha20Rng::from_seed([19; 32]))
                })
            });
            let mut source = PipelineSource {
                ready: ready_rx,
                recycle: recycle_tx,
                current: None,
                expected: 0,
                observer: Noop,
            };
            let mut expected = ChaCha20Rng::from_seed([19; 32]);
            let mut allocations = std::collections::HashSet::new();
            for _ in 0..20 {
                let keys = source.next(&stop)?.unwrap();
                allocations.insert(keys.as_ptr() as usize);
                assert_eq!(keys.len(), 33);
                for key in keys {
                    assert!(*key == generate_secret_key(&mut expected));
                }
                source.recycle(&stop)?;
            }
            assert_eq!(allocations.len(), 2);
            drop(guard);
            producer.join().unwrap()?;
            Ok(())
        })
    }

    #[test]
    fn full_and_empty_queues_cancel_without_drain() {
        for full in [false, true] {
            let stop = AtomicBool::new(false);
            let (tx, rx) = channel::bounded(1);
            if full {
                tx.send(1).unwrap();
            }
            std::thread::scope(|scope| {
                let (entered_tx, entered_rx) = channel::bounded(0);
                let stop = &stop;
                let worker = scope.spawn(move || {
                    entered_tx.send(()).unwrap();
                    if full {
                        assert!(!send(&tx, 2, stop).unwrap());
                    } else {
                        assert!(receive(&rx, stop).unwrap().is_none());
                    }
                });
                entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                stop.store(true, Ordering::Relaxed);
                worker.join().unwrap();
            });
        }
    }

    #[test]
    fn pipelined_hits_preserve_empty_odd_and_overlapping_targets() {
        for mode in 0..3 {
            let secp = Arc::new(Secp256k1::new());
            let first = generate_secret_key(&mut ChaCha20Rng::from_seed([17; 32]));
            let nibs = address_nibbles(&cpu::derive_address(&first, &secp));
            let targets = match mode {
                0 => Targets::default(),
                1 => Targets {
                    prefix: nibs[..3].to_vec(),
                    suffix: nibs[37..].to_vec(),
                },
                _ => Targets {
                    prefix: nibs.to_vec(),
                    suffix: nibs.to_vec(),
                },
            };
            let mut reports = vec![];
            let (result, total, _) = exercise(
                &mut cpu::CpuBackend::new(secp),
                || Ok(ChaCha20Rng::from_seed([17; 32])),
                &targets,
                |p| reports.push(p),
            );
            let hit = result.unwrap().unwrap();
            assert_eq!((hit.tries, hit.worker_id, total), (1, 0, 8));
            assert_eq!((reports.len(), reports[0].completed), (1, 8));
        }
    }

    #[test]
    fn producer_seed_failure_or_panic_propagates_before_success() {
        for panic in [false, true] {
            let (result, total, best) = exercise(
                &mut cpu::CpuBackend::new(Arc::new(Secp256k1::new())),
                move || -> Result<ChaCha20Rng> {
                    assert!(!panic, "injected seed panic");
                    anyhow::bail!("injected entropy failure")
                },
                &Targets::default(),
                |_| {},
            );
            assert!(result.is_err());
            assert_eq!(total, 0);
            assert!(best.snapshot().is_none());
        }
    }

    #[test]
    fn compute_and_verification_errors_publish_nothing() {
        struct Failing(bool);
        impl AddressBackend for Failing {
            fn derive_batch(&mut self, _: &[SecretKey], addresses: &mut [Address]) -> Result<()> {
                ensure!(!self.0, "injected compute failure");
                addresses.fill([0xab; 20]);
                Ok(())
            }
        }
        for compute in [true, false] {
            let (result, total, best) = exercise(
                &mut Failing(compute),
                || Ok(ChaCha20Rng::from_seed([17; 32])),
                &Targets {
                    prefix: vec![10, 11],
                    suffix: vec![],
                },
                |_| {},
            );
            assert!(result.is_err());
            assert_eq!(total, if compute { 0 } else { 8 });
            assert!(best.snapshot().is_none());
        }
    }

    #[test]
    fn consumer_unwind_cancels_and_joins_the_producer() {
        struct OwnedRng {
            rng: ChaCha20Rng,
            dropped: channel::Sender<()>,
        }
        impl Drop for OwnedRng {
            fn drop(&mut self) {
                self.dropped.send(()).unwrap();
            }
        }
        impl CryptoRng for OwnedRng {}
        impl RngCore for OwnedRng {
            fn next_u32(&mut self) -> u32 {
                self.rng.next_u32()
            }
            fn next_u64(&mut self) -> u64 {
                self.rng.next_u64()
            }
            fn fill_bytes(&mut self, bytes: &mut [u8]) {
                self.rng.fill_bytes(bytes);
            }
            fn try_fill_bytes(&mut self, bytes: &mut [u8]) -> std::result::Result<(), rand::Error> {
                self.rng.try_fill_bytes(bytes)
            }
        }
        struct Panicking;
        impl AddressBackend for Panicking {
            fn derive_batch(&mut self, _: &[SecretKey], _: &mut [Address]) -> Result<()> {
                panic!("injected consumer unwind");
            }
        }
        let (tx, rx) = channel::bounded(1);
        let result = std::panic::catch_unwind(|| {
            exercise(
                &mut Panicking,
                || {
                    Ok(OwnedRng {
                        rng: ChaCha20Rng::from_seed([17; 32]),
                        dropped: tx,
                    })
                },
                &Targets::default(),
                |_| {},
            )
        });
        assert!(result.is_err());
        // Already delivered: run's scope has joined before unwinding reaches us.
        rx.try_recv().unwrap();
    }

    #[test]
    fn cancellation_counts_only_completed_batches() {
        struct Cancel<'a>(&'a AtomicBool);
        impl AddressBackend for Cancel<'_> {
            fn derive_batch(&mut self, _: &[SecretKey], _: &mut [Address]) -> Result<()> {
                self.0.store(true, Ordering::Relaxed);
                Ok(())
            }
        }
        for already_stopped in [false, true] {
            let stop = AtomicBool::new(already_stopped);
            let total = AtomicU64::new(0);
            let best = BestState::default();
            let (updates, receiver) = channel::bounded(1);
            let result = run(
                &mut Cancel(&stop),
                33,
                WorkerContext {
                    worker_id: 0,
                    stop: &stop,
                    total: &total,
                    best: &best,
                    updates: &updates,
                    targets: &Targets::default(),
                    report_every: 1,
                    verifier: &Secp256k1::new(),
                },
                |_| {},
                Noop,
                || Ok(ChaCha20Rng::from_seed([17; 32])),
            )
            .unwrap();
            assert!(result.is_none());
            assert_eq!(
                total.load(Ordering::Relaxed),
                if already_stopped { 0 } else { 33 }
            );
            assert!(receiver.try_recv().is_err());
            assert!(best.snapshot().is_none());
        }
    }

    #[test]
    fn preparation_cancels_at_the_next_1024_key_boundary() {
        struct CancellingRng<'a> {
            generated: &'a AtomicU64,
            stop: &'a AtomicBool,
        }
        impl CryptoRng for CancellingRng<'_> {}
        impl RngCore for CancellingRng<'_> {
            fn next_u32(&mut self) -> u32 {
                unreachable!()
            }
            fn next_u64(&mut self) -> u64 {
                unreachable!()
            }
            fn fill_bytes(&mut self, bytes: &mut [u8]) {
                bytes.fill(0);
                *bytes.last_mut().unwrap() = 1;
                if self.generated.fetch_add(1, Ordering::Relaxed) + 1 == 1024 {
                    self.stop.store(true, Ordering::Relaxed);
                }
            }
            fn try_fill_bytes(&mut self, bytes: &mut [u8]) -> std::result::Result<(), rand::Error> {
                self.fill_bytes(bytes);
                Ok(())
            }
        }
        let stop = AtomicBool::new(false);
        let generated = AtomicU64::new(0);
        let (recycle_tx, recycle_rx) = channel::bounded(2);
        let (ready_tx, ready_rx) = channel::bounded(1);
        recycle_tx
            .send(PreparedBatch {
                sequence: 0,
                keys: KeyBatch(Vec::with_capacity(2048)),
            })
            .unwrap();
        produce(recycle_rx, ready_tx, &stop, 2048, Noop, || {
            Ok(CancellingRng {
                generated: &generated,
                stop: &stop,
            })
        })
        .unwrap();
        assert_eq!(generated.load(Ordering::Relaxed), 1024);
        assert!(ready_rx.try_recv().is_err());
    }

    #[test]
    fn out_of_order_and_disconnected_batches_fail() {
        let stop = AtomicBool::new(false);
        let (ready, rx) = channel::bounded(1);
        let (recycle, _) = channel::bounded(2);
        ready
            .send(PreparedBatch {
                sequence: 1,
                keys: KeyBatch(vec![]),
            })
            .unwrap();
        let mut source = PipelineSource {
            ready: rx,
            recycle,
            current: None,
            expected: 0,
            observer: Noop,
        };
        assert!(source.next(&stop).is_err());
        let (tx, rx) = channel::bounded::<()>(1);
        drop(tx);
        assert!(receive(&rx, &stop).is_err());
    }

    #[test]
    fn late_producer_panic_overrides_consumer_hit() {
        // The producer blocks inside the second batch. The consumer completes a
        // valid first batch and signals from progress, immediately before its hit.
        struct GatedRng {
            calls: usize,
            entered: channel::Sender<()>,
            release: channel::Receiver<()>,
        }
        impl CryptoRng for GatedRng {}
        impl RngCore for GatedRng {
            fn next_u32(&mut self) -> u32 {
                unreachable!()
            }
            fn next_u64(&mut self) -> u64 {
                unreachable!()
            }
            fn fill_bytes(&mut self, bytes: &mut [u8]) {
                self.calls += 1;
                if self.calls == 9 {
                    self.entered.send(()).unwrap();
                    self.release.recv_timeout(Duration::from_secs(5)).unwrap();
                    panic!("injected late producer panic");
                }
                bytes.fill(0);
                *bytes.last_mut().unwrap() = self.calls as u8;
            }
            fn try_fill_bytes(&mut self, bytes: &mut [u8]) -> std::result::Result<(), rand::Error> {
                self.fill_bytes(bytes);
                Ok(())
            }
        }
        struct GatedBackend {
            entered: channel::Receiver<()>,
            cpu: cpu::CpuBackend,
        }
        impl AddressBackend for GatedBackend {
            fn derive_batch(
                &mut self,
                keys: &[SecretKey],
                addresses: &mut [Address],
            ) -> Result<()> {
                self.entered.recv_timeout(Duration::from_secs(5))?;
                self.cpu.derive_batch(keys, addresses)
            }
        }
        let (entered_tx, entered_rx) = channel::bounded(1);
        let (release_tx, release_rx) = channel::bounded(1);
        let (result, total, _) = exercise(
            &mut GatedBackend {
                entered: entered_rx,
                cpu: cpu::CpuBackend::new(Arc::new(Secp256k1::new())),
            },
            || {
                Ok(GatedRng {
                    calls: 0,
                    entered: entered_tx,
                    release: release_rx,
                })
            },
            &Targets::default(),
            |_| release_tx.send(()).unwrap(),
        );
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("producer panicked")
        );
        assert_eq!(total, 8);
    }
}
