//! Statically dispatched diagnostics. Normal searches never read a profiling
//! clock or acquire a profiling lock; the recorder exists only in test builds.

#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Prepare,
    Upload,
    EncodeSubmit,
    Wait,
    ReadbackCleanup,
    SampleVerify,
    MatchVerify,
    QueueWait,
}

pub(crate) trait Observer: Clone + Send {
    const ENABLED: bool;
    type Stamp;
    fn start(&self) -> Self::Stamp;
    fn finish(&self, stage: Stage, started: Self::Stamp);
    fn gpu_seconds(&self, start: f64, end: f64);
}

#[derive(Clone, Copy)]
pub(crate) struct Noop;

impl Observer for Noop {
    const ENABLED: bool = false;
    type Stamp = ();
    #[inline(always)]
    fn start(&self) {}
    #[inline(always)]
    fn finish(&self, _: Stage, _: ()) {}
    #[inline(always)]
    fn gpu_seconds(&self, _: f64, _: f64) {}
}

#[cfg(test)]
#[derive(Clone, Default)]
pub(crate) struct Recorder(std::sync::Arc<std::sync::Mutex<Measurements>>);

#[cfg(test)]
#[derive(Default)]
struct Measurements {
    seconds: [f64; 8],
    gpu_seconds: f64,
    gpu_valid: u64,
    gpu_unavailable: u64,
}

#[cfg(test)]
impl Observer for Recorder {
    const ENABLED: bool = true;
    type Stamp = std::time::Instant;
    fn start(&self) -> Self::Stamp {
        std::time::Instant::now()
    }
    fn finish(&self, stage: Stage, started: Self::Stamp) {
        self.0.lock().unwrap().seconds[stage as usize] += started.elapsed().as_secs_f64();
    }
    fn gpu_seconds(&self, start: f64, end: f64) {
        let mut data = self.0.lock().unwrap();
        if start.is_finite() && end.is_finite() && start > 0.0 && end > start {
            data.gpu_seconds += end - start;
            data.gpu_valid += 1;
        } else {
            data.gpu_unavailable += 1;
        }
    }
}

#[cfg(test)]
impl Recorder {
    pub(crate) fn snapshot(&self) -> serde_json::Value {
        let data = self.0.lock().unwrap();
        let names = [
            "prepare",
            "upload",
            "encode_submit",
            "cpu_wait",
            "readback_cleanup",
            "sample_verify",
            "match_verify",
            "queue_wait",
        ];
        let stages: serde_json::Map<_, _> = names
            .into_iter()
            .zip(data.seconds)
            .map(|(name, seconds)| (name.into(), seconds.into()))
            .collect();
        serde_json::json!({"stage_seconds": stages, "gpu_seconds": (data.gpu_valid > 0).then_some(data.gpu_seconds), "gpu_valid_batches": data.gpu_valid, "gpu_unavailable_batches": data.gpu_unavailable})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_gpu_timestamps_are_not_cpu_wait_times() {
        let recorder = Recorder::default();
        recorder.gpu_seconds(0.0, 1.0);
        recorder.gpu_seconds(2.0, 1.0);
        recorder.gpu_seconds(f64::NAN, 3.0);
        assert!(recorder.snapshot()["gpu_seconds"].is_null());
        recorder.gpu_seconds(10.0, 10.25);
        let result = recorder.snapshot();
        assert_eq!(result["gpu_seconds"], 0.25);
        assert_eq!(result["gpu_unavailable_batches"], 3);
    }
}
