use std::{
    fs,
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use secp256k1::{PublicKey, Secp256k1, SecretKey};
use tiny_keccak::{Hasher, Keccak};

fn run(args: &[&str], out: &std::path::Path) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_vanity-rs"))
        .args(args)
        .arg("--out")
        .arg(out)
        .arg("--report-every")
        .arg("0")
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if child.try_wait().unwrap().is_some() {
            return child.wait_with_output().unwrap();
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let _ = child.wait();
            panic!("CLI did not terminate within 20 seconds");
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn check_wallet(record: &serde_json::Value, prefix: &str, suffix: &str, output: &Output) {
    let key = record["private_key"].as_str().unwrap();
    let secret = key
        .strip_prefix("0x")
        .unwrap()
        .parse::<SecretKey>()
        .unwrap();
    let public = PublicKey::from_secret_key(&Secp256k1::new(), &secret).serialize_uncompressed();
    let mut keccak = Keccak::v256();
    keccak.update(&public[1..]);
    let mut hash = [0; 32];
    keccak.finalize(&mut hash);
    let address = format!(
        "0x{}",
        hash[12..]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    assert_eq!(record["address"], address);
    assert!(address[2..].starts_with(prefix));
    assert!(address.ends_with(suffix));
    assert!(!String::from_utf8_lossy(&output.stderr).contains(key));
    assert!(!String::from_utf8_lossy(&output.stdout).contains(key));
}

fn check_permissions(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

fn exercise(backend: &str, batch: &str) {
    let directory = tempfile::tempdir().unwrap();
    let out = directory.path().join("result.jsonl");
    for _ in 0..2 {
        let result = run(
            &[
                "--backend",
                backend,
                "--gpu-batch-size",
                batch,
                "--workers",
                "2",
                "--prefix",
                "a",
                "--suffix",
                "f",
            ],
            &out,
        );
        assert!(
            result.status.success(),
            "CLI failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let contents = fs::read_to_string(&out).unwrap();
        for line in contents.lines() {
            let record: serde_json::Value = serde_json::from_str(line).unwrap();
            check_wallet(&record, "a", "f", &result);
            assert_eq!(record.as_object().unwrap().len(), 6);
            assert!(record["tries"].as_u64().unwrap() > 0);
        }
        let closest = directory.path().join("result-closest.json");
        let record = serde_json::from_str(&fs::read_to_string(&closest).unwrap()).unwrap();
        check_wallet(&record, "a", "f", &result);
        check_permissions(&closest);
        check_permissions(&out);
        if backend == "metal" || backend == "cuda" || backend == "vulkan" || backend == "auto" {
            let stderr = String::from_utf8_lossy(&result.stderr);
            if backend == "auto" {
                assert!(
                    stderr.contains("Backend: metal")
                        || stderr.contains("Backend: cuda")
                        || stderr.contains("Backend: vulkan"),
                    "{stderr}"
                );
            } else {
                assert!(stderr.contains(&format!("Backend: {backend}")), "{stderr}");
            }
            assert!(stderr.contains("--workers only applies to CPU"), "{stderr}");
        }
    }
    assert_eq!(fs::read_to_string(&out).unwrap().lines().count(), 2);
    let txt = directory.path().join("result.txt");
    let first = run(
        &[
            "--backend",
            backend,
            "--gpu-batch-size",
            batch,
            "--format",
            "txt",
        ],
        &txt,
    );
    assert!(first.status.success());
    let second = run(
        &[
            "--backend",
            backend,
            "--gpu-batch-size",
            batch,
            "--format",
            "txt",
            "--append",
        ],
        &txt,
    );
    assert!(second.status.success());
    assert_eq!(
        fs::read_to_string(&txt)
            .unwrap()
            .matches("=== MATCH FOUND ===")
            .count(),
        2
    );
    assert!(!directory.path().join("result-closest.json").exists());
    check_permissions(&txt);

    // A closest-path error must cancel an otherwise practically unbounded search.
    let blocked = directory.path().join("blocked-closest.json");
    fs::create_dir(&blocked).unwrap();
    let failure = run(
        &[
            "--backend",
            backend,
            "--gpu-batch-size",
            batch,
            "--prefix",
            "0000000000000000000000000000000000000000",
        ],
        &directory.path().join("blocked.jsonl"),
    );
    assert!(!failure.status.success());
    assert!(
        String::from_utf8_lossy(&failure.stderr).contains("cannot save closest-candidate snapshot")
    );
    assert_eq!(
        fs::metadata(directory.path().join("blocked.jsonl"))
            .unwrap()
            .len(),
        0
    );
}

#[test]
fn cpu_cli_compatibility_and_persistence() {
    exercise("cpu", "4096");
}

#[test]
#[ignore = "requires a real Metal device and runs the production binary"]
fn metal_cli_compatibility_and_persistence() {
    for batch in ["1", "33", "4096", "65536", "131072", "262144"] {
        exercise("metal", batch);
    }
    exercise("auto", "4096");
    // Exercise the actual compiled default instead of spelling its value here.
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("default.jsonl");
    let result = run(
        &["--backend", "metal", "--prefix", "a", "--suffix", "f"],
        &path,
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let record = serde_json::from_str(fs::read_to_string(&path).unwrap().trim()).unwrap();
    check_wallet(&record, "a", "f", &result);
    assert_eq!(record["worker_id"], 0);
    check_permissions(&path);
}

#[test]
#[ignore = "requires a real Vulkan compute device and runs the production binary"]
fn vulkan_cli_compatibility_and_persistence() {
    for batch in ["1", "33", "4096", "65536", "131072", "262144"] {
        exercise("vulkan", batch);
    }
    exercise("auto", "4096");
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("default.jsonl");
    let result = run(
        &["--backend", "vulkan", "--prefix", "a", "--suffix", "f"],
        &path,
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let record = serde_json::from_str(fs::read_to_string(&path).unwrap().trim()).unwrap();
    check_wallet(&record, "a", "f", &result);
    assert_eq!(record["worker_id"], 0);
    check_permissions(&path);
}

#[test]
#[ignore = "requires a real CUDA compute device and runs the production binary"]
fn cuda_cli_compatibility_and_persistence() {
    for batch in ["1", "33", "4096", "65536", "131072", "262144"] {
        exercise("cuda", batch);
    }
    exercise("auto", "4096");
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("default.jsonl");
    let result = run(
        &["--backend", "cuda", "--prefix", "a", "--suffix", "f"],
        &path,
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let record = serde_json::from_str(fs::read_to_string(&path).unwrap().trim()).unwrap();
    check_wallet(&record, "a", "f", &result);
    assert_eq!(record["worker_id"], 0);
    check_permissions(&path);
}
