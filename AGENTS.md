# Repository Guidelines

## Project Structure & Module Organization
- `src/main.rs` — CLI, progress UI, and file output.
- `src/backend/` — address derivation: CPU (`cpu.rs`), Apple Silicon Metal (`metal.rs`, `shader.metal`), and Vulkan (`vulkan.rs`, `shader.comp`, `shader.spv`). Shared fixed-base table construction lives in `table.rs`. Backends fill address slices from valid `SecretKey` batches; they do not match, rank, or write files.
- `src/search.rs` and `src/search/pipeline.rs` — CSPRNG, matching, ranking, cancellation, and optional GPU key-prep overlap.
- `src/timing.rs` — optional stage timing for benchmarks.
- `tests/cli.rs` — binary-level output, permissions, and persistence checks.
- `docs/` — performance notes and GPU implementation bounds. Benchmark artifacts belong in `target/` or other gitignored scratch space, never in `src/`.

## Build, Test, and Development Commands
- `cargo run --release -- --help` prints CLI usage and verifies argument wiring.
- `cargo run --release -- --prefix dead --suffix beef` exercises the full pipeline.
- `cargo build` compiles on the default toolchain; run before opening a PR.
- `cargo fmt` applies standard formatting; run prior to commits.
- `cargo clippy --all-targets -- -D warnings` enforces lint cleanliness.
- `cargo test` runs unit and CPU CLI tests. Metal and Vulkan hardware tests are `#[ignore]` and require a real device.

## Coding Style & Naming Conventions
Follow Rust 2024 idioms with four-space indentation and `snake_case` for functions, variables, and files. Public CLI enums (e.g., `OutFmt`) should use `PascalCase`. Prefer `?` for error propagation and favor slices over owned buffers when feasible. User-facing copy and errors are English; code comments may be Chinese. Document safety-sensitive sections, especially concurrency and GPU buffer lifetime.

## Testing Guidelines
Keep deterministic helpers next to their modules with `#[cfg(test)]`. Wrap long-running statistical or hardware checks behind `#[ignore]` so CI stays fast. Cover prefix/suffix edge cases, error paths, and private-key file permissions.

## Commit & Pull Request Guidelines
Use present-tense, imperative summaries (e.g., `Add JSON output option`). When grouping changes, prefer small, reviewable commits with descriptive bodies for performance-sensitive tweaks. Pull requests must include: scope description, testing notes (commands and outcomes), and any benchmarks if performance claims are made.
