# Repository Guidelines

## Project Structure & Module Organization

- `src/`: Rust library and CLI (`src/bin/menace.rs`) with core modules such as `tictactoe/`, `menace/`, `active_inference/`, `pipeline/`, and `cli/`.
- `tests/`: integration tests; `examples/`: runnable demos (e.g., `cargo run --example symmetry_demo`).
- `scripts/`: Python automation/reporting; `configs/`: experiment YAML; `docs/`: architecture and analysis docs; `thesis/`: LaTeX sources; `resources/`: reference material.
- Generated outputs in `menace_data/`, `menace_reports/`, `results/`, and `target/` are git-ignored.

## Build, Test, and Development Commands

- `cargo build` / `cargo build --release` to compile (release builds are used by experiments).
- `cargo run --bin menace -- --help` for the CLI; examples via `cargo run --example symmetry_demo`.
- `cargo fmt` and `cargo clippy --all-targets --all-features -- -D warnings` for format/lint.
- `cargo test`, or `cargo test --lib`/`cargo test --test workspace_serialization`.
- `make check` runs fmt + clippy + tests; `make results` runs the full experiment + analysis + report pipeline.
- Python tooling: `make venv` (uses `uv` with `pyproject.toml`); experiments via `python -m scripts.automation.experiment_driver run --config-file configs/experiments.yaml`.

## Coding Style & Naming Conventions

- Rust edition 2024; use rustfmt defaults (4-space indentation).
- Follow `docs/rust-style-guide.md` and `docs/rust-hexagonal-architecture-guide.md` for error handling, newtypes, and hexagonal boundaries.
- Naming: modules/files `snake_case`, types `CamelCase`, constants `SCREAMING_SNAKE_CASE`.

## Testing Guidelines

- Unit tests live next to code under `src/` with `#[cfg(test)]`; integration tests are in `tests/` with descriptive snake_case filenames.
- Keep tests deterministic; use `cargo test -- --nocapture` for debugging output.

## Commit & Pull Request Guidelines

- Commit messages are short, capitalized, and single-topic (e.g., "Docs", "Configs").
- PRs: describe the change clearly, link relevant configs or run directories (e.g., `menace_data/<run_name>`) when experiments are involved, and note the commands/tests executed.
