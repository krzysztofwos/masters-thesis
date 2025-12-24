# MENACE Documentation

This directory contains documentation for the MENACE (Matchbox Educable Noughts And Crosses Engine) Rust implementation.

## Directory Structure

### `analysis/`

Game-theoretic analysis documents:

- **menace-analysis.md** - MENACE algorithm analysis and mathematical foundations
- **menace-state-count-analysis.md** - State space complexity and reduction analysis
- **tic-tac-toe-analysis.md** - Complete tic-tac-toe game-theoretic analysis
- **tic-tac-toe-complete-decomposition.md** - Categorical decomposition of tic-tac-toe state space
- **connect-4-analysis.md** - Connect-4 game analysis and complexity bounds

## Development Shortcuts

- `make check` – runs `cargo fmt`, `cargo clippy`, and `cargo test` in one pass (ideal for CI).
- `make results` – rebuilds the release binary, reruns all experiments in `configs/experiments.yaml`, analyzes metrics, and refreshes Markdown/PNG reports.
- `make results-report` – skips re-running experiments; re-analyzes the directories specified via `RUN_DIRS` (defaults to `menace_data/*`) and regenerates the consolidated performance report.

Tips:

- Override the default experiment config with `EXPERIMENT_CONFIG=path/to/config.yaml`.
- Use `RUN_DIRS="menace_data/Restock_Strategy_Comparison menace_data/AIF_Showdown"` to target specific runs.
- Set `PYTHON=python3.11` (or another interpreter) if your environment relies on a non-default Python.

## CLI Quick Reference

### Training Commands

```bash
# Train MENACE agent against random opponent
menace train menace -o random -g 500 --seed 42

# Train with progress indicators
menace train menace -o optimal -g 1000 --progress

# Save trained agent (MessagePack format)
menace train menace -o random -g 500 --output trained_agent.msgpack

# Save training summary alongside the agent artifact
menace train pure-active-inference \
  --opponent uniform \
  --games 500 \
  --summary outputs/pure_aif_summary.json \
  --output outputs/pure_aif_agent.msgpack

# Train with custom configuration and save
menace train menace \
  --opponent optimal \
  --games 1000 \
  --filter michie \
  --restock box \
  --init-beads 4,3,2,1 \
  --reward "win=3,draw=1,loss=-1" \
  --output menace_optimal_1000.msgpack

# Train Active Inference agent with canonical preferences and save
menace train active-inference \
  --opponent minimax \
  --games 500 \
  --ai-epistemic-weight 0.5 \
  --ai-ambiguity-weight 0.1 \
  --policy-lambda 0.25 \
  --policy-beads-scale 40 \
  --output aif_agent.msgpack
```

### Evaluation Commands

```bash
# Evaluate a trained agent against optimal play
menace evaluate trained_agent.msgpack -o optimal -g 100

# Evaluate with custom configuration
menace evaluate menace_optimal_1000.msgpack \
  --opponent random \
  --games 500 \
  --seed 42

# Evaluate and export results
menace evaluate aif_agent.msgpack \
  --opponent defensive \
  --games 200 \
  --export results.json
```

### Comparison Commands

```bash
# Compare multiple learners
menace compare menace random optimal -g 100

# Round-robin tournament
menace compare menace random optimal defensive -g 200
```

### Analysis Commands

```bash
# Game tree analysis with filters
menace analyze game-tree --filter canonical      # 765 canonical states
menace analyze game-tree --filter all            # 338 X-to-move states
menace analyze game-tree --filter decision-only  # 304 decision states
menace analyze game-tree --filter michie         # 287 MENACE states

# Optimal policy computation
menace analyze optimal --export optimal_policy.json

# Symmetry visualization
menace analyze symmetry --state "X........_O" --visualize

# Trajectory enumeration
menace analyze trajectories --detailed --export trajectories.csv

# Mathematical structures
menace analyze structures --structure magic-square
menace analyze structures --structure positional-values
menace analyze structures --structure categorical

# First move analysis
menace analyze first-moves --export openings.csv

# State space validation
menace analyze validate --verify --show-invalid

# Active Inference EFE decomposition
menace analyze active-inference --opponent minimax --beta 0.5 --export efe.csv
```

## Features

### Core Functionality

- **MENACE Agent** - Matchbox-based reinforcement learning with configurable restock strategies
- **Active Inference** - Bayesian decision-making with Expected Free Energy minimization
  - Canonical priors: `P(win)=0.60`, `P(draw)=0.35`, `P(loss)=0.05`
  - Default ambiguity weight `β_amb = 0.1`, policy temperature `λ = 0.25`, bead scale `40`
- **Optimal Policy** - Minimax algorithm for perfect Tic-tac-toe play
- **Random & Defensive Baselines** - Simple strategies for comparison
- **O-First Game Support** - Full support for non-standard O-first games (validation and game logic automatically adapt)

### Analysis Tools

- **Game Tree Analysis** - State space enumeration with filtering (all/decision-only/Michie)
- **Trajectory Analysis** - Complete enumeration of 255,168 game sequences
- **Symmetry Analysis** - D4 dihedral group transformations and stabilizers
- **First Move Analysis** - Opening strategy evaluation (corner/edge/center)
- **State Validation** - Verification against known mathematical bounds
- **Mathematical Structures** - Magic square isomorphism and positional values

### Training Infrastructure

- **Pipeline Abstraction** - Learner trait for uniform training interface
- **Observer Pattern** - Progress tracking, metrics collection, and JSONL export
- **Curriculum Training** - Staged opponent progression
- **Comparison Framework** - Round-robin tournament evaluation

### Data Export

- **CSV** - Trajectory data, opening analysis, Active Inference decompositions
- **JSON** - Optimal policies, learned strategies, evaluation results
- **Summary JSON** - High-level training/validation metrics via `--summary`
- **JSONL** - Training observations with move weights
- **MessagePack** - Agent serialization for save/load (workspace + metadata)

### Agent Serialization

Trained agents can be saved and loaded using MessagePack binary format:

#### CLI Usage

```bash
# Train and save an agent
menace train menace -o optimal -g 1000 --output agent.msgpack

# Load and evaluate the saved agent
menace evaluate agent.msgpack -o random -g 100

# Export a compact training summary without saving the workspace
menace train menace -o random -g 100 --summary summaries/menace_random.json
```

#### Programmatic API

```rust
use menace::{MenaceAgent, SavedMenaceAgent, TrainingMetadata, StateFilter};

// Create and train an agent
let mut agent = MenaceAgent::builder()
    .filter(StateFilter::Michie)
    .seed(42)
    .build()?;

// ... train the agent ...

// Save the trained agent
let metadata = TrainingMetadata {
    games_trained: Some(1000),
    opponents: vec!["optimal".to_string()],
    seed: Some(42),
    saved_at: None,
    agent_player: Some(Player::X),
    first_player: Some(Player::X),
};

let saved = SavedMenaceAgent::from_agent(&agent, metadata)?;
saved.save_to_file("agent.msgpack")?;

// Load the agent
let loaded = SavedMenaceAgent::load_from_file("agent.msgpack")?;
let restored_agent = loaded.to_agent()?;
```

#### Workspace-Only Serialization

For lightweight serialization of just the learned weights (without metadata):

```rust
use menace::MenaceWorkspace;

// Save workspace
let workspace = agent.workspace().clone();
workspace.save(Path::new("workspace.msgpack"))?;

// Load workspace
let loaded_workspace = MenaceWorkspace::load(Path::new("workspace.msgpack"))?;
```

The saved agent format includes:

- Complete learned workspace (state-action weights)
- State filter configuration (All/DecisionOnly/Michie/Both)
- Restock mode (None/Move/Box)
- Initial bead schedule
- Algorithm type (ClassicMenace/ActiveInference)
- Algorithm-specific parameters (reinforcement values, beliefs, preferences)
- Training metadata (games, opponents, seed, optional timestamps)

## Experiment Automation & Evaluation

- `scripts/automation/experiment_driver.py` orchestrates batch training suites. Each seed now emits:
  - `metrics.jsonl` — detailed episode observations
  - `training_summary.json` — aggregate metrics for automation pipelines
  - `agent.msgpack` — serialized learner workspace + metadata
- Optional `evaluation_configs` entries in `configs/experiments.yaml` automatically trigger `scripts/automation/run_evaluations.py` after training, producing `results/evaluation_summary.json`.
- On demand, run `uv run python -m scripts.automation.run_evaluations --config configs/evaluate.yml` to regenerate evaluation summaries (expects agent artifacts produced by the driver).

## Testing

The test suite includes:

- **Unit tests** (105 library tests) - Core functionality validation
- **Integration tests** (75 tests) - Training, comparison, and analysis workflows
  - Serialization tests (8 tests) - Agent save/load and workspace persistence
  - Validation tests - Mathematical property verification (state counts, symmetry reductions)
  - Active Inference tests - Convergence and decision-making validation
  - Pipeline tests - Training and evaluation workflows
  - Regression tests - MENACE behavioral consistency

Run tests with:

```bash
# Run all tests (180 total)
cargo test

# Run only library tests (105 tests)
cargo test --lib

# Run only integration tests (75 tests)
cargo test --test '*'

# Run specific test suite
cargo test workspace_serialization

# Run with output
cargo test -- --nocapture
```

## References

- [Michie, D. (1961). "Trial and Error" in Science Survey](https://people.cs.umass.edu/~barto/courses/cs687/Michie-trialanderror.pdf)
- [Friston, K. et al. (2015). "Active inference and epistemic value"](https://doi.org/10.1007/s11571-015-9330-7)
