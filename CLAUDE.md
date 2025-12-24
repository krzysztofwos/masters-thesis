# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MENACE (Matchbox Educable Noughts And Crosses Engine) is a Rust implementation of Donald Michie's 1961 reinforcement learning algorithm for Tic-tac-toe, extended with Active Inference capabilities. This is a research-oriented crate combining classical reinforcement learning with modern Bayesian decision theory.

## Essential Commands

### Building and Testing

```bash
# Build the project
cargo build
cargo build --release          # Release build with optimizations

# Run all tests (108 unit tests + 116 integration tests)
cargo test

# Run specific test categories
cargo test --lib               # Unit tests only (108)
cargo test --test '*'          # Integration tests only (116)
cargo test test_name           # Specific test by name
cargo test --test filename     # Tests in specific file

# Test with detailed output
cargo test -- --nocapture --test-threads=1

# Build and run the CLI
cargo run --bin menace -- --help
```

### CLI Usage

```bash
# Train MENACE agent
cargo run --bin menace -- train menace -o random -g 500 --seed 42
cargo run --bin menace -- train menace --opponent optimal --games 1000 --filter michie

# Train Active Inference agent
cargo run --bin menace -- train active-inference --opponent minimax --games 500 --ai-epistemic-weight 0.5

# Train Q-learning agent (off-policy TD)
cargo run --bin menace -- train q-learning --opponent random --games 1000 --seed 42

# Train SARSA agent (on-policy TD)
cargo run --bin menace -- train sarsa --opponent random --games 1000 --seed 42

# Compare multiple learners (round-robin tournament)
cargo run --bin menace -- compare menace random optimal -g 100

# Evaluate trained agent
cargo run --bin menace -- evaluate agent.msgpack -o optimal -g 100

# Analyze game tree (different filters produce different state counts)
cargo run --bin menace -- analyze game-tree --filter michie         # 287 states
cargo run --bin menace -- analyze game-tree --filter decision-only  # 304 states
cargo run --bin menace -- analyze game-tree --filter all            # 338 states
cargo run --bin menace -- analyze game-tree --filter canonical      # 765 states

# Analyze Active Inference EFE decomposition
cargo run --bin menace -- analyze active-inference --opponent minimax --beta 0.5
cargo run --bin menace -- analyze active-inference --opponent minimax --beta 0.5 --export efe.csv

# Export trained model
cargo run --bin menace -- export agent.msgpack --format json --output agent.json
```

### Examples

```bash
# Educational demonstrations
cargo run --example symmetry_demo                    # D4 dihedral group tutorial
cargo run --example active_inference_demo            # Active Inference introduction
cargo run --example free_energy_demo                 # Free energy analysis basics
cargo run --example free_energy_convergence          # Convergence analysis over training
cargo run --example free_energy_component_analysis   # EFE component breakdown
cargo run --example free_energy_perfect_vs_perfect   # Optimal vs optimal analysis
cargo run --example efe_micro                        # Minimal EFE computation example
```

### Development Commands

```bash
cargo fmt                                   # Format code
cargo clippy --all-targets -- -D warnings  # Lint
make check                                 # Format + Lint + Test
make results                               # Run full experiment suite with analysis
```

### Experiment Automation

```bash
# Run experiments defined in configs/experiments.yaml
make results-run                          # Execute all experiments (requires release build)

# Analyze existing experiment data
make results-report                       # Generate reports for current runs

# Full pipeline (run + analyze + report)
make results                              # Complete experimental workflow

# Python-based experiment management (requires uv or Python environment)
python -m scripts.automation.experiment_driver run --config-file configs/experiments.yaml
python -m scripts.automation.experiment_driver analyze --run-dir menace_data/<run_name>
python -m scripts.automation.experiment_driver report --run-dir menace_data/<run_name>
```

### Thesis (LaTeX) Pipeline

```bash
# Run the thesis experiment suite (build + experiments + evaluations + reports)
make thesis-results

# Run stages separately
make thesis-results-run
make thesis-results-report

# Build the LaTeX thesis PDF
make -C thesis

# Package seed-level artefacts for external audit
make thesis-package-minimal
make thesis-package-curves
```

The thesis experiment configuration lives in `configs/thesis_experiments.yaml` and post-training evaluation configuration in `configs/evaluate_thesis.yml`.

Note: The legacy Quarto thesis was archived outside the repo; LaTeX sources live in `thesis/`.

## Architecture Overview

The project follows **Hexagonal Architecture** (Ports & Adapters) with clear separation of concerns.

### Core Module Organization

**Game Core (tictactoe/)**

- `BoardState`: Core 10-byte state (9 cells + player to move) with copy semantics for zero-cost passing
- `D4Transform`: Dihedral group symmetry operations for board canonicalization (8 transformations)
- `game_tree.rs`: Builds reduced game tree with symmetry reduction (765 total states → 287 Michie states)
- Board encoding format: `"XXXXXXXXX_P"` where P is X or O (e.g., `"X........_O"`)

**Workspace Management (workspace.rs)**

- `MenaceWorkspace`: Central data structure managing matchbox weights and morphisms
- Uses HashMap for efficient weight storage and retrieval
- Handles state filtering (All/DecisionOnly/Michie/Both) and restocking strategies (None/Move/Box)
- Implements matchbox-based reinforcement learning with configurable bead schedules

**Learning Algorithms (menace/)**

- `ClassicMenace`: Original MENACE algorithm with reinforcement learning
- `ActiveInferenceLearner`: Bayesian agent using Expected Free Energy minimization
- Multiple builders for different configurations

**Temporal Difference Learning (q_learning/)**

- `QLearningAgent`: Off-policy TD control learning optimal Q\* values
- `SarsaAgent`: On-policy TD control learning Q^π for the followed policy
- `QTable`: State-action value storage with canonical state handling
- Parameters: learning_rate, discount_factor, epsilon (exploration), epsilon_decay, min_epsilon, q_init

**Active Inference Framework (active_inference/)**

- `GenerativeModel`: Bayesian agent using Expected Free Energy (EFE) minimization
- `OpponentKind`: Different opponent models (Uniform/Adversarial/Minimax)
- `PreferenceModel`: Utility over outcomes (win/draw/loss)
- EFE decomposes into: Risk + β_ambiguity × Ambiguity - β_epistemic × Epistemic

**Training Pipeline (pipeline/)**

- `Learner` trait: Uniform interface for all agent types (MENACE, Active Inference, Q-learning, SARSA, optimal, random, defensive)
- `Observer` trait: Progress tracking, metrics collection, JSONL export
- `TrainingRegimen`: Curriculum training with staged opponent progression
- `ComparisonFramework`: Round-robin tournament evaluation

**CLI and Commands (cli/)**

- `commands/train.rs`: Training with configurable opponents and observers
- `commands/compare.rs`: Round-robin tournament comparison
- `commands/evaluate.rs`: Evaluation against fixed opponents
- `commands/analyze/`: Analysis subcommands (game-tree, optimal, symmetry, trajectories, structures, active-inference)
- `commands/export.rs`: Data export in CSV/JSON/JSONL/MessagePack formats

### Key Architectural Concepts

**State Space and Filtering**

- All board states reduced to canonical forms via D4 symmetry group (8 transformations)
- StateFilter::Michie (287 states): Excludes forced moves and double-threat positions (original MENACE)
- StateFilter::DecisionOnly (304 states): Excludes only forced moves
- StateFilter::All (338 states): All X-to-move decision states (forced moves included)
- StateFilter::Both: Player-agnostic, includes both X and O decision states
- CLI analyze/export also supports `canonical` (765 states): full reachable symmetry-reduced state space (both players + terminal)

**Matchbox Learning Mechanics**

- Each decision state has a "matchbox" mapping legal moves to weights (probabilities)
- Weights start with bead schedule [4.0, 3.0, 2.0, 1.0] for plies [0-1, 2-3, 4-5, 6+]
- Reinforcement updates: Win → +strength, Loss → -strength, Draw → neutral
- Restocking prevents empty matchboxes: RestockMode::Box (whole box), Move (single bead), None

**Canonical Labels and Move Mapping**

- `CanonicalContext`: Caches expensive canonicalization for repeated operations
- `map_move_to_canonical()`: Transforms moves from original to canonical coordinates
- `map_canonical_to_original()`: Inverse transformation for move execution
- Always canonicalize states before workspace lookups

**Missing Matchboxes**

- States can legitimately lack matchboxes (forced moves, filtered states, wrong player)
- `handle_missing_matchbox()`: Plays through without learning (random or forced move)
- Returns dummy morphism labels excluded from `learnable_morphisms` set
- Enables gameplay through entire trajectory while learning only at designated decision points

### Data Flow Patterns

1. **Training**: Game → Trajectory (BoardStates) → Canonical labels → MenaceWorkspace lookups → Weighted sampling → Reinforcement updates
2. **Move Selection**: BoardState → Canonical context → Workspace.sample_move() → SampledMove (position + morphism label + distribution)
3. **Learning Update**: Game outcome → Reinforcement signal → Path of morphism labels → Workspace.apply_reinforcement() → Weight updates + restocking

## Module Relationships

```
CLI (bin/menace.rs) ──── Commands ──── Learner trait ──┬─ MenaceAgent
                                        (ports/)        ├─ OptimalPolicy
                                                        ├─ ActiveInference
                                                        ├─ QLearning/SARSA
                                                        └─ Random/Defensive

Pipeline ──────────── TrainingPipeline ──── uses Learner + Observer
                      ComparisonFramework ── multiple Learners

MenaceWorkspace ────── Central data structure for weights/morphisms
                      Used by ClassicMenace and ActiveInferenceLearner

GenerativeModel ────── EFE computation for Active Inference
                      Opponent modeling, preference optimization

BoardState/Game ────── Core game logic and state representation
GameTree ───────────── Symmetry-reduced state space construction
```

## Testing Strategy

### Unit Tests (108 tests)

Embedded in source files alongside implementation:

- BoardState operations, symmetry transformations
- Learning updates and weight management
- EFE computation and opponent models
- Pipeline operations and observers

### Integration Tests (116 tests in tests/)

- `exact_count_validation.rs`: Validates state space mathematics (765 canonical states, 5478 total states)
- `game_tree_structure.rs`: Verifies game tree properties and symmetry reductions
- `menace_regressions.rs`: Regression tests for training convergence
- `active_inference_minimax.rs`: Active Inference agent validation
- `double_threat_golden.rs`: Double-threat position detection
- `tictactoe_validation.rs`: Core game logic correctness
- `workspace_serialization.rs`: Save/load functionality
- `pipeline_tests.rs`: Full training/comparison workflows
- `ambiguity_computation.rs`: EFE component verification
- `active_inference_prefs.rs`: Preference model validation

## Mathematical Foundations

### Game Tree Enumeration

- 255,168 total game trajectories (sequences from root to terminal)
- 5,478 distinct states (with symmetry)
- 765 canonical states after D4 reduction
- 287 Michie decision states (excludes forced + double-threat)

### D4 Dihedral Group

- Identity + 3 rotations (90°, 180°, 270°) + 4 reflections
- Board position mapping: transforms applied via lookup tables
- Canonical form: lexicographically smallest encoding under all transformations

### Active Inference EFE

```
EFE = Risk + β_ambiguity × Ambiguity - β_epistemic × Epistemic

Where:
- Risk: KL divergence between predicted and preferred outcomes
- Ambiguity: Shannon entropy H[Q(o|π)] of outcome distribution
- Epistemic: Information gain I(θ; o|π) about opponent parameters
- Beta parameters control exploration-exploitation trade-offs
```

## Key Types and Data Structures

### Core Types

- `BoardState`: 10-byte Copy type (9 cells + player to move)
- `CanonicalLabel`: Newtype for canonical state representation
- `Position`: Validated board position (0-8)
- `Weight`: Non-negative weight value
- `SampledMove`: Move with probability distribution

### Central Structures

- `MenaceWorkspace`: HashMap of state→weights with morphism tracking
- `GenerativeModel`: Active Inference agent with belief updates
- `TrainingPipeline`: Orchestrates training with learners and observers
- `ComparisonFramework`: Round-robin tournament evaluation

## Key Invariants

1. **Piece Count Consistency**: X count == O count (X to move) or X count == O count + 1 (O to move)
2. **Canonical State Uniqueness**: Each canonical label maps to exactly one decision state
3. **Morphism Source Tracking**: Every morphism in the workspace has an entry in morphism_sources
4. **Weight Non-Negativity**: All weights clamped to [0.0, ∞) after updates
5. **Learnable Morphisms**: Only morphisms from decision states (per StateFilter) are updated during learning

## Common Pitfalls

### Canonicalization Errors

- Always canonicalize states before workspace lookups
- Use CanonicalContext for repeated operations to avoid redundant canonicalization
- Move indices must be transformed between original and canonical coordinates

### Matchbox Lifecycle

- Check `has_matchbox()` before assuming state has learnable moves
- Don't panic on missing matchboxes; handle gracefully (see `handle_missing_matchbox`)
- Restocking only applies during learning updates, not during initial construction

### Player Perspective

- BoardState.to_move indicates whose turn it is, not whose perspective
- Use `flip_perspective()` for opponent agents trained as X to evaluate O positions
- StateFilter::Both enables player-agnostic learning (both X and O decision states)

## External Dependencies

### Core Dependencies

- `rand`: Random number generation
- `serde` + `serde_json`: Serialization
- `rmp-serde`: MessagePack serialization
- `clap`: CLI argument parsing
- `thiserror`: Error handling (library)
- `anyhow`: Error handling (binary)
- `statrs`: Statistical computations
- `indicatif`: Progress bars

### Python Analysis Tools

Python environment managed via `uv` or standard virtualenv with dependencies in `pyproject.toml`:

- `matplotlib`, `seaborn`: Visualization
- `pandas`, `pyarrow`: Data processing
- `pyyaml`: Experiment configuration
- `ipykernel`, `markitdown`: Development tools

## Coding Standards

This project follows strict Rust best practices documented in `docs/rust-style-guide.md` and `docs/rust-hexagonal-architecture-guide.md`. Key principles:

### Error Handling

- Use `thiserror` for all library errors (see `src/error.rs`)
- Structured error types at module boundaries with descriptive variants
- Never use `Result<T, String>` - always use typed errors
- Binary (`src/bin/menace.rs`) uses `anyhow` for application-level errors

### Type Safety and Domain Modeling

- Newtype pattern for domain identifiers: `Position`, `Weight`, `CanonicalLabel`
- Parse, don't validate: Make invalid states unrepresentable
- Value objects enforce invariants at construction time
- Example: `BoardState::from_label()` validates piece counts and turn consistency

### Architecture Principles

- Domain logic depends on nothing external (pure business logic)
- Trait boundaries for external concerns (see `Learner`, `Observer` traits)
- Services orchestrate domain logic (`TrainingPipeline`, `ComparisonFramework`)
- Clear separation between core game logic (tictactoe/) and learning mechanisms (workspace.rs, pipeline/)

### Performance Considerations

- `BoardState` is Copy (10 bytes) for zero-cost moves through call stack
- Pre-allocated collections where size is known
- Iterator chains over intermediate allocations
- Canonical state caching to avoid redundant symmetry searches

## Documentation

Extensive documentation in `docs/`:

- `docs/README.md`: CLI reference and feature overview
- `docs/rust-style-guide.md`: Error handling, type design, architecture patterns
- `docs/rust-hexagonal-architecture-guide.md`: Domain modeling, ports, adapters, testing
- `docs/analysis/`: Game-theoretic analysis and mathematical foundations

Experiment scripts and analysis tools in `scripts/`:

- `scripts/automation/experiment_driver.py`: Main experiment orchestration
- `configs/experiments.yaml`: Default experiment configuration
- `scripts/reporting/generate_performance_report.py`: Performance metrics report generation
- Various analysis scripts for Active Inference, ambiguity, and opponent models
