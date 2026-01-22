# MENACE as a Bayesian Observer

**A Technical Analysis through the Free Energy Principle**

Master's Thesis — Nakayama Laboratory, Graduate School of Information Science and Technology, The University of Tokyo

## Abstract

This thesis uses Donald Michie's MENACE — an interpretable matchbox-and-bead learner for Tic-Tac-Toe — as a concrete, fully analysable bridge between a working learning mechanism and an Active Inference interpretation under the Free Energy Principle.

We map MENACE's 287 matchboxes and bead updates to a Dirichlet–categorical model: bead counts act as Dirichlet pseudo-counts and random bead draws implement posterior predictive probability matching. Under explicit modelling commitments, MENACE corresponds to an instrumental special case of expected free energy minimisation (λ = 0), while Active Inference variants introduce epistemic value via mutual information.

## Thesis Contributions

1. **Dirichlet-Categorical Mapping**: A precise mapping from MENACE's mechanics to Active Inference using Dirichlet-categorical distributions
2. **Instrumental Equivalence**: Identification of MENACE as an instrumental Active Inference agent (λ = 0)
3. **Empirical Validation**: Systematic comparison against Active Inference variants and Reinforcement Learning baselines
4. **Information Cost Framework**: A generative model in which λ parameterizes the cost of information acquisition — directly answering Michie's 1966 question

## Repository Structure

```
├── src/                    # Rust implementation
│   ├── tictactoe/          # Game core, symmetry, game tree
│   ├── menace/             # Classic MENACE algorithm
│   ├── active_inference/   # Active Inference agents
│   ├── q_learning/         # Q-learning and SARSA
│   └── pipeline/           # Training and evaluation
├── thesis/                 # LaTeX thesis source
├── presentation/           # Quarto/Reveal.js slides
├── configs/                # Experiment configurations
├── scripts/                # Analysis and automation
├── tests/                  # Integration tests
└── docs/                   # Additional documentation
```

## Building

### Rust Implementation

```bash
cargo build --release
cargo test
```

### Thesis (LaTeX)

```bash
make -C thesis
```

### Presentation (Quarto)

```bash
cd presentation
quarto preview presentation.qmd
```

## Running Experiments

```bash
# Train MENACE agent
cargo run --bin menace -- train menace -o random -g 500 --seed 42

# Train Active Inference agent
cargo run --bin menace -- train active-inference --opponent minimax --games 500 --ai-epistemic-weight 0.5

# Compare agents
cargo run --bin menace -- compare menace random optimal -g 100

# Run full experiment suite
make thesis-results
```

## Key Results

| Algorithm              | Draw Rate (%) | Loss Rate (%) |
| ---------------------- | :-----------: | :-----------: |
| MENACE (box restock)   |  84.5 ± 8.1   |  15.5 ± 8.1   |
| Instrumental AIF (λ=0) |  88.1 ± 3.9   |  11.9 ± 3.9   |

MENACE and an Active Inference agent with epistemic drive suppressed achieve comparable performance, supporting the identification of MENACE as the purely instrumental special case.

## Presentation

View the slides online: [krzysztofwos.github.io/masters-thesis](https://krzysztofwos.github.io/masters-thesis/)

## Documentation

- [CLAUDE.md](CLAUDE.md) — Detailed technical documentation
- [docs/](docs/) — Architecture guides and analysis

## Author

Krzysztof Woś
