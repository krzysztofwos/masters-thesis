# MENACE Examples

Educational examples demonstrating key concepts in the MENACE implementation.

## Available Examples

### symmetry_demo.rs - D4 Symmetry Group Tutorial

```bash
cargo run --example symmetry_demo
```

A visual walkthrough of the D4 dihedral group as applied to Tic-tac-toe:

- All 8 D4 transformations with detailed explanations
- Board canonicalization process
- Symmetry reduction mathematics
- Stabilizer subgroups
- Practical MENACE applications

### active_inference_demo.rs - Active Inference Introduction

```bash
cargo run --example active_inference_demo
```

A minimal demonstration of Active Inference concepts:

- Active Inference agent creation
- Training configuration
- Results interpretation

### efe_micro.rs - One-Matchbox Expected Free Energy

```bash
cargo run --example efe_micro
```

Numerically reproduces the Dirichlet-categorical walkthrough from the thesis:
computes risk, entropy, mutual information, and one-step updates for a single
matchbox with closed-form values. Useful for sanity-checking the Active
Inference formulas without running the full game.

### free_energy_demo.rs - Free Energy Reporting Walkthrough

```bash
cargo run --example free_energy_demo
```

Trains MENACE for 100 games versus the random opponent and prints the raw and
per-state Free Energy components along with the training win/draw/loss counts.

### free_energy_component_analysis.rs - Uniform-Prior Diagnostic

```bash
cargo run --example free_energy_component_analysis
```

Computes F(π) against the _uniform_ MENACE prior (the untrained matchboxes) to
show why Free Energy can increase even while gameplay improves. Use this to
illustrate that KL[q‖p_uniform] grows as the policy sharpens; the result is a
counter-example to "F always decreases" when the prior is a poor fit.

### free_energy_convergence.rs - Optimal-Prior Convergence

```bash
cargo run --example free_energy_convergence
```

Computes F(π) against the minimax workspace and demonstrates the Lyapunov-style
decrease discussed in the thesis. This is the companion to the uniform-prior
diagnostic above and highlights why the choice of prior matters.

### free_energy_perfect_vs_perfect.rs - Nash Baseline

```bash
cargo run --example free_energy_perfect_vs_perfect
```

Computes Free Energy for π\* versus both a uniform and optimal opponent to show
how the opponent model affects the surprise term even when the agent already
plays optimally. This represents the game-theoretic Nash equilibrium reference.

## CLI Equivalents

The `menace` CLI provides production-ready implementations of all example functionality:

```bash
# Symmetry analysis
menace analyze symmetry --visualize --stabilizers

# Active Inference training
menace train --algorithm active-inference --opponent minimax

# Complete analysis suite
menace analyze --help
```

See the [CLI documentation](../docs/README.md) for complete command reference.
