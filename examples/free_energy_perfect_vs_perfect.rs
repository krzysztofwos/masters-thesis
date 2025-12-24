//! Free Energy Analysis: Perfect vs Perfect
//!
//! This example analyzes Free Energy when BOTH agents use optimal policies,
//! representing the game-theoretic equilibrium ("AlphaZero-style" but static).
//!
//! # Research Question
//!
//! What is Free Energy F(π*) when:
//! - Agent policy: π* (minimax optimal)
//! - Opponent model: π* (also optimal)
//! - Prior: π₀ (initial uniform)
//!
//! This represents the theoretical limit of co-evolutionary learning where
//! both agents converge to Nash equilibrium.
//!
//! # Theoretical Predictions
//!
//! - **Surprise**: Should be minimal (optimal play against optimal opponent)
//! - **KL Divergence**: Maximum divergence from uniform prior
//! - **Total F(π*)**: Balance between performance and complexity
//!
//! # Comparison
//!
//! We compare three scenarios:
//! 1. F_uniform(π*): Optimal policy vs uniform opponent (original)
//! 2. F_optimal(π*): Optimal policy vs optimal opponent (new)
//! 3. Component breakdown to understand differences

mod common;

use common::workspaces::{OptimalPolicyMode, build_optimal_workspace};
use menace::{
    Result,
    active_inference::{PreferenceModel, preferences::CANONICAL_PREFERENCE_PROBS},
    analysis::{FreeEnergyAnalysis, WorkspaceOpponent},
    tictactoe::Player,
    workspace::{InitialBeadSchedule, MenaceWorkspace, RestockMode, StateFilter},
};

fn main() -> Result<()> {
    println!("\n=== Free Energy Analysis: Perfect vs Perfect ===\n");
    println!("Analyzing Free Energy when both agents use optimal minimax policies");
    println!("This represents the game-theoretic Nash equilibrium\n");

    // Configuration
    let (win_pref, draw_pref, loss_pref) = CANONICAL_PREFERENCE_PROBS;
    let preferences = PreferenceModel::from_probabilities(win_pref, draw_pref, loss_pref);

    println!("Configuration:");
    println!("  Agent: Minimax optimal (π*)");
    println!("  Opponent Model: Minimax optimal (π*)");
    println!("  Prior: Uniform initial distribution (π₀)");
    println!("  Player: X (first mover)\n");

    // Step 1: Build optimal policy workspace
    println!("=== Building Optimal Policy Workspace ===\n");
    let optimal_workspace =
        build_optimal_workspace(StateFilter::Both, true, OptimalPolicyMode::SingleBest)?;
    println!("  ✓ Optimal workspace created\n");

    // Step 2: Build uniform prior workspace for KL divergence
    let prior_workspace = MenaceWorkspace::with_config(
        StateFilter::Michie,
        RestockMode::default(),
        InitialBeadSchedule::default(),
    )?;

    // Step 3: Analyze with THREE different opponent models
    println!("=== Scenario 1: Optimal Policy vs Uniform Opponent ===\n");

    let analysis_uniform = FreeEnergyAnalysis::new(preferences.clone(), Player::X);
    let fe_uniform = analysis_uniform.compute_free_energy(&optimal_workspace, &prior_workspace)?;

    println!("  F_uniform(π*) = {:.2} nats", fe_uniform.total);
    println!(
        "  Expected Surprise: {:.2} nats ({:.1}%)",
        fe_uniform.expected_surprise,
        100.0 * fe_uniform.expected_surprise / fe_uniform.total
    );
    println!(
        "  KL[π* || π₀]: {:.2} nats ({:.1}%)",
        fe_uniform.kl_divergence,
        100.0 * fe_uniform.kl_divergence / fe_uniform.total
    );
    println!(
        "  Per-state: {:.4} nats/state\n",
        fe_uniform.total / fe_uniform.num_states as f64
    );

    println!("=== Scenario 2: Optimal Policy vs Optimal Opponent ===\n");

    let optimal_opponent = WorkspaceOpponent::new(optimal_workspace.clone());
    let analysis_optimal =
        FreeEnergyAnalysis::with_opponent_model(preferences.clone(), Player::X, optimal_opponent);
    let fe_optimal = analysis_optimal.compute_free_energy(&optimal_workspace, &prior_workspace)?;

    println!("  F_optimal(π*) = {:.2} nats", fe_optimal.total);
    println!(
        "  Expected Surprise: {:.2} nats ({:.1}%)",
        fe_optimal.expected_surprise,
        100.0 * fe_optimal.expected_surprise / fe_optimal.total
    );
    println!(
        "  KL[π* || π₀]: {:.2} nats ({:.1}%)",
        fe_optimal.kl_divergence,
        100.0 * fe_optimal.kl_divergence / fe_optimal.total
    );
    println!(
        "  Per-state: {:.4} nats/state\n",
        fe_optimal.total / fe_optimal.num_states as f64
    );

    // Step 4: Comparative Analysis
    println!("=== Comparative Analysis ===\n");

    let delta_surprise = fe_optimal.expected_surprise - fe_uniform.expected_surprise;
    let delta_kl = fe_optimal.kl_divergence - fe_uniform.kl_divergence;
    let delta_total = fe_optimal.total - fe_uniform.total;

    println!("Differences (Optimal Opponent - Uniform Opponent):");
    println!(
        "  ΔSurprise = {:+.2} nats ({:+.1}%)",
        delta_surprise,
        100.0 * delta_surprise / fe_uniform.expected_surprise
    );
    println!(
        "  ΔKL = {:+.2} nats ({:+.1}%)",
        delta_kl,
        if fe_uniform.kl_divergence > 0.0 {
            100.0 * delta_kl / fe_uniform.kl_divergence
        } else {
            0.0
        }
    );
    println!(
        "  ΔF(π*) = {:+.2} nats ({:+.1}%)\n",
        delta_total,
        100.0 * delta_total / fe_uniform.total
    );

    // Step 5: Component Ratios
    println!("Component Ratios:");
    println!("  Uniform opponent:");
    println!(
        "    Surprise: {:.1}% | KL: {:.1}%",
        100.0 * fe_uniform.expected_surprise / fe_uniform.total,
        100.0 * fe_uniform.kl_divergence / fe_uniform.total
    );
    println!("  Optimal opponent:");
    println!(
        "    Surprise: {:.1}% | KL: {:.1}%\n",
        100.0 * fe_optimal.expected_surprise / fe_optimal.total,
        100.0 * fe_optimal.kl_divergence / fe_optimal.total
    );

    // Step 6: Theoretical Interpretation
    println!("=== Theoretical Interpretation ===\n");

    if delta_surprise < 0.0 {
        println!("Surprise DECREASES with optimal opponent:");
        println!("  ✓ Optimal vs optimal → deterministic outcome (always draw)");
        println!("  ✓ Lower uncertainty reduces expected surprise");
        println!("  ✓ Uniform opponent → many possible outcomes → higher uncertainty");
        println!("  ✓ This validates that opponent model critically affects surprise\n");
    } else {
        println!("Surprise increases with optimal opponent:");
        println!("  ✓ Optimal opponent is harder to beat than random");
        println!("  ✓ Even optimal policy has less preferred outcomes\n");
    }

    if delta_kl.abs() < 1e-6 {
        println!("KL divergence unchanged:");
        println!("  ✓ KL[π* || π₀] depends only on policy, not opponent");
        println!("  ✓ Complexity cost is independent of opponent model\n");
    } else {
        println!("KL divergence changes:");
        println!("  ? Unexpected - KL should be opponent-independent\n");
    }

    println!("Free Energy Decomposition:");
    println!("  F(π*) = Surprise(π*, opponent) + KL[π* || π₀]");
    println!("          \\___depends on opponent___/   \\__independent__/\n");

    println!("Key Insights:");
    println!("  1. Opponent model DOES affect expected surprise");
    println!("  2. Optimal vs optimal = game-theoretic equilibrium");
    println!("  3. This enables analyzing co-evolutionary dynamics");
    println!("  4. \"Perfect vs perfect\" represents learning convergence limit\n");

    // Step 7: Game-Theoretic Interpretation
    println!("=== Game-Theoretic Interpretation ===\n");
    println!("In tic-tac-toe with perfect play:");
    println!("  • Both agents play optimally → always draw");
    println!("  • Surprise reflects draw probability under preferences");
    println!("  • KL reflects how far optimal policy is from uniform\n");

    println!("For co-evolutionary learning:");
    println!("  • Self-play training should converge to this equilibrium");
    println!("  • F(π_t) trajectory depends on both agents' learning");
    println!("  • This provides a reference point for analyzing convergence\n");

    Ok(())
}
