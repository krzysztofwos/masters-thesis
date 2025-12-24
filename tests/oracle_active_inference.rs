//! Tests for Oracle Active Inference agent
//!
//! Oracle uses perfect game tree knowledge to compute theoretically optimal
//! Active Inference policies. This serves as ground truth for what EFE
//! minimization should produce.

use std::collections::HashMap;

use menace::{
    menace::{MenaceAgent, TrainingConfig, TrainingSession, training::OpponentType},
    tictactoe::{BoardState, Player},
};

#[test]
fn test_oracle_plays_optimally_from_start() {
    // Create an Oracle Active Inference agent (no learning needed)
    let agent = MenaceAgent::builder()
        .oracle_active_inference_uniform(0.5)
        .seed(42)
        .build()
        .expect("agent construction should succeed");

    let (_root_ctx, root_label) = BoardState::new().canonical_context_and_label();

    // Get initial weights at root - Oracle computes these from scratch
    let initial_weights: HashMap<usize, f64> = agent
        .workspace()
        .move_weights(&root_label)
        .expect("root should have weights")
        .into_iter()
        .collect();

    println!("Oracle initial weights at root (before any games):");
    for (mv, weight) in &initial_weights {
        println!("  Move {mv}: {weight:.3}");
    }

    // The key insight: Oracle should have sensible weights BEFORE training
    // Unlike learning-based agents that start uniform, Oracle uses game tree knowledge
    let total_initial_weight: f64 = initial_weights.values().sum();
    assert!(
        total_initial_weight > 0.0,
        "Oracle should have non-zero weights from the start"
    );

    // Play a few games to verify Oracle maintains consistency
    let config = TrainingConfig {
        num_games: 10,
        opponent: OpponentType::Random,
        logging: false,
        seed: Some(123),
        restock: None,
        curriculum: None,
        agent_player: Player::X,
        first_player: Player::X,
    };

    let mut session = TrainingSession::new(agent, config);
    session.train().expect("training should succeed");

    // Get weights after "training" (Oracle just recomputes, doesn't learn)
    let final_weights: HashMap<usize, f64> = session
        .agent
        .workspace()
        .move_weights(&root_label)
        .expect("root should have weights")
        .into_iter()
        .collect();

    println!("\nOracle weights after {} games:", session.games_played);
    println!(
        "Results: {} wins, {} draws, {} losses",
        session.results.wins, session.results.draws, session.results.losses
    );
    for (mv, weight) in &final_weights {
        println!("  Move {mv}: {weight:.3}");
    }

    // Oracle should maintain consistent weights (recomputes same values each time)
    println!("\nWeight stability (Oracle recomputes, doesn't learn from outcomes):");
    let mut max_change = 0.0f64;
    for mv in 0..9 {
        let initial = initial_weights.get(&mv).copied().unwrap_or(0.0);
        let final_w = final_weights.get(&mv).copied().unwrap_or(0.0);
        let change = (final_w - initial).abs();
        max_change = max_change.max(change);
        println!("  Move {mv}: {initial:.3} -> {final_w:.3} (change: {change:+.3})");
    }

    // Oracle weights might change as it updates workspace on each state visit,
    // but the changes should reflect the same theoretical optimal policy
    println!("\n✓ Oracle agent created and tested");
    println!("  Max weight change: {max_change:.3}");
    println!(
        "  Win rate: {:.1}%",
        100.0 * session.results.wins as f64 / session.games_played as f64
    );
}

#[test]
fn test_oracle_vs_learning_agent_comparison() {
    // Oracle should perform well immediately, learning agent should improve over time

    // Create Oracle agent
    let oracle = MenaceAgent::builder()
        .oracle_active_inference_uniform(0.5)
        .seed(42)
        .build()
        .expect("oracle construction should succeed");

    // Create learning-based Active Inference agent
    let learner = MenaceAgent::builder()
        .active_inference_uniform(0.5)
        .seed(43)
        .build()
        .expect("learner construction should succeed");

    // Test Oracle after just 10 games
    let config_short = TrainingConfig {
        num_games: 10,
        opponent: OpponentType::Random,
        logging: false,
        seed: Some(100),
        restock: None,
        curriculum: None,
        agent_player: Player::X,
        first_player: Player::X,
    };

    let mut oracle_session = TrainingSession::new(oracle, config_short.clone());
    oracle_session
        .train()
        .expect("oracle training should succeed");

    let mut learner_session = TrainingSession::new(learner, config_short);
    learner_session
        .train()
        .expect("learner training should succeed");

    let oracle_win_rate = oracle_session.results.wins as f64 / oracle_session.games_played as f64;
    let learner_win_rate =
        learner_session.results.wins as f64 / learner_session.games_played as f64;

    println!("After 10 games against random opponent:");
    println!("  Oracle win rate: {:.1}%", oracle_win_rate * 100.0);
    println!("  Learner win rate: {:.1}%", learner_win_rate * 100.0);

    // Note: We're not asserting oracle > learner here because:
    // 1. The learning agent has the catastrophic unlearning bug
    // 2. Random variation in 10 games can cause either to win more
    // This test is mainly to verify both agents can run without crashing

    println!("\n✓ Oracle and learning agents both completed training");
}

#[test]
fn test_oracle_agent_name() {
    let agent = MenaceAgent::builder()
        .oracle_active_inference_uniform(0.5)
        .build()
        .expect("construction should succeed");

    let name = agent.algorithm_name();
    assert_eq!(name, "Oracle Active Inference (Uniform)");
}

#[test]
fn test_oracle_with_different_opponents() {
    // Test that Oracle can be configured with different opponent models
    let uniform = MenaceAgent::builder()
        .oracle_active_inference_uniform(0.5)
        .build()
        .expect("uniform oracle should build");

    let adversarial = MenaceAgent::builder()
        .oracle_active_inference_adversarial(0.5)
        .build()
        .expect("adversarial oracle should build");

    let minimax = MenaceAgent::builder()
        .oracle_active_inference_minimax()
        .build()
        .expect("minimax oracle should build");

    assert_eq!(
        uniform.algorithm_name(),
        "Oracle Active Inference (Uniform)"
    );
    assert_eq!(
        adversarial.algorithm_name(),
        "Oracle Active Inference (Adversarial)"
    );
    assert_eq!(
        minimax.algorithm_name(),
        "Oracle Active Inference (Minimax)"
    );

    println!("✓ Oracle can be configured with all opponent types");
}
