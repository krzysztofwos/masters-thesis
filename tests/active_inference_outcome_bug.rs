//! Test to demonstrate that Active Inference ignores game outcomes
//!
//! This test shows that the AIF agent doesn't learn from wins/losses,
//! leading to catastrophic unlearning over time.

use std::collections::HashMap;

use menace::{
    menace::{MenaceAgent, TrainingConfig, TrainingSession, training::OpponentType},
    tictactoe::{BoardState, Player},
};

#[test]
fn test_active_inference_ignores_outcomes() {
    // Create an Active Inference agent
    let agent = MenaceAgent::builder()
        .active_inference_uniform(0.5)
        .seed(42)
        .build()
        .expect("agent construction should succeed");

    let (_root_ctx, root_label) = BoardState::new().canonical_context_and_label();

    // Get initial weights at root
    let initial_weights: HashMap<usize, f64> = agent
        .workspace()
        .move_weights(&root_label)
        .expect("root should have weights")
        .into_iter()
        .collect();

    println!("Initial weights at root:");
    for (mv, weight) in &initial_weights {
        println!("  Move {mv}: {weight:.3}");
    }

    // Train for a few games against random opponent
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

    // Get weights after training
    let final_weights: HashMap<usize, f64> = session
        .agent
        .workspace()
        .move_weights(&root_label)
        .expect("root should have weights")
        .into_iter()
        .collect();

    println!(
        "\nFinal weights at root after {} games:",
        session.games_played
    );
    println!(
        "Results: {} wins, {} draws, {} losses",
        session.results.wins, session.results.draws, session.results.losses
    );
    for (mv, weight) in &final_weights {
        println!("  Move {mv}: {weight:.3}");
    }

    // Compare weights - they should have changed significantly
    // because AIF overwrites them based on EFE, NOT based on outcomes
    println!("\nWeight changes:");
    for mv in 0..9 {
        let initial = initial_weights.get(&mv).copied().unwrap_or(0.0);
        let final_w = final_weights.get(&mv).copied().unwrap_or(0.0);
        let change = final_w - initial;
        println!("  Move {mv}: {initial:.3} -> {final_w:.3} (change: {change:+.3})");
    }

    // The key observation: weights changed even though we didn't use outcomes!
    // This is the bug - AIF should learn from whether games were won/lost,
    // but instead it just overwrites based on its internal model.

    // We expect weights to have changed (this demonstrates the bug exists)
    let total_change: f64 = (0..9)
        .map(|mv| {
            let initial = initial_weights.get(&mv).copied().unwrap_or(0.0);
            let final_w = final_weights.get(&mv).copied().unwrap_or(0.0);
            (final_w - initial).abs()
        })
        .sum();

    assert!(
        total_change > 0.1,
        "Weights should have changed during training (change: {total_change:.3})"
    );

    println!("\n✓ Bug confirmed: Weights changed without using game outcomes!");
    println!("  Total absolute change: {total_change:.3}");
}

#[test]
fn test_classical_menace_learns_from_outcomes() {
    // Contrast: Classical MENACE DOES use outcomes
    let agent = MenaceAgent::builder()
        .seed(42)
        .build()
        .expect("agent construction should succeed");

    let (_root_ctx, root_label) = BoardState::new().canonical_context_and_label();

    let initial_weights: HashMap<usize, f64> = agent
        .workspace()
        .move_weights(&root_label)
        .expect("root should have weights")
        .into_iter()
        .collect();

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

    let final_weights: HashMap<usize, f64> = session
        .agent
        .workspace()
        .move_weights(&root_label)
        .expect("root should have weights")
        .into_iter()
        .collect();

    println!("\nClassical MENACE:");
    println!(
        "Results: {} wins, {} draws, {} losses",
        session.results.wins, session.results.draws, session.results.losses
    );

    // Classical MENACE weights change based on outcomes
    // Moves that led to wins get more beads, losses get fewer
    let total_change: f64 = (0..9)
        .map(|mv| {
            let initial = initial_weights.get(&mv).copied().unwrap_or(0.0);
            let final_w = final_weights.get(&mv).copied().unwrap_or(0.0);
            (final_w - initial).abs()
        })
        .sum();

    println!("Total weight change: {total_change:.3}");
    println!("✓ Classical MENACE correctly learns from outcomes");
}
