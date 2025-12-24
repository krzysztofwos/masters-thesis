//! Demonstration of Active Inference learning algorithm
//!
//! This example shows how the Active Inference algorithm learns through
//! Expected Free Energy minimization and belief updates about the opponent.

use menace::{
    menace::{
        builder::MenaceAgentBuilder,
        training::{OpponentType, TrainingConfig, TrainingSession},
    },
    tictactoe::Player,
};

fn main() {
    println!("Active Inference Learning Demonstration");
    println!("=======================================\n");

    // Create Active Inference agent with uniform opponent model
    println!("Creating Active Inference agent with uniform opponent model...");
    let agent = MenaceAgentBuilder::new()
        .seed(42)
        .active_inference_uniform(1.0)
        .build()
        .expect("agent construction should succeed");
    println!("Algorithm: {}\n", agent.algorithm_name());

    // Training configuration
    let config = TrainingConfig {
        num_games: 200,
        opponent: OpponentType::Random,
        logging: false,
        seed: Some(42),
        restock: None,
        curriculum: None,
        agent_player: Player::X,
        first_player: Player::X,
    };

    println!("Training configuration:");
    println!("  Games: {}", config.num_games);
    println!("  Opponent: Random");
    println!("  Agent plays as: X");
    println!("  Epistemic weight: 1.0\n");

    // Train the agent
    println!("Training...");
    let mut session = TrainingSession::new(agent, config);
    session.train().expect("Training failed");

    // Report results
    println!("\nTraining complete!");
    let total_games = session.results.wins + session.results.draws + session.results.losses;
    println!("Results:");
    println!(
        "  Wins:   {} ({:.1}%)",
        session.results.wins,
        session.results.wins as f64 / total_games as f64 * 100.0
    );
    println!(
        "  Draws:  {} ({:.1}%)",
        session.results.draws,
        session.results.draws as f64 / total_games as f64 * 100.0
    );
    println!(
        "  Losses: {} ({:.1}%)",
        session.results.losses,
        session.results.losses as f64 / total_games as f64 * 100.0
    );

    // Show learned statistics
    println!("\nLearned statistics:");
    let stats = session.agent.algorithm_stats();
    for (key, value) in stats.iter() {
        println!("  {key}: {value:.2}");
    }

    // Show workspace statistics
    println!("\nWorkspace statistics:");
    let agent_stats = session.agent.stats();
    println!("  Total matchboxes: {}", agent_stats.total_matchboxes);
    println!("  Total beads: {:.0}", agent_stats.total_beads);
    println!("  Average entropy: {:.3}", agent_stats.avg_entropy);

    println!("\nComparison with Classic MENACE:");
    println!("  Active Inference uses Bayesian inference and EFE minimization");
    println!("  Classic MENACE uses reinforcement learning with bead updates");
    println!("  Both can learn optimal Tic-tac-toe strategies!");
}
