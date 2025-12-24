//! Free Energy analysis demonstration
//!
//! This example demonstrates how to compute Free Energy for MENACE's learned policies
//! and track how it evolves during training.

use menace::{
    Result,
    active_inference::{PreferenceModel, preferences::CANONICAL_PREFERENCE_PROBS},
    analysis::FreeEnergyAnalysis,
    pipeline::{MenaceLearner, RandomLearner, TrainingConfig, TrainingPipeline},
    tictactoe::Player,
    workspace::{InitialBeadSchedule, MenaceWorkspace, RestockMode, StateFilter},
};

fn main() -> Result<()> {
    println!("\n=== Free Energy Analysis Demo ===\n");

    // Create preferences (strong preference for winning)
    let (win_pref, draw_pref, loss_pref) = CANONICAL_PREFERENCE_PROBS;
    let preferences = PreferenceModel::from_probabilities(win_pref, draw_pref, loss_pref);
    let analysis = FreeEnergyAnalysis::new(preferences, Player::X);

    // Create initial workspace (prior)
    let initial_workspace = MenaceWorkspace::with_config(
        StateFilter::Michie,
        RestockMode::default(),
        InitialBeadSchedule::default(),
    )?;

    // Create MENACE agent
    let menace_agent = menace::menace::MenaceAgent::new(Some(42))?;
    let mut agent = MenaceLearner::new(menace_agent, "MENACE".to_string());

    // Train against random opponent
    let config = TrainingConfig {
        num_games: 100,
        seed: Some(42),
        agent_player: Player::X,
        first_player: Player::X,
    };

    let mut pipeline = TrainingPipeline::new(config);
    let mut opponent = RandomLearner::new("Random".to_string());

    println!("Training MENACE for 100 games against random opponent...\n");
    let result = pipeline.run(&mut agent, &mut opponent)?;

    println!("Training Results:");
    println!(
        "  Wins:   {} ({:.1}%)",
        result.wins,
        result.win_rate * 100.0
    );
    println!(
        "  Draws:  {} ({:.1}%)",
        result.draws,
        result.draw_rate * 100.0
    );
    println!(
        "  Losses: {} ({:.1}%)\n",
        result.losses,
        result.loss_rate * 100.0
    );

    // Get trained workspace
    let trained_workspace = &agent.agent().workspace;

    // Compute Free Energy
    println!("Computing Free Energy...\n");
    let fe = analysis.compute_free_energy(trained_workspace, &initial_workspace)?;

    println!("Free Energy Analysis:");
    println!("  States analyzed: {}", fe.num_states);
    println!("  Expected surprise: {:.4} nats", fe.expected_surprise);
    println!("  KL divergence: {:.4} nats", fe.kl_divergence);
    println!("  Total Free Energy: {:.4} nats\n", fe.total);

    // Normalized (per-state average)
    let fe_norm = fe.normalized();
    println!("Normalized (per-state) Free Energy:");
    println!(
        "  Expected surprise: {:.4} nats/state",
        fe_norm.expected_surprise
    );
    println!("  KL divergence: {:.4} nats/state", fe_norm.kl_divergence);
    println!("  Total: {:.4} nats/state\n", fe_norm.total);

    println!("=== Interpretation ===");
    println!("Free Energy = Expected Surprise + Policy Complexity");
    println!("- Expected surprise measures how much the agent is surprised by outcomes");
    println!("- KL divergence measures how much the policy differs from the prior");
    println!("- Lower Free Energy indicates better alignment with preferences");

    Ok(())
}
