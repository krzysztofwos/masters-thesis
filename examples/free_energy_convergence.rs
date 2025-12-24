//! Free Energy Convergence Analysis (Optimal Prior)
//!
//! This example validates that F(π) decreases monotonically when it is measured
//! against the minimax policy (π*). It pairs with `free_energy_component_analysis.rs`,
//! which shows divergence under the uniform prior.
//!
//! Experiment Design:
//! - Train MENACE for 1000 games
//! - Track F(π) at checkpoints: 50, 100, 200, 500, 1000
//! - Visualize trajectory to validate monotonic decrease
//! - Multiple replications for statistical confidence

mod common;

use std::path::Path;

use common::workspaces::{OptimalPolicyMode, build_optimal_workspace};
use menace::{
    Result,
    active_inference::{PreferenceModel, preferences::CANONICAL_PREFERENCE_PROBS},
    analysis::{FreeEnergyAnalysis, FreeEnergyComponents},
    menace::MenaceAgent,
    pipeline::{MenaceLearner, RandomLearner, TrainingConfig, TrainingPipeline},
    tictactoe::Player,
    workspace::StateFilter,
};

fn main() -> Result<()> {
    println!("\n=== Free Energy Convergence Analysis ===\n");
    println!("Validating hypothesis: F(π) decreases monotonically during training\n");

    // Configuration
    let (win_pref, draw_pref, loss_pref) = CANONICAL_PREFERENCE_PROBS;
    let preferences = PreferenceModel::from_probabilities(win_pref, draw_pref, loss_pref);
    let analysis = FreeEnergyAnalysis::new(preferences, Player::X);
    let checkpoints = vec![50, 100, 200, 500, 1000];
    let num_replications = 3; // Use 3 for demo (report suggests 10 for full study)

    println!("Configuration:");
    println!("  Checkpoints: {checkpoints:?}");
    println!("  Replications: {num_replications}");
    println!("  Preferences: P(win)={win_pref}, P(draw)={draw_pref}, P(loss)={loss_pref}\n");
    println!("  Prior for FE measurement: Optimal minimax policy (π*)\n");

    // Store results from all replications
    let mut all_results: Vec<Vec<(usize, FreeEnergyComponents)>> = Vec::new();

    for replication in 0..num_replications {
        println!(
            "--- Replication {}/{} ---",
            replication + 1,
            num_replications
        );

        // Optimal prior (minimax policy) used for Free Energy evaluation
        let optimal_prior =
            build_optimal_workspace(StateFilter::Michie, false, OptimalPolicyMode::SingleBest)?;

        // Create MENACE agent
        let seed = 42 + replication as u64;
        let menace_agent = MenaceAgent::new(Some(seed))?;
        let mut agent = MenaceLearner::new(menace_agent, format!("MENACE-{replication}"));

        // Track Free Energy at checkpoints
        let mut checkpoint_results: Vec<(usize, FreeEnergyComponents)> = Vec::new();
        let mut games_trained = 0;

        // Create a single opponent that persists across training segments
        let mut opponent = RandomLearner::new("Random".to_string());

        for &num_games in &checkpoints {
            let games_to_train = num_games - games_trained;

            // Train for this segment (cumulative)
            let config = TrainingConfig {
                num_games: games_to_train,
                seed: Some(seed),
                agent_player: Player::X,
                first_player: Player::X,
            };

            let mut pipeline = TrainingPipeline::new(config);
            pipeline.run(&mut agent, &mut opponent)?;

            games_trained = num_games;

            // Compute Free Energy at this checkpoint
            let trained_workspace = &agent.agent().workspace;
            let fe = analysis.compute_free_energy(trained_workspace, &optimal_prior)?;
            let fe_normalized = fe.normalized();

            println!(
                "  Checkpoint {}: F(π) = {:.2} nats ({:.4} nats/state)",
                num_games, fe.total, fe_normalized.total,
            );
            println!(
                "    Expected surprise: {:.2} nats ({:.4} nats/state)",
                fe.expected_surprise, fe_normalized.expected_surprise,
            );
            println!(
                "    KL divergence: {:.2} nats ({:.4} nats/state)",
                fe.kl_divergence, fe_normalized.kl_divergence,
            );

            checkpoint_results.push((num_games, fe));
        }

        all_results.push(checkpoint_results);
        println!();
    }

    // Analysis: Check for monotonic decrease
    println!("=== Convergence Analysis ===\n");

    // Average across replications
    let mut avg_total = vec![0.0; checkpoints.len()];
    let mut avg_expected = vec![0.0; checkpoints.len()];
    let mut avg_kl = vec![0.0; checkpoints.len()];
    let mut state_counts = vec![0usize; checkpoints.len()];

    for results in &all_results {
        for (i, (_, fe)) in results.iter().enumerate() {
            avg_total[i] += fe.total;
            avg_expected[i] += fe.expected_surprise;
            avg_kl[i] += fe.kl_divergence;
            if state_counts[i] == 0 {
                state_counts[i] = fe.num_states;
            }
        }
    }

    let replications_f = num_replications as f64;
    for i in 0..checkpoints.len() {
        avg_total[i] /= replications_f;
        avg_expected[i] /= replications_f;
        avg_kl[i] /= replications_f;
    }

    println!("Average Free Energy across {num_replications} replications:");
    for (i, &checkpoint) in checkpoints.iter().enumerate() {
        let states = state_counts[i].max(1) as f64;
        println!(
            "  {checkpoint} games: total={:.2} nats (surprise={:.2}, KL={:.2})",
            avg_total[i], avg_expected[i], avg_kl[i],
        );
        println!(
            "    per-state: total={:.4}, surprise={:.4}, KL={:.4}",
            avg_total[i] / states,
            avg_expected[i] / states,
            avg_kl[i] / states,
        );
    }
    println!();

    // Check monotonicity
    let mut is_monotonic = true;
    let mut decreases = Vec::new();
    for window in avg_total.windows(2) {
        let decrease = window[0] - window[1];
        decreases.push(decrease);
        if decrease < 0.0 {
            is_monotonic = false;
        }
    }

    println!("Monotonic Decrease Test:");
    println!(
        "  Result: {}",
        if is_monotonic { "✓ PASS" } else { "✗ FAIL" }
    );
    println!("  Changes:");
    for (i, &decrease) in decreases.iter().enumerate() {
        let from = checkpoints[i];
        let to = checkpoints[i + 1];
        let symbol = if decrease >= 0.0 { "↓" } else { "↑" };
        println!(
            "    {from} → {to} games: {symbol} {:.2} nats",
            decrease.abs()
        );
    }
    println!();

    // Total decrease
    let initial_total = avg_total.first().copied().unwrap_or(0.0);
    let final_total = avg_total.last().copied().unwrap_or(0.0);
    let total_decrease = initial_total - final_total;
    let percent_decrease = if initial_total.abs() > f64::EPSILON {
        100.0 * total_decrease / initial_total
    } else {
        0.0
    };
    println!("Total Free Energy Reduction:");
    println!("  Initial: {initial_total:.2} nats");
    println!("  Final: {final_total:.2} nats");
    println!("  Decrease: {total_decrease:.2} nats ({percent_decrease:.1}%)");
    println!();

    // Export results for visualization
    export_results(
        &all_results,
        &checkpoints,
        "results/free_energy_convergence.json",
    )?;

    println!("✓ Results exported to results/free_energy_convergence.json");
    println!(
        "  Use: python -m scripts.reporting.visualize_free_energy results/free_energy_convergence.json"
    );

    Ok(())
}

fn export_results(
    all_results: &[Vec<(usize, FreeEnergyComponents)>],
    checkpoints: &[usize],
    path: &str,
) -> Result<()> {
    use std::fs;

    // Create results directory if it doesn't exist
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }

    // Average the results across replications
    let mut avg_results = Vec::new();
    for (i, &checkpoint) in checkpoints.iter().enumerate() {
        let mut sum_expected = 0.0;
        let mut sum_kl = 0.0;
        let mut sum_total = 0.0;
        let mut num_states = 0usize;

        for results in all_results {
            let (_, fe) = &results[i];
            sum_expected += fe.expected_surprise;
            sum_kl += fe.kl_divergence;
            sum_total += fe.total;
            if num_states == 0 {
                num_states = fe.num_states;
            }
        }

        let count = all_results.len() as f64;
        let avg_expected = sum_expected / count;
        let avg_kl = sum_kl / count;
        let avg_total = sum_total / count;
        let states_f = num_states as f64;

        let avg_data = serde_json::json!({
            "expected_surprise": avg_expected,
            "expected_surprise_normalized": if num_states > 0 { avg_expected / states_f } else { 0.0 },
            "kl_divergence": avg_kl,
            "kl_divergence_normalized": if num_states > 0 { avg_kl / states_f } else { 0.0 },
            "total": avg_total,
            "total_normalized": if num_states > 0 { avg_total / states_f } else { 0.0 },
            "num_states": num_states,
            "run": "Free Energy Convergence (avg)",
            "game_num": checkpoint,
            "step": checkpoint,
        });

        avg_results.push((checkpoint, avg_data));
    }

    let json = serde_json::to_string_pretty(&avg_results)?;
    fs::write(path, json)?;

    Ok(())
}
