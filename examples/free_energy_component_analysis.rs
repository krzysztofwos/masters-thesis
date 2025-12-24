//! Free Energy Component Analysis (Uniform Prior Diagnostic)
//!
//! This example purposefully measures F(π) against the *uniform* MENACE prior
//! (the untrained workspace) to show why Free Energy can increase even as
//! gameplay improves: KL[q‖p_uniform] grows faster than expected surprise falls.
//! Use it as a counter-example when discussing why the choice of prior matters.

use menace::{
    Result,
    active_inference::{PreferenceModel, preferences::CANONICAL_PREFERENCE_PROBS},
    analysis::FreeEnergyAnalysis,
    menace::MenaceAgent,
    pipeline::{MenaceLearner, RandomLearner, TrainingConfig, TrainingPipeline},
    tictactoe::Player,
    workspace::{InitialBeadSchedule, MenaceWorkspace, RestockMode, StateFilter},
};

fn main() -> Result<()> {
    println!("\n=== Free Energy Component Analysis ===\n");
    println!("Investigating: Why does F(π) increase during training?\n");

    // Configuration
    let (win_pref, draw_pref, loss_pref) = CANONICAL_PREFERENCE_PROBS;
    let preferences = PreferenceModel::from_probabilities(win_pref, draw_pref, loss_pref);
    let analysis = FreeEnergyAnalysis::new(preferences, Player::X);
    let checkpoints = vec![0, 50, 100, 200, 500, 1000];

    println!("Configuration:");
    println!("  Checkpoints: {checkpoints:?}");
    println!("  Preferences: P(win)={win_pref}, P(draw)={draw_pref}, P(loss)={loss_pref}\n");

    // Create initial workspace (prior)
    let initial_workspace = MenaceWorkspace::with_config(
        StateFilter::Michie,
        RestockMode::default(),
        InitialBeadSchedule::default(),
    )?;

    // Compute F(π) for initial (untrained) workspace
    println!("=== Checkpoint 0 (Untrained) ===");
    let fe_initial = analysis.compute_free_energy(&initial_workspace, &initial_workspace)?;
    println!("  Total F(π): {:.2} nats", fe_initial.total);
    println!(
        "  Expected Surprise: {:.2} nats ({:.1}%)",
        fe_initial.expected_surprise,
        100.0 * fe_initial.expected_surprise / fe_initial.total
    );
    println!(
        "  KL Divergence: {:.2} nats ({:.1}%)",
        fe_initial.kl_divergence,
        100.0 * fe_initial.kl_divergence / fe_initial.total
    );
    println!(
        "  Per-state: {:.4} nats/state\n",
        fe_initial.total / fe_initial.num_states as f64
    );

    // Create MENACE agent
    let seed = 42;
    let menace_agent = MenaceAgent::new(Some(seed))?;
    let mut agent = MenaceLearner::new(menace_agent, "MENACE".to_string());
    let mut opponent = RandomLearner::new("Random".to_string());

    let mut games_trained = 0;
    let mut results = Vec::new();

    // Track components at each checkpoint
    for &num_games in &checkpoints[1..] {
        // Skip checkpoint 0 (already computed)
        let games_to_train = num_games - games_trained;

        // Train
        let config = TrainingConfig {
            num_games: games_to_train,
            seed: Some(seed),
            agent_player: Player::X,
            first_player: Player::X,
        };

        let mut pipeline = TrainingPipeline::new(config);
        pipeline.run(&mut agent, &mut opponent)?;

        games_trained = num_games;

        // Compute Free Energy with full component breakdown
        let trained_workspace = &agent.agent().workspace;
        let fe = analysis.compute_free_energy(trained_workspace, &initial_workspace)?;

        println!("=== Checkpoint {num_games} ===");
        println!("  Total F(π): {:.2} nats", fe.total);
        println!(
            "  Expected Surprise: {:.2} nats ({:.1}%)",
            fe.expected_surprise,
            100.0 * fe.expected_surprise / fe.total
        );
        println!(
            "  KL Divergence: {:.2} nats ({:.1}%)",
            fe.kl_divergence,
            100.0 * fe.kl_divergence / fe.total
        );
        println!(
            "  Per-state: {:.4} nats/state",
            fe.total / fe.num_states as f64
        );

        // Changes from initial
        let delta_total = fe.total - fe_initial.total;
        let delta_surprise = fe.expected_surprise - fe_initial.expected_surprise;
        let delta_kl = fe.kl_divergence - fe_initial.kl_divergence;

        println!("  Changes from initial:");
        println!(
            "    ΔF(π) = {:+.2} nats ({:+.1}%)",
            delta_total,
            100.0 * delta_total / fe_initial.total
        );
        println!(
            "    ΔSurprise = {:+.2} nats ({:+.1}%)",
            delta_surprise,
            100.0 * delta_surprise / fe_initial.expected_surprise
        );
        println!(
            "    ΔKL = {:+.2} nats ({:+.1}%)",
            delta_kl,
            100.0 * delta_kl / fe_initial.kl_divergence
        );
        println!();

        results.push((
            num_games,
            fe.total,
            fe.expected_surprise,
            fe.kl_divergence,
            delta_total,
            delta_surprise,
            delta_kl,
        ));
    }

    // Summary analysis
    println!("=== Summary Analysis ===\n");

    println!("Trajectory:");
    println!("  Games | Total F(π) | Surprise | KL Div | ΔF(π) | ΔSurprise | ΔKL");
    println!("  ------|-----------|----------|--------|-------|-----------|-----");
    println!(
        "  {:5} | {:9.2} | {:8.2} | {:6.2} |   N/A |       N/A |  N/A",
        0, fe_initial.total, fe_initial.expected_surprise, fe_initial.kl_divergence
    );
    for (games, total, surprise, kl, dt, ds, dk) in &results {
        println!(
            "  {games:5} | {total:9.2} | {surprise:8.2} | {kl:6.2} | {dt:+5.1} | {ds:+9.1} | {dk:+4.1}"
        );
    }
    println!();

    println!("Key Findings:");

    // Check which component dominates the increase
    let final_result = results.last().unwrap();
    let (_, _, _, _, delta_total, delta_surprise, delta_kl) = final_result;

    if delta_surprise.abs() > delta_kl.abs() {
        println!(
            "  ✓ Surprise change dominates ({:.1} vs {:.1} nats)",
            delta_surprise.abs(),
            delta_kl.abs()
        );
        if *delta_surprise > 0.0 {
            println!("  ⚠ Surprise INCREASES (unexpected - policy should improve outcomes)");
        }
    } else {
        println!(
            "  ✓ KL divergence change dominates ({:.1} vs {:.1} nats)",
            delta_kl.abs(),
            delta_surprise.abs()
        );
        if *delta_kl > 0.0 {
            println!(
                "  ✓ Policy diverges from uniform prior (expected - learning sharpens policy)"
            );
        }
    }

    println!();
    println!("Interpretation:");
    if *delta_total > 0.0 {
        println!("  F(π) increases during training, suggesting:");
        println!("  1. MENACE does NOT minimize F(π) as currently formulated");
        println!("  2. Policy sharpening (↑KL) outweighs outcome improvement (↓Surprise)");
        println!("  3. Theoretical framework requires revision");
        println!();
        println!("  Possible explanations:");
        println!("  - Wrong choice of prior p(a) (uniform may not be appropriate)");
        println!("  - Bead mechanics optimize different objective than F(π)");
        println!("  - Opponent model mismatch (computing F assuming random opponent)");
    }

    Ok(())
}
