//! Evaluate command - Evaluate trained learners against opponents

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use serde::Serialize;

use crate::{cli::commands::train::parse_player_token, tictactoe::Player};

#[derive(Parser, Debug)]
#[command(about = "Evaluate a trained learner")]
pub struct EvaluateArgs {
    /// Path to trained learner file
    pub learner: PathBuf,

    /// Opponent to evaluate against
    #[arg(long, short = 'o', default_value = "optimal")]
    pub opponent: String,

    /// Number of evaluation games
    #[arg(long, short = 'g', default_value_t = 100)]
    pub games: usize,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Separate validation seed (defaults to training seed + 1)
    #[arg(long)]
    pub validation_seed: Option<u64>,

    /// Which token the evaluated agent controls (`x` or `o`)
    #[arg(long)]
    pub agent_player: Option<String>,

    /// Which token makes the first move in evaluation (`x` or `o`)
    #[arg(long)]
    pub first_player: Option<String>,

    /// Export results to file
    #[arg(long)]
    pub export: Option<PathBuf>,
}

struct AgentInfo {
    filter: Option<String>,
    algorithm: String,
    restock_mode: Option<String>,
    games_trained: Option<usize>,
    trained_against: Vec<String>,
    saved_at: Option<String>,
}

struct LoadedAgent {
    learner: Box<dyn crate::ports::Learner>,
    info: AgentInfo,
    agent_player: Player,
    first_player: Player,
    seed: Option<u64>,
}

pub fn execute(args: EvaluateArgs) -> Result<()> {
    use crate::pipeline::{
        DefensiveLearner, FrozenLearner, Learner, MetricsObserver, OptimalLearner,
        ProgressObserver, RandomLearner, TrainingConfig, TrainingPipeline,
    };

    // Load trained agent
    println!("Loading trained agent from: {}", args.learner.display());
    let loaded = load_agent(&args.learner)?;

    println!("\n=== Loaded Agent Info ===");
    if let Some(filter) = &loaded.info.filter {
        println!("Filter: {filter}");
    }
    println!("Algorithm: {}", loaded.info.algorithm);
    if let Some(restock) = &loaded.info.restock_mode {
        println!("Restock mode: {restock}");
    }
    if let Some(games) = loaded.info.games_trained {
        println!("Games trained: {games}");
    }
    if !loaded.info.trained_against.is_empty() {
        println!(
            "Trained against: {}",
            loaded.info.trained_against.join(", ")
        );
    }

    // Wrap in learner trait
    let saved_seed = loaded.seed;
    let mut trained_learner = loaded.learner;
    let saved_agent_player = loaded.agent_player;
    let saved_first_player = loaded.first_player;

    let agent_player = if let Some(ref value) = args.agent_player {
        parse_player_token(value, "--agent-player")?
    } else {
        saved_agent_player
    };

    let first_player = if let Some(ref value) = args.first_player {
        parse_player_token(value, "--first-player")?
    } else {
        saved_first_player
    };

    // Create opponent
    let mut opponent: Box<dyn Learner> = match args.opponent.to_lowercase().as_str() {
        "random" => Box::new(RandomLearner::new("Random".to_string())),
        "optimal" => Box::new(OptimalLearner::new("Optimal".to_string())),
        "defensive" => Box::new(DefensiveLearner::new("Defensive".to_string())),
        other => {
            return Err(anyhow::anyhow!(
                "Unknown opponent type: '{other}'. Supported: random, optimal, defensive"
            ));
        }
    };

    println!("\n=== Evaluation Configuration ===");
    println!("Opponent: {}", opponent.name());
    println!("Agent plays as: {agent_player:?} (first player: {first_player:?})");
    println!("Games: {}", args.games);

    let evaluation_seed = args
        .validation_seed
        .or(args.seed)
        .or_else(|| saved_seed.map(|s| s.wrapping_add(1)));
    if let Some(seed) = evaluation_seed {
        println!("Seed: {seed}");
    }

    // Create evaluation pipeline
    let config = TrainingConfig {
        num_games: args.games,
        seed: evaluation_seed,
        agent_player,
        first_player,
    };

    let mut pipeline = TrainingPipeline::new(config);

    // Add progress bar
    pipeline = pipeline.with_observer(Box::new(ProgressObserver::new()));

    // Add metrics observer
    let metrics_observer = MetricsObserver::new();
    pipeline = pipeline.with_observer(Box::new(metrics_observer));

    // Run evaluation
    println!("\n=== Running Evaluation ===");
    let mut frozen_learner = FrozenLearner::new(trained_learner.as_mut());
    let result = pipeline.run(&mut frozen_learner, opponent.as_mut())?;

    // Print results
    println!("\n=== Evaluation Results ===");
    println!("Total games: {}", result.total_games);
    println!("Wins: {} ({:.1}%)", result.wins, result.win_rate * 100.0);
    println!("Draws: {} ({:.1}%)", result.draws, result.draw_rate * 100.0);
    println!(
        "Losses: {} ({:.1}%)",
        result.losses,
        result.loss_rate * 100.0
    );

    // Export if requested
    if let Some(export_path) = &args.export {
        export_results(&result, &loaded.info, &args, export_path)?;
        println!("\n✓ Results exported to: {}", export_path.display());
    }

    Ok(())
}

/// Export evaluation results to JSON
fn export_results(
    result: &crate::pipeline::TrainingResult,
    saved: &AgentInfo,
    args: &EvaluateArgs,
    path: &PathBuf,
) -> Result<()> {
    use std::fs::File;

    #[derive(Serialize)]
    struct EvaluationExport {
        evaluation: EvaluationSection,
        agent: AgentSection,
    }

    #[derive(Serialize)]
    struct EvaluationSection {
        agent_file: String,
        opponent: String,
        total_games: usize,
        wins: usize,
        draws: usize,
        losses: usize,
        win_rate: f64,
        draw_rate: f64,
        loss_rate: f64,
    }

    #[derive(Serialize)]
    struct AgentSection {
        #[serde(skip_serializing_if = "Option::is_none")]
        filter: Option<String>,
        algorithm: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        restock_mode: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        games_trained: Option<usize>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        trained_against: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        saved_at: Option<String>,
    }

    let export = EvaluationExport {
        evaluation: EvaluationSection {
            agent_file: args.learner.display().to_string(),
            opponent: args.opponent.clone(),
            total_games: result.total_games,
            wins: result.wins,
            draws: result.draws,
            losses: result.losses,
            win_rate: result.win_rate,
            draw_rate: result.draw_rate,
            loss_rate: result.loss_rate,
        },
        agent: AgentSection {
            filter: saved.filter.clone(),
            algorithm: saved.algorithm.clone(),
            restock_mode: saved.restock_mode.clone(),
            games_trained: saved.games_trained,
            trained_against: saved.trained_against.clone(),
            saved_at: saved.saved_at.clone(),
        },
    };

    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, &export)?;
    Ok(())
}

fn load_agent(path: &PathBuf) -> Result<LoadedAgent> {
    use crate::{menace::SavedMenaceAgent, pipeline::MenaceLearner, q_learning::SavedTdAgent};

    match SavedMenaceAgent::load_from_file(path) {
        Ok(saved) => {
            let agent = saved.to_agent()?;
            let learner: Box<dyn crate::ports::Learner> =
                Box::new(MenaceLearner::new(agent, "Trained Agent".to_string()));

            let info = AgentInfo {
                filter: Some(format!("{:?}", saved.state_filter)),
                algorithm: format!("{:?}", saved.algorithm_type),
                restock_mode: Some(format!("{:?}", saved.restock_mode)),
                games_trained: saved.metadata.games_trained,
                trained_against: saved.metadata.opponents.clone(),
                saved_at: saved.metadata.saved_at.clone(),
            };

            let agent_player = saved.metadata.agent_player.unwrap_or(Player::X);
            let first_player = saved.metadata.first_player.unwrap_or(Player::X);
            let seed = saved.metadata.seed;

            Ok(LoadedAgent {
                learner,
                info,
                agent_player,
                first_player,
                seed,
            })
        }
        Err(menace_err) => {
            let saved_td = SavedTdAgent::load_from_file(path).map_err(|td_err| {
                anyhow::anyhow!(
                    "Failed to load agent as MENACE ({menace_err:#}) or TD learner ({td_err:#})"
                )
            })?;
            let agent = saved_td.to_agent()?;
            let learner = agent.into_box();

            let info = AgentInfo {
                filter: None,
                algorithm: match saved_td.algorithm {
                    crate::q_learning::TdAlgorithm::QLearning => "Q-learning".to_string(),
                    crate::q_learning::TdAlgorithm::Sarsa => "SARSA".to_string(),
                },
                restock_mode: None,
                games_trained: saved_td.metadata.games_trained,
                trained_against: saved_td.metadata.opponents.clone(),
                saved_at: saved_td.metadata.saved_at.clone(),
            };

            let agent_player = saved_td.metadata.agent_player.unwrap_or(Player::X);
            let first_player = saved_td.metadata.first_player.unwrap_or(Player::X);
            let seed = saved_td.metadata.seed;

            Ok(LoadedAgent {
                learner,
                info,
                agent_player,
                first_player,
                seed,
            })
        }
    }
}
