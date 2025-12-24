//! Compare command - Compare multiple learners side-by-side

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::Parser;

use crate::{
    cli::commands::train::parse_player_token,
    menace::MenaceAgent,
    pipeline::{ComparisonFramework, Learner, MenaceLearner, OptimalLearner, RandomLearner},
};

#[derive(Parser, Debug)]
#[command(about = "Compare multiple learners")]
pub struct CompareArgs {
    /// Learners to compare (format: type:path or type:param=value)
    /// Examples: menace:agent.msgpack, active-inference:beta=0.5, optimal
    #[arg(required = true)]
    pub learners: Vec<String>,

    /// Number of games per matchup
    #[arg(long, short = 'g', default_value_t = 100)]
    pub games: usize,

    /// Export comparison results to CSV
    #[arg(long, short = 'o')]
    pub output: Option<PathBuf>,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Which token makes the first move in each matchup (`x` or `o`)
    #[arg(long = "first-player", default_value = "x")]
    pub first_player: String,
}

pub fn execute(args: CompareArgs) -> Result<()> {
    if args.learners.len() < 2 {
        return Err(anyhow!("Need at least 2 learners to compare"));
    }

    // Create learners
    let mut learners: Vec<Box<dyn Learner>> = Vec::new();
    for (i, spec) in args.learners.iter().enumerate() {
        let learner = create_learner(spec, args.seed, i)?;
        learners.push(learner);
    }

    println!("Comparing {} learners:", learners.len());
    for (i, learner) in learners.iter().enumerate() {
        println!("  {}: {}", i + 1, learner.name());
    }
    println!("\nGames per matchup: {}", args.games);

    // Run round-robin comparison
    let first_player = parse_player_token(&args.first_player, "--first-player")?;

    let mut framework = ComparisonFramework::new(learners).with_first_player(first_player);
    let result = framework.compare_round_robin(args.games)?;

    // Display results
    println!("\n=== Comparison Results ===");
    println!("Total games: {}", result.total_games);
    println!();

    // Print matchup table
    println!("Head-to-Head Results:");
    println!("(Format: Player1 vs Player2 | W-D-L from Player1's perspective)");
    println!();

    for i in 0..result.learners.len() {
        for j in (i + 1)..result.learners.len() {
            if let Some((wins, draws, losses)) = result.head_to_head.get(&(i, j)) {
                println!(
                    "{} vs {} | {}-{}-{} ({:.1}% wins)",
                    result.learners[i],
                    result.learners[j],
                    wins,
                    draws,
                    losses,
                    (*wins as f64 / (wins + draws + losses) as f64) * 100.0
                );
            }
        }
    }

    println!();
    println!("Overall Win Rates:");
    for (i, name) in result.learners.iter().enumerate() {
        let win_rate = result.win_rate(i);
        println!("  {}: {:.1}%", name, win_rate * 100.0);
    }

    // Export to CSV if requested
    if let Some(output_path) = &args.output {
        export_csv(&result, output_path)?;
        println!("\nResults exported to: {}", output_path.display());
    }

    Ok(())
}

fn create_learner(spec: &str, seed: Option<u64>, index: usize) -> Result<Box<dyn Learner>> {
    let spec_lower = spec.to_lowercase();

    match spec_lower.as_str() {
        "menace" => {
            let agent = MenaceAgent::new(seed)?;
            Ok(Box::new(MenaceLearner::new(
                agent,
                format!("MENACE-{}", index + 1),
            )))
        }
        "random" => Ok(Box::new(RandomLearner::new(format!(
            "Random-{}",
            index + 1
        )))),
        "optimal" => Ok(Box::new(OptimalLearner::new(format!(
            "Optimal-{}",
            index + 1
        )))),
        _ => Err(anyhow!(
            "Unknown learner type: '{spec}'. Supported: menace, random, optimal"
        )),
    }
}

fn export_csv(result: &crate::pipeline::ComparisonResult, path: &PathBuf) -> Result<()> {
    use std::{fs::File, io::Write};

    let mut file = File::create(path)?;

    // Write header
    writeln!(file, "Player1,Player2,Wins,Draws,Losses,WinRate")?;

    // Write matchup data
    for i in 0..result.learners.len() {
        for j in (i + 1)..result.learners.len() {
            if let Some((wins, draws, losses)) = result.head_to_head.get(&(i, j)) {
                let total = wins + draws + losses;
                let win_rate = if total > 0 {
                    (*wins as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                writeln!(
                    file,
                    "{},{},{},{},{},{:.2}",
                    result.learners[i], result.learners[j], wins, draws, losses, win_rate
                )?;
            }
        }
    }

    Ok(())
}
