//! MENACE CLI - Production-grade research toolkit for learnable game agents
//!
//! This CLI provides a unified interface for:
//! - Training different learner types (MENACE, Active Inference, etc.)
//! - Evaluating learned policies
//! - Comparing learners side-by-side
//! - Analyzing game trees and strategies
//! - Exporting data for further analysis

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "menace")]
#[command(version, about = "Research toolkit for learnable game agents", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a learner (MENACE, Active Inference, etc.)
    Train(Box<menace::cli::commands::train::TrainArgs>),

    /// Evaluate a trained learner against opponents
    Evaluate(menace::cli::commands::evaluate::EvaluateArgs),

    /// Compare multiple learners side-by-side
    Compare(menace::cli::commands::compare::CompareArgs),

    /// Analyze game trees, strategies, and policies
    Analyze(menace::cli::commands::analyze::AnalyzeArgs),

    /// Export data in various formats
    Export(menace::cli::commands::export::ExportArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Train(args) => menace::cli::commands::train::execute(*args),
        Commands::Evaluate(args) => menace::cli::commands::evaluate::execute(args),
        Commands::Compare(args) => menace::cli::commands::compare::execute(args),
        Commands::Analyze(args) => menace::cli::commands::analyze::execute(args),
        Commands::Export(args) => menace::cli::commands::export::execute(args),
    }
}
