//! Analyze command - Analyze game trees, strategies, and policies
//!
//! This module provides various analysis tools for Tic-Tac-Toe game trees,
//! learned strategies, and mathematical properties.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

mod active_inference;
mod first_moves;
mod game_tree;
pub mod optimal;
mod strategy;
mod structures;
mod symmetry;
mod trajectories;
mod validate;

#[derive(Parser, Debug)]
#[command(about = "Analyze game trees, strategies, and policies")]
pub struct AnalyzeArgs {
    #[command(subcommand)]
    pub command: AnalyzeCommand,
}

#[derive(Subcommand, Debug)]
pub enum AnalyzeCommand {
    /// Analyze the complete game tree
    GameTree {
        /// State filter (canonical, all, decision-only, michie, both)
        #[arg(long, default_value = "michie")]
        filter: String,

        /// Export game tree to file
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Analyze a learned strategy
    Strategy {
        /// Path to trained agent
        agent: PathBuf,

        /// Export analysis to file
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Compute optimal policy
    Optimal {
        /// Custom state to analyze
        #[arg(long)]
        state: Option<String>,

        /// Export optimal policy to file
        #[arg(long)]
        export: Option<PathBuf>,

        /// How to represent optimal moves (single canonical move or full distribution)
        #[arg(long, value_enum, default_value = "single")]
        policy_mode: optimal::PolicyMode,
    },

    /// Analyze symmetries
    Symmetry {
        /// Board state to analyze
        #[arg(long)]
        state: Option<String>,

        /// Visualize symmetries
        #[arg(long)]
        visualize: bool,

        /// Show stabilizer subgroup analysis
        #[arg(long)]
        stabilizers: bool,
    },

    /// Analyze game trajectories and sequences
    Trajectories {
        /// Export trajectories to file
        #[arg(long)]
        export: Option<PathBuf>,

        /// Show detailed statistics
        #[arg(long)]
        detailed: bool,
    },

    /// Analyze mathematical structures (magic square, positional values)
    Structures {
        /// Show only specific structure (magic-square, positional-values, categorical)
        #[arg(long)]
        structure: Option<String>,
    },

    /// Analyze first move strategies
    FirstMoves {
        /// Export analysis to file
        #[arg(long)]
        export: Option<PathBuf>,
    },

    /// Validate state space and verify mathematical properties
    Validate {
        /// Show invalid continuation examples
        #[arg(long)]
        show_invalid: bool,

        /// Show verification table against expected values
        #[arg(long)]
        verify: bool,
    },

    /// Analyze Active Inference EFE decomposition
    ActiveInference {
        /// Board state to analyze (e.g., "........._X" for empty board)
        #[arg(long)]
        state: Option<String>,

        /// Opponent model (uniform, adversarial, or minimax)
        #[arg(long, default_value = "uniform")]
        opponent: String,

        /// Epistemic weight (0.0-1.0)
        #[arg(long, default_value_t = 0.5)]
        beta: f64,

        /// Export CSV with EFE decompositions for all/selected states
        #[arg(long)]
        export: Option<PathBuf>,

        /// Comma-separated list of states to export
        #[arg(long)]
        states: Option<String>,
    },
}

pub fn execute(args: AnalyzeArgs) -> Result<()> {
    match args.command {
        AnalyzeCommand::GameTree { filter, export } => game_tree::analyze(&filter, export),
        AnalyzeCommand::Strategy { agent, export } => strategy::analyze(agent, export),
        AnalyzeCommand::Optimal {
            state,
            export,
            policy_mode,
        } => optimal::analyze(state, export, policy_mode),
        AnalyzeCommand::Symmetry {
            state,
            visualize,
            stabilizers,
        } => symmetry::analyze(state, visualize, stabilizers),
        AnalyzeCommand::Trajectories { export, detailed } => {
            trajectories::analyze(export, detailed)
        }
        AnalyzeCommand::Structures { structure } => structures::analyze(structure),
        AnalyzeCommand::FirstMoves { export } => first_moves::analyze(export),
        AnalyzeCommand::Validate {
            show_invalid,
            verify,
        } => validate::analyze(show_invalid, verify),
        AnalyzeCommand::ActiveInference {
            state,
            opponent,
            beta,
            export,
            states,
        } => active_inference::analyze(state, opponent, beta, export, states),
    }
}

/// Format numbers with comma separators
///
/// This is a shared utility function used across multiple analysis modules.
pub(super) fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let mut count = 0;

    for c in s.chars().rev() {
        if count == 3 {
            result.insert(0, ',');
            count = 0;
        }
        result.insert(0, c);
        count += 1;
    }

    result
}
