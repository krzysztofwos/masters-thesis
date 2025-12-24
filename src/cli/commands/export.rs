//! Export command - Export data in various formats
//!
//! Note: Most export functionality is available through specific commands:
//! - Game tree: `menace analyze game-tree --export`
//! - Optimal policy: `menace analyze optimal --export`
//! - Observations: `menace train --observations`
//!
//! This command provides a unified export interface for convenience.

use std::{path::PathBuf, str::FromStr};

use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};

use crate::cli::commands::analyze::optimal::{self, PolicyMode};

#[derive(Parser, Debug)]
#[command(about = "Export data in various formats")]
pub struct ExportArgs {
    /// Type of data to export
    #[arg(value_enum)]
    pub data_type: DataType,

    /// Input source
    /// - For game-tree: filter name (canonical, all, decision-only, michie, both)
    /// - For policy: 'optimal' (other learners not yet supported)
    /// - For observations: path to JSONL file to convert
    pub source: String,

    /// Output file path
    #[arg(long, short = 'o')]
    pub output: PathBuf,

    /// Export format (some combinations not supported)
    #[arg(long, short = 'f', default_value = "csv")]
    pub format: ExportFormat,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum DataType {
    /// Game tree structure
    GameTree,
    /// Learned policy/strategy
    Policy,
    /// Training observations (convert JSONL)
    Observations,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum ExportFormat {
    /// JSON format
    Json,
    /// CSV format
    Csv,
    /// JSONL format (JSON Lines)
    Jsonl,
    /// DOT format (for graph visualization)
    Dot,
}

pub fn execute(args: ExportArgs) -> Result<()> {
    match args.data_type {
        DataType::GameTree => export_game_tree(&args.source, &args.output, &args.format),
        DataType::Policy => export_policy(&args.source, &args.output, &args.format),
        DataType::Observations => export_observations(&args.source, &args.output, &args.format),
    }
}

fn export_game_tree(filter_str: &str, output: &PathBuf, format: &ExportFormat) -> Result<()> {
    use crate::{
        tictactoe::{BoardState, collect_reachable_canonical_labels},
        workspace::{MenaceWorkspace, StateFilter},
    };

    let mut normalised = filter_str.trim().to_ascii_lowercase();
    if normalised == "decision" {
        normalised = "decision-only".to_string();
    }
    let (filter_label, states): (String, Vec<String>) = match normalised.as_str() {
        "canonical" | "full" | "765" => (
            "canonical".to_string(),
            collect_reachable_canonical_labels(),
        ),
        _ => {
            let filter = StateFilter::from_str(&normalised).map_err(|_| {
                anyhow!(
                    "Unknown filter: '{filter_str}'. Use: canonical/765, all/338, decision-only/304, michie/287, both"
                )
            })?;
            let workspace = MenaceWorkspace::new(filter)?;
            let mut states: Vec<String> = workspace.decision_labels().cloned().collect();
            states.sort();
            (filter.to_string(), states)
        }
    };

    println!("Building game tree with filter: {filter_label}...");
    println!("Total states: {}", states.len());

    // Export based on format
    match format {
        ExportFormat::Csv => {
            use std::{fs::File, io::Write};

            let mut file = File::create(output)?;
            writeln!(file, "# Tic-Tac-Toe Game Tree")?;
            writeln!(file, "# Filter: {filter_label}")?;
            writeln!(file, "# Total states: {}", states.len())?;
            writeln!(file)?;
            writeln!(file, "State,Depth,LegalMoves")?;

            for state_str in &states {
                let state = BoardState::from_string(state_str)?;
                let depth = state.occupied_count();
                let legal_moves = state.legal_moves().len();
                writeln!(file, "{state_str},{depth},{legal_moves}")?;
            }

            println!("✓ Game tree exported to: {}", output.display());
            Ok(())
        }
        ExportFormat::Json => {
            use std::{fs::File, io::Write};

            let mut file = File::create(output)?;
            writeln!(file, "{{")?;
            writeln!(file, "  \"filter\": \"{filter_label}\",")?;
            writeln!(file, "  \"total_states\": {},", states.len())?;
            writeln!(file, "  \"states\": [")?;

            for (i, state_label) in states.iter().enumerate() {
                let state_str = state_label.as_str();
                let state = BoardState::from_string(state_str)?;
                let depth = state.occupied_count();
                let legal_moves = state.legal_moves().len();
                let comma = if i < states.len() - 1 { "," } else { "" };
                writeln!(
                    file,
                    "    {{\"state\": \"{state_str}\", \"depth\": {depth}, \"legal_moves\": {legal_moves}}}{comma}"
                )?;
            }

            writeln!(file, "  ]")?;
            writeln!(file, "}}")?;

            println!("✓ Game tree exported to: {}", output.display());
            Ok(())
        }
        ExportFormat::Dot => {
            println!("Note: DOT format not yet implemented for game trees.");
            println!("      This would require graph layout algorithms.");
            println!("      Consider using CSV and importing into Gephi or similar tools.");
            Ok(())
        }
        ExportFormat::Jsonl => Err(anyhow!(
            "JSONL format not suitable for game trees. Use CSV or JSON."
        )),
    }
}

fn export_policy(source: &str, output: &PathBuf, format: &ExportFormat) -> Result<()> {
    if source.to_lowercase() != "optimal" {
        return Err(anyhow!(
            "Only 'optimal' policy export is currently supported.\n\
             For learned agent policies, use the analyze strategy command (requires agent serialization)."
        ));
    }

    match format {
        ExportFormat::Json => {
            use crate::pipeline::OptimalLearner;

            let mut optimal = OptimalLearner::new("Optimal".to_string());
            optimal::export_optimal_policy(&mut optimal, output, PolicyMode::Single)?;
            println!("✓ Optimal policy exported to: {}", output.display());
            Ok(())
        }
        ExportFormat::Csv => {
            println!("Note: CSV format for policies is less useful than JSON.");
            println!("      Use --format json for better structure.");
            Err(anyhow!(
                "CSV format not supported for policies. Use --format json"
            ))
        }
        _ => Err(anyhow!("Only JSON format is supported for policy export")),
    }
}

fn export_observations(source: &str, output: &PathBuf, format: &ExportFormat) -> Result<()> {
    use std::{
        fs::File,
        io::{BufRead, BufReader, Write},
    };

    println!("Reading observations from: {source}");

    let input = File::open(source)?;
    let reader = BufReader::new(input);

    let mut observations: Vec<serde_json::Value> = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&line) {
            Ok(obs) => observations.push(obs),
            Err(e) => {
                println!("Warning: Failed to parse line {}: {}", i + 1, e);
            }
        }
    }

    println!("Loaded {} observations", observations.len());

    match format {
        ExportFormat::Jsonl => {
            // Just copy the file
            std::fs::copy(source, output)?;
            println!("✓ Observations copied to: {}", output.display());
            Ok(())
        }
        ExportFormat::Json => {
            // Convert JSONL to JSON array
            let mut file = File::create(output)?;
            writeln!(file, "{{")?;
            writeln!(file, "  \"total_observations\": {},", observations.len())?;
            writeln!(file, "  \"observations\": [")?;

            for (i, obs) in observations.iter().enumerate() {
                let comma = if i < observations.len() - 1 { "," } else { "" };
                let obs_str = serde_json::to_string(obs)?;
                writeln!(file, "    {obs_str}{comma}")?;
            }

            writeln!(file, "  ]")?;
            writeln!(file, "}}")?;

            println!("✓ Observations exported to: {}", output.display());
            Ok(())
        }
        ExportFormat::Csv => {
            println!("Note: CSV format for observations loses structure.");
            println!("      Consider using --format json instead.");
            Err(anyhow!(
                "CSV format not suitable for complex observation data. Use --format json"
            ))
        }
        ExportFormat::Dot => Err(anyhow!("DOT format not applicable to observations")),
    }
}
