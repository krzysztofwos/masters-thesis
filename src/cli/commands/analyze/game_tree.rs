//! Game tree analysis
//!
//! This module provides analysis and export functionality for the Tic-Tac-Toe game tree.

use std::{path::PathBuf, str::FromStr};

use anyhow::{Result, anyhow};

use crate::{
    tictactoe::{BoardState, collect_reachable_canonical_labels},
    workspace::{MenaceWorkspace, StateFilter},
};

/// Analyze the complete game tree
pub fn analyze(filter_str: &str, export: Option<PathBuf>) -> Result<()> {
    let mut normalised = filter_str.trim().to_ascii_lowercase();
    if normalised == "decision" {
        normalised = "decision-only".to_string();
    }
    let (filter_label, states): (String, Vec<String>) = match normalised.as_str() {
        // Full canonical state space under D4 symmetry reduction (includes both players + terminal states).
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

    println!("Building game tree with filter: {filter_str}...");
    match filter_label.as_str() {
        "canonical" => println!(
            "(All canonical states under symmetry reduction; reachable from empty board with X to move)"
        ),
        "all" => println!("(All canonical X-to-move decision states; forced moves included)"),
        "decision-only" => {
            println!("(X-to-move decision states; forced single-move states excluded)")
        }
        "michie" => {
            println!("(Michie filter: decision-only excluding forced + double-threat positions)")
        }
        "both" => println!(
            "(Player-agnostic decision states; excludes forced + double-threat positions for both players)"
        ),
        _ => {}
    }

    // Display statistics
    println!("\n=== Game Tree Statistics ===");
    println!("Filter: {filter_label}");
    println!("Total states: {}", states.len());

    // Count by depth
    let mut depth_counts = std::collections::HashMap::new();
    for state_label in &states {
        let state = BoardState::from_string(state_label)?;
        let depth = state.occupied_count();
        *depth_counts.entry(depth).or_insert(0) += 1;
    }

    println!("\nStates by depth:");
    for depth in 0..=9 {
        if let Some(count) = depth_counts.get(&depth) {
            println!("  Depth {depth}: {count} states");
        }
    }

    // Export if requested
    if let Some(path) = export {
        export_game_tree(&states, &path)?;
        println!("\nGame tree exported to: {}", path.display());
    }

    Ok(())
}

/// Export game tree to CSV file
fn export_game_tree(states: &[String], path: &PathBuf) -> Result<()> {
    use std::{fs::File, io::Write};

    let mut file = File::create(path)?;

    writeln!(file, "# Tic-Tac-Toe Game Tree")?;
    writeln!(file, "# Total states: {}", states.len())?;
    writeln!(file)?;
    writeln!(file, "State,Depth,LegalMoves")?;

    for state_str in states {
        let state = BoardState::from_string(state_str)?;
        let depth = state.occupied_count();
        let legal_moves = state.legal_moves().len();

        writeln!(file, "{state_str},{depth},{legal_moves}")?;
    }

    Ok(())
}
