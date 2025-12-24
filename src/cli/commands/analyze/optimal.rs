//! Optimal policy analysis
//!
//! This module computes and analyzes the optimal (minimax) policy for Tic-Tac-Toe.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use clap::ValueEnum;
use serde::Serialize;

use crate::{pipeline::OptimalLearner, tictactoe::BoardState};

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum PolicyMode {
    /// Encode a single canonical optimal move per state
    Single,
    /// Encode all moves with minimax-optimal value
    Full,
}

impl PolicyMode {
    fn as_str(&self) -> &'static str {
        match self {
            PolicyMode::Single => "single",
            PolicyMode::Full => "full",
        }
    }
}

/// Compute and analyze optimal policy
pub fn analyze(state_str: Option<String>, export: Option<PathBuf>, mode: PolicyMode) -> Result<()> {
    let mut optimal = OptimalLearner::new("Optimal".to_string());

    // If custom state provided, analyze just that state
    if let Some(s) = state_str {
        let state = BoardState::from_string(&s)?;
        println!("=== Optimal Analysis for Custom State ===\n");
        analyze_position(&mut optimal, &state, "Custom state", mode)?;
        return Ok(());
    }

    // Otherwise, analyze key positions
    println!("Computing optimal (minimax) policy...");
    println!("\n=== Optimal Policy Analysis ===");
    println!("Showing optimal moves for key positions:\n");

    // Empty board
    let empty = BoardState::new();
    analyze_position(&mut optimal, &empty, "Empty board", mode)?;

    // Center taken
    let center = BoardState::from_string("....X...._O")?;
    analyze_position(&mut optimal, &center, "Center taken by X", mode)?;

    // Corner taken
    let corner = BoardState::from_string("X........_O")?;
    analyze_position(&mut optimal, &corner, "Corner taken by X", mode)?;

    // Export if requested
    if let Some(path) = export {
        export_optimal_policy(&mut optimal, &path, mode)?;
        println!("\nOptimal policy exported to: {}", path.display());
    }

    Ok(())
}

/// Analyze a single position
fn analyze_position(
    optimal: &mut OptimalLearner,
    state: &BoardState,
    description: &str,
    mode: PolicyMode,
) -> Result<()> {
    println!("{description}:");
    println!("{state}");

    let best_moves = compute_best_moves(optimal, state);
    if best_moves.is_empty() {
        println!("  (state is terminal)\n");
        return Ok(());
    }

    if mode == PolicyMode::Single {
        let best_move = best_moves[0];
        let (row, col) = (best_move / 3, best_move % 3);
        println!("Optimal move: position {best_move} (row {row}, col {col})\n");
    } else {
        println!("Optimal moves (all minimax-equivalent):");
        for mv in &best_moves {
            println!("  - position {} (row {}, col {})", mv, mv / 3, mv % 3);
        }
        println!();
    }

    Ok(())
}

#[derive(Serialize)]
struct OptimalPolicyExport {
    description: &'static str,
    player: &'static str,
    mode: &'static str,
    total_states: usize,
    policy: HashMap<String, PolicyEntry>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum PolicyEntry {
    Single(usize),
    Multiple(Vec<usize>),
}

/// Export optimal policy to JSON file
pub fn export_optimal_policy(
    optimal: &mut OptimalLearner,
    path: &PathBuf,
    mode: PolicyMode,
) -> Result<()> {
    println!("\nComputing optimal policy for all X decision states...");

    let tree = crate::tictactoe::build_reduced_game_tree(true, false);
    let states: Vec<_> = tree.states.iter().collect();
    let mut policy = HashMap::new();

    for (i, state_label) in states.iter().enumerate() {
        let state_str = state_label.to_string();
        let state = BoardState::from_string(&state_str)?;

        if !state.is_terminal() {
            let best_moves = compute_best_moves(optimal, &state);
            if best_moves.is_empty() {
                continue;
            }
            let entry = match mode {
                PolicyMode::Single => PolicyEntry::Single(best_moves[0]),
                PolicyMode::Full => PolicyEntry::Multiple(best_moves.clone()),
            };
            policy.insert(state_str, entry);
        }

        if (i + 1).is_multiple_of(100) {
            println!("  Processed {}/{} states...", i + 1, states.len());
        }
    }

    println!("  Processed {}/{} states", states.len(), states.len());
    println!("  Total policy entries: {}", policy.len());

    let export = OptimalPolicyExport {
        description: "Optimal (minimax) policy for Tic-Tac-Toe",
        player: "X",
        mode: mode.as_str(),
        total_states: policy.len(),
        policy,
    };

    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(file, &export)?;

    Ok(())
}

fn compute_best_moves(optimal: &mut OptimalLearner, state: &BoardState) -> Vec<usize> {
    let moves_with_values = optimal.evaluate_moves(state);
    if moves_with_values.is_empty() {
        return Vec::new();
    }
    let is_x = state.to_move == crate::tictactoe::Player::X;
    let best_value = if is_x {
        moves_with_values
            .iter()
            .map(|(_, value)| *value)
            .max()
            .unwrap_or(0)
    } else {
        moves_with_values
            .iter()
            .map(|(_, value)| *value)
            .min()
            .unwrap_or(0)
    };
    let mut best_moves: Vec<usize> = moves_with_values
        .into_iter()
        .filter(|(_, value)| *value == best_value)
        .map(|(mv, _)| mv)
        .collect();
    best_moves.sort_unstable();
    best_moves
}
