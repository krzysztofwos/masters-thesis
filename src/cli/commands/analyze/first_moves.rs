//! First move analysis
//!
//! This module analyzes outcomes and strategies for different opening moves.

use std::path::PathBuf;

use anyhow::Result;

use crate::tictactoe::{BoardState, Player};

/// Analyze first move strategies
pub fn analyze(export: Option<PathBuf>) -> Result<()> {
    println!("=== First Move Analysis ===\n");

    println!("Essentially different first moves: 3");
    println!("  1. Corner (positions 0, 2, 6, 8)");
    println!("  2. Edge (positions 1, 3, 5, 7)");
    println!("  3. Center (position 4)");

    // Analyze outcomes for each first move type
    let positions = vec![(0, "Corner"), (1, "Edge"), (4, "Center")];
    let mut results = Vec::new();

    for (pos, name) in positions {
        let mut state = BoardState::new();
        state = state.make_move(pos)?;

        let (wins, draws, losses) = count_subtree_outcomes(state);
        let total = wins + draws + losses;

        println!("\n{name} first move outcomes:");
        println!(
            "  X wins: {} ({:.1}%)",
            super::format_number(wins),
            wins as f64 / total as f64 * 100.0
        );
        println!(
            "  Draws: {} ({:.1}%)",
            super::format_number(draws),
            draws as f64 / total as f64 * 100.0
        );
        println!(
            "  O wins: {} ({:.1}%)",
            super::format_number(losses),
            losses as f64 / total as f64 * 100.0
        );

        results.push((name, wins, draws, losses, total));
    }

    // Strategic analysis
    println!("\n=== Strategic Analysis ===\n");
    println!("First-player advantage:");
    let total_games: usize = results.iter().map(|(_, w, d, l, _)| w + d + l).sum();
    let total_wins: usize = results.iter().map(|(_, w, _, _, _)| w).sum();
    let total_draws: usize = results.iter().map(|(_, _, d, _, _)| d).sum();
    println!(
        "  Overall X win rate: {:.1}%",
        total_wins as f64 / total_games as f64 * 100.0
    );
    println!(
        "  Overall draw rate: {:.1}%",
        total_draws as f64 / total_games as f64 * 100.0
    );

    // Export if requested
    if let Some(path) = export {
        use std::{fs::File, io::Write};

        let mut file = File::create(&path)?;
        writeln!(file, "# First Move Analysis")?;
        writeln!(file)?;
        writeln!(file, "Move,X_Wins,Draws,O_Wins,Total,X_Win_Pct")?;

        for (name, wins, draws, losses, total) in results {
            writeln!(
                file,
                "{},{},{},{},{},{:.2}",
                name,
                wins,
                draws,
                losses,
                total,
                wins as f64 / total as f64 * 100.0
            )?;
        }

        println!("\nAnalysis exported to: {}", path.display());
    }

    Ok(())
}

/// Count outcomes in a subtree
fn count_subtree_outcomes(initial: BoardState) -> (usize, usize, usize) {
    let mut x_wins = 0;
    let mut o_wins = 0;
    let mut draws = 0;

    fn count_games(state: BoardState, x_wins: &mut usize, o_wins: &mut usize, draws: &mut usize) {
        if state.is_terminal() {
            if state.has_won(Player::X) {
                *x_wins += 1;
            } else if state.has_won(Player::O) {
                *o_wins += 1;
            } else {
                *draws += 1;
            }
            return;
        }

        for pos in state.empty_positions() {
            if let Ok(next) = state.make_move(pos) {
                count_games(next, x_wins, o_wins, draws);
            }
        }
    }

    count_games(initial, &mut x_wins, &mut o_wins, &mut draws);
    (x_wins, draws, o_wins)
}
