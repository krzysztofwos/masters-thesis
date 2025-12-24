//! Game trajectory analysis
//!
//! This module analyzes all possible game sequences and their outcomes.

use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;

use crate::tictactoe::{BoardState, Player};

#[derive(Debug, Clone, Copy)]
enum GameOutcome {
    XWin,
    OWin,
    Draw,
}

struct GameTrajectory {
    moves: Vec<usize>,
    outcome: GameOutcome,
}

/// Analyze all game trajectories
pub fn analyze(export: Option<PathBuf>, detailed: bool) -> Result<()> {
    println!("=== Game Trajectory Analysis ===");
    println!("Enumerating all possible game sequences...\n");

    fn enumerate_games(state: BoardState, moves: Vec<usize>, all_games: &mut Vec<GameTrajectory>) {
        if state.is_terminal() {
            let outcome = if state.has_won(Player::X) {
                GameOutcome::XWin
            } else if state.has_won(Player::O) {
                GameOutcome::OWin
            } else {
                GameOutcome::Draw
            };
            all_games.push(GameTrajectory { moves, outcome });
            return;
        }

        for pos in state.empty_positions() {
            if let Ok(next) = state.make_move(pos) {
                let mut next_moves = moves.clone();
                next_moves.push(pos);
                enumerate_games(next, next_moves, all_games);
            }
        }
    }

    let mut trajectories = Vec::new();
    enumerate_games(BoardState::new(), vec![], &mut trajectories);

    println!(
        "Total game trajectories: {}",
        super::format_number(trajectories.len())
    );

    // Count by outcome
    let mut x_wins = 0;
    let mut o_wins = 0;
    let mut draws = 0;
    let mut total_length = 0;
    let mut games_by_length: HashMap<usize, usize> = HashMap::new();

    for traj in &trajectories {
        total_length += traj.moves.len();
        *games_by_length.entry(traj.moves.len()).or_insert(0) += 1;

        match traj.outcome {
            GameOutcome::XWin => x_wins += 1,
            GameOutcome::OWin => o_wins += 1,
            GameOutcome::Draw => draws += 1,
        }
    }

    let avg_length = total_length as f64 / trajectories.len() as f64;

    println!("\nGame outcomes:");
    println!(
        "  X wins: {} ({:.1}%)",
        super::format_number(x_wins),
        x_wins as f64 / trajectories.len() as f64 * 100.0
    );
    println!(
        "  O wins: {} ({:.1}%)",
        super::format_number(o_wins),
        o_wins as f64 / trajectories.len() as f64 * 100.0
    );
    println!(
        "  Draws: {} ({:.1}%)",
        super::format_number(draws),
        draws as f64 / trajectories.len() as f64 * 100.0
    );

    println!("\nAverage game length: {avg_length:.2} moves");

    if detailed {
        println!("\nGame length distribution:");
        let mut lengths: Vec<_> = games_by_length.iter().collect();
        lengths.sort_by_key(|(len, _)| *len);

        for (len, count) in lengths {
            let percentage = *count as f64 / trajectories.len() as f64 * 100.0;
            println!(
                "  {} moves: {} games ({:.2}%)",
                len,
                super::format_number(*count),
                percentage
            );
        }
    }

    // Compute canonical trajectories
    println!("\nComputing canonical trajectories under D4 symmetry...");
    let transforms = crate::tictactoe::symmetry::D4Transform::all();
    let mut canonical_trajectories = std::collections::HashSet::new();

    for traj in &trajectories {
        let mut best: Option<Vec<usize>> = None;

        for transform in &transforms {
            let mapped: Vec<usize> = traj
                .moves
                .iter()
                .map(|&pos| {
                    let (mut row, mut col) = (pos / 3, pos % 3);
                    if transform.reflection {
                        col = 2 - col;
                    }
                    for _ in 0..(transform.rotation / 90) {
                        let new_row = col;
                        let new_col = 2 - row;
                        row = new_row;
                        col = new_col;
                    }
                    row * 3 + col
                })
                .collect();

            if best.as_ref().is_none() || mapped < *best.as_ref().unwrap() {
                best = Some(mapped);
            }
        }

        if let Some(canonical) = best {
            canonical_trajectories.insert(canonical);
        }
    }

    println!(
        "Canonical trajectories: {}",
        super::format_number(canonical_trajectories.len())
    );
    println!(
        "Reduction factor: {:.2}x",
        trajectories.len() as f64 / canonical_trajectories.len() as f64
    );

    // Export if requested
    if let Some(path) = export {
        use std::{fs::File, io::Write};

        let mut file = File::create(&path)?;
        writeln!(file, "# Tic-Tac-Toe Game Trajectories")?;
        writeln!(file, "# Total: {}", trajectories.len())?;
        writeln!(file, "# X wins: {x_wins}, O wins: {o_wins}, Draws: {draws}")?;
        writeln!(file)?;
        writeln!(file, "Trajectory,Outcome,Length")?;

        for traj in &trajectories {
            let moves_str: Vec<String> = traj.moves.iter().map(|m| m.to_string()).collect();
            let outcome_str = match traj.outcome {
                GameOutcome::XWin => "X",
                GameOutcome::OWin => "O",
                GameOutcome::Draw => "D",
            };
            writeln!(
                file,
                "{},{},{}",
                moves_str.join("-"),
                outcome_str,
                traj.moves.len()
            )?;
        }

        println!("\nTrajectories exported to: {}", path.display());
    }

    Ok(())
}
