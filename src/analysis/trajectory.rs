//! Game trajectory analysis

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::tictactoe::{BoardState, Game, Player};

/// A trajectory through game states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub states: Vec<BoardState>,
    pub moves: Vec<usize>,
    pub player: Player,
}

/// Analysis of game trajectories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryAnalysis {
    pub total_trajectories: usize,
    pub unique_trajectories: usize,
    pub canonical_trajectories: usize,
    pub trajectory_frequencies: HashMap<String, usize>,
}

impl Trajectory {
    /// Create a new trajectory
    pub fn new(player: Player) -> Self {
        Trajectory {
            states: vec![BoardState::new()],
            moves: Vec::new(),
            player,
        }
    }

    /// Add a move to the trajectory
    pub fn add_move(&mut self, position: usize) -> Result<(), crate::Error> {
        let current = self.states.last().ok_or(crate::Error::EmptyTrajectory)?;

        let next = current.make_move(position)?;
        self.states.push(next);
        self.moves.push(position);
        Ok(())
    }

    /// Get canonical form of trajectory
    ///
    /// # Errors
    ///
    /// Returns error if any move in the trajectory is invalid.
    /// This indicates corrupted trajectory data.
    pub fn canonical(&self) -> Result<Vec<usize>, crate::Error> {
        // Convert to game and get canonical form
        let game = Game {
            initial: BoardState::new(),
            moves: self
                .moves
                .iter()
                .enumerate()
                .map(|(i, &pos)| crate::tictactoe::game::Move {
                    position: pos,
                    player: if i.is_multiple_of(2) {
                        Player::X
                    } else {
                        Player::O
                    },
                })
                .collect(),
            outcome: None,
        };
        game.canonical()
    }

    /// Calculate trajectory entropy
    pub fn entropy(&self) -> f64 {
        // Shannon entropy of move distribution
        let mut move_counts = HashMap::new();
        for &m in &self.moves {
            *move_counts.entry(m).or_insert(0) += 1;
        }

        let total = self.moves.len() as f64;
        let mut entropy = 0.0;

        for count in move_counts.values() {
            let p = *count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }

        entropy
    }
}

impl TrajectoryAnalysis {
    /// Analyze a collection of trajectories
    ///
    /// # Errors
    ///
    /// Returns error if any trajectory contains invalid moves.
    pub fn from_trajectories(trajectories: Vec<Trajectory>) -> Result<Self, crate::Error> {
        let total = trajectories.len();
        let mut canonical_set = std::collections::HashSet::new();
        let mut frequencies = HashMap::new();

        for traj in trajectories {
            let canonical = traj.canonical()?;
            let key = format!("{canonical:?}");

            canonical_set.insert(canonical);
            *frequencies.entry(key).or_insert(0) += 1;
        }

        Ok(TrajectoryAnalysis {
            total_trajectories: total,
            unique_trajectories: frequencies.len(),
            canonical_trajectories: canonical_set.len(),
            trajectory_frequencies: frequencies,
        })
    }

    /// Find most common trajectories
    pub fn most_common(&self, n: usize) -> Vec<(&String, &usize)> {
        let mut items: Vec<_> = self.trajectory_frequencies.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1));
        items.into_iter().take(n).collect()
    }
}
