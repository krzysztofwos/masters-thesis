//! Classic MENACE learning algorithm
//!
//! Original reinforcement-based learning using fixed reward values.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::{
    game_utils,
    learning::{LearningAlgorithm, ReinforcementBased},
};
use crate::{
    tictactoe::{BoardState, GameOutcome, Player},
    workspace::{MenaceWorkspace, Reinforcement},
};

/// MENACE reinforcement values
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReinforcementValues {
    pub win: i16,
    pub draw: i16,
    pub loss: i16,
}

impl Default for ReinforcementValues {
    fn default() -> Self {
        // MENACE's original values
        ReinforcementValues {
            win: 3,
            draw: 1,
            loss: -1,
        }
    }
}

/// Classic MENACE learning algorithm
///
/// Uses fixed reinforcement values to adjust matchbox weights
/// based on game outcomes.
#[derive(Debug, Clone)]
pub struct ClassicMenace {
    reinforcement: ReinforcementValues,
    games_trained: usize,
}

impl ClassicMenace {
    /// Create with default reinforcement values
    pub fn new() -> Self {
        Self::with_reinforcement(ReinforcementValues::default())
    }

    /// Create with custom reinforcement values
    pub fn with_reinforcement(reinforcement: ReinforcementValues) -> Self {
        Self {
            reinforcement,
            games_trained: 0,
        }
    }

    /// Get current reinforcement values
    pub fn reinforcement(&self) -> ReinforcementValues {
        self.reinforcement
    }

    /// Set reinforcement values
    pub fn set_reinforcement(&mut self, reinforcement: ReinforcementValues) {
        self.reinforcement = reinforcement;
    }

    /// Calculate reinforcement value based on game outcome
    fn calculate_reinforcement(&self, outcome: GameOutcome, player: Player) -> i16 {
        match outcome {
            GameOutcome::Win(winner) if winner == player => self.reinforcement.win,
            GameOutcome::Win(_) => self.reinforcement.loss,
            GameOutcome::Draw => self.reinforcement.draw,
        }
    }

    /// Convert reinforcement value to signal enum
    fn reinforcement_to_signal(&self, reinforcement: i16) -> Reinforcement {
        let reinforcement_value = reinforcement as f64;
        if reinforcement_value > 0.0 {
            Reinforcement::Positive(reinforcement_value)
        } else if reinforcement_value < 0.0 {
            Reinforcement::Negative(reinforcement_value.abs())
        } else {
            Reinforcement::Neutral
        }
    }
}

impl Default for ClassicMenace {
    fn default() -> Self {
        Self::new()
    }
}

impl LearningAlgorithm for ClassicMenace {
    fn train_from_game(
        &mut self,
        workspace: &mut MenaceWorkspace,
        states: &[BoardState],
        moves: &[usize],
        outcome: GameOutcome,
        player: Player,
    ) -> crate::Result<()> {
        let reinforcement = self.calculate_reinforcement(outcome, player);
        let path = game_utils::build_move_path_for_player(workspace, states, moves, player);
        let signal = self.reinforcement_to_signal(reinforcement);
        workspace.apply_reinforcement(path, signal)?;
        self.games_trained += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "Classic MENACE"
    }

    fn stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("games_trained".to_string(), self.games_trained as f64);
        stats.insert(
            "win_reinforcement".to_string(),
            self.reinforcement.win as f64,
        );
        stats.insert(
            "draw_reinforcement".to_string(),
            self.reinforcement.draw as f64,
        );
        stats.insert(
            "loss_reinforcement".to_string(),
            self.reinforcement.loss as f64,
        );
        stats
    }

    fn reset(&mut self) {
        self.games_trained = 0;
    }
}

impl ReinforcementBased for ClassicMenace {
    fn reinforcement_values(&self) -> ReinforcementValues {
        self.reinforcement
    }

    fn set_reinforcement_values(&mut self, values: ReinforcementValues) {
        self.reinforcement = values;
    }
}
