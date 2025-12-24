//! Learning algorithm trait and implementations
//!
//! This module defines a common interface for different learning approaches
//! that can be used with the MENACE agent.

use std::collections::HashMap;

use super::{
    active::{ActiveInference, OracleActiveInference, PureActiveInference},
    classic::{ClassicMenace, ReinforcementValues},
};
use crate::{
    tictactoe::{BoardState, GameOutcome, Player},
    workspace::MenaceWorkspace,
};

/// A learning algorithm that can train from game experience
///
/// This is the base trait for all learning algorithms, defining the core
/// operations that any learning approach must support.
pub trait LearningAlgorithm: Send {
    /// Train from a complete game history
    fn train_from_game(
        &mut self,
        workspace: &mut MenaceWorkspace,
        states: &[BoardState],
        moves: &[usize],
        outcome: GameOutcome,
        player: Player,
    ) -> crate::Result<()>;

    /// Get a human-readable name for this algorithm
    fn name(&self) -> &str;

    /// Get algorithm-specific statistics
    fn stats(&self) -> HashMap<String, f64>;

    /// Reset algorithm state (e.g., learning rates, counters)
    fn reset(&mut self);
}

/// Extension trait for algorithms based on reinforcement learning
///
/// Algorithms that implement this trait learn by adjusting weights based
/// on explicit reinforcement values for different game outcomes.
pub trait ReinforcementBased: LearningAlgorithm {
    /// Get the current reinforcement values
    ///
    /// Returns the reward/penalty values used for wins, losses, and draws.
    fn reinforcement_values(&self) -> ReinforcementValues;

    /// Set new reinforcement values
    ///
    /// Allows dynamic adjustment of learning parameters.
    fn set_reinforcement_values(&mut self, values: ReinforcementValues);
}

/// Enumeration of available learning strategies
///
/// This enum provides type-safe dispatch to different learning algorithms
/// without requiring runtime downcasting. Each variant encapsulates a specific
/// learning approach with its own parameters and behavior.
#[derive(Debug)]
pub enum LearningStrategy {
    /// Classic MENACE with reinforcement learning
    ClassicMenace(ClassicMenace),
    /// Hybrid Active Inference (EFE-based policy + Bayesian opponent beliefs)
    ActiveInference(Box<ActiveInference>),
    /// Oracle Active Inference using perfect game tree knowledge (no learning)
    OracleActiveInference(Box<OracleActiveInference>),
    /// Pure Active Inference using Bayesian beliefs for both opponent and action outcomes
    PureActiveInference(Box<PureActiveInference>),
}

impl LearningStrategy {
    /// Get reinforcement values if this strategy supports them
    ///
    /// Returns `Some(values)` for reinforcement-based algorithms (ClassicMenace),
    /// `None` for other algorithms (ActiveInference, OracleActiveInference, PureActiveInference).
    pub fn reinforcement_values(&self) -> Option<ReinforcementValues> {
        match self {
            Self::ClassicMenace(algo) => Some(algo.reinforcement_values()),
            Self::ActiveInference(_)
            | Self::OracleActiveInference(_)
            | Self::PureActiveInference(_) => None,
        }
    }

    /// Set reinforcement values if this strategy supports them
    ///
    /// Returns `true` if the values were set successfully (reinforcement-based algorithm),
    /// `false` if the algorithm doesn't support reinforcement values.
    pub fn set_reinforcement_values(&mut self, values: ReinforcementValues) -> bool {
        match self {
            Self::ClassicMenace(algo) => {
                algo.set_reinforcement_values(values);
                true
            }
            Self::ActiveInference(_)
            | Self::OracleActiveInference(_)
            | Self::PureActiveInference(_) => false,
        }
    }

    /// Check if this strategy uses reinforcement learning
    pub fn is_reinforcement_based(&self) -> bool {
        matches!(self, Self::ClassicMenace(_))
    }

    /// Ensure the strategy is configured for the specified agent player.
    pub fn set_agent_player(&mut self, player: Player) {
        match self {
            Self::ClassicMenace(_) => {}
            Self::ActiveInference(algo) => {
                algo.set_agent_player(player);
            }
            Self::OracleActiveInference(algo) => {
                algo.set_agent_player(player);
            }
            Self::PureActiveInference(algo) => {
                algo.set_agent_player(player);
            }
        }
    }
}

impl LearningAlgorithm for LearningStrategy {
    fn train_from_game(
        &mut self,
        workspace: &mut MenaceWorkspace,
        states: &[BoardState],
        moves: &[usize],
        outcome: GameOutcome,
        player: Player,
    ) -> crate::Result<()> {
        match self {
            Self::ClassicMenace(algo) => {
                algo.train_from_game(workspace, states, moves, outcome, player)
            }
            Self::ActiveInference(algo) => {
                algo.train_from_game(workspace, states, moves, outcome, player)
            }
            Self::OracleActiveInference(algo) => {
                algo.train_from_game(workspace, states, moves, outcome, player)
            }
            Self::PureActiveInference(algo) => {
                algo.train_from_game(workspace, states, moves, outcome, player)
            }
        }
    }

    fn name(&self) -> &str {
        match self {
            Self::ClassicMenace(algo) => algo.name(),
            Self::ActiveInference(algo) => algo.name(),
            Self::OracleActiveInference(algo) => algo.name(),
            Self::PureActiveInference(algo) => algo.name(),
        }
    }

    fn stats(&self) -> HashMap<String, f64> {
        match self {
            Self::ClassicMenace(algo) => algo.stats(),
            Self::ActiveInference(algo) => algo.stats(),
            Self::OracleActiveInference(algo) => algo.stats(),
            Self::PureActiveInference(algo) => algo.stats(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::ClassicMenace(algo) => algo.reset(),
            Self::ActiveInference(algo) => algo.reset(),
            Self::OracleActiveInference(algo) => algo.reset(),
            Self::PureActiveInference(algo) => algo.reset(),
        }
    }
}
