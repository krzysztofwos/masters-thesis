//! Opponent models for Active Inference
//!
//! This module defines different opponent strategies used in game tree evaluation,
//! including uniform random play, adversarial play, and minimax-optimal play.

use super::{generative_model::GenerativeModel, types::OpponentKind};
use crate::beliefs::Beliefs;

/// Trait for opponent behavior in Active Inference
pub trait Opponent: Send {
    /// Returns the kind of opponent
    fn kind(&self) -> OpponentKind;

    /// Computes predictive distribution and expected information gain
    ///
    /// Returns:
    /// - Predictive weights over actions
    /// - Expected information gain from opponent's perspective
    fn predictive_and_eig(
        &self,
        gm: &GenerativeModel,
        state_label: &str,
        beliefs: &Beliefs,
    ) -> Option<(Vec<f64>, f64)>;
}

/// Opponent that plays uniformly at random
#[derive(Debug, Clone, Copy)]
pub struct UniformOpponent;

impl Opponent for UniformOpponent {
    fn kind(&self) -> OpponentKind {
        OpponentKind::Uniform
    }

    fn predictive_and_eig(
        &self,
        gm: &GenerativeModel,
        state_label: &str,
        beliefs: &Beliefs,
    ) -> Option<(Vec<f64>, f64)> {
        let n = gm.legal_action_count(state_label);
        Some((
            beliefs.predictive(state_label, n),
            beliefs.opponent_eig(state_label, n),
        ))
    }
}

/// Opponent that maximizes agent's risk (worst-case scenario)
///
/// This opponent model represents a pessimistic, robust decision-making strategy.
/// By returning `None` for `predictive_and_eig`, it signals the generative model
/// to assume the worst-case outcome for each action.
///
/// # Expected Free Energy Interpretation
///
/// In Active Inference, agents minimize Expected Free Energy (EFE), which represents
/// the cost of taking an action. The adversarial opponent causes the agent to:
/// 1. Assume the opponent will choose moves that **maximize** the agent's EFE (cost)
/// 2. Select actions that minimize this worst-case EFE
/// 3. Effectively implement a minimax-like robust strategy
///
/// This is conceptually similar to game-theoretic minimax, but formulated through
/// the lens of free energy minimization rather than explicit value functions.
#[derive(Debug, Clone, Copy)]
pub struct AdversarialOpponent;

impl Opponent for AdversarialOpponent {
    fn kind(&self) -> OpponentKind {
        OpponentKind::Adversarial
    }

    /// Returns `None` to signal worst-case evaluation.
    ///
    /// When `None` is returned, the generative model treats each opponent action
    /// as maximizing the agent's risk, leading to pessimistic (robust) policies.
    fn predictive_and_eig(
        &self,
        _gm: &GenerativeModel,
        _state_label: &str,
        _beliefs: &Beliefs,
    ) -> Option<(Vec<f64>, f64)> {
        None
    }
}

/// Opponent that plays optimally according to minimax
#[derive(Debug, Clone, Copy)]
pub struct MinimaxOpponent;

impl Opponent for MinimaxOpponent {
    fn kind(&self) -> OpponentKind {
        OpponentKind::Minimax
    }

    fn predictive_and_eig(
        &self,
        _gm: &GenerativeModel,
        _state_label: &str,
        _beliefs: &Beliefs,
    ) -> Option<(Vec<f64>, f64)> {
        None
    }
}

impl OpponentKind {
    /// Creates a boxed opponent trait object from the kind
    pub fn into_boxed_opponent(self) -> Box<dyn Opponent> {
        match self {
            OpponentKind::Uniform => Box::new(UniformOpponent),
            OpponentKind::Adversarial => Box::new(AdversarialOpponent),
            OpponentKind::Minimax => Box::new(MinimaxOpponent),
        }
    }
}
