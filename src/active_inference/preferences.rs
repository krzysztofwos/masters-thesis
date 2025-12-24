//! Preference models and policy priors for Active Inference
//!
//! This module defines preference models that encode an agent's utility over outcomes
//! and policy priors that bias action selection.

use serde::{Deserialize, Serialize};

use super::{
    state::StateNode,
    types::{EFEMode, RiskModel, TerminalOutcome},
};
use crate::{tictactoe::Player, workspace::InitialBeadSchedule};

/// Canonical win/draw/loss outcome preferences used across CLI defaults and reports.
pub const CANONICAL_PREFERENCE_PROBS: (f64, f64, f64) = (0.6, 0.35, 0.05);

/// Prior distribution over policies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyPrior {
    /// Uniform prior over all actions
    Uniform,
    /// MENACE-style positional prior based on move positions
    MenacePositional,
    /// MENACE-style initial bead schedule
    MenaceInitial(InitialBeadSchedule),
    /// Explicit prior weights
    Explicit(Vec<f64>),
}

/// Preference model encoding utilities and risk attitudes
///
/// # Perspective Handling
///
/// This struct stores preferences over **terminal outcomes** in absolute coordinates:
/// $P(\text{XWin})$, $P(\text{Draw})$, and $P(\text{OWin})$.
///
/// When creating a model with `from_probabilities(win, draw, loss)` or
/// `from_utilities(win, draw, loss)`, the arguments are interpreted in the
/// canonical outcome order **Win/Draw/Loss for an agent playing as X**
/// (i.e., `win` corresponds to $\text{XWin}$ and `loss` corresponds to $\text{OWin}$).
///
/// To reuse the same win/draw/loss preferences for an agent playing as O, call
/// `for_player(Player::O)`, which swaps the XWin/OWin fields so that the agent
/// still prefers its own wins (now $\text{OWin}$) and dislikes its losses (now $\text{XWin}$).
///
/// For agent-relative access, use `agent_win_preference_for(Player)` /
/// `agent_loss_preference_for(Player)` or `agent_relative_preference_distribution(Player)`.
///
/// # Example
///
/// ```
/// use menace::active_inference::PreferenceModel;
/// use menace::tictactoe::Player;
///
/// // Preferences are specified in win/draw/loss order for an X-playing agent.
/// let base = PreferenceModel::from_probabilities(0.9, 0.05, 0.05);
///
/// // Re-orient for an O-playing agent (swap XWin/OWin).
/// let o_prefs = base.for_player(Player::O);
/// assert!(o_prefs.agent_win_preference_for(Player::O) > o_prefs.agent_loss_preference_for(Player::O));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferenceModel {
    pub(crate) risk_win: f64,
    pub(crate) risk_draw: f64,
    pub(crate) risk_loss: f64,
    pub(crate) risk_model: RiskModel,
    /// Preferred outcome distribution (for KL divergence risk model).
    ///
    /// These fields are stored in absolute coordinates: XWin/Draw/OWin.
    /// To reuse the same win/draw/loss preferences for an O-playing agent, call
    /// `for_player(Player::O)` to swap XWin/OWin.
    pub(crate) preferred_dist_x_win: f64,
    pub(crate) preferred_dist_draw: f64,
    pub(crate) preferred_dist_o_win: f64,
    /// Weight for epistemic (information-seeking) value in EFE: G = Risk - β_epistemic × Epistemic + β_ambiguity × Ambiguity
    pub epistemic_weight: f64,
    /// Weight for ambiguity (outcome uncertainty) in EFE: G = Risk - β_epistemic × Epistemic + β_ambiguity × Ambiguity
    ///
    /// Positive values increase cost for uncertain outcomes (prefer deterministic predictions).
    /// Negative values decrease cost for uncertain outcomes (prefer keeping options open).
    /// Zero (default) ignores ambiguity entirely (classic formulation).
    pub ambiguity_weight: f64,
    /// Mode for computing expected free energy
    pub efe_mode: EFEMode,
    /// Temperature parameter for policy distribution
    pub policy_lambda: f64,
    /// Prior distribution over policies
    pub policy_prior: PolicyPrior,
    /// Dirichlet concentration for opponent model
    pub opponent_dirichlet_alpha: f64,
    /// Scale factor for converting policy probabilities to bead weights.
    /// Default is 100.0, which maps a uniform policy over 9 actions (~0.11 each)
    /// to approximately 11 beads per action. This should ideally be tied to the
    /// InitialBeadSchedule for consistency across experiments.
    pub policy_to_beads_scale: f64,
}

impl PreferenceModel {
    /// Creates a preference model from (possibly unnormalised) outcome probabilities.
    ///
    /// Uses canonical Active Inference formulation where risk is computed as
    /// KL divergence between predicted and preferred outcome distributions.
    ///
    /// If the inputs do not sum to 1, they are renormalised. Zero entries are
    /// clamped to a tiny ε to avoid division by zero in KL calculations.
    pub fn from_probabilities(win: f64, draw: f64, loss: f64) -> Self {
        assert!(
            win >= 0.0 && draw >= 0.0 && loss >= 0.0,
            "preferences must be non-negative"
        );
        let sum = win + draw + loss;
        assert!(
            sum.is_finite() && sum > 0.0,
            "at least one preference must be > 0"
        );

        let (mut pw, mut pd, mut pl) = if (sum - 1.0).abs() <= f64::EPSILON {
            (win, draw, loss)
        } else {
            (win / sum, draw / sum, loss / sum)
        };

        // Clamp to avoid division by zero in KL divergence
        let eps = 1e-12_f64;
        if pw == 0.0 {
            pw = eps;
        }
        if pd == 0.0 {
            pd = eps;
        }
        if pl == 0.0 {
            pl = eps;
        }

        Self {
            // These are now unused for KL model but kept for backward compat with utilities
            risk_win: 0.0,
            risk_draw: 0.0,
            risk_loss: 0.0,
            risk_model: RiskModel::KLDivergence, // Canonical AIF default
            preferred_dist_x_win: pw,
            preferred_dist_draw: pd,
            preferred_dist_o_win: pl,
            epistemic_weight: 1.0,
            ambiguity_weight: 0.0, // Default: don't penalize ambiguity
            efe_mode: EFEMode::Approx,
            policy_lambda: 1.0,
            policy_prior: PolicyPrior::Uniform,
            opponent_dirichlet_alpha: 1.0,
            policy_to_beads_scale: 100.0,
        }
    }

    /// Creates a preference model from utilities
    pub fn from_utilities(win: f64, draw: f64, loss: f64) -> Self {
        assert!(win.is_finite(), "win utility must be finite");
        assert!(draw.is_finite(), "draw utility must be finite");
        assert!(loss.is_finite(), "loss utility must be finite");

        // For utility-based model, convert to probability distribution via softmax
        // This allows KL divergence to be computed if needed
        let max_u = win.max(draw).max(loss);
        let exp_win = (win - max_u).exp();
        let exp_draw = (draw - max_u).exp();
        let exp_loss = (loss - max_u).exp();
        let sum = exp_win + exp_draw + exp_loss;

        Self {
            risk_win: -win,
            risk_draw: -draw,
            risk_loss: -loss,
            risk_model: RiskModel::NegativeUtility,
            preferred_dist_x_win: exp_win / sum,
            preferred_dist_draw: exp_draw / sum,
            preferred_dist_o_win: exp_loss / sum,
            epistemic_weight: 1.0,
            ambiguity_weight: 0.0, // Default: don't penalize ambiguity
            efe_mode: EFEMode::Approx,
            policy_lambda: 1.0,
            policy_prior: PolicyPrior::Uniform,
            opponent_dirichlet_alpha: 1.0,
            policy_to_beads_scale: 100.0,
        }
    }

    /// Returns risk for a terminal outcome
    pub fn risk(&self, outcome: TerminalOutcome) -> f64 {
        match outcome {
            TerminalOutcome::XWin => self.risk_win,
            TerminalOutcome::Draw => self.risk_draw,
            TerminalOutcome::OWin => self.risk_loss,
        }
    }

    /// Return a preference model aligned to the specified player perspective.
    ///
    /// # Perspective Transformation
    ///
    /// For `Player::X`, returns a clone with no changes.
    /// For `Player::O`, swaps the XWin/OWin preference fields so that the agent
    /// still prefers its own wins (now OWin) and dislikes its losses (now XWin).
    ///
    /// This transformation allows the same model to be used for both players while
    /// maintaining consistent absolute field names.
    ///
    /// # Example
    ///
    /// ```
    /// use menace::active_inference::PreferenceModel;
    /// use menace::tictactoe::Player;
    ///
    /// let prefs = PreferenceModel::from_probabilities(0.7, 0.2, 0.1);
    /// let o_prefs = prefs.for_player(Player::O);
    ///
    /// // XWin/OWin preferences are swapped for O
    /// assert!((o_prefs.x_win_preference() - 0.1).abs() < 1e-6);
    /// assert!((o_prefs.o_win_preference() - 0.7).abs() < 1e-6);
    /// assert!((o_prefs.agent_win_preference_for(Player::O) - 0.7).abs() < 1e-6);
    /// ```
    pub fn for_player(&self, player: Player) -> PreferenceModel {
        if player == Player::X {
            self.clone()
        } else {
            let mut mirrored = self.clone();
            std::mem::swap(&mut mirrored.risk_win, &mut mirrored.risk_loss);
            std::mem::swap(
                &mut mirrored.preferred_dist_x_win,
                &mut mirrored.preferred_dist_o_win,
            );
            mirrored
        }
    }

    /// Returns the risk model type
    pub fn risk_model(&self) -> RiskModel {
        self.risk_model
    }

    /// Sets epistemic weight (builder pattern)
    pub fn with_epistemic_weight(mut self, weight: f64) -> Self {
        self.epistemic_weight = weight;
        self
    }

    /// Sets ambiguity weight (builder pattern)
    ///
    /// Controls how much the agent penalizes or rewards outcome uncertainty.
    ///
    /// # Arguments
    ///
    /// * `weight` - The ambiguity weight β_ambiguity
    ///   - Positive: Prefer deterministic outcomes (risk-averse to uncertainty)
    ///   - Negative: Prefer uncertain outcomes (keep options open)
    ///   - Zero (default): Ignore ambiguity
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::active_inference::PreferenceModel;
    ///
    /// // Risk-averse: penalize uncertain outcomes
    /// let risk_averse = PreferenceModel::from_probabilities(0.9, 0.5, 0.1)
    ///     .with_ambiguity_weight(0.5);
    ///
    /// // Risk-seeking: prefer keeping options open
    /// let risk_seeking = PreferenceModel::from_probabilities(0.9, 0.5, 0.1)
    ///     .with_ambiguity_weight(-0.5);
    /// ```
    pub fn with_ambiguity_weight(mut self, weight: f64) -> Self {
        assert!(weight.is_finite(), "ambiguity weight must be finite");
        self.ambiguity_weight = weight;
        self
    }

    /// Sets EFE computation mode (builder pattern)
    pub fn with_efe_mode(mut self, mode: EFEMode) -> Self {
        self.efe_mode = mode;
        self
    }

    /// Sets policy temperature (builder pattern)
    pub fn with_policy_lambda(mut self, lambda: f64) -> Self {
        assert!(
            lambda.is_finite() && lambda > 0.0,
            "lambda must be positive"
        );
        self.policy_lambda = lambda;
        self
    }

    /// Sets policy prior (builder pattern)
    pub fn with_policy_prior(mut self, prior: PolicyPrior) -> Self {
        self.policy_prior = prior;
        self
    }

    /// Sets opponent Dirichlet concentration (builder pattern)
    pub fn with_opponent_dirichlet_alpha(mut self, alpha: f64) -> Self {
        assert!(alpha > 0.0, "Dirichlet concentration must be positive");
        self.opponent_dirichlet_alpha = alpha;
        self
    }

    /// Sets policy-to-beads scale factor (builder pattern)
    ///
    /// This controls how policy probabilities are converted to bead weights.
    /// Default is 100.0, which maps probabilities to roughly 0-100 range.
    /// Consider tying this to InitialBeadSchedule for consistency.
    pub fn with_policy_to_beads_scale(mut self, scale: f64) -> Self {
        assert!(
            scale > 0.0 && scale.is_finite(),
            "scale must be positive and finite"
        );
        self.policy_to_beads_scale = scale;
        self
    }

    /// Returns the preference for the terminal outcome $\text{XWin}$.
    ///
    /// If you need the agent-relative win/loss preferences, use
    /// `agent_win_preference_for(Player)` / `agent_loss_preference_for(Player)` instead.
    pub fn agent_win_preference(&self) -> f64 {
        self.preferred_dist_x_win
    }

    /// Returns the preference for the terminal outcome $\text{OWin}$.
    pub fn agent_loss_preference(&self) -> f64 {
        self.preferred_dist_o_win
    }

    /// Get the draw preference (same for both players).
    pub fn draw_preference(&self) -> f64 {
        self.preferred_dist_draw
    }

    /// Returns the preference for $\text{XWin}$ (alias for `agent_win_preference()`).
    pub fn x_win_preference(&self) -> f64 {
        self.preferred_dist_x_win
    }

    /// Returns the preference for $\text{OWin}$ (alias for `agent_loss_preference()`).
    pub fn o_win_preference(&self) -> f64 {
        self.preferred_dist_o_win
    }

    /// Returns the agent-relative preference for a win.
    ///
    /// This is the preference weight for the terminal outcome where `player` wins.
    pub fn agent_win_preference_for(&self, player: Player) -> f64 {
        match player {
            Player::X => self.preferred_dist_x_win,
            Player::O => self.preferred_dist_o_win,
        }
    }

    /// Returns the agent-relative preference for a loss.
    ///
    /// This is the preference weight for the terminal outcome where `player` loses.
    pub fn agent_loss_preference_for(&self, player: Player) -> f64 {
        match player {
            Player::X => self.preferred_dist_o_win,
            Player::O => self.preferred_dist_x_win,
        }
    }

    /// Returns the agent-relative preference distribution in win/draw/loss order.
    pub fn agent_relative_preference_distribution(&self, player: Player) -> [f64; 3] {
        [
            self.agent_win_preference_for(player),
            self.preferred_dist_draw,
            self.agent_loss_preference_for(player),
        ]
    }

    /// Computes policy prior weights for a state
    pub fn policy_prior_weights(&self, node: &StateNode) -> Vec<f64> {
        match &self.policy_prior {
            PolicyPrior::Uniform => vec![1.0; node.actions.len()],
            PolicyPrior::MenacePositional => menace_positional_prior(node),
            PolicyPrior::MenaceInitial(schedule) => menace_initial_prior(node, *schedule),
            PolicyPrior::Explicit(weights) => {
                assert_eq!(weights.len(), node.actions.len());
                weights.clone()
            }
        }
    }
}

/// Computes MENACE-style positional prior weights
pub(crate) fn menace_positional_prior(node: &StateNode) -> Vec<f64> {
    node.actions
        .iter()
        .map(|edge| {
            let pos = edge.action;
            match pos {
                4 => 8.0,             // Center
                0 | 2 | 6 | 8 => 4.0, // Corners
                1 | 3 | 5 | 7 => 2.0, // Edges
                _ => 1.0,
            }
        })
        .collect()
}

/// Computes MENACE-style initial bead schedule prior
pub(crate) fn menace_initial_prior(node: &StateNode, schedule: InitialBeadSchedule) -> Vec<f64> {
    vec![schedule.weight_for_piece_count(node.state.occupied_count()); node.actions.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tictactoe::Player;

    #[test]
    fn for_player_mirrors_preferences_for_o() {
        let model = PreferenceModel::from_probabilities(0.7, 0.2, 0.1)
            .with_epistemic_weight(1.5)
            .with_ambiguity_weight(0.25);

        let mirrored = model.for_player(Player::O);

        assert!((mirrored.preferred_dist_x_win - model.preferred_dist_o_win).abs() <= 1e-12);
        assert!((mirrored.preferred_dist_o_win - model.preferred_dist_x_win).abs() <= 1e-12);
        assert!((mirrored.preferred_dist_draw - model.preferred_dist_draw).abs() <= 1e-12);
        assert!((mirrored.risk_win - model.risk_loss).abs() <= f64::EPSILON);
        assert!((mirrored.risk_loss - model.risk_win).abs() <= f64::EPSILON);
        assert_eq!(mirrored.risk_model, model.risk_model);
        assert_eq!(mirrored.epistemic_weight, model.epistemic_weight);
        assert_eq!(mirrored.ambiguity_weight, model.ambiguity_weight);
    }
}
