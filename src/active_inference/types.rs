//! Core types for Active Inference framework
//!
//! This module defines the fundamental types used throughout the Active Inference
//! implementation, including opponent models, game outcomes, and computation modes.

use serde::{Deserialize, Serialize};

/// Specifies the type of opponent in the Active Inference model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpponentKind {
    /// Opponent plays uniformly at random
    Uniform,
    /// Opponent plays to maximize agent's risk (worst-case)
    Adversarial,
    /// Opponent plays optimally according to minimax
    Minimax,
}

/// Strategy for breaking ties when multiple moves have equal value
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TieBreak {
    /// Split probability mass uniformly across optimal moves
    Uniform,
}

/// Terminal game outcomes from the perspective of player X
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TerminalOutcome {
    XWin,
    OWin,
    Draw,
}

/// Mode for computing Expected Free Energy
///
/// Determines whether to use approximate (greedy) or exact (policy-optimized)
/// EFE computation. The distinction matters for research applications comparing
/// different planning strategies.
///
/// # Modes
///
/// **Approx** (Approximate/Greedy):
/// - Evaluates EFE for each action: G(a) = Risk(a) - β × Epistemic(a)
/// - Selects action with minimum EFE: a* = argmin_a G(a)
/// - Faster computation, suitable for real-time decision-making
/// - No explicit policy distribution computed
///
/// **Exact** (Policy-Optimized):
/// - Computes optimal policy distribution: q*(a) ∝ p(a) exp(-G(a)/λ)
/// - Incorporates policy prior p(a) via KL regularization
/// - Lambda (λ) parameter controls exploration-exploitation tradeoff
/// - Returns full policy distribution (useful for analysis and entropy computation)
/// - Implements control-as-inference framework (Levine 2018, Todorov 2008)
///
/// # Mathematical Formulation
///
/// For Exact mode, the optimal policy minimizes:
/// ```text
/// KL[q(a) || p(a)] + E_q[G(a)]
/// ```
///
/// This yields: `q*(a) = p(a) exp(-G(a)/λ) / Z`
///
/// where Z is the partition function (normalization constant).
///
/// # Example
///
/// ```rust
/// use menace::active_inference::{EFEMode, PreferenceModel};
///
/// let mut prefs = PreferenceModel::from_probabilities(0.6, 0.35, 0.05);
///
/// // Use approximate mode for faster computation
/// prefs.efe_mode = EFEMode::Approx;
///
/// // Use exact mode for policy optimization with λ = 0.25
/// prefs.efe_mode = EFEMode::Exact;
/// prefs.policy_lambda = 0.25;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EFEMode {
    /// Approximate EFE: Greedy selection `a* = argmin_a G(a)`
    ///
    /// Faster computation, no policy distribution computed.
    /// Suitable for real-time decision-making.
    Approx,

    /// Exact EFE: Optimal policy `q*(a) ∝ p(a) exp(-G(a)/λ)`
    ///
    /// Computes full policy distribution with KL-regularization.
    /// Enables control-as-inference and policy entropy analysis.
    Exact,
}

/// Risk computation model for preferences
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskModel {
    /// Risk as negative log preference: -ln(p)
    NegativeLogPreference,
    /// Risk as negative utility: -u
    NegativeUtility,
    /// Risk as KL divergence: KL[Q(o|π) || P(o)] between predicted and preferred outcomes
    KLDivergence,
}
