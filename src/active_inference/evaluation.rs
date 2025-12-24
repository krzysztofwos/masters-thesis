//! Evaluation structures for Active Inference
//!
//! This module contains types for representing state evaluations, action evaluations,
//! and outcome distributions in the Active Inference framework.

use super::{preferences::PreferenceModel, types::TerminalOutcome};
use crate::tictactoe::Player;

/// Distribution over terminal outcomes
#[derive(Debug, Clone, Copy)]
pub struct OutcomeDistribution {
    pub x_win: f64,
    pub o_win: f64,
    pub draw: f64,
}

impl OutcomeDistribution {
    /// Creates a zero distribution
    pub fn zero() -> Self {
        Self {
            x_win: 0.0,
            o_win: 0.0,
            draw: 0.0,
        }
    }

    /// Creates a deterministic distribution for a terminal outcome
    pub fn terminal(winner: Option<Player>) -> Self {
        match winner {
            Some(Player::X) => Self {
                x_win: 1.0,
                o_win: 0.0,
                draw: 0.0,
            },
            Some(Player::O) => Self {
                x_win: 0.0,
                o_win: 1.0,
                draw: 0.0,
            },
            None => Self {
                x_win: 0.0,
                o_win: 0.0,
                draw: 1.0,
            },
        }
    }

    /// Adds another distribution weighted by a factor
    pub fn add_weighted(&mut self, other: &Self, weight: f64) {
        self.x_win += weight * other.x_win;
        self.o_win += weight * other.o_win;
        self.draw += weight * other.draw;
    }

    /// Computes the expected risk (negative utility) given preference model.
    ///
    /// The risk calculation depends on the risk model in preferences:
    /// - NegativeLogPreference: Weighted sum of -ln(p) values
    /// - NegativeUtility: Weighted sum of negative utilities
    /// - KLDivergence: KL[Q(o|π) || P(o)] where Q is this distribution, P is preferred
    ///
    /// # Arguments
    ///
    /// * `preferences` - The preference model defining risk values for each outcome
    ///
    /// # Returns
    ///
    /// The expected risk as a weighted average of outcome risks (for non-KL models)
    /// or as KL divergence (for KL model).
    ///
    /// Computes the entropy (ambiguity) of the outcome distribution.
    ///
    /// Ambiguity quantifies the uncertainty about which specific outcome will occur,
    /// independent of whether those outcomes are preferred or not. In the canonical
    /// Active Inference formulation (Friston et al., 2015), ambiguity H[Q(o|π)] is
    /// computed as the Shannon entropy of the outcome distribution:
    ///
    /// H[Q(o|π)] = -Σ Q(o) ln Q(o)
    ///
    /// Higher entropy means more uncertainty about outcomes. A deterministic distribution
    /// (one outcome has probability 1.0) has zero ambiguity, while a uniform distribution
    /// has maximum ambiguity.
    ///
    /// # Returns
    ///
    /// The Shannon entropy in nats (natural units).
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::active_inference::OutcomeDistribution;
    /// use menace::tictactoe::Player;
    ///
    /// // Deterministic outcome: zero ambiguity
    /// let certain = OutcomeDistribution::terminal(Some(Player::X));
    /// assert_eq!(certain.ambiguity(), 0.0);
    ///
    /// // Uncertain outcome: positive ambiguity
    /// let uncertain = OutcomeDistribution {
    ///     x_win: 0.33,
    ///     draw: 0.34,
    ///     o_win: 0.33,
    /// };
    /// assert!(uncertain.ambiguity() > 0.0);
    /// ```
    pub fn ambiguity(&self) -> f64 {
        let eps = 1e-12;
        let mut h = 0.0;

        if self.x_win > eps {
            h -= self.x_win * self.x_win.ln();
        }
        if self.draw > eps {
            h -= self.draw * self.draw.ln();
        }
        if self.o_win > eps {
            h -= self.o_win * self.o_win.ln();
        }

        h
    }

    /// Computes the expected risk (negative utility) given preference model.
    ///
    /// The risk calculation depends on the risk model in preferences:
    /// - NegativeLogPreference: Weighted sum of -ln(p) values
    /// - NegativeUtility: Weighted sum of negative utilities
    /// - KLDivergence: KL[Q(o|π) || P(o)] where Q is this distribution, P is preferred
    ///
    /// # Arguments
    ///
    /// * `preferences` - The preference model defining risk values for each outcome
    ///
    /// # Returns
    ///
    /// The expected risk as a weighted average of outcome risks (for non-KL models)
    /// or as KL divergence (for KL model).
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::active_inference::{OutcomeDistribution, PreferenceModel};
    /// use menace::tictactoe::Player;
    ///
    /// // Create preference model where winning is good (low risk/negative utility)
    /// let preferences = PreferenceModel::from_utilities(1.0, 0.0, -1.0);
    /// let dist = OutcomeDistribution::terminal(Some(Player::X));
    /// let risk = dist.expected_risk(&preferences);
    /// // X winning should have low (negative) risk when win utility is positive
    /// assert!(risk < 0.0);
    /// ```
    pub fn expected_risk(&self, preferences: &PreferenceModel) -> f64 {
        use super::types::RiskModel;

        match preferences.risk_model() {
            RiskModel::NegativeLogPreference | RiskModel::NegativeUtility => {
                // Traditional weighted sum approach
                self.x_win * preferences.risk(TerminalOutcome::XWin)
                    + self.draw * preferences.risk(TerminalOutcome::Draw)
                    + self.o_win * preferences.risk(TerminalOutcome::OWin)
            }
            RiskModel::KLDivergence => {
                // KL[Q || P] = Σ Q(i) * ln(Q(i) / P(i))
                // where Q = predicted (this distribution), P = preferred
                let eps = 1e-12_f64;

                let kl_x_win = if self.x_win > eps {
                    self.x_win * (self.x_win / preferences.preferred_dist_x_win.max(eps)).ln()
                } else {
                    0.0
                };

                let kl_draw = if self.draw > eps {
                    self.draw * (self.draw / preferences.preferred_dist_draw.max(eps)).ln()
                } else {
                    0.0
                };

                let kl_o_win = if self.o_win > eps {
                    self.o_win * (self.o_win / preferences.preferred_dist_o_win.max(eps)).ln()
                } else {
                    0.0
                };

                kl_x_win + kl_draw + kl_o_win
            }
        }
    }
}

/// Internal state value representation
///
/// Represents the value of a state in Active Inference, including both pragmatic
/// (risk) and epistemic (information gain) components.
#[derive(Debug, Clone)]
pub(crate) struct StateValue {
    /// Pragmatic value: KL divergence from preferred outcome distribution
    pub risk: f64,

    /// Epistemic value: Information gain about opponent parameters (θ)
    /// Computed as I(θ; o|π) for Uniform opponent model, 0.0 otherwise
    pub epistemic: f64,

    /// Ambiguity: Entropy of outcome distribution H[Q(o|π)]
    ///
    /// In canonical Active Inference (Friston et al., 2015), ambiguity represents
    /// uncertainty about which specific outcome will occur, computed as:
    ///
    /// H[Q(o|π)] = -Σ Q(o) ln Q(o)
    ///
    /// This is distinct from epistemic value (information gain about parameters).
    /// Ambiguity quantifies outcome uncertainty, while epistemic value quantifies
    /// learning potential.
    pub ambiguity: f64,

    /// Distribution over terminal outcomes (win/draw/loss)
    pub distribution: OutcomeDistribution,
}

impl StateValue {
    /// Computes expected free energy with epistemic and ambiguity weights
    ///
    /// Implements canonical Active Inference EFE (Friston et al., 2015):
    ///
    /// G = Risk + β_ambiguity × Ambiguity - β_epistemic × Epistemic
    ///
    /// where:
    /// - Risk: Expected divergence from preferred outcomes (pragmatic value)
    /// - Ambiguity: Entropy of outcome distribution H[Q(o|π)] (outcome uncertainty)
    /// - Epistemic: Information gain about parameters I(θ; o|π) (learning potential)
    ///
    /// All terms are in nats (natural units of information).
    ///
    /// # Arguments
    ///
    /// * `beta_epistemic` - Weight for information gain (typically positive)
    /// * `beta_ambiguity` - Weight for outcome uncertainty (positive = risk-averse, negative = risk-seeking)
    pub fn expected_free_energy(&self, beta_epistemic: f64, beta_ambiguity: f64) -> f64 {
        self.risk + beta_ambiguity * self.ambiguity - beta_epistemic * self.epistemic
    }

    /// Factory method for terminal states
    ///
    /// Creates a StateValue for a terminal game state with the appropriate
    /// outcome distribution and risk based on the game outcome.
    pub(crate) fn terminal(outcome: Option<Player>, preferences: &PreferenceModel) -> Self {
        let distribution = OutcomeDistribution::terminal(outcome);
        let risk = distribution.expected_risk(preferences);

        Self {
            risk,
            epistemic: 0.0,
            ambiguity: 0.0,
            distribution,
        }
    }
}

/// Evaluation of a single action from a state
#[derive(Debug, Clone)]
pub struct ActionEvaluation {
    /// The action (move position)
    pub action: usize,
    /// The resulting state label
    pub next_state: String,
    /// Expected free energy of this action
    pub free_energy: f64,
    /// Risk (expected negative utility)
    pub risk: f64,
    /// Epistemic value (information gain about preferences)
    pub epistemic: f64,
    /// Ambiguity (expected entropy of outcomes)
    pub ambiguity: f64,
    /// Distribution over terminal outcomes
    pub outcome_distribution: OutcomeDistribution,
    /// Opponent's expected information gain
    pub opponent_eig: f64,
    /// Policy prior weight for this action
    pub policy_prior: f64,
}

/// Evaluation of an opponent's action
#[derive(Debug, Clone)]
pub struct OpponentActionEvaluation {
    pub action: usize,
    pub next_state: String,
    pub predictive_weight: f64,
    pub free_energy: f64,
    pub risk: f64,
    pub epistemic: f64,
    pub ambiguity: f64,
    pub outcome_distribution: OutcomeDistribution,
}

impl OpponentActionEvaluation {
    /// Create a new OpponentActionEvaluation with default values
    pub(crate) fn new(action: usize, next_state: String) -> Self {
        Self {
            action,
            next_state,
            predictive_weight: 0.0,
            free_energy: 0.0,
            risk: 0.0,
            epistemic: 0.0,
            ambiguity: 0.0,
            outcome_distribution: OutcomeDistribution::zero(),
        }
    }

    /// Create from an ActionEdge (convenience constructor)
    pub(crate) fn from_edge(edge: &crate::active_inference::state::ActionEdge) -> Self {
        Self::new(edge.action, edge.next_label.clone())
    }

    /// Set values from a StateValue and weights
    pub(crate) fn with_state_value(
        mut self,
        value: StateValue,
        weight: f64,
        epistemic_weight: f64,
        ambiguity_weight: f64,
    ) -> Self {
        self.predictive_weight = weight;
        self.free_energy = value.expected_free_energy(epistemic_weight, ambiguity_weight);
        self.risk = value.risk;
        self.epistemic = value.epistemic;
        self.ambiguity = value.ambiguity;
        self.outcome_distribution = value.distribution;
        self
    }

    /// Set values from an outcome distribution (for Minimax)
    pub(crate) fn with_distribution(
        mut self,
        distribution: OutcomeDistribution,
        weight: f64,
        risk: f64,
    ) -> Self {
        self.predictive_weight = weight;
        self.free_energy = risk;
        self.risk = risk;
        self.epistemic = 0.0;
        self.ambiguity = 0.0;
        self.outcome_distribution = distribution;
        self
    }
}

/// Summary of opponent state evaluation
#[derive(Debug, Clone)]
pub struct OpponentStateSummary {
    pub information_gain: f64,
    pub actions: Vec<OpponentActionEvaluation>,
}

/// Complete state summary with exact policy computation
#[derive(Debug, Clone)]
pub struct ExactStateSummary {
    pub actions: Vec<ActionEvaluation>,
    pub policy: crate::efe::ExactPolicySummary,
}
