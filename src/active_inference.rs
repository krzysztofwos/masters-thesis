//! Active Inference framework for game-playing agents
//!
//! This module implements Active Inference, a Bayesian approach to decision-making
//! that combines risk minimization with information-seeking behavior. The framework
//! evaluates game positions by computing Expected Free Energy (EFE) which balances
//! outcome preferences with epistemic value.
//!
//! ## Key Concepts
//!
//! - **Generative Model**: Represents the agent's beliefs about game dynamics
//! - **Preferences**: Encodes the agent's utility over outcomes (win/draw/loss)
//! - **Opponent Models**: Different assumptions about opponent behavior (uniform/adversarial/minimax)
//! - **Expected Free Energy**: Combines risk and epistemic value to guide action selection
//!
//! ## Module Structure
//!
//! - [`types`]: Core enums and type definitions
//! - [`opponents`]: Opponent models and strategies
//! - [`preferences`]: Preference models and policy priors
//! - [`evaluation`]: Evaluation structures and results
//! - [`generative_model`]: Main generative model implementation
//! - [`state`]: Internal state representation

pub mod evaluation;
pub mod generative_model;
pub mod opponents;
pub mod preferences;
pub mod state;
pub mod types;

// Public re-exports
pub use evaluation::{
    ActionEvaluation, ExactStateSummary, OpponentActionEvaluation, OpponentStateSummary,
    OutcomeDistribution,
};
pub use generative_model::GenerativeModel;
pub use opponents::{AdversarialOpponent, MinimaxOpponent, Opponent, UniformOpponent};
pub use preferences::{PolicyPrior, PreferenceModel};
pub use types::{EFEMode, OpponentKind, RiskModel, TerminalOutcome, TieBreak};
