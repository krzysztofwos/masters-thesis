//! MENACE (Matchbox Educable Noughts And Crosses Engine) implementation
//!
//! This crate provides:
//! - Complete Tic-Tac-Toe game implementation with validation
//! - MENACE learning agent with matchbox-based reinforcement learning
//! - Game analysis tools and statistics
//! - Active Inference baseline for comparison
//! - Integration with categorical modeling

pub mod active_inference;
pub mod adapters;
pub mod analysis;
pub mod app;
pub mod beliefs;
pub mod cli;
pub mod efe;
pub mod error;
pub mod export;
pub mod identifiers;
pub mod menace;
pub mod pipeline;
pub mod ports;
pub mod q_learning;
pub mod tictactoe;
pub mod types;
pub mod utils;
pub mod workspace;

pub use active_inference::{
    ActionEvaluation as ActiveActionEvaluation, AdversarialOpponent as ActiveAdversarialOpponent,
    EFEMode as ActiveEFEMode, ExactStateSummary as ActiveExactStateSummary,
    GenerativeModel as ActiveGenerativeModel, MinimaxOpponent as ActiveMinimaxOpponent,
    Opponent as ActiveOpponent, OpponentKind as ActiveOpponentKind,
    OutcomeDistribution as ActiveOutcomeDistribution, PolicyPrior as ActivePolicyPrior,
    PreferenceModel as ActivePreferenceModel, RiskModel as ActiveRiskModel, TerminalOutcome,
    TieBreak as ActiveTieBreak, UniformOpponent as ActiveUniformOpponent,
};
pub use beliefs::Beliefs as ActiveBeliefs;
pub use efe::{
    ExactPolicySummary as ActiveExactPolicySummary, decompose_policy as active_decompose_policy,
};
pub use error::{Error, Result};
pub use types::CanonicalLabel;
pub use workspace::{
    InitialBeadSchedule, MenaceLearningRule, MenaceWorkspace, Reinforcement, RestockMode,
    StateFilter,
};
