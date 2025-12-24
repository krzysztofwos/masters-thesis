//! MENACE learning system
//!
//! This module provides the MENACE learning agent and various learning algorithms.

pub mod active;
pub mod agent;
pub mod builder;
pub mod classic;
pub mod game_utils;
pub mod learning;
pub mod matchbox;
pub mod optimal;
pub mod serialization;
pub mod training;

// Re-export main types
pub use active::ActiveInference;
pub use agent::MenaceAgent;
pub use builder::MenaceAgentBuilder;
pub use classic::{ClassicMenace, ReinforcementValues};
pub use learning::LearningAlgorithm;
pub use matchbox::Matchbox;
pub use optimal::{
    OptimalPolicy, compute_optimal_policy, kl_divergence, kl_divergence_weighted,
    optimal_move_distribution,
};
pub use serialization::{AlgorithmParams, AlgorithmType, SavedMenaceAgent, TrainingMetadata};
pub use training::{TrainingConfig, TrainingSession};
