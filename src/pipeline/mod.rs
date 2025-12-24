//! Training and evaluation pipeline abstractions
//!
//! This module provides composable pipelines for:
//! - Training learners with different opponents
//! - Evaluating learned policies
//! - Comparing multiple learners
//! - Recording observations during training

pub mod comparison;
pub mod observers;
pub mod regimen;
pub mod training;

// Re-export ports (trait definitions)
// Re-export learner implementations (adapters)
pub use comparison::{
    ComparisonFramework, ComparisonResult, DefensiveLearner, FrozenLearner, MenaceLearner,
    OptimalLearner, RandomLearner, SharedMenaceLearner,
};
// Re-export observer implementations (adapters)
pub use observers::{
    FreeEnergyObserver, JsonlObserver, MetricsObserver, MilestoneObserver, Observation,
    ProgressObserver, StepObservation,
};
pub use regimen::{CurriculumConfig, OpponentType, TrainingBlock, TrainingRegimen};
pub use training::{TrainingConfig, TrainingPipeline, TrainingResult};

pub use crate::ports::{Learner, Observer};
