//! Analysis tools for studying MENACE and Active Inference agents
//!
//! This module provides tools for analyzing learned policies, computing
//! theoretical quantities, and validating theoretical predictions.

pub mod free_energy;
pub mod stats;
pub mod trajectory;

pub use free_energy::{
    FreeEnergyAnalysis, FreeEnergyComponents, OpponentModel, UniformOpponent, WorkspaceOpponent,
};
pub use stats::{GameAnalysis, GameStats};
pub use trajectory::{Trajectory, TrajectoryAnalysis};
