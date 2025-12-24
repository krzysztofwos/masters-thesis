//! Shared configuration types for CLI commands

use serde::{Deserialize, Serialize};

/// Common configuration shared across commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonConfig {
    /// Random seed for reproducibility
    pub seed: Option<u64>,

    /// Whether to show progress bars
    pub progress: bool,

    /// Verbose output
    pub verbose: bool,
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            seed: None,
            progress: true,
            verbose: false,
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training games
    pub games: usize,

    /// Opponent type
    pub opponent: String,

    /// Restock mode
    pub restock: Option<String>,

    /// Initial bead schedule
    pub initial_beads: Option<String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            games: 500,
            opponent: "random".to_string(),
            restock: None,
            initial_beads: None,
        }
    }
}

/// Evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Number of evaluation games
    pub games: usize,

    /// Opponent type
    pub opponent: String,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            games: 100,
            opponent: "optimal".to_string(),
        }
    }
}
