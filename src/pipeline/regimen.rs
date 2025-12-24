//! Training regimen and curriculum learning support
//!
//! This module provides types for organizing training into sequential blocks
//! against different opponents, enabling curriculum learning strategies.

use serde::{Deserialize, Serialize};

/// A single block of training against a specific opponent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingBlock {
    /// Opponent type for this training block
    pub opponent: OpponentType,
    /// Number of games to play in this block
    pub games: usize,
}

impl TrainingBlock {
    /// Create a new training block
    pub fn new(opponent: OpponentType, games: usize) -> Self {
        Self { opponent, games }
    }
}

/// Opponent type for training
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpponentType {
    /// Random opponent
    Random,
    /// Optimal (minimax) opponent
    Optimal,
    /// Defensive opponent (blocks wins)
    Defensive,
}

impl OpponentType {
    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            OpponentType::Random => "Random",
            OpponentType::Optimal => "Optimal",
            OpponentType::Defensive => "Defensive",
        }
    }

    /// Get short label
    pub fn label(&self) -> &'static str {
        match self {
            OpponentType::Random => "random",
            OpponentType::Optimal => "optimal",
            OpponentType::Defensive => "defensive",
        }
    }
}

/// Training regimen strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingRegimen {
    /// Train exclusively against optimal opponent
    Optimal,
    /// Train exclusively against random opponent
    Random,
    /// Train exclusively against defensive opponent
    Defensive,
    /// Mixed curriculum with multiple opponents
    Mixed,
}

impl TrainingRegimen {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            TrainingRegimen::Optimal => "Single optimal opponent (benchmark)",
            TrainingRegimen::Random => "Purely random opponent (exploration stress test)",
            TrainingRegimen::Defensive => "Defensive opponent prioritizing blocks",
            TrainingRegimen::Mixed => {
                "Mixed curriculum blending random, defensive, and optimal play"
            }
        }
    }

    /// Get short label
    pub fn label(&self) -> &'static str {
        match self {
            TrainingRegimen::Optimal => "optimal",
            TrainingRegimen::Random => "random",
            TrainingRegimen::Defensive => "defensive",
            TrainingRegimen::Mixed => "mixed",
        }
    }

    /// Generate training schedule based on regimen and curriculum config
    pub fn schedule(&self, total_games: usize, config: &CurriculumConfig) -> Vec<TrainingBlock> {
        match self {
            TrainingRegimen::Optimal => {
                vec![TrainingBlock::new(OpponentType::Optimal, total_games)]
            }
            TrainingRegimen::Random => {
                vec![TrainingBlock::new(OpponentType::Random, total_games)]
            }
            TrainingRegimen::Defensive => {
                vec![TrainingBlock::new(OpponentType::Defensive, total_games)]
            }
            TrainingRegimen::Mixed => build_mixed_curriculum(total_games, config),
        }
    }
}

/// Configuration for mixed curriculum
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct CurriculumConfig {
    /// Override: number of random games (default: 40% of total)
    pub mixed_random_games: Option<usize>,
    /// Override: number of defensive games (default: 20% of total)
    pub mixed_defensive_games: Option<usize>,
    /// Override: number of optimal games (default: remaining after random + defensive)
    pub mixed_optimal_games: Option<usize>,
}

/// Build mixed curriculum schedule
///
/// Default allocation:
/// - 40% Random (exploration)
/// - 20% Defensive (tactical play)
/// - 40% Optimal (convergence)
fn build_mixed_curriculum(total_games: usize, config: &CurriculumConfig) -> Vec<TrainingBlock> {
    if total_games == 0 {
        return vec![];
    }

    // Calculate game allocation with defaults
    let mut random_games = config
        .mixed_random_games
        .unwrap_or(total_games.saturating_mul(2) / 5); // 40%

    let mut defensive_games = config.mixed_defensive_games.unwrap_or(total_games / 5); // 20%

    let mut optimal_games = config.mixed_optimal_games.unwrap_or(
        total_games.saturating_sub(random_games + defensive_games), // 40% or remaining
    );

    let mut remaining = total_games;
    let mut blocks = Vec::new();

    // Add random block
    if random_games == 0 && remaining > 0 {
        random_games = 1; // Ensure at least 1 game if space available
    }
    random_games = random_games.min(remaining);
    if random_games > 0 {
        blocks.push(TrainingBlock::new(OpponentType::Random, random_games));
        remaining -= random_games;
    }

    // Add defensive block
    if defensive_games == 0 && remaining > 0 {
        defensive_games = 1;
    }
    defensive_games = defensive_games.min(remaining);
    if defensive_games > 0 {
        blocks.push(TrainingBlock::new(OpponentType::Defensive, defensive_games));
        remaining -= defensive_games;
    }

    // Add optimal block
    if optimal_games == 0 && remaining > 0 {
        optimal_games = remaining;
    }
    optimal_games = optimal_games.min(remaining);
    if optimal_games > 0 {
        blocks.push(TrainingBlock::new(OpponentType::Optimal, optimal_games));
        remaining -= optimal_games;
    }

    // Allocate any remaining games to the last block
    if remaining > 0 {
        if let Some(last) = blocks.last_mut() {
            last.games += remaining;
        } else {
            // Fallback: create a random block with remaining games
            blocks.push(TrainingBlock::new(OpponentType::Random, remaining));
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixed_curriculum_default_allocation() {
        let config = CurriculumConfig::default();
        let blocks = build_mixed_curriculum(500, &config);

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].opponent, OpponentType::Random);
        assert_eq!(blocks[0].games, 200); // 40%
        assert_eq!(blocks[1].opponent, OpponentType::Defensive);
        assert_eq!(blocks[1].games, 100); // 20%
        assert_eq!(blocks[2].opponent, OpponentType::Optimal);
        assert_eq!(blocks[2].games, 200); // 40%
    }

    #[test]
    fn test_mixed_curriculum_custom_allocation() {
        let config = CurriculumConfig {
            mixed_random_games: Some(300),
            mixed_defensive_games: Some(100),
            mixed_optimal_games: Some(100),
        };
        let blocks = build_mixed_curriculum(500, &config);

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].games, 300);
        assert_eq!(blocks[1].games, 100);
        assert_eq!(blocks[2].games, 100);
    }

    #[test]
    fn test_single_regimen_schedules() {
        let config = CurriculumConfig::default();

        let optimal = TrainingRegimen::Optimal.schedule(100, &config);
        assert_eq!(optimal.len(), 1);
        assert_eq!(optimal[0].opponent, OpponentType::Optimal);
        assert_eq!(optimal[0].games, 100);

        let random = TrainingRegimen::Random.schedule(100, &config);
        assert_eq!(random.len(), 1);
        assert_eq!(random[0].opponent, OpponentType::Random);

        let defensive = TrainingRegimen::Defensive.schedule(100, &config);
        assert_eq!(defensive.len(), 1);
        assert_eq!(defensive[0].opponent, OpponentType::Defensive);
    }
}
