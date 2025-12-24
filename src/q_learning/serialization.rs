//! Serialization support for temporal difference learning agents.

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::{
    menace::TrainingMetadata,
    ports::Learner,
    q_learning::agent::{QLearningAgent, SarsaAgent, TdAgentState},
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TdAlgorithm {
    QLearning,
    Sarsa,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedTdAgent {
    pub version: u32,
    pub algorithm: TdAlgorithm,
    state: TdAgentState,
    pub metadata: TrainingMetadata,
}

pub enum TdLearner {
    QLearning(QLearningAgent),
    Sarsa(SarsaAgent),
}

impl TdLearner {
    pub fn into_box(self) -> Box<dyn Learner> {
        match self {
            TdLearner::QLearning(agent) => Box::new(agent),
            TdLearner::Sarsa(agent) => Box::new(agent),
        }
    }
}

impl SavedTdAgent {
    pub const VERSION: u32 = 1;

    pub fn from_q_learning(agent: &QLearningAgent, metadata: TrainingMetadata) -> Self {
        Self {
            version: Self::VERSION,
            algorithm: TdAlgorithm::QLearning,
            state: agent.export_state(),
            metadata,
        }
    }

    pub fn from_sarsa(agent: &SarsaAgent, metadata: TrainingMetadata) -> Self {
        Self {
            version: Self::VERSION,
            algorithm: TdAlgorithm::Sarsa,
            state: agent.export_state(),
            metadata,
        }
    }

    pub fn to_agent(&self) -> Result<TdLearner> {
        if self.version != Self::VERSION {
            return Err(anyhow!(
                "Unsupported TD save format version: {}. Expected {}",
                self.version,
                Self::VERSION
            ));
        }

        match self.algorithm {
            TdAlgorithm::QLearning => Ok(TdLearner::QLearning(QLearningAgent::from_state(
                self.state.clone(),
            ))),
            TdAlgorithm::Sarsa => Ok(TdLearner::Sarsa(SarsaAgent::from_state(self.state.clone()))),
        }
    }

    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create file: {}", path.as_ref().display()))?;
        let mut writer = BufWriter::new(file);

        rmp_serde::encode::write(&mut writer, self).context("Failed to serialize TD agent")?;

        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open file: {}", path.as_ref().display()))?;
        let reader = BufReader::new(file);

        rmp_serde::decode::from_read(reader).context("Failed to deserialize TD agent")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ports::Learner,
        tictactoe::{GameOutcome, Player},
    };

    fn train_once<L: Learner>(agent: &mut L, outcome: GameOutcome, role: Player) -> Result<()> {
        let moves = vec![0, 4, 1, 3, 2];
        agent.learn(Player::X, &moves, outcome, role)?;
        Ok(())
    }

    #[test]
    fn test_q_learning_roundtrip() -> Result<()> {
        let mut agent = QLearningAgent::new(0.5, 0.99, 0.5, 0.995, 0.01, 0.0).with_seed(7);
        train_once(&mut agent, GameOutcome::Win(Player::X), Player::X)?;
        assert!(agent.q_table_size() > 0);

        let saved = SavedTdAgent::from_q_learning(&agent, TrainingMetadata::default());
        let bytes = rmp_serde::to_vec(&saved)?;
        let loaded: SavedTdAgent = rmp_serde::from_slice(&bytes)?;
        let restored = loaded.to_agent()?;

        match restored {
            TdLearner::QLearning(restored_agent) => {
                assert_eq!(restored_agent.q_table_size(), agent.q_table_size());
            }
            TdLearner::Sarsa(_) => panic!("Expected Q-learning agent"),
        }

        Ok(())
    }

    #[test]
    fn test_sarsa_roundtrip() -> Result<()> {
        let mut agent = SarsaAgent::new(0.5, 0.99, 0.5, 0.995, 0.01, 0.0).with_seed(11);
        train_once(&mut agent, GameOutcome::Draw, Player::X)?;
        assert!(agent.q_table_size() > 0);

        let saved = SavedTdAgent::from_sarsa(&agent, TrainingMetadata::default());
        let bytes = rmp_serde::to_vec(&saved)?;
        let loaded: SavedTdAgent = rmp_serde::from_slice(&bytes)?;
        let restored = loaded.to_agent()?;

        match restored {
            TdLearner::Sarsa(restored_agent) => {
                assert_eq!(restored_agent.q_table_size(), agent.q_table_size());
            }
            TdLearner::QLearning(_) => panic!("Expected SARSA agent"),
        }

        Ok(())
    }
}
