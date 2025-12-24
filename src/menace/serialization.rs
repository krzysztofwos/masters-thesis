//! Agent serialization support
//!
//! Provides save/load functionality for trained MENACE agents.

use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use super::{
    active::{ActionOutcomeBeliefs, ActiveInference, PureActiveInference},
    agent::MenaceAgent,
    classic::ReinforcementValues,
    learning::ReinforcementBased,
};
use crate::{
    beliefs::Beliefs,
    tictactoe::Player,
    workspace::{InitialBeadSchedule, MenaceWorkspace, RestockMode, StateFilter},
};

/// Serializable representation of a trained MENACE agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SavedMenaceAgent {
    /// Version of the save format (for future compatibility)
    pub version: u32,
    /// The learned workspace containing all state-action weights
    pub workspace: MenaceWorkspace,
    /// State filtering strategy used during training
    pub state_filter: StateFilter,
    /// Restocking mode configuration
    pub restock_mode: RestockMode,
    /// Initial bead schedule
    pub initial_beads: InitialBeadSchedule,
    /// Learning algorithm type
    pub algorithm_type: AlgorithmType,
    /// Algorithm-specific parameters
    pub algorithm_params: AlgorithmParams,
    /// Training metadata
    pub metadata: TrainingMetadata,
}

/// Supported learning algorithm types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmType {
    /// Classic MENACE with reinforcement learning
    ClassicMenace,
    /// Hybrid Active Inference (EFE-based policy + Bayesian opponent beliefs)
    ActiveInference,
    /// Pure Active Inference using Bayesian beliefs for action outcomes
    PureActiveInference,
}

/// Algorithm-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AlgorithmParams {
    /// Parameters for Classic MENACE
    ClassicMenace {
        /// Reinforcement values for win/draw/loss
        reinforcement: ReinforcementValues,
    },
    /// Parameters for Hybrid Active Inference
    ActiveInference {
        /// Opponent model type
        opponent_kind: crate::active_inference::OpponentKind,
        /// Complete preference model
        preferences: crate::active_inference::PreferenceModel,
        /// Learned opponent beliefs
        beliefs: Beliefs,
        /// Player controlled by the agent
        agent_player: Player,
    },
    /// Parameters for Pure Active Inference
    PureActiveInference {
        /// Opponent model type
        opponent_kind: crate::active_inference::OpponentKind,
        /// Complete preference model
        preferences: crate::active_inference::PreferenceModel,
        /// Learned opponent beliefs
        opponent_beliefs: Beliefs,
        /// Learned action-outcome beliefs
        action_beliefs: ActionOutcomeBeliefs,
        /// Player controlled by the agent
        agent_player: Player,
    },
}

/// Metadata about the training process
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetadata {
    /// Number of games trained
    pub games_trained: Option<usize>,
    /// Opponent(s) trained against
    pub opponents: Vec<String>,
    /// Random seed used (if any)
    pub seed: Option<u64>,
    /// Timestamp when saved
    pub saved_at: Option<String>,
    /// Which token the agent was trained to control
    pub agent_player: Option<Player>,
    /// Which token opened games during training
    pub first_player: Option<Player>,
}

impl SavedMenaceAgent {
    /// Current save format version
    pub const VERSION: u32 = 1;

    /// Create from a trained agent
    pub fn from_agent(agent: &MenaceAgent, metadata: TrainingMetadata) -> Result<Self> {
        let (algorithm_type, algorithm_params) = match &agent.algorithm {
            super::learning::LearningStrategy::ClassicMenace(classic) => {
                let reinforcement = classic.reinforcement_values();
                (
                    AlgorithmType::ClassicMenace,
                    AlgorithmParams::ClassicMenace { reinforcement },
                )
            }
            super::learning::LearningStrategy::ActiveInference(aif) => {
                // Extract complete AIF configuration
                let opponent_kind = aif.opponent_kind();
                let preferences = aif.preference_model().clone();
                let beliefs = aif.beliefs().clone();

                (
                    AlgorithmType::ActiveInference,
                    AlgorithmParams::ActiveInference {
                        opponent_kind,
                        preferences,
                        beliefs,
                        agent_player: aif.agent_player(),
                    },
                )
            }
            super::learning::LearningStrategy::OracleActiveInference(_) => {
                // Oracle agents don't learn, so they don't need serialization.
                // They can be reconstructed from scratch with their configuration.
                return Err(anyhow!(
                    "Oracle Active Inference agents don't support serialization. \
                     They compute optimal policies from scratch and don't learn."
                ));
            }
            super::learning::LearningStrategy::PureActiveInference(pure) => {
                // Extract complete Pure AIF configuration including both belief types
                let opponent_kind = pure.opponent_kind();
                let preferences = pure.preference_model().clone();
                let opponent_beliefs = pure.opponent_beliefs().clone();
                let action_beliefs = pure.action_beliefs().clone();

                (
                    AlgorithmType::PureActiveInference,
                    AlgorithmParams::PureActiveInference {
                        opponent_kind,
                        preferences,
                        opponent_beliefs,
                        action_beliefs,
                        agent_player: pure.agent_player(),
                    },
                )
            }
        };

        Ok(Self {
            version: Self::VERSION,
            workspace: agent.workspace.clone(),
            state_filter: agent.state_filter,
            restock_mode: agent.restock_mode,
            initial_beads: agent.initial_beads,
            algorithm_type,
            algorithm_params,
            metadata,
        })
    }

    /// Reconstruct an agent from saved data
    ///
    /// All agent types and their learned state are fully restored:
    /// - **Classic MENACE**: Workspace weights and configuration
    /// - **Hybrid Active Inference**: Policy weights and opponent beliefs
    /// - **Pure Active Inference**: Action-outcome beliefs and opponent beliefs
    ///
    /// Oracle agents cannot be serialized as they compute optimal policies from scratch.
    pub fn to_agent(&self) -> Result<MenaceAgent> {
        self.to_agent_with_belief_reset(false)
    }

    /// Reconstruct an agent, optionally resetting Active Inference beliefs to their priors.
    ///
    /// # Arguments
    ///
    /// * `reset_beliefs` - If true, Hybrid Active Inference opponent beliefs are reset to priors.
    ///   Pure Active Inference agents are unaffected (beliefs always fully restored).
    pub fn to_agent_with_belief_reset(&self, reset_beliefs: bool) -> Result<MenaceAgent> {
        use super::learning::LearningStrategy;

        if self.version != Self::VERSION {
            return Err(anyhow!(
                "Unsupported save format version: {}. Expected {}",
                self.version,
                Self::VERSION
            ));
        }

        let algorithm = match (&self.algorithm_type, &self.algorithm_params) {
            (AlgorithmType::ClassicMenace, AlgorithmParams::ClassicMenace { reinforcement }) => {
                let classic = super::classic::ClassicMenace::with_reinforcement(*reinforcement);
                LearningStrategy::ClassicMenace(classic)
            }
            (
                AlgorithmType::ActiveInference,
                AlgorithmParams::ActiveInference {
                    opponent_kind,
                    preferences,
                    beliefs,
                    agent_player,
                },
            ) => {
                let aif = if reset_beliefs {
                    ActiveInference::with_custom_preferences_for_player(
                        *opponent_kind,
                        preferences.clone(),
                        *agent_player,
                    )
                } else {
                    ActiveInference::with_custom_preferences_and_beliefs_for_player(
                        *opponent_kind,
                        preferences.clone(),
                        beliefs.clone(),
                        *agent_player,
                    )
                };
                LearningStrategy::ActiveInference(Box::new(aif))
            }
            (
                AlgorithmType::PureActiveInference,
                AlgorithmParams::PureActiveInference {
                    opponent_kind,
                    preferences,
                    opponent_beliefs,
                    action_beliefs,
                    agent_player,
                },
            ) => {
                // Restore Pure AIF with full belief state
                let pure = PureActiveInference::with_custom_beliefs(
                    *opponent_kind,
                    preferences.clone(),
                    opponent_beliefs.clone(),
                    action_beliefs.clone(),
                    *agent_player,
                );
                LearningStrategy::PureActiveInference(Box::new(pure))
            }
            _ => {
                return Err(anyhow!("Mismatched algorithm type and parameters"));
            }
        };

        let agent_player = match &self.algorithm_params {
            AlgorithmParams::ActiveInference { agent_player, .. } => *agent_player,
            AlgorithmParams::PureActiveInference { agent_player, .. } => *agent_player,
            _ => self.metadata.agent_player.unwrap_or(Player::X),
        };

        Ok(MenaceAgent {
            workspace: self.workspace.clone(),
            state_filter: self.state_filter,
            rng: None, // Will be reseeded when needed
            restock_mode: self.restock_mode,
            initial_beads: self.initial_beads,
            algorithm,
            agent_player,
        })
    }

    /// Save agent to a file
    ///
    /// Uses MessagePack format for efficient serialization with support for complex types.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("Failed to create file: {}", path.as_ref().display()))?;
        let mut writer = BufWriter::new(file);

        rmp_serde::encode::write(&mut writer, self).context("Failed to serialize agent")?;

        Ok(())
    }

    /// Load agent from a file
    ///
    /// Reads MessagePack-serialized agent data.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open file: {}", path.as_ref().display()))?;
        let reader = BufReader::new(file);

        rmp_serde::decode::from_read(reader).context("Failed to deserialize agent")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_roundtrip() -> Result<()> {
        // Create a simple agent
        let agent = MenaceAgent::builder()
            .filter(StateFilter::Michie)
            .seed(42)
            .build()?;

        // Save to memory (MessagePack format)
        let metadata = TrainingMetadata {
            games_trained: Some(100),
            opponents: vec!["random".to_string()],
            seed: Some(42),
            saved_at: None,
            agent_player: Some(Player::X),
            first_player: Some(Player::X),
        };

        let saved = SavedMenaceAgent::from_agent(&agent, metadata)?;

        // Test roundtrip
        let bytes = rmp_serde::to_vec(&saved)?;
        let loaded: SavedMenaceAgent = rmp_serde::from_slice(&bytes)?;
        let restored_agent = loaded.to_agent()?;

        // Verify basic properties match
        assert_eq!(restored_agent.state_filter, agent.state_filter);
        assert_eq!(restored_agent.restock_mode, agent.restock_mode);

        Ok(())
    }

    #[test]
    fn test_active_inference_belief_roundtrip_and_reset() -> Result<()> {
        use crate::menace::learning::LearningStrategy;

        let mut agent = MenaceAgent::builder()
            .filter(StateFilter::Both)
            .seed(7)
            .active_inference_uniform(0.4)
            .build()?;

        // Inject custom beliefs so we can verify persistence across serialization.
        if let LearningStrategy::ActiveInference(ref mut boxed) = agent.algorithm {
            let opponent_kind = boxed.opponent_kind();
            let preferences = boxed.preference_model().clone();
            let mut beliefs = Beliefs::symmetric(preferences.opponent_dirichlet_alpha);
            beliefs.observe_opponent_action("........._O", 9, 0);

            *boxed = Box::new(ActiveInference::with_custom_preferences_and_beliefs(
                opponent_kind,
                preferences,
                beliefs,
            ));
        } else {
            panic!("expected Active Inference algorithm");
        }

        let metadata = TrainingMetadata::default();
        let saved = SavedMenaceAgent::from_agent(&agent, metadata)?;

        let restored = saved.to_agent()?;
        if let LearningStrategy::ActiveInference(ref boxed) = restored.algorithm {
            assert_eq!(boxed.beliefs().tracked_states(), 1);
        } else {
            panic!("restored agent should use Active Inference");
        }

        let reset = saved.to_agent_with_belief_reset(true)?;
        if let LearningStrategy::ActiveInference(ref boxed) = reset.algorithm {
            assert_eq!(boxed.beliefs().tracked_states(), 0);
        } else {
            panic!("reset agent should use Active Inference");
        }

        Ok(())
    }

    #[test]
    fn test_pure_active_inference_belief_roundtrip() -> Result<()> {
        use crate::{
            menace::learning::{LearningAlgorithm, LearningStrategy},
            tictactoe::{BoardState, GameOutcome},
        };

        // Create a Pure Active Inference agent
        let mut agent = MenaceAgent::builder()
            .filter(StateFilter::Both)
            .seed(42)
            .pure_active_inference_uniform(0.5)
            .build()?;

        // Train it for a few games to accumulate beliefs
        let initial_state = BoardState::new();
        let states = vec![initial_state, initial_state.make_move(0)?];
        let moves = vec![0, 4];

        for _ in 0..5 {
            if let LearningStrategy::PureActiveInference(ref mut pure) = agent.algorithm {
                pure.train_from_game(
                    &mut agent.workspace,
                    &states,
                    &moves,
                    GameOutcome::Draw,
                    Player::X,
                )
                .unwrap();
            }
        }

        // Verify the agent has learned something
        let tracked_pairs_before =
            if let LearningStrategy::PureActiveInference(ref pure) = agent.algorithm {
                let pairs = pure.action_beliefs().tracked_pairs();
                let opponent_version = pure.opponent_beliefs().version();
                assert!(pairs > 0, "Agent should have tracked some action pairs");
                assert!(
                    opponent_version > 0,
                    "Opponent beliefs should have been updated"
                );
                pairs
            } else {
                panic!("Expected Pure Active Inference agent");
            };

        // Serialize and deserialize
        let metadata = TrainingMetadata::default();
        let saved = SavedMenaceAgent::from_agent(&agent, metadata)?;
        let bytes = rmp_serde::to_vec(&saved)?;
        let loaded: SavedMenaceAgent = rmp_serde::from_slice(&bytes)?;
        let restored = loaded.to_agent()?;

        // Verify beliefs were preserved
        if let LearningStrategy::PureActiveInference(ref pure) = restored.algorithm {
            assert_eq!(
                pure.action_beliefs().tracked_pairs(),
                tracked_pairs_before,
                "Action-outcome beliefs should be preserved"
            );
            assert!(
                pure.opponent_beliefs().tracked_states() > 0
                    || pure.opponent_beliefs().version() > 0,
                "Opponent beliefs should be preserved"
            );
        } else {
            panic!("Restored agent should be Pure Active Inference");
        }

        Ok(())
    }
}
