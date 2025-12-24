//! Builder pattern for MenaceAgent construction
//!
//! Provides a fluent API for configuring and creating MENACE agents,
//! following the builder pattern commonly used in Rust (e.g., std::thread::Builder).

use rand::{SeedableRng, rngs::StdRng};

use super::{
    active::{ActiveInference, OracleActiveInference, PureActiveInference},
    agent::MenaceAgent,
    classic::{ClassicMenace, ReinforcementValues},
    learning::LearningStrategy,
};
use crate::{
    tictactoe::Player,
    workspace::{InitialBeadSchedule, MenaceWorkspace, RestockMode, StateFilter},
};

/// Builder for constructing MenaceAgent instances with custom configuration.
///
/// # Examples
///
/// ```
/// use menace::menace::builder::MenaceAgentBuilder;
/// use menace::{StateFilter, RestockMode};
///
/// // Simple construction with defaults (Classic MENACE)
/// let agent = MenaceAgentBuilder::new().build();
///
/// // Custom configuration with Classic MENACE
/// let agent = MenaceAgentBuilder::new()
///     .seed(42)
///     .filter(StateFilter::Michie)
///     .restock_mode(RestockMode::Box)
///     .build();
///
/// // Active Inference with uniform opponent model
/// let agent = MenaceAgentBuilder::new()
///     .seed(42)
///     .active_inference_uniform(0.5)
///     .build();
///
/// // Active Inference with adversarial opponent model
/// let agent = MenaceAgentBuilder::new()
///     .active_inference_adversarial(0.7)
///     .build();
///
/// // Active Inference with minimax opponent model
/// let agent = MenaceAgentBuilder::new()
///     .active_inference_minimax()
///     .build();
/// ```
#[derive(Debug)]
pub struct MenaceAgentBuilder {
    seed: Option<u64>,
    filter: StateFilter,
    restock_mode: RestockMode,
    initial_beads: InitialBeadSchedule,
    reinforcement: ReinforcementValues,
    algorithm: Option<LearningStrategy>,
    agent_player: Player,
    workspace: Option<MenaceWorkspace>,
}

impl MenaceAgentBuilder {
    /// Create a new builder with default configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the random seed for deterministic behavior.
    ///
    /// # Arguments
    /// * `seed` - The seed value for the random number generator
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the state filter for determining which positions become matchboxes.
    ///
    /// # Arguments
    /// * `filter` - The filter strategy to use
    pub fn filter(mut self, filter: StateFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Set the restock mode for handling depleted matchboxes.
    ///
    /// # Arguments
    /// * `mode` - The restock strategy to use
    pub fn restock_mode(mut self, mode: RestockMode) -> Self {
        self.restock_mode = mode;
        self
    }

    /// Set the initial bead schedule for matchboxes.
    ///
    /// # Arguments
    /// * `schedule` - The initial bead counts by ply
    pub fn initial_beads(mut self, schedule: InitialBeadSchedule) -> Self {
        self.initial_beads = schedule;
        self
    }

    /// Set the reinforcement values for training.
    ///
    /// # Arguments
    /// * `values` - The reinforcement values for win/draw/loss
    pub fn reinforcement(mut self, values: ReinforcementValues) -> Self {
        self.reinforcement = values;
        self
    }

    /// Set win reinforcement value.
    ///
    /// # Arguments
    /// * `value` - Reinforcement for winning (typically positive)
    pub fn win_reinforcement(mut self, value: i16) -> Self {
        self.reinforcement.win = value;
        self
    }

    /// Set draw reinforcement value.
    ///
    /// # Arguments
    /// * `value` - Reinforcement for drawing (typically small positive)
    pub fn draw_reinforcement(mut self, value: i16) -> Self {
        self.reinforcement.draw = value;
        self
    }

    /// Set loss reinforcement value.
    ///
    /// # Arguments
    /// * `value` - Reinforcement for losing (typically negative)
    pub fn loss_reinforcement(mut self, value: i16) -> Self {
        self.reinforcement.loss = value;
        self
    }

    /// Set which player the agent controls (X or O).
    pub fn agent_player(mut self, player: Player) -> Self {
        self.agent_player = player;
        if let Some(strategy) = self.algorithm.as_mut() {
            strategy.set_agent_player(player);
        }
        self
    }

    /// Use an existing workspace instead of creating a new one.
    ///
    /// This is primarily used when loading a trained agent from persistent storage.
    /// The workspace's restock mode will be used, overriding any previously set mode.
    ///
    /// # Arguments
    /// * `workspace` - An existing workspace to use
    ///
    /// # Examples
    /// ```no_run
    /// use menace::menace::builder::MenaceAgentBuilder;
    /// use menace::{MenaceWorkspace, StateFilter, adapters::MsgPackRepository, ports::WorkspaceRepository};
    ///
    /// // Load a workspace from file using repository pattern (hexagonal architecture)
    /// let repo = MsgPackRepository::new();
    /// let workspace = repo.load(std::path::Path::new("agent.msgpack"))?;
    ///
    /// // Create agent with the loaded workspace
    /// let agent = MenaceAgentBuilder::new()
    ///     .with_workspace(workspace)
    ///     .build()?;
    /// # Ok::<(), menace::Error>(())
    /// ```
    pub fn with_workspace(mut self, workspace: MenaceWorkspace) -> Self {
        self.restock_mode = workspace.restock_mode();
        self.workspace = Some(workspace);
        self
    }

    /// Configure agent to use Active Inference with uniform opponent model.
    ///
    /// # Arguments
    /// * `epistemic_weight` - Weight for epistemic (information-seeking) value (typically 0.0-1.0)
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .active_inference_uniform(0.5)
    ///     .build();
    /// ```
    pub fn active_inference_uniform(mut self, epistemic_weight: f64) -> Self {
        self.algorithm = Some(LearningStrategy::ActiveInference(Box::new(
            ActiveInference::with_uniform_opponent_for_player(epistemic_weight, self.agent_player),
        )));
        self
    }

    /// Configure agent to use Active Inference with adversarial opponent model.
    ///
    /// # Arguments
    /// * `epistemic_weight` - Weight for epistemic (information-seeking) value (typically 0.0-1.0)
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .active_inference_adversarial(0.7)
    ///     .build();
    /// ```
    pub fn active_inference_adversarial(mut self, epistemic_weight: f64) -> Self {
        self.algorithm = Some(LearningStrategy::ActiveInference(Box::new(
            ActiveInference::with_adversarial_opponent_for_player(
                epistemic_weight,
                self.agent_player,
            ),
        )));
        self
    }

    /// Configure agent to use Active Inference with minimax opponent model.
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .active_inference_minimax()
    ///     .build();
    /// ```
    pub fn active_inference_minimax(mut self) -> Self {
        self.algorithm = Some(LearningStrategy::ActiveInference(Box::new(
            ActiveInference::with_minimax_opponent_for_player(self.agent_player),
        )));
        self
    }

    /// Configure agent to use Active Inference with custom preferences and parameters.
    ///
    /// This is the most flexible Active Inference configuration method, allowing full control
    /// over all EFE parameters, opponent models, and policy priors.
    ///
    /// # Arguments
    /// * `opponent_kind` - The opponent model to use
    /// * `preferences` - Fully configured PreferenceModel with all parameters set
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    /// use menace::active_inference::{PreferenceModel, OpponentKind, EFEMode, PolicyPrior};
    ///
    /// let prefs = PreferenceModel::from_probabilities(0.9, 0.5, 0.1)
    ///     .with_epistemic_weight(0.5)
    ///     .with_policy_lambda(2.0)
    ///     .with_opponent_dirichlet_alpha(0.5);
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .active_inference_custom(OpponentKind::Adversarial, prefs)
    ///     .build();
    /// ```
    pub fn active_inference_custom(
        mut self,
        opponent_kind: crate::active_inference::OpponentKind,
        preferences: crate::active_inference::PreferenceModel,
    ) -> Self {
        self.algorithm = Some(LearningStrategy::ActiveInference(Box::new(
            ActiveInference::with_custom_preferences_for_player(
                opponent_kind,
                preferences,
                self.agent_player,
            ),
        )));
        self
    }

    /// Configure agent to use Oracle Active Inference with uniform opponent model.
    ///
    /// Oracle Active Inference uses perfect game tree knowledge rather than learning.
    /// It serves as the theoretical ground truth for what optimal EFE minimization
    /// should produce in Tic-Tac-Toe.
    ///
    /// # Arguments
    /// * `epistemic_weight` - Weight for epistemic (information-seeking) value (typically 0.0-1.0)
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .oracle_active_inference_uniform(0.5)
    ///     .build();
    /// ```
    pub fn oracle_active_inference_uniform(mut self, epistemic_weight: f64) -> Self {
        self.algorithm = Some(LearningStrategy::OracleActiveInference(Box::new(
            OracleActiveInference::new_for_player(
                crate::active_inference::OpponentKind::Uniform,
                epistemic_weight,
                self.agent_player,
            ),
        )));
        self
    }

    /// Configure agent to use Oracle Active Inference with adversarial opponent model.
    ///
    /// # Arguments
    /// * `epistemic_weight` - Weight for epistemic (information-seeking) value (typically 0.0-1.0)
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .oracle_active_inference_adversarial(0.7)
    ///     .build();
    /// ```
    pub fn oracle_active_inference_adversarial(mut self, epistemic_weight: f64) -> Self {
        self.algorithm = Some(LearningStrategy::OracleActiveInference(Box::new(
            OracleActiveInference::new_for_player(
                crate::active_inference::OpponentKind::Adversarial,
                epistemic_weight,
                self.agent_player,
            ),
        )));
        self
    }

    /// Configure agent to use Oracle Active Inference with minimax opponent model.
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .oracle_active_inference_minimax()
    ///     .build();
    /// ```
    pub fn oracle_active_inference_minimax(mut self) -> Self {
        self.algorithm = Some(LearningStrategy::OracleActiveInference(Box::new(
            OracleActiveInference::new_for_player(
                crate::active_inference::OpponentKind::Minimax,
                0.0,
                self.agent_player,
            ),
        )));
        self
    }

    /// Configure agent to use Oracle Active Inference with custom preferences.
    ///
    /// # Arguments
    /// * `opponent_kind` - The opponent model to use
    /// * `preferences` - Fully configured PreferenceModel with all parameters set
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    /// use menace::active_inference::{PreferenceModel, OpponentKind};
    ///
    /// let prefs = PreferenceModel::from_probabilities(0.9, 0.5, 0.1)
    ///     .with_epistemic_weight(0.5)
    ///     .with_policy_lambda(2.0);
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .oracle_active_inference_custom(OpponentKind::Adversarial, prefs)
    ///     .build();
    /// ```
    pub fn oracle_active_inference_custom(
        mut self,
        opponent_kind: crate::active_inference::OpponentKind,
        preferences: crate::active_inference::PreferenceModel,
    ) -> Self {
        self.algorithm = Some(LearningStrategy::OracleActiveInference(Box::new(
            OracleActiveInference::with_custom_preferences(
                opponent_kind,
                preferences,
                self.agent_player,
            ),
        )));
        self
    }

    /// Configure agent to use Pure Active Inference with uniform opponent model.
    ///
    /// Pure Active Inference learns beliefs about action-outcome distributions and uses
    /// those learned beliefs to compute EFE-based policies. This is the theoretically
    /// pure Active Inference approach that should converge toward Oracle performance.
    ///
    /// # Arguments
    /// * `epistemic_weight` - Weight for epistemic (information-seeking) value (typically 0.0-1.0)
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .pure_active_inference_uniform(0.5)
    ///     .build();
    /// ```
    pub fn pure_active_inference_uniform(mut self, epistemic_weight: f64) -> Self {
        self.algorithm = Some(LearningStrategy::PureActiveInference(Box::new(
            PureActiveInference::new_for_player(
                crate::active_inference::OpponentKind::Uniform,
                epistemic_weight,
                self.agent_player,
            ),
        )));
        self
    }

    /// Configure agent to use Pure Active Inference with adversarial opponent model.
    ///
    /// # Arguments
    /// * `epistemic_weight` - Weight for epistemic (information-seeking) value (typically 0.0-1.0)
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .pure_active_inference_adversarial(0.7)
    ///     .build();
    /// ```
    pub fn pure_active_inference_adversarial(mut self, epistemic_weight: f64) -> Self {
        self.algorithm = Some(LearningStrategy::PureActiveInference(Box::new(
            PureActiveInference::new_for_player(
                crate::active_inference::OpponentKind::Adversarial,
                epistemic_weight,
                self.agent_player,
            ),
        )));
        self
    }

    /// Configure agent to use Pure Active Inference with minimax opponent model.
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .pure_active_inference_minimax()
    ///     .build();
    /// ```
    pub fn pure_active_inference_minimax(mut self) -> Self {
        self.algorithm = Some(LearningStrategy::PureActiveInference(Box::new(
            PureActiveInference::new_for_player(
                crate::active_inference::OpponentKind::Minimax,
                0.0,
                self.agent_player,
            ),
        )));
        self
    }

    /// Configure agent to use Pure Active Inference with custom preferences.
    ///
    /// # Arguments
    /// * `opponent_kind` - The opponent model to use
    /// * `preferences` - Fully configured PreferenceModel with all parameters set
    ///
    /// # Examples
    /// ```
    /// use menace::menace::builder::MenaceAgentBuilder;
    /// use menace::active_inference::{PreferenceModel, OpponentKind};
    ///
    /// let prefs = PreferenceModel::from_probabilities(0.9, 0.5, 0.1)
    ///     .with_epistemic_weight(0.5)
    ///     .with_policy_lambda(2.0);
    ///
    /// let agent = MenaceAgentBuilder::new()
    ///     .pure_active_inference_custom(OpponentKind::Adversarial, prefs)
    ///     .build();
    /// ```
    pub fn pure_active_inference_custom(
        mut self,
        opponent_kind: crate::active_inference::OpponentKind,
        preferences: crate::active_inference::PreferenceModel,
    ) -> Self {
        self.algorithm = Some(LearningStrategy::PureActiveInference(Box::new(
            PureActiveInference::with_custom_preferences(
                opponent_kind,
                preferences,
                self.agent_player,
            ),
        )));
        self
    }

    /// Build the MenaceAgent with the configured parameters.
    ///
    /// # Returns
    /// A configured MenaceAgent ready for training and playing.
    ///
    /// # Errors
    /// Returns an error if workspace construction fails.
    pub fn build(self) -> crate::Result<MenaceAgent> {
        if self.agent_player == Player::O && self.filter != StateFilter::Both {
            return Err(crate::Error::InvalidConfiguration {
                message: format!(
                    "StateFilter '{}' does not expose O-to-move matchboxes; use StateFilter::Both when agent_player is O",
                    self.filter
                ),
            });
        }

        // Use provided workspace or create a new one
        let workspace = match self.workspace {
            Some(ws) => ws,
            None => {
                MenaceWorkspace::with_config(self.filter, self.restock_mode, self.initial_beads)?
            }
        };

        let rng = self.seed.map(StdRng::seed_from_u64);

        // Use configured algorithm or default to Classic MENACE
        let algorithm = self.algorithm.unwrap_or_else(|| {
            LearningStrategy::ClassicMenace(ClassicMenace::with_reinforcement(self.reinforcement))
        });

        Ok(MenaceAgent {
            workspace,
            state_filter: self.filter,
            rng,
            restock_mode: self.restock_mode,
            initial_beads: self.initial_beads,
            algorithm,
            agent_player: self.agent_player,
        })
    }
}

impl Default for MenaceAgentBuilder {
    fn default() -> Self {
        Self {
            seed: None,
            filter: StateFilter::default(),
            restock_mode: RestockMode::default(),
            initial_beads: InitialBeadSchedule::default(),
            reinforcement: ReinforcementValues::default(),
            algorithm: None,
            agent_player: Player::X,
            workspace: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::menace::learning::LearningAlgorithm;

    #[test]
    fn test_builder_defaults() {
        let agent = MenaceAgentBuilder::new()
            .build()
            .expect("build should succeed");
        assert_eq!(agent.state_filter, StateFilter::default());
        assert_eq!(agent.restock_mode(), RestockMode::default());
    }

    #[test]
    fn test_builder_custom_config() {
        let agent = MenaceAgentBuilder::new()
            .seed(42)
            .filter(StateFilter::All)
            .restock_mode(RestockMode::Move)
            .win_reinforcement(5)
            .draw_reinforcement(2)
            .loss_reinforcement(-2)
            .build()
            .expect("build should succeed");

        assert_eq!(agent.state_filter, StateFilter::All);
        assert_eq!(agent.restock_mode(), RestockMode::Move);

        // Check reinforcement values through the accessor
        let reinforcement = agent
            .reinforcement_values()
            .expect("Should have reinforcement values");
        assert_eq!(reinforcement.win, 5);
        assert_eq!(reinforcement.draw, 2);
        assert_eq!(reinforcement.loss, -2);
        assert!(agent.rng.is_some());
    }

    #[test]
    fn test_builder_chainable() {
        // Test that all builder methods are chainable
        let _agent = MenaceAgentBuilder::new()
            .seed(1)
            .filter(StateFilter::Michie)
            .restock_mode(RestockMode::Box)
            .initial_beads(InitialBeadSchedule::default())
            .reinforcement(ReinforcementValues::default())
            .win_reinforcement(3)
            .draw_reinforcement(1)
            .loss_reinforcement(-1)
            .build()
            .expect("build should succeed");
    }

    #[test]
    fn test_builder_active_inference_uniform() {
        let agent = MenaceAgentBuilder::new()
            .seed(42)
            .active_inference_uniform(0.5)
            .build()
            .expect("build should succeed");

        assert!(agent.rng.is_some());
        // Verify it's using Active Inference algorithm
        assert_eq!(agent.algorithm.name(), "Active Inference (Uniform)");
    }

    #[test]
    fn test_builder_active_inference_adversarial() {
        let agent = MenaceAgentBuilder::new()
            .active_inference_adversarial(0.7)
            .build()
            .expect("build should succeed");

        assert_eq!(agent.algorithm.name(), "Active Inference (Adversarial)");
    }

    #[test]
    fn test_builder_active_inference_minimax() {
        let agent = MenaceAgentBuilder::new()
            .active_inference_minimax()
            .build()
            .expect("build should succeed");

        assert_eq!(agent.algorithm.name(), "Active Inference (Minimax)");
    }

    #[test]
    fn test_builder_chainable_with_active_inference() {
        // Test that Active Inference methods are chainable with other configurations
        let _agent = MenaceAgentBuilder::new()
            .seed(1)
            .filter(StateFilter::All)
            .restock_mode(RestockMode::Move)
            .active_inference_uniform(0.6)
            .build()
            .expect("build should succeed");
    }

    #[test]
    fn builder_disallows_o_player_without_both_filter() {
        let result = MenaceAgentBuilder::new().agent_player(Player::O).build();
        assert!(
            matches!(result, Err(crate::Error::InvalidConfiguration { .. })),
            "expected builder to reject incompatible filter/player configuration"
        );
    }
}
