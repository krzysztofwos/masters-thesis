//! Configuration types for agent creation.

use crate::workspace::{InitialBeadSchedule, RestockMode, StateFilter};

/// Configuration for creating a MENACE agent.
///
/// This type provides a type-safe, builder-style API for configuring agents
/// before creation through the dependency injection container.
///
/// # Examples
///
/// ```
/// use menace::app::AgentConfig;
/// use menace::{StateFilter, RestockMode, InitialBeadSchedule};
///
/// let config = AgentConfig::new(StateFilter::Michie)
///     .with_seed(42)
///     .with_restock_mode(RestockMode::Box)
///     .with_initial_beads(InitialBeadSchedule::new([5.0, 4.0, 3.0, 2.0]));
/// ```
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// State filtering strategy for decision states
    pub filter: StateFilter,
    /// Restocking mode for depleted matchboxes
    pub restock_mode: RestockMode,
    /// Initial bead weights per ply
    pub initial_beads: InitialBeadSchedule,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl AgentConfig {
    /// Create a new agent configuration with the given state filter.
    ///
    /// Uses default values for other parameters:
    /// - Restock mode: `RestockMode::Box`
    /// - Initial beads: MENACE default (4, 3, 2, 1)
    /// - Seed: None (non-deterministic)
    pub fn new(filter: StateFilter) -> Self {
        Self {
            filter,
            restock_mode: RestockMode::default(),
            initial_beads: InitialBeadSchedule::default(),
            seed: None,
        }
    }

    /// Set the restock mode.
    pub fn with_restock_mode(mut self, mode: RestockMode) -> Self {
        self.restock_mode = mode;
        self
    }

    /// Set the initial bead schedule.
    pub fn with_initial_beads(mut self, beads: InitialBeadSchedule) -> Self {
        self.initial_beads = beads;
        self
    }

    /// Set the random seed for deterministic behavior.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self::new(StateFilter::default())
    }
}
