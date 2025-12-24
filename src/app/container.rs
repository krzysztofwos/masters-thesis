//! Dependency injection container for MENACE application.
//!
//! This module provides centralized dependency management following hexagonal
//! architecture principles. The container owns infrastructure dependencies and
//! provides factory methods for creating domain objects.

use std::{path::Path, sync::Arc};

use super::config::AgentConfig;
use crate::{Result, adapters::MsgPackRepository, menace::MenaceAgent, ports::WorkspaceRepository};

/// Application with dependency injection.
///
/// Centralizes creation and wiring of dependencies following hexagonal architecture.
/// All infrastructure dependencies are owned by the app and injected into
/// domain objects and use cases.
///
/// # Examples
///
/// ## Production usage
///
/// ```
/// use menace::app::{App, AgentConfig};
/// use menace::StateFilter;
///
/// let app = App::new();
///
/// let config = AgentConfig::new(StateFilter::Michie).with_seed(42);
/// let agent = app.create_agent(config)?;
/// # Ok::<(), menace::Error>(())
/// ```
///
/// ## Testing with dependency injection
///
/// ```
/// use menace::app::App;
/// use menace::adapters::InMemoryRepository;
///
/// let app = App::for_testing()
///     .with_repository(InMemoryRepository::new())
///     .with_default_seed(42)
///     .build();
/// ```
pub struct App {
    /// Repository for workspace persistence
    workspace_repository: Arc<dyn WorkspaceRepository + Send + Sync>,
    /// Default random seed (None = non-deterministic)
    default_seed: Option<u64>,
}

impl App {
    /// Create a new app with production defaults.
    ///
    /// Uses:
    /// - `MsgPackRepository` for workspace persistence
    /// - No default seed (non-deterministic RNG)
    pub fn new() -> Self {
        Self {
            workspace_repository: Arc::new(MsgPackRepository::new()),
            default_seed: None,
        }
    }

    /// Create a builder for constructing app with custom dependencies.
    ///
    /// Primarily used for testing with mock/in-memory dependencies.
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::app::App;
    /// use menace::adapters::InMemoryRepository;
    ///
    /// let app = App::for_testing()
    ///     .with_repository(InMemoryRepository::new())
    ///     .with_default_seed(42)
    ///     .build();
    /// ```
    pub fn for_testing() -> AppBuilder {
        AppBuilder::new()
    }

    /// Get the workspace repository.
    ///
    /// Returns an Arc-wrapped repository that can be shared across threads.
    pub fn workspace_repository(&self) -> Arc<dyn WorkspaceRepository + Send + Sync> {
        Arc::clone(&self.workspace_repository)
    }

    /// Create a new MENACE agent with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Agent configuration including filter, restock mode, and seed
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::app::{App, AgentConfig};
    /// use menace::{StateFilter, RestockMode};
    ///
    /// let app = App::new();
    ///
    /// let config = AgentConfig::new(StateFilter::Michie)
    ///     .with_seed(42)
    ///     .with_restock_mode(RestockMode::Box);
    ///
    /// let agent = app.create_agent(config)?;
    /// # Ok::<(), menace::Error>(())
    /// ```
    pub fn create_agent(&self, config: AgentConfig) -> Result<MenaceAgent> {
        let mut builder = MenaceAgent::builder()
            .filter(config.filter)
            .restock_mode(config.restock_mode)
            .initial_beads(config.initial_beads);

        // Apply seed from config or use container default
        if let Some(seed) = config.seed.or(self.default_seed) {
            builder = builder.seed(seed);
        }

        builder.build()
    }

    /// Load an agent from persistent storage.
    ///
    /// Uses the configured repository to load the workspace. The loaded workspace
    /// is used as-is, preserving all its learned weights and restock mode
    /// configuration. The agent is created with default Classic MENACE learning
    /// strategy unless configured otherwise.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the saved workspace file
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use menace::app::App;
    /// use std::path::Path;
    ///
    /// let app = App::new();
    /// let agent = app.load_agent(Path::new("trained_agent.msgpack"))?;
    /// # Ok::<(), menace::Error>(())
    /// ```
    pub fn load_agent(&self, path: &Path) -> Result<MenaceAgent> {
        let workspace = self.workspace_repository.load(path)?;

        // Create agent using builder with the loaded workspace
        let mut builder = MenaceAgent::builder().with_workspace(workspace);

        // Apply default seed if configured
        if let Some(seed) = self.default_seed {
            builder = builder.seed(seed);
        }

        builder.build()
    }

    /// Save an agent to persistent storage.
    ///
    /// Uses the configured repository to persist the agent's workspace.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to save
    /// * `path` - Path where the workspace should be saved
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use menace::app::{App, AgentConfig};
    /// use menace::StateFilter;
    /// use std::path::Path;
    ///
    /// let app = App::new();
    /// let config = AgentConfig::new(StateFilter::Michie);
    /// let agent = app.create_agent(config)?;
    ///
    /// // Train the agent...
    ///
    /// app.save_agent(&agent, Path::new("trained_agent.msgpack"))?;
    /// # Ok::<(), menace::Error>(())
    /// ```
    pub fn save_agent(&self, agent: &MenaceAgent, path: &Path) -> Result<()> {
        self.workspace_repository.save(agent.workspace(), path)
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing app with custom dependencies.
///
/// Primarily used for testing to inject mock repositories and control randomness.
///
/// # Examples
///
/// ```
/// use menace::app::AppBuilder;
/// use menace::adapters::InMemoryRepository;
///
/// let app = AppBuilder::new()
///     .with_repository(InMemoryRepository::new())
///     .with_default_seed(42)
///     .build();
/// ```
pub struct AppBuilder {
    workspace_repository: Option<Arc<dyn WorkspaceRepository + Send + Sync>>,
    default_seed: Option<u64>,
}

impl AppBuilder {
    /// Create a new app builder.
    pub fn new() -> Self {
        Self {
            workspace_repository: None,
            default_seed: None,
        }
    }

    /// Set a custom workspace repository.
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::app::AppBuilder;
    /// use menace::adapters::InMemoryRepository;
    ///
    /// let builder = AppBuilder::new()
    ///     .with_repository(InMemoryRepository::new());
    /// ```
    pub fn with_repository<R: WorkspaceRepository + Send + Sync + 'static>(
        mut self,
        repo: R,
    ) -> Self {
        self.workspace_repository = Some(Arc::new(repo));
        self
    }

    /// Set a default random seed for all agents created by this container.
    ///
    /// Useful for creating deterministic tests.
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::app::AppBuilder;
    ///
    /// let builder = AppBuilder::new()
    ///     .with_default_seed(42);  // All agents will use seed 42
    /// ```
    pub fn with_default_seed(mut self, seed: u64) -> Self {
        self.default_seed = Some(seed);
        self
    }

    /// Build the app with the configured dependencies.
    ///
    /// If no repository was specified, uses `MsgPackRepository` by default.
    pub fn build(self) -> App {
        App {
            workspace_repository: self
                .workspace_repository
                .unwrap_or_else(|| Arc::new(MsgPackRepository::new())),
            default_seed: self.default_seed,
        }
    }
}

impl Default for AppBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateFilter;

    #[test]
    fn test_app_creates_agent() {
        let app = App::new();
        let config = AgentConfig::new(StateFilter::Michie);
        let agent = app.create_agent(config);
        assert!(agent.is_ok());
    }

    #[test]
    fn test_app_applies_seed() {
        let app = App::for_testing().with_default_seed(42).build();

        let config = AgentConfig::new(StateFilter::Michie);
        let agent = app.create_agent(config).unwrap();

        // Agent should have been created with seed 42
        // (We can't directly verify the seed, but we can verify creation succeeded)
        assert_eq!(
            agent.workspace().decision_labels().count(),
            287 // Michie filter produces 287 states
        );
    }

    #[test]
    fn test_config_seed_overrides_app_default() {
        let app = App::for_testing().with_default_seed(42).build();

        // Config seed should override app default
        let config = AgentConfig::new(StateFilter::Michie).with_seed(123);
        let agent = app.create_agent(config).unwrap();

        // Verify agent was created (seed applied internally)
        assert!(agent.workspace().decision_labels().count() > 0);
    }
}
