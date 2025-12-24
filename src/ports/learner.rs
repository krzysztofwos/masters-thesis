//! Learner port - abstraction for different learning approaches
//!
//! This port defines the interface that all learners must implement,
//! allowing the system to work with:
//! - Reinforcement learners (MENACE)
//! - Bayesian learners (Active Inference)
//! - Optimal policies (minimax)
//! - Baselines (random, defensive)

use crate::{
    Result,
    tictactoe::{BoardState, GameOutcome, Player},
};

/// Learner trait - Unified interface for all learning approaches
///
/// This abstraction enables comparison between different approaches:
/// - Reinforcement learners (MENACE)
/// - Bayesian learners (Active Inference)
/// - Optimal policies (minimax)
/// - Baselines (random, defensive)
///
/// # Design Philosophy
///
/// This trait represents a **port** in hexagonal architecture - a boundary
/// between the application core and external implementations. Different
/// learning strategies are **adapters** that implement this port.
///
/// # Examples
///
/// ```no_run
/// use menace::{
///     ports::Learner,
///     tictactoe::BoardState,
/// };
///
/// fn train_agent<L: Learner>(mut agent: L, opponent: L, games: usize) {
///     // Training logic that works with any Learner implementation
/// }
/// ```
pub trait Learner: Send {
    /// Select a move for the given board state.
    ///
    /// The learner should analyze the current state and return the
    /// position (0-8) where it wants to place its piece.
    ///
    /// # Errors
    ///
    /// Returns an error if no valid moves are available (terminal state).
    fn select_move(&mut self, state: &BoardState) -> Result<usize>;

    /// Update the learner after a game completes.
    ///
    /// This method is called at the end of each game, allowing adaptive
    /// learners (like MENACE) to update their policy based on the outcome.
    ///
    /// # Parameters
    ///
    /// * `first_player` - Which player moved first in the game
    /// * `moves` - Sequence of move positions (0-8) made during the game
    /// * `outcome` - Final outcome of the game
    /// * `role` - Which player this learner was playing as
    ///
    /// # Default Implementation
    ///
    /// The default implementation does nothing, suitable for non-adaptive
    /// learners like optimal or random policies.
    fn learn(
        &mut self,
        _first_player: Player,
        _moves: &[usize],
        _outcome: GameOutcome,
        _role: Player,
    ) -> Result<()> {
        // Default: no learning (for non-adaptive learners like optimal/random)
        Ok(())
    }

    /// Get the learner's name.
    ///
    /// Used for identification in comparisons and logging.
    fn name(&self) -> &str;

    /// Reset learner state to initial conditions.
    ///
    /// This method is called when resetting experiments or starting
    /// fresh training runs. Adaptive learners should clear their
    /// learned policy, while stateless learners can use the default
    /// no-op implementation.
    ///
    /// # Default Implementation
    ///
    /// The default implementation does nothing, suitable for stateless
    /// learners.
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }

    /// Enable downcasting to concrete types.
    ///
    /// This method allows accessing concrete learner implementations
    /// when needed (e.g., for serialization or introspection).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use menace::ports::Learner;
    /// use menace::pipeline::MenaceLearner;
    ///
    /// fn save_if_menace(learner: &dyn Learner) {
    ///     if let Some(menace) = learner.as_any().downcast_ref::<MenaceLearner>() {
    ///         // Save MENACE-specific state
    ///     }
    /// }
    /// ```
    fn as_any(&self) -> &dyn std::any::Any;

    /// Get move weights for a specific state, if available.
    ///
    /// Returns the probability distribution over moves for the given state.
    /// This is useful for:
    /// - Observing learned policies
    /// - Debugging training progress
    /// - Analyzing decision-making
    ///
    /// # Returns
    ///
    /// * `Some(Vec<(position, weight)>)` - For learners with explicit policies
    /// * `None` - For learners without explicit move weights (e.g., minimax)
    ///
    /// # Default Implementation
    ///
    /// Returns `None`, indicating no explicit move weights available.
    fn move_weights(&self, _state: &BoardState) -> Option<Vec<(usize, f64)>> {
        None
    }

    /// Seed the learner's internal random number generator.
    ///
    /// Training pipelines call this method when supplied with a deterministic
    /// seed to ensure reproducible results. Stateless learners can ignore it.
    ///
    /// # Default Implementation
    ///
    /// Does nothing and returns `Ok(())`.
    fn set_rng_seed(&mut self, _seed: u64) -> Result<()> {
        Ok(())
    }

    /// Provide a snapshot of the learner's internal workspace, if available.
    ///
    /// Observers that compute analytics (e.g., Free Energy) can request a
    /// workspace snapshot via this hook. Learners that do not expose an
    /// inspectable workspace can return `None`.
    ///
    /// # Returns
    ///
    /// * `Some(&dyn Any)` – Reference to the workspace value
    /// * `None` – No workspace available
    fn workspace_snapshot(&self) -> Option<&dyn std::any::Any> {
        None
    }
}
