//! Observer port - abstraction for training observation and data collection
//!
//! This port defines the interface for observing training events,
//! allowing composable data collection without coupling training
//! logic to specific output formats or metrics.

use crate::{
    Result,
    tictactoe::{BoardState, GameOutcome},
};

/// Observer trait for monitoring training
///
/// Observers can be composed to collect different types of data during training.
/// Examples include:
/// - Progress bars for user feedback
/// - JSONL export for analysis
/// - Metrics tracking for evaluation
/// - Active Inference baselines for comparison
///
/// # Design Philosophy
///
/// This trait represents a **port** in hexagonal architecture - a boundary
/// between the training pipeline and external observation mechanisms.
/// Different observation strategies are **adapters** that implement this port.
///
/// # Event Sequence
///
/// The observer methods are called in the following order:
/// 1. `on_training_start(total_games)` - Once at the beginning
/// 2. For each game:
///    - `on_game_start(game_num)`
///    - `on_move(...)` - For each move in the game
///    - `on_weights_updated(...)` - After learning updates (optional)
///    - `on_game_end(game_num, outcome)`
/// 3. `on_training_end()` - Once at the end
///
/// # Examples
///
/// ```no_run
/// use menace::{
///     ports::Observer,
///     tictactoe::GameOutcome,
/// };
///
/// struct CustomObserver {
///     game_count: usize,
/// }
///
/// impl Observer for CustomObserver {
///     fn on_game_end(
///         &mut self,
///         _game_num: usize,
///         _outcome: GameOutcome
///     ) -> menace::Result<()> {
///         self.game_count += 1;
///         Ok(())
///     }
/// }
/// ```
pub trait Observer: Send {
    /// Called when training starts.
    ///
    /// This is the first method called in the observation lifecycle.
    ///
    /// # Parameters
    ///
    /// * `total_games` - Total number of games that will be played
    ///
    /// # Default Implementation
    ///
    /// Does nothing. Override to initialize observation state.
    fn on_training_start(&mut self, _total_games: usize) -> Result<()> {
        Ok(())
    }

    /// Called when a game starts.
    ///
    /// # Parameters
    ///
    /// * `game_num` - Index of the game (0-based)
    ///
    /// # Default Implementation
    ///
    /// Does nothing. Override to reset per-game state.
    fn on_game_start(&mut self, _game_num: usize) -> Result<()> {
        Ok(())
    }

    /// Called for each move in a game.
    ///
    /// This method is invoked after a move is selected but before
    /// any learning updates are applied.
    ///
    /// # Parameters
    ///
    /// * `game_num` - Index of the current game
    /// * `step_num` - Step number within the game (0-based)
    /// * `state` - Board state before the move
    /// * `canonical_state` - Canonical form of the board state
    /// * `move_pos` - Position (0-8) where the move was made
    /// * `weights_before` - Move weights before any learning updates
    ///
    /// # Default Implementation
    ///
    /// Does nothing. Override to observe move selection.
    fn on_move(
        &mut self,
        _game_num: usize,
        _step_num: usize,
        _state: &BoardState,
        _canonical_state: &BoardState,
        _move_pos: usize,
        _weights_before: &[(usize, f64)],
    ) -> Result<()> {
        Ok(())
    }

    /// Called when a game ends.
    ///
    /// This method is invoked after the game reaches a terminal state.
    ///
    /// # Parameters
    ///
    /// * `game_num` - Index of the completed game
    /// * `outcome` - Final outcome (win/draw/loss)
    ///
    /// # Default Implementation
    ///
    /// Does nothing. Override to record game outcomes.
    fn on_game_end(&mut self, _game_num: usize, _outcome: GameOutcome) -> Result<()> {
        Ok(())
    }

    /// Called when training completes.
    ///
    /// This is the last method called in the observation lifecycle.
    /// Use this to finalize outputs, close files, or display summaries.
    ///
    /// # Default Implementation
    ///
    /// Does nothing. Override to perform cleanup or final reporting.
    fn on_training_end(&mut self) -> Result<()> {
        Ok(())
    }

    /// Called after weights are updated (for post-update observations).
    ///
    /// This method is optional and only called for adaptive learners
    /// that update their policy after games. It allows observers to
    /// track how weights change during learning.
    ///
    /// # Parameters
    ///
    /// * `game_num` - Index of the current game
    /// * `canonical_state` - Canonical form of the board state
    /// * `weights_after` - Move weights after learning updates
    ///
    /// # Default Implementation
    ///
    /// Does nothing. Override to observe weight changes.
    fn on_weights_updated(
        &mut self,
        _game_num: usize,
        _canonical_state: &BoardState,
        _weights_after: &[(usize, f64)],
    ) -> Result<()> {
        Ok(())
    }

    /// Called to provide workspace snapshot for Free Energy analysis.
    ///
    /// This method is optional and only called when the training pipeline
    /// works with workspace-based agents (like MENACE). It allows specialized
    /// observers to capture workspace state for analysis.
    ///
    /// # Parameters
    ///
    /// * `workspace` - Reference to the current workspace state
    ///
    /// # Default Implementation
    ///
    /// Does nothing. Override to capture workspace snapshots.
    ///
    /// # Note
    ///
    /// This method requires the `workspace` module to be imported.
    /// It uses dynamic dispatch to avoid tight coupling with MenaceWorkspace.
    #[allow(unused_variables)]
    fn on_workspace_snapshot(&mut self, workspace: &dyn std::any::Any) -> Result<()> {
        Ok(())
    }
}
