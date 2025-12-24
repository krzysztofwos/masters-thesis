//! Observer pattern for training pipelines
//!
//! Observers allow composable data collection during training without coupling
//! training logic to specific output formats.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

use crate::{
    Result,
    analysis::free_energy::{FreeEnergyAnalysis, FreeEnergyComponents},
    ports::Observer,
    tictactoe::{BoardState, GameOutcome, Player},
    workspace::MenaceWorkspace,
};

/// Observation of a single step during a game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepObservation {
    /// Game number
    pub game_num: usize,
    /// Step number within game
    pub step_num: usize,
    /// Board state
    pub state: String,
    /// Canonical board state
    pub canonical_state: String,
    /// Move selected
    pub move_position: usize,
    /// Move weights before selection
    pub weights_before: Vec<(usize, f64)>,
    /// Move weights after update (if applicable)
    pub weights_after: Option<Vec<(usize, f64)>>,
}

/// Complete observation of a training game
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Game number
    pub game_num: usize,
    /// Final outcome
    pub outcome: String,
    /// Steps in the game
    pub steps: Vec<StepObservation>,
    /// Total moves in game
    pub total_moves: usize,
}

/// Progress bar observer - Shows training progress
pub struct ProgressObserver {
    progress_bar: Option<ProgressBar>,
    wins: usize,
    draws: usize,
    losses: usize,
}

impl ProgressObserver {
    /// Create a new progress observer
    pub fn new() -> Self {
        Self {
            progress_bar: None,
            wins: 0,
            draws: 0,
            losses: 0,
        }
    }
}

impl Default for ProgressObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer for ProgressObserver {
    fn on_training_start(&mut self, total_games: usize) -> Result<()> {
        let pb = ProgressBar::new(total_games as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} games (W:{msg})")
                .map_err(|e| crate::Error::ProgressBarTemplate {
                    message: e.to_string(),
                })?
                .progress_chars("=>-"),
        );
        self.progress_bar = Some(pb);
        Ok(())
    }

    fn on_game_end(&mut self, game_num: usize, outcome: GameOutcome) -> Result<()> {
        match outcome {
            GameOutcome::Win(Player::X) => self.wins += 1,
            GameOutcome::Win(Player::O) => self.losses += 1, // Loss for X
            GameOutcome::Draw => self.draws += 1,
        }

        if let Some(pb) = &self.progress_bar {
            pb.set_position(game_num as u64);
            pb.set_message(format!("{} D:{} L:{}", self.wins, self.draws, self.losses));
        }
        Ok(())
    }

    fn on_training_end(&mut self) -> Result<()> {
        if let Some(pb) = &self.progress_bar {
            pb.finish_with_message(format!("{} D:{} L:{}", self.wins, self.draws, self.losses));
        }
        Ok(())
    }
}

/// Metrics observer - Tracks training metrics
pub struct MetricsObserver {
    wins: usize,
    draws: usize,
    losses: usize,
    total_games: usize,
    move_counts: Vec<usize>,
}

impl MetricsObserver {
    /// Create a new metrics observer
    pub fn new() -> Self {
        Self {
            wins: 0,
            draws: 0,
            losses: 0,
            total_games: 0,
            move_counts: Vec::new(),
        }
    }

    /// Get current win rate
    pub fn win_rate(&self) -> f64 {
        if self.total_games == 0 {
            0.0
        } else {
            self.wins as f64 / self.total_games as f64
        }
    }

    /// Get current draw rate
    pub fn draw_rate(&self) -> f64 {
        if self.total_games == 0 {
            0.0
        } else {
            self.draws as f64 / self.total_games as f64
        }
    }

    /// Get current loss rate
    pub fn loss_rate(&self) -> f64 {
        if self.total_games == 0 {
            0.0
        } else {
            self.losses as f64 / self.total_games as f64
        }
    }

    /// Get average game length
    pub fn avg_game_length(&self) -> f64 {
        if self.move_counts.is_empty() {
            0.0
        } else {
            self.move_counts.iter().sum::<usize>() as f64 / self.move_counts.len() as f64
        }
    }

    /// Get metrics summary
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            total_games: self.total_games,
            wins: self.wins,
            draws: self.draws,
            losses: self.losses,
            win_rate: self.win_rate(),
            draw_rate: self.draw_rate(),
            loss_rate: self.loss_rate(),
            avg_game_length: self.avg_game_length(),
        }
    }
}

/// Summary of training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_games: usize,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub win_rate: f64,
    pub draw_rate: f64,
    pub loss_rate: f64,
    pub avg_game_length: f64,
}

impl Default for MetricsObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl Observer for MetricsObserver {
    fn on_game_start(&mut self, _game_num: usize) -> Result<()> {
        self.move_counts.push(0);
        Ok(())
    }

    fn on_move(
        &mut self,
        _game_num: usize,
        _step_num: usize,
        _state: &BoardState,
        _canonical_state: &BoardState,
        _move_pos: usize,
        _weights_before: &[(usize, f64)],
    ) -> Result<()> {
        if let Some(last) = self.move_counts.last_mut() {
            *last += 1;
        }
        Ok(())
    }

    fn on_game_end(&mut self, _game_num: usize, outcome: GameOutcome) -> Result<()> {
        self.total_games += 1;
        match outcome {
            GameOutcome::Win(Player::X) => self.wins += 1,
            GameOutcome::Win(Player::O) => self.losses += 1, // Loss for X
            GameOutcome::Draw => self.draws += 1,
        }
        Ok(())
    }
}

/// JSONL observer - Exports observations to JSON Lines format
pub struct JsonlObserver {
    writer: BufWriter<File>,
    current_game_steps: Vec<StepObservation>,
    current_game_num: usize,
}

impl JsonlObserver {
    /// Create a new JSONL observer
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            writer,
            current_game_steps: Vec::new(),
            current_game_num: 0,
        })
    }
}

impl Observer for JsonlObserver {
    fn on_game_start(&mut self, game_num: usize) -> Result<()> {
        self.current_game_num = game_num;
        self.current_game_steps.clear();
        Ok(())
    }

    fn on_move(
        &mut self,
        game_num: usize,
        step_num: usize,
        state: &BoardState,
        canonical_state: &BoardState,
        move_pos: usize,
        weights_before: &[(usize, f64)],
    ) -> Result<()> {
        self.current_game_steps.push(StepObservation {
            game_num,
            step_num,
            state: state.encode(),
            canonical_state: canonical_state.encode(),
            move_position: move_pos,
            weights_before: weights_before.to_vec(),
            weights_after: None,
        });
        Ok(())
    }

    fn on_weights_updated(
        &mut self,
        _game_num: usize,
        canonical_state: &BoardState,
        weights_after: &[(usize, f64)],
    ) -> Result<()> {
        // Update the last step's weights_after
        let canonical_str = canonical_state.encode();
        if let Some(step) = self
            .current_game_steps
            .iter_mut()
            .rfind(|s| s.canonical_state == canonical_str)
        {
            step.weights_after = Some(weights_after.to_vec());
        }
        Ok(())
    }

    fn on_game_end(&mut self, game_num: usize, outcome: GameOutcome) -> Result<()> {
        let observation = Observation {
            game_num,
            outcome: format!("{outcome:?}"),
            total_moves: self.current_game_steps.len(),
            steps: self.current_game_steps.clone(),
        };

        // Write as JSONL (one JSON object per line)
        serde_json::to_writer(&mut self.writer, &observation)?;
        writeln!(&mut self.writer)?;
        self.writer.flush()?;

        Ok(())
    }
}

/// Milestone observer - Tracks key learning achievements
///
/// This observer tracks important milestones during training against optimal opponents,
/// such as the first draw achieved and the last loss encountered. These milestones
/// are useful for understanding learning convergence.
pub struct MilestoneObserver {
    /// Opponent type being tracked (for milestone context)
    opponent_name: String,
    /// First game where a draw was achieved (global index)
    first_draw: Option<usize>,
    /// Last game where a loss occurred (global index)
    last_loss: Option<usize>,
    /// Current game being played
    current_game: usize,
    /// Track games played against this specific opponent
    games_vs_opponent: usize,
}

impl MilestoneObserver {
    /// Create a new milestone observer for tracking games against a specific opponent
    pub fn new(opponent_name: String) -> Self {
        Self {
            opponent_name,
            first_draw: None,
            last_loss: None,
            current_game: 0,
            games_vs_opponent: 0,
        }
    }

    /// Get the first draw milestone (if achieved)
    pub fn first_draw(&self) -> Option<usize> {
        self.first_draw
    }

    /// Get the last loss milestone (if any losses occurred)
    pub fn last_loss(&self) -> Option<usize> {
        self.last_loss
    }

    /// Get the number of games played against this opponent
    pub fn games_vs_opponent(&self) -> usize {
        self.games_vs_opponent
    }

    /// Get the opponent name
    pub fn opponent_name(&self) -> &str {
        &self.opponent_name
    }

    /// Display milestone summary
    pub fn display_summary(&self) {
        println!("\n=== Learning Milestones vs {} ===", self.opponent_name);

        match self.first_draw {
            Some(game) => {
                println!(
                    "  First draw: Game #{} ({} games vs {})",
                    game + 1,
                    self.games_vs_opponent,
                    self.opponent_name
                );
            }
            None => println!("  First draw: Not achieved"),
        }

        match self.last_loss {
            Some(game) => {
                println!("  Last loss: Game #{}", game + 1);
                if let Some(first_draw) = self.first_draw {
                    if game < first_draw {
                        println!("  ✓ No losses after achieving first draw!");
                    } else {
                        let games_since = self.current_game - game;
                        println!("  Games since last loss: {games_since}");
                    }
                }
            }
            None => println!("  Last loss: No losses recorded"),
        }

        if self.first_draw.is_some() && self.last_loss.is_some() {
            let first_draw = self.first_draw.unwrap();
            let last_loss = self.last_loss.unwrap();
            if last_loss < first_draw {
                println!("  ✓ Achieved competent play (consistent draws, no losses)");
            }
        }
    }
}

impl Observer for MilestoneObserver {
    fn on_game_end(&mut self, game_num: usize, outcome: GameOutcome) -> Result<()> {
        self.current_game = game_num;
        self.games_vs_opponent += 1;

        match outcome {
            GameOutcome::Draw => {
                if self.first_draw.is_none() {
                    self.first_draw = Some(game_num);
                }
            }
            GameOutcome::Win(Player::O) => {
                // Agent lost (playing as X)
                self.last_loss = Some(game_num);
            }
            _ => {}
        }

        Ok(())
    }
}

/// Free Energy observer - Tracks Free Energy minimization during training
///
/// This observer computes Free Energy at regular intervals during training
/// to validate the hypothesis that MENACE's reinforcement learning minimizes
/// Free Energy according to the Free Energy Principle.
///
/// # Usage
///
/// This observer requires workspace snapshots to compute Free Energy. The
/// training pipeline must call `on_workspace_snapshot()` to provide workspace
/// state. The observer captures the initial workspace on first snapshot and
/// computes Free Energy relative to that baseline.
pub struct FreeEnergyObserver {
    /// Free Energy analysis engine
    analysis: FreeEnergyAnalysis,
    /// Initial workspace (for computing KL divergence)
    initial_workspace: Option<MenaceWorkspace>,
    /// Checkpoint interval (compute FE every N games)
    checkpoint_interval: usize,
    /// Checkpoints: (game_num, FE components)
    checkpoints: Vec<(usize, FreeEnergyComponents)>,
    /// Current workspace snapshot (updated via on_workspace_snapshot)
    current_workspace: Option<MenaceWorkspace>,
    /// Total games played so far (1-based count)
    current_game: usize,
}

impl FreeEnergyObserver {
    /// Create a new Free Energy observer
    ///
    /// # Arguments
    /// * `analysis` - Free Energy analysis engine
    /// * `checkpoint_interval` - Compute FE every N games (e.g., 100)
    pub fn new(analysis: FreeEnergyAnalysis, checkpoint_interval: usize) -> Self {
        assert!(
            checkpoint_interval > 0,
            "checkpoint interval must be a positive integer"
        );
        Self {
            analysis,
            initial_workspace: None,
            checkpoint_interval,
            checkpoints: Vec::new(),
            current_workspace: None,
            current_game: 0,
        }
    }

    /// Get collected checkpoints
    pub fn checkpoints(&self) -> &[(usize, FreeEnergyComponents)] {
        &self.checkpoints
    }

    /// Export checkpoints to JSON file
    pub fn export_checkpoints<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.checkpoints)?;
        Ok(())
    }

    /// Compute Free Energy at current checkpoint
    fn compute_checkpoint(&mut self) -> Result<()> {
        if let (Some(current), Some(initial)) = (&self.current_workspace, &self.initial_workspace) {
            let fe = self.analysis.compute_free_energy(current, initial)?;
            self.checkpoints.push((self.current_game, fe));
        }
        Ok(())
    }
}

impl Observer for FreeEnergyObserver {
    fn on_training_start(&mut self, _total_games: usize) -> Result<()> {
        // Workspace will be captured on first snapshot
        self.current_game = 0;
        Ok(())
    }

    fn on_game_end(&mut self, game_num: usize, _outcome: GameOutcome) -> Result<()> {
        let games_played = game_num + 1;
        self.current_game = games_played;

        // Check if this is a checkpoint
        if games_played.is_multiple_of(self.checkpoint_interval) {
            self.compute_checkpoint()?;
        }
        Ok(())
    }

    fn on_training_end(&mut self) -> Result<()> {
        // Final checkpoint
        if self.current_game > 0 {
            if let Some(&(last_game, _)) = self.checkpoints.last() {
                if last_game != self.current_game {
                    self.compute_checkpoint()?;
                }
            } else {
                // No checkpoints yet, add final one
                self.compute_checkpoint()?;
            }
        }
        Ok(())
    }

    fn on_workspace_snapshot(&mut self, workspace: &dyn std::any::Any) -> Result<()> {
        // Try to downcast to MenaceWorkspace
        if let Some(ws) = workspace.downcast_ref::<MenaceWorkspace>() {
            // Capture initial workspace on first snapshot
            if self.initial_workspace.is_none() {
                self.initial_workspace = Some(ws.clone());
            }

            // Update current workspace
            self.current_workspace = Some(ws.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_observer() {
        let mut observer = MetricsObserver::new();

        assert_eq!(observer.win_rate(), 0.0);

        // Simulate 3 games
        observer
            .on_game_end(1, GameOutcome::Win(crate::tictactoe::Player::X))
            .unwrap();
        observer.on_game_end(2, GameOutcome::Draw).unwrap();
        observer
            .on_game_end(3, GameOutcome::Win(crate::tictactoe::Player::X))
            .unwrap();

        assert_eq!(observer.total_games, 3);
        assert_eq!(observer.wins, 2);
        assert_eq!(observer.draws, 1);
        assert_eq!(observer.losses, 0);
        assert!((observer.win_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_milestone_observer() {
        let mut observer = MilestoneObserver::new("Optimal".to_string());

        // Game 0: Loss
        observer
            .on_game_end(0, GameOutcome::Win(Player::O))
            .unwrap();
        assert_eq!(observer.last_loss(), Some(0));
        assert_eq!(observer.first_draw(), None);

        // Game 1: Loss
        observer
            .on_game_end(1, GameOutcome::Win(Player::O))
            .unwrap();
        assert_eq!(observer.last_loss(), Some(1));

        // Game 2: First draw!
        observer.on_game_end(2, GameOutcome::Draw).unwrap();
        assert_eq!(observer.first_draw(), Some(2));
        assert_eq!(observer.last_loss(), Some(1));

        // Game 3: Another draw
        observer.on_game_end(3, GameOutcome::Draw).unwrap();
        assert_eq!(observer.first_draw(), Some(2)); // Still the first
        assert_eq!(observer.last_loss(), Some(1)); // No new loss

        assert_eq!(observer.games_vs_opponent(), 4);
    }
}
