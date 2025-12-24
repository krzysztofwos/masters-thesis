//! Training pipeline for learnable agents

use serde::{Deserialize, Serialize};

use super::regimen::{OpponentType, TrainingBlock};
use crate::{
    Error, Result,
    ports::{Learner, Observer},
    tictactoe::{BoardState, GameOutcome, Player},
};

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training games
    pub num_games: usize,

    /// Random seed
    pub seed: Option<u64>,

    /// Whether the agent plays as X or O
    pub agent_player: Player,

    /// Which player opens the game (determines move sequence parity)
    pub first_player: Player,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_games: 500,
            seed: None,
            agent_player: Player::X,
            first_player: Player::X,
        }
    }
}

/// Result of a training run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Total games played
    pub total_games: usize,

    /// Number of wins
    pub wins: usize,

    /// Number of draws
    pub draws: usize,

    /// Number of losses
    pub losses: usize,

    /// Win rate
    pub win_rate: f64,

    /// Draw rate
    pub draw_rate: f64,

    /// Loss rate
    pub loss_rate: f64,
}

impl TrainingResult {
    /// Create a new training result
    pub fn new(total_games: usize, wins: usize, draws: usize, losses: usize) -> Self {
        let win_rate = if total_games > 0 {
            wins as f64 / total_games as f64
        } else {
            0.0
        };
        let draw_rate = if total_games > 0 {
            draws as f64 / total_games as f64
        } else {
            0.0
        };
        let loss_rate = if total_games > 0 {
            losses as f64 / total_games as f64
        } else {
            0.0
        };

        Self {
            total_games,
            wins,
            draws,
            losses,
            win_rate,
            draw_rate,
            loss_rate,
        }
    }

    /// Save result to JSON file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Load result from JSON file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let result = serde_json::from_reader(file)?;
        Ok(result)
    }
}

/// Training pipeline for a single learner against an opponent
pub struct TrainingPipeline {
    config: TrainingConfig,
    observers: Vec<Box<dyn Observer>>,
}

impl TrainingPipeline {
    /// Create a new training pipeline
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            observers: Vec::new(),
        }
    }

    /// Add an observer to the pipeline
    pub fn with_observer(mut self, observer: Box<dyn Observer>) -> Self {
        self.observers.push(observer);
        self
    }

    /// Run training curriculum with multiple sequential blocks
    ///
    /// This method enables curriculum learning by training against different
    /// opponents in sequence. The agent accumulates learning across all blocks.
    pub fn run_curriculum(
        &mut self,
        agent: &mut dyn Learner,
        schedule: &[TrainingBlock],
    ) -> Result<TrainingResult> {
        use super::comparison::{DefensiveLearner, OptimalLearner, RandomLearner};

        self.seed_learner(agent, 0)?;

        let mut total_wins = 0;
        let mut total_draws = 0;
        let mut total_losses = 0;
        let mut total_games = 0;
        let mut depleted_label: Option<String> = None;

        // Notify observers of total training games
        let all_games: usize = schedule.iter().map(|b| b.games).sum();
        for observer in &mut self.observers {
            observer.on_training_start(all_games)?;
        }

        // Run each training block
        for (block_idx, block) in schedule.iter().enumerate() {
            if block.games == 0 {
                continue;
            }

            // Create opponent for this block
            let mut opponent: Box<dyn Learner> = match block.opponent {
                OpponentType::Random => {
                    Box::new(RandomLearner::new(format!("Random-Block{}", block_idx + 1)))
                }
                OpponentType::Optimal => Box::new(OptimalLearner::new(format!(
                    "Optimal-Block{}",
                    block_idx + 1
                ))),
                OpponentType::Defensive => Box::new(DefensiveLearner::new(format!(
                    "Defensive-Block{}",
                    block_idx + 1
                ))),
            };

            self.seed_learner(opponent.as_mut(), (block_idx as u64) + 1)?;

            // Play games in this block
            for _ in 0..block.games {
                let global_game_num = total_games;
                let outcome = match self.play_game(global_game_num, agent, opponent.as_mut()) {
                    Ok(outcome) => outcome,
                    Err(Error::DepletedMatchbox { label }) => {
                        depleted_label = Some(label);
                        break;
                    }
                    Err(err) => return Err(err),
                };

                // Count from agent's perspective
                match outcome {
                    GameOutcome::Win(winner) if winner == self.config.agent_player => {
                        total_wins += 1
                    }
                    GameOutcome::Win(_) => total_losses += 1,
                    GameOutcome::Draw => total_draws += 1,
                }

                // Notify observers of game end
                for observer in &mut self.observers {
                    observer.on_game_end(global_game_num, outcome)?;
                }

                total_games += 1;
            }
            if depleted_label.is_some() {
                break;
            }
        }

        // Notify observers of training end
        for observer in &mut self.observers {
            observer.on_training_end()?;
        }

        if let Some(label) = depleted_label {
            eprintln!(
                "Warning: training stopped early because matchbox '{label}' depleted under restock=none."
            );
        }

        Ok(TrainingResult::new(
            total_games,
            total_wins,
            total_draws,
            total_losses,
        ))
    }

    /// Run training with the given agent and opponent
    pub fn run(
        &mut self,
        agent: &mut dyn Learner,
        opponent: &mut dyn Learner,
    ) -> Result<TrainingResult> {
        self.seed_pair(agent, opponent)?;

        let mut wins = 0;
        let mut draws = 0;
        let mut losses = 0;
        let mut total_games = 0;
        let mut depleted_label: Option<String> = None;

        // Notify observers of training start
        for observer in &mut self.observers {
            observer.on_training_start(self.config.num_games)?;
        }

        // Play games
        for game_num in 0..self.config.num_games {
            let outcome = match self.play_game(game_num, agent, opponent) {
                Ok(outcome) => outcome,
                Err(Error::DepletedMatchbox { label }) => {
                    depleted_label = Some(label);
                    break;
                }
                Err(err) => return Err(err),
            };

            // Count from agent's perspective
            match outcome {
                GameOutcome::Win(winner) if winner == self.config.agent_player => wins += 1,
                GameOutcome::Win(_) => losses += 1, // Other player won, so agent lost
                GameOutcome::Draw => draws += 1,
            }

            // Notify observers of game end
            for observer in &mut self.observers {
                observer.on_game_end(game_num, outcome)?;
            }

            total_games += 1;
        }

        // Notify observers of training end
        for observer in &mut self.observers {
            observer.on_training_end()?;
        }

        if let Some(label) = depleted_label {
            eprintln!(
                "Warning: training stopped early because matchbox '{label}' depleted under restock=none."
            );
        }

        Ok(TrainingResult::new(total_games, wins, draws, losses))
    }

    fn seed_pair(&self, agent: &mut dyn Learner, opponent: &mut dyn Learner) -> Result<()> {
        if let Some(seed) = self.config.seed {
            agent.set_rng_seed(seed)?;
            opponent.set_rng_seed(seed.wrapping_add(1))?;
        }
        Ok(())
    }

    fn seed_learner(&self, learner: &mut dyn Learner, offset: u64) -> Result<()> {
        if let Some(seed) = self.config.seed {
            learner.set_rng_seed(seed.wrapping_add(offset))?;
        }
        Ok(())
    }

    fn notify_weights_updated(
        &mut self,
        game_num: usize,
        states: &[BoardState],
        learner: &dyn Learner,
    ) -> Result<()> {
        for state in states {
            if let Some(weights) = learner.move_weights(state) {
                let canonical_state = state.canonical();
                for observer in &mut self.observers {
                    observer.on_weights_updated(game_num, &canonical_state, &weights)?;
                }
            }
        }
        Ok(())
    }

    fn notify_workspace_snapshot(&mut self, learner: &dyn Learner) -> Result<()> {
        if let Some(snapshot) = learner.workspace_snapshot() {
            for observer in &mut self.observers {
                observer.on_workspace_snapshot(snapshot)?;
            }
        }
        Ok(())
    }

    fn play_game(
        &mut self,
        game_num: usize,
        agent: &mut dyn Learner,
        opponent: &mut dyn Learner,
    ) -> Result<GameOutcome> {
        // Notify observers of game start
        for observer in &mut self.observers {
            observer.on_game_start(game_num)?;
        }

        let mut state = BoardState::new_with_player(self.config.first_player);
        let mut moves = Vec::new();
        let mut step_num = 0;
        let mut agent_states = Vec::new();

        while !state.is_terminal() {
            let current_player = state.to_move;
            let is_agent_turn = current_player == self.config.agent_player;

            let learner: &mut dyn Learner = if is_agent_turn { agent } else { opponent };

            if is_agent_turn {
                agent_states.push(state);
            }

            // Get canonical state for observation
            let canonical_state = state.canonical();

            // Get move weights before selecting move (for observation)
            let weights_before = learner.move_weights(&state).unwrap_or_default();

            // Get move from learner
            let move_pos = learner.select_move(&state)?;

            // Notify observers of move
            for observer in &mut self.observers {
                observer.on_move(
                    game_num,
                    step_num,
                    &state,
                    &canonical_state,
                    move_pos,
                    &weights_before,
                )?;
            }

            // Make move
            moves.push(move_pos);
            state = state.make_move(move_pos)?;
            step_num += 1;
        }

        // Determine outcome
        let outcome = if let Some(winner) = state.winner() {
            GameOutcome::Win(winner)
        } else {
            GameOutcome::Draw
        };

        // Let agent learn from the game
        agent.learn(
            self.config.first_player,
            &moves,
            outcome,
            self.config.agent_player,
        )?;

        // Let opponent learn too (flip outcome for opponent)
        let opponent_outcome = outcome.swap_players();
        opponent.learn(
            self.config.first_player,
            &moves,
            opponent_outcome,
            self.config.agent_player.opponent(),
        )?;

        self.notify_weights_updated(game_num, &agent_states, &*agent)?;
        self.notify_workspace_snapshot(&*agent)?;
        self.notify_workspace_snapshot(&*opponent)?;

        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::comparison::RandomLearner;

    #[test]
    fn test_training_pipeline() {
        let config = TrainingConfig {
            num_games: 10,
            seed: Some(42),
            agent_player: Player::X,
            first_player: Player::X,
        };

        let mut pipeline = TrainingPipeline::new(config);
        let mut agent = RandomLearner::new("Agent".to_string());
        let mut opponent = RandomLearner::new("Opponent".to_string());

        let result = pipeline.run(&mut agent, &mut opponent).unwrap();

        assert_eq!(result.total_games, 10);
        assert!(result.wins + result.draws + result.losses == 10);
    }
}
