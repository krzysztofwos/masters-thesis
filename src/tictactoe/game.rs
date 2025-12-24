//! High-level game management

use serde::{Deserialize, Serialize};

use super::board::{BoardState, Player};

/// A move in the game
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Move {
    pub position: usize,
    pub player: Player,
}

/// Outcome of a game
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameOutcome {
    Win(Player),
    Draw,
}

impl GameOutcome {
    /// Swap the winner perspective (X ↔ O). Useful when mirroring games.
    pub fn swap_players(self) -> Self {
        match self {
            GameOutcome::Win(player) => GameOutcome::Win(player.opponent()),
            GameOutcome::Draw => GameOutcome::Draw,
        }
    }
}

/// A complete game with history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Game {
    pub initial: BoardState,
    pub moves: Vec<Move>,
    pub outcome: Option<GameOutcome>,
}

impl Game {
    /// Create a new game from initial position
    pub fn new() -> Self {
        Game {
            initial: BoardState::new(),
            moves: Vec::new(),
            outcome: None,
        }
    }

    /// Play a move
    pub fn play(&mut self, position: usize) -> Result<(), crate::Error> {
        if self.outcome.is_some() {
            return Err(crate::Error::GameOver);
        }

        let current = self.current_state()?;
        let new_state = current.make_move(position)?;

        self.moves.push(Move {
            position,
            player: current.to_move,
        });

        if new_state.is_terminal() {
            self.outcome = Some(if let Some(winner) = new_state.winner() {
                GameOutcome::Win(winner)
            } else {
                GameOutcome::Draw
            });
        }

        Ok(())
    }

    /// Replay moves up to a given index (exclusive)
    ///
    /// Returns the board state after applying moves[0..end_index].
    /// If end_index >= moves.len(), all moves are applied.
    ///
    /// # Errors
    ///
    /// Returns error if any move in the history is invalid for the current state.
    /// This indicates corrupted game data.
    fn replay_moves_until(&self, end_index: usize) -> Result<BoardState, crate::Error> {
        let mut state = self.initial;
        for (i, m) in self.moves.iter().take(end_index).enumerate() {
            state = state.make_move_with_context(
                m.position,
                &format!("Invalid move in game history at position {i}"),
            )?;
        }
        Ok(state)
    }

    /// Get current board state
    ///
    /// # Errors
    ///
    /// Returns error if any move in the history is invalid for the current state.
    /// This indicates corrupted game data.
    pub fn current_state(&self) -> Result<BoardState, crate::Error> {
        self.replay_moves_until(self.moves.len())
    }

    /// Get the sequence of board states
    ///
    /// # Errors
    ///
    /// Returns error if any move in the history is invalid for the current state.
    /// This indicates corrupted game data.
    pub fn state_sequence(&self) -> Result<Vec<BoardState>, crate::Error> {
        let mut states = Vec::with_capacity(self.moves.len() + 1);
        states.push(self.initial);

        for i in 1..=self.moves.len() {
            states.push(self.replay_moves_until(i)?);
        }

        Ok(states)
    }

    /// Get canonical representation of the game
    ///
    /// Returns a canonical move sequence by transforming each move based on the
    /// canonical form of the board state at that point in the game. This approach
    /// allows more games to map to the same canonical trajectory compared to
    /// applying a single transformation to the entire move sequence.
    ///
    /// The algorithm:
    /// 1. For each move in the game:
    ///    - Get the canonical state once using [`BoardState::canonical_context`]
    ///      (avoids redundant D4 search during the transformation selection)
    ///    - Among all D4 transforms that map to the canonical state, find the one
    ///      that produces the lexicographically minimal move position
    ///    - Add the minimal move position to the canonical trajectory
    ///
    /// # Errors
    ///
    /// Returns error if any move in the history is invalid for the current state.
    /// This indicates corrupted game data.
    pub fn canonical(&self) -> Result<Vec<usize>, crate::Error> {
        use super::symmetry::D4Transform;

        let mut canonical_moves = Vec::with_capacity(self.moves.len());
        let mut state = self.initial;

        for mv in &self.moves {
            // Get canonical state once (avoids redundant canonicalization in the search)
            let ctx = state.canonical_context();
            let canonical_state = &ctx.state;

            // Among all transforms that map to canonical, find the one giving the
            // lexicographically minimal move position (with tie-breaking on transform rank)
            let mut best_position = usize::MAX;
            let mut best_rank = (u16::MAX, true);

            for transform in D4Transform::all() {
                if &state.transform(&transform) == canonical_state {
                    let candidate = transform.transform_position(mv.position);
                    let rank = (transform.rotation, transform.reflection);

                    if candidate < best_position || (candidate == best_position && rank < best_rank)
                    {
                        best_position = candidate;
                        best_rank = rank;
                    }
                }
            }

            canonical_moves.push(best_position);
            state = state.make_move(mv.position)?;
        }

        Ok(canonical_moves)
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}
