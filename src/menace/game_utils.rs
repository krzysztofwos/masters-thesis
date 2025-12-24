//! Shared game utilities for learning algorithms
//!
//! This module contains common operations used by multiple learning algorithms
//! to avoid code duplication (DRY principle).

use crate::{
    identifiers::MoveId,
    tictactoe::{BoardState, Player},
    workspace::MenaceWorkspace,
};

/// Build a path of move identifiers from game states and moves for a specific player
///
/// This is a common operation used by learning algorithms to map game trajectories
/// to move identifier paths. It filters states to only include those where
/// the specified player is to move, then maps each state-move pair to a move ID.
///
/// # Arguments
///
/// * `workspace` - The MENACE workspace containing move mappings
/// * `states` - Sequence of board states from the game
/// * `moves` - Sequence of moves made (aligned with states)
/// * `player` - Filter to only include states where this player is to move
///
/// # Returns
///
/// A vector of move identifiers representing the player's decision path
pub fn build_move_path_for_player(
    workspace: &MenaceWorkspace,
    states: &[BoardState],
    moves: &[usize],
    player: Player,
) -> Vec<MoveId> {
    states
        .iter()
        .zip(moves.iter())
        .filter(|(state, _)| state.to_move == player)
        .filter_map(|(state, &mov)| {
            let (ctx, label) = state.canonical_context_and_label();
            let canonical_move = ctx.map_move_to_canonical(mov);
            workspace.move_for_position(&label, canonical_move).cloned()
        })
        .collect()
}
