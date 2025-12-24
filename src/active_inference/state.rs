//! Internal state representation for game tree nodes
//!
//! This module contains internal types used by the GenerativeModel to represent
//! the game tree structure.

use crate::tictactoe::{BoardState, Player};

/// Edge representing an action from a state
#[derive(Debug, Clone)]
pub(crate) struct ActionEdge {
    pub action: usize,
    pub next_label: String,
}

/// Node in the game tree
#[derive(Debug, Clone)]
pub struct StateNode {
    pub state: BoardState,
    pub outcome: Option<Player>,
    pub(crate) actions: Vec<ActionEdge>,
}
