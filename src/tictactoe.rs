//! Tic-Tac-Toe game implementation

pub mod board;
pub mod game;
pub mod game_tree;
pub mod lines;
pub mod symmetry;
pub mod validation;

pub use board::{BoardState, CanonicalContext, Cell, Player};
pub use game::{Game, GameOutcome, Move};
pub use game_tree::{
    MenacePositionStats, analyze_menace_positions, build_reduced_game_tree,
    collect_reachable_canonical_labels, format_board,
};
pub use lines::{LineAnalyzer, WINNING_LINES};
pub use symmetry::D4Transform;
