//! Winning line analysis for Tic-Tac-Toe

use std::collections::HashSet;

use super::{Cell, Player};

/// Winning line indices on the 3x3 board
pub const WINNING_LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8], // rows
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8], // columns
    [0, 4, 8],
    [2, 4, 6], // diagonals
];

/// Utility for analyzing winning lines in Tic-Tac-Toe
pub struct LineAnalyzer;

impl LineAnalyzer {
    /// Check if a player has won by having three in a row
    pub fn has_won(cells: &[Cell; 9], player: Player) -> bool {
        let target = player.to_cell();
        WINNING_LINES
            .iter()
            .any(|line| line.iter().all(|&idx| cells[idx] == target))
    }

    /// Find all positions that would immediately win for the player
    pub fn winning_moves(cells: &[Cell; 9], player: Player) -> HashSet<usize> {
        let mut moves = HashSet::new();
        for &line in &WINNING_LINES {
            if let Some(pos) = Self::winning_move_in_line(cells, player, &line) {
                moves.insert(pos);
            }
        }
        moves
    }

    /// Check if a player has an immediate winning move available (2 in a line with 1 empty)
    pub fn has_immediate_win(cells: &[Cell; 9], player: Player) -> bool {
        WINNING_LINES
            .iter()
            .any(|line| Self::winning_move_in_line(cells, player, line).is_some())
    }

    /// Find the winning move position in a specific line, if one exists
    fn winning_move_in_line(cells: &[Cell; 9], player: Player, line: &[usize; 3]) -> Option<usize> {
        let target = player.to_cell();
        let mut count = 0;
        let mut empty_pos = None;

        for &idx in line {
            match cells[idx] {
                Cell::Empty => {
                    if empty_pos.is_some() {
                        // More than one empty cell, not a winning move
                        return None;
                    }
                    empty_pos = Some(idx);
                }
                c if c == target => count += 1,
                _ => return None, // Opponent piece in line
            }
        }

        if count == 2 && empty_pos.is_some() {
            empty_pos
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_won_horizontal() {
        let mut cells = [Cell::Empty; 9];
        cells[0] = Cell::X;
        cells[1] = Cell::X;
        cells[2] = Cell::X;

        assert!(LineAnalyzer::has_won(&cells, Player::X));
        assert!(!LineAnalyzer::has_won(&cells, Player::O));
    }

    #[test]
    fn test_has_won_vertical() {
        let mut cells = [Cell::Empty; 9];
        cells[0] = Cell::O;
        cells[3] = Cell::O;
        cells[6] = Cell::O;

        assert!(LineAnalyzer::has_won(&cells, Player::O));
        assert!(!LineAnalyzer::has_won(&cells, Player::X));
    }

    #[test]
    fn test_has_won_diagonal() {
        let mut cells = [Cell::Empty; 9];
        cells[0] = Cell::X;
        cells[4] = Cell::X;
        cells[8] = Cell::X;

        assert!(LineAnalyzer::has_won(&cells, Player::X));
        assert!(!LineAnalyzer::has_won(&cells, Player::O));
    }

    #[test]
    fn test_winning_moves() {
        // X.X
        // ...
        // ...
        let mut cells = [Cell::Empty; 9];
        cells[0] = Cell::X;
        cells[2] = Cell::X;

        let moves = LineAnalyzer::winning_moves(&cells, Player::X);
        assert_eq!(moves.len(), 1);
        assert!(moves.contains(&1));
    }

    #[test]
    fn test_winning_moves_multiple() {
        // XX.
        // X..
        // ...
        let mut cells = [Cell::Empty; 9];
        cells[0] = Cell::X;
        cells[1] = Cell::X;
        cells[3] = Cell::X;

        let moves = LineAnalyzer::winning_moves(&cells, Player::X);
        assert_eq!(moves.len(), 2);
        assert!(moves.contains(&2)); // Complete top row
        assert!(moves.contains(&6)); // Complete left column
    }

    #[test]
    fn test_has_immediate_win() {
        let mut cells = [Cell::Empty; 9];
        cells[0] = Cell::X;
        cells[1] = Cell::X;

        assert!(LineAnalyzer::has_immediate_win(&cells, Player::X));
        assert!(!LineAnalyzer::has_immediate_win(&cells, Player::O));
    }

    #[test]
    fn test_no_immediate_win() {
        let mut cells = [Cell::Empty; 9];
        cells[0] = Cell::X;

        assert!(!LineAnalyzer::has_immediate_win(&cells, Player::X));
        assert!(!LineAnalyzer::has_immediate_win(&cells, Player::O));
    }
}
