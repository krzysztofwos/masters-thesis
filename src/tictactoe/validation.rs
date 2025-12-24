//! Board state validation logic

use super::board::{BoardState, Cell, Player};

impl BoardState {
    /// Check if the board state is valid according to Tic-Tac-Toe rules
    pub fn is_valid(&self) -> bool {
        // Check turn counts
        let x_count = self.cells.iter().filter(|&&c| c == Cell::X).count();
        let o_count = self.cells.iter().filter(|&&c| c == Cell::O).count();

        // Piece counts must differ by at most 1 in either direction.
        // Allow O-first sequences by permitting O to have one extra move.
        if !(x_count == o_count || x_count == o_count + 1 || o_count == x_count + 1) {
            return false;
        }

        let diff = x_count as isize - o_count as isize;

        // Ensure the recorded turn matches the piece counts
        match self.to_move {
            Player::X => {
                if !(diff == 0 || diff == -1) {
                    return false;
                }
            }
            Player::O => {
                if !(diff == 0 || diff == 1) {
                    return false;
                }
            }
        }

        // Check for invalid continuations (both players winning)
        let x_wins = self.has_won(Player::X);
        let o_wins = self.has_won(Player::O);

        if x_wins && o_wins {
            return false; // Both can't win
        }

        // If someone won, they must have moved last and the recorded turn must
        // belong to their opponent.
        if x_wins {
            if self.to_move != Player::O {
                return false;
            }
            if !(x_count == o_count + 1 || x_count == o_count) {
                return false;
            }
        }
        if o_wins {
            if self.to_move != Player::X {
                return false;
            }
            if !(o_count == x_count || o_count == x_count + 1) {
                return false;
            }
        }

        // Check for multiple winning lines that don't share a cell
        // (indicates an invalid continuation after a win)
        if x_wins && !self.winning_lines_share_cell(Player::X) {
            return false;
        }
        if o_wins && !self.winning_lines_share_cell(Player::O) {
            return false;
        }

        true
    }

    /// Check if all winning lines for a player share at least one cell
    /// This is necessary for multiple lines to be formed in a single move
    pub fn winning_lines_share_cell(&self, player: Player) -> bool {
        let cell = player.to_cell();
        let mut winning_lines = Vec::new();

        // Collect all winning lines
        // Check rows
        for row in 0..3 {
            if self.cells[row * 3] == cell
                && self.cells[row * 3 + 1] == cell
                && self.cells[row * 3 + 2] == cell
            {
                winning_lines.push(vec![row * 3, row * 3 + 1, row * 3 + 2]);
            }
        }

        // Check columns
        for col in 0..3 {
            if self.cells[col] == cell && self.cells[col + 3] == cell && self.cells[col + 6] == cell
            {
                winning_lines.push(vec![col, col + 3, col + 6]);
            }
        }

        // Check diagonals
        if self.cells[0] == cell && self.cells[4] == cell && self.cells[8] == cell {
            winning_lines.push(vec![0, 4, 8]);
        }
        if self.cells[2] == cell && self.cells[4] == cell && self.cells[6] == cell {
            winning_lines.push(vec![2, 4, 6]);
        }

        // If fewer than 2 lines, trivially true
        if winning_lines.len() < 2 {
            return true;
        }

        // Check if there's a cell that appears in all winning lines
        for pos in 0..9 {
            if winning_lines.iter().all(|line| line.contains(&pos)) {
                return true;
            }
        }

        false
    }

    /// Count valid states reachable from empty board
    pub fn count_valid_states() -> usize {
        let mut count = 0;
        let mut stack = vec![BoardState::new()];
        let mut seen = std::collections::HashSet::new();

        while let Some(state) = stack.pop() {
            let key = state.encode();
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);

            if state.is_valid() {
                count += 1;

                if !state.is_terminal() {
                    for pos in state.empty_positions() {
                        if let Ok(next) = state.make_move(pos) {
                            stack.push(next);
                        }
                    }
                }
            }
        }

        count
    }
}
