//! Board state representation and basic operations

use std::{collections::HashSet, fmt};

use serde::{Deserialize, Serialize};

use super::symmetry::D4Transform;

/// A cell on the Tic-Tac-Toe board
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Cell {
    Empty,
    X,
    O,
}

impl Cell {
    pub fn to_char(self) -> char {
        match self {
            Cell::Empty => '.',
            Cell::X => 'X',
            Cell::O => 'O',
        }
    }

    pub fn from_char(c: char) -> Option<Cell> {
        match c {
            '.' | ' ' => Some(Cell::Empty),
            'X' | 'x' => Some(Cell::X),
            'O' | 'o' | '0' => Some(Cell::O),
            _ => None,
        }
    }
}

/// A player in the game
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Player {
    X,
    O,
}

impl Player {
    /// Get the opponent player
    pub fn opponent(self) -> Player {
        match self {
            Player::X => Player::O,
            Player::O => Player::X,
        }
    }

    /// Convert player to cell
    pub fn to_cell(self) -> Cell {
        match self {
            Player::X => Cell::X,
            Player::O => Cell::O,
        }
    }
}

/// Complete board state including cells and whose turn it is
///
/// This type implements `Copy` for efficiency since it's only 10 bytes
/// (9 bytes for cells + 1 byte for player enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoardState {
    pub cells: [Cell; 9],
    pub to_move: Player,
}

/// Count of each piece type on the board
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PieceCount {
    x: usize,
    o: usize,
    empty: usize,
}

/// Cached result of canonicalization for efficient repeated operations.
///
/// This struct caches the result of the expensive canonicalization search,
/// allowing multiple transformations to be performed without recomputing
/// the canonical form.
#[derive(Debug, Clone)]
pub struct CanonicalContext {
    /// The canonical board state
    pub state: BoardState,
    /// The transform that maps the original state to the canonical state
    pub transform: D4Transform,
    /// The string encoding of the canonical state
    pub encoding: String,
}

impl CanonicalContext {
    /// Get the canonical encoding
    pub fn encoding(&self) -> &str {
        &self.encoding
    }

    /// Map a move from original coordinates to canonical coordinates
    pub fn map_move_to_canonical(&self, original_move: usize) -> usize {
        self.transform.transform_position(original_move)
    }

    /// Map a move from canonical coordinates back to original coordinates
    pub fn map_canonical_to_original(&self, canonical_move: usize) -> usize {
        self.transform.inverse().transform_position(canonical_move)
    }

    /// Get the canonical state and encoding as a tuple (for compatibility)
    pub fn label_and_transform(&self) -> (&str, &D4Transform) {
        (&self.encoding, &self.transform)
    }
}

impl BoardState {
    /// Create a new empty board with X to move
    pub fn new() -> Self {
        Self::new_with_player(Player::X)
    }

    /// Create a new empty board with a specified player to move first.
    ///
    /// # O-First Games
    ///
    /// While standard Tic-Tac-Toe has X moving first, this method supports
    /// O-first games for analysis and training scenarios. The game logic and
    /// validation correctly handle both cases:
    ///
    /// - **X-first (standard):** Valid states have `x_count == o_count` or `x_count == o_count + 1`
    /// - **O-first:** Valid states have `o_count == x_count` or `o_count == x_count + 1`
    ///
    /// Terminal state validation adapts automatically:
    /// - X wins: requires `x_count == o_count + 1` (X moved last)
    /// - O wins (X-first): requires `o_count == x_count` (O moved last)
    /// - O wins (O-first): requires `o_count == x_count + 1` (O moved last)
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::tictactoe::{BoardState, Player};
    ///
    /// // Standard X-first game
    /// let board = BoardState::new();
    /// assert_eq!(board.to_move, Player::X);
    ///
    /// // O-first game (non-standard but supported)
    /// let o_first = BoardState::new_with_player(Player::O);
    /// assert_eq!(o_first.to_move, Player::O);
    ///
    /// // Both support full game play
    /// let after_move = o_first.make_move(4).unwrap();
    /// assert_eq!(after_move.to_move, Player::X);
    /// ```
    pub fn new_with_player(first_player: Player) -> Self {
        BoardState {
            cells: [Cell::Empty; 9],
            to_move: first_player,
        }
    }

    /// Helper: Parse 9 cells from a slice of characters.
    ///
    /// # Errors
    ///
    /// Returns error if fewer than 9 characters or any character is invalid.
    fn parse_cells(chars: &[char], context: &str) -> Result<[Cell; 9], crate::Error> {
        if chars.len() < 9 {
            return Err(crate::Error::InvalidBoardLength {
                expected: 9,
                got: chars.len(),
                context: context.to_string(),
            });
        }

        let mut cells = [Cell::Empty; 9];
        for (i, &c) in chars.iter().take(9).enumerate() {
            cells[i] = Cell::from_char(c).ok_or_else(|| crate::Error::InvalidCellCharacter {
                character: c,
                position: i,
                context: context.to_string(),
            })?;
        }

        Ok(cells)
    }

    /// Helper: Count pieces on the board.
    fn count_pieces(cells: &[Cell; 9]) -> PieceCount {
        let mut count = PieceCount {
            x: 0,
            o: 0,
            empty: 0,
        };
        for cell in cells {
            match cell {
                Cell::X => count.x += 1,
                Cell::O => count.o += 1,
                Cell::Empty => count.empty += 1,
            }
        }
        count
    }

    /// Helper: Parse a player string ("X" or "O").
    ///
    /// # Errors
    ///
    /// Returns error if the string is not "X" or "O".
    fn parse_player(player_str: &str, context: &str) -> Result<Player, crate::Error> {
        match player_str {
            "X" => Ok(Player::X),
            "O" => Ok(Player::O),
            _ => Err(crate::Error::InvalidPlayerString {
                player: player_str.to_string(),
                label: context.to_string(),
            }),
        }
    }

    fn determine_turn_from_counts(count: &PieceCount) -> Result<Player, crate::Error> {
        if count.x == count.o {
            Ok(Player::X)
        } else if count.x == count.o + 1 {
            Ok(Player::O)
        } else {
            Err(crate::Error::InvalidPieceCounts {
                x_count: count.x,
                o_count: count.o,
            })
        }
    }

    fn ensure_turn_consistent_with_counts(
        count: &PieceCount,
        player: Player,
        context: &str,
    ) -> Result<(), crate::Error> {
        let valid = match player {
            Player::X => count.x == count.o || count.o == count.x + 1,
            Player::O => count.x == count.o || count.x == count.o + 1,
        };

        if valid {
            Ok(())
        } else {
            Err(crate::Error::InvalidConfiguration {
                message: format!(
                    "piece counts (X={}, O={}) are inconsistent with {} to move in '{}'",
                    count.x,
                    count.o,
                    match player {
                        Player::X => "X",
                        Player::O => "O",
                    },
                    context
                ),
            })
        }
    }

    /// Create a board from a string representation.
    ///
    /// The string should contain 9 characters (whitespace is filtered out) and
    /// may optionally include a suffix `_X` or `_O` to explicitly set the player
    /// to move. When the suffix is omitted, the player is inferred from the piece
    /// counts, defaulting to X-first semantics for ambiguous cases.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - The board part has fewer than 9 non-whitespace characters
    /// - Any character is not a valid cell representation
    /// - The piece counts are invalid (difference greater than 1)
    /// - A provided `_X`/`_O` suffix conflicts with the piece counts
    pub fn from_string(s: &str) -> Result<Self, crate::Error> {
        let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        let (board_part, specified_turn) = Self::split_board_and_turn(&cleaned)?;
        let chars: Vec<char> = board_part.chars().collect();
        let cells = Self::parse_cells(&chars, s)?;
        let count = Self::count_pieces(&cells);

        let to_move = if let Some(turn) = specified_turn {
            Self::ensure_turn_consistent_with_counts(&count, turn, s).map(|_| turn)?
        } else {
            Self::determine_turn_from_counts(&count)?
        };

        Ok(BoardState { cells, to_move })
    }

    fn split_board_and_turn(cleaned: &str) -> Result<(&str, Option<Player>), crate::Error> {
        if let Some(idx) = cleaned.find('_') {
            let board = &cleaned[..idx];
            let suffix = &cleaned[idx + 1..];
            if suffix.is_empty() {
                return Err(crate::Error::InvalidPlayerString {
                    player: String::new(),
                    label: cleaned.to_string(),
                });
            }
            let player = Self::parse_player(suffix, cleaned)?;
            Ok((board, Some(player)))
        } else {
            Ok((cleaned, None))
        }
    }

    /// Create a board from label format "XXXXXXXXX_P" where P is X or O.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - The label format is invalid (not "board_player")
    /// - The board part is not exactly 9 characters
    /// - Any character is not a valid cell representation
    /// - The player part is not "X" or "O"
    pub fn from_label(label: &str) -> Result<Self, crate::Error> {
        let mut parts = label.split('_');
        let board = parts.next().ok_or_else(|| crate::Error::MissingLabelPart {
            part: "board".to_string(),
            label: label.to_string(),
        })?;
        let to_move_str = parts.next().ok_or_else(|| crate::Error::MissingLabelPart {
            part: "player".to_string(),
            label: label.to_string(),
        })?;

        if parts.next().is_some() || board.len() != 9 {
            return Err(crate::Error::InvalidLabelFormat {
                label: label.to_string(),
                expected: "XXXXXXXXX_P".to_string(),
            });
        }

        // Reuse parse_cells helper
        let chars: Vec<char> = board.chars().collect();
        let cells = Self::parse_cells(&chars, label)?;

        // Use parse_player helper for consistency
        let to_move = Self::parse_player(to_move_str, label)?;

        let counts = Self::count_pieces(&cells);
        let diff = counts.x as isize - counts.o as isize;

        let invalid_label = |reason: &str| crate::Error::InvalidConfiguration {
            message: format!("invalid board label '{label}': {reason}"),
        };

        if diff.abs() > 1 {
            return Err(invalid_label(&format!(
                "piece counts must differ by at most 1 (X={}, O={})",
                counts.x, counts.o
            )));
        }

        match diff {
            1 if to_move != Player::O => {
                return Err(invalid_label(
                    "X has an extra move, so it must be O's turn in the label",
                ));
            }
            -1 if to_move != Player::X => {
                return Err(invalid_label(
                    "O has an extra move, so it must be X's turn in the label",
                ));
            }
            _ => {}
        }

        let board = BoardState { cells, to_move };
        let x_wins = board.has_won(Player::X);
        let o_wins = board.has_won(Player::O);

        if x_wins && o_wins {
            return Err(invalid_label("both players cannot have winning lines"));
        }

        if x_wins && diff != 1 && diff != 0 {
            return Err(invalid_label(
                "X winning requires X to have the same number of moves as O (O opened) or exactly one more move (X opened)",
            ));
        }

        if o_wins && diff != -1 && diff != 0 {
            return Err(invalid_label(
                "O winning requires O to have the same number of moves as X (X opened) or exactly one more move (O opened)",
            ));
        }

        Ok(board)
    }

    /// Count the number of occupied cells on the board.
    pub fn occupied_count(&self) -> usize {
        let count = Self::count_pieces(&self.cells);
        count.x + count.o
    }

    /// Get cell at position (0-8)
    pub fn get(&self, pos: usize) -> Cell {
        self.cells[pos]
    }

    /// Check if a position is empty
    pub fn is_empty(&self, pos: usize) -> bool {
        self.cells[pos] == Cell::Empty
    }

    /// Get all empty positions
    pub fn empty_positions(&self) -> Vec<usize> {
        self.cells
            .iter()
            .enumerate()
            .filter(|&(_, &cell)| cell == Cell::Empty)
            .map(|(i, _)| i)
            .collect()
    }

    /// Make a move and return a new board state
    #[must_use = "make_move returns a new board state; the original is unchanged"]
    pub fn make_move(&self, pos: usize) -> Result<BoardState, crate::Error> {
        if pos >= 9 {
            return Err(crate::Error::InvalidMove { position: pos });
        }

        if !self.is_empty(pos) {
            return Err(crate::Error::InvalidMove { position: pos });
        }

        let mut new_state = *self;
        new_state.cells[pos] = self.to_move.to_cell();
        new_state.to_move = self.to_move.opponent();
        Ok(new_state)
    }

    /// Apply a move with enhanced error context
    #[must_use = "make_move_with_context returns a new board state; the original is unchanged"]
    pub fn make_move_with_context(
        &self,
        pos: usize,
        context: &str,
    ) -> Result<BoardState, crate::Error> {
        self.make_move(pos)
            .map_err(|e| crate::Error::IllegalGeneratedMove {
                position: pos,
                context: format!("{context}: {e}"),
            })
    }

    /// Apply a move assuming it is legal (used for enumerations)
    #[must_use = "make_move_force returns a new board state; the original is unchanged"]
    pub fn make_move_force(&self, pos: usize) -> Result<BoardState, crate::Error> {
        self.make_move_with_context(
            pos,
            &format!("Generated tree contained illegal move at position {pos}"),
        )
    }

    /// Get legal moves in this position (empty cells when game not terminal)
    pub fn legal_moves(&self) -> Vec<usize> {
        if self.is_terminal() {
            return Vec::new();
        }
        self.empty_positions()
    }

    /// Swap X and O pieces on the board, preserving the turn structure.
    ///
    /// This creates a new board state where all X pieces become O pieces and
    /// vice versa. The player to move is also swapped to their opponent.
    ///
    /// This is useful for analyzing positions from the opposite player's perspective
    /// while maintaining the game's turn structure.
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::tictactoe::{BoardState, Player};
    ///
    /// let mut board = BoardState::new();
    /// board = board.make_move(0).unwrap(); // X at position 0, now O to move
    ///
    /// // Swap players - O becomes X, X becomes O
    /// let swapped = board.swap_players();
    /// assert_eq!(swapped.to_move, Player::X); // Original was O, now X
    /// ```
    #[must_use = "swap_players returns a new board state; the original is unchanged"]
    pub fn swap_players(&self) -> Self {
        let mut swapped = *self;
        for cell in &mut swapped.cells {
            *cell = match cell {
                Cell::X => Cell::O,
                Cell::O => Cell::X,
                Cell::Empty => Cell::Empty,
            };
        }
        swapped.to_move = self.to_move.opponent();
        swapped
    }

    /// Flip the board perspective by swapping X and O pieces.
    ///
    /// This creates a new board state where all X pieces become O pieces and
    /// vice versa. The player to move is always set to X in the flipped board.
    ///
    /// This is useful for opponent agents that are trained to play as X, allowing
    /// them to evaluate positions where they play as O by flipping the perspective.
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::tictactoe::{BoardState, Player, Cell};
    ///
    /// // Create a board with X to move
    /// let mut board = BoardState::new();
    /// board = board.make_move(0).unwrap(); // X at position 0
    ///
    /// // Flip perspective - O becomes X
    /// let flipped = board.flip_perspective();
    /// assert_eq!(flipped.to_move, Player::X);
    /// ```
    #[must_use = "flip_perspective returns a new board state; the original is unchanged"]
    pub fn flip_perspective(&self) -> Self {
        let mut flipped = self.swap_players();
        flipped.to_move = Player::X;
        flipped
    }

    /// Check if only a single move is available
    pub fn has_forced_move(&self) -> bool {
        self.legal_moves().len() == 1
    }

    /// Determine if the opponent currently threatens two winning moves
    pub fn opponent_has_double_threat(&self) -> bool {
        if self.has_immediate_win() {
            return false;
        }

        let opponent_piece = match self.to_move {
            Player::X => Cell::O,
            Player::O => Cell::X,
        };

        let mut winning_moves = HashSet::new();
        for pos in 0..9 {
            if self.cells[pos] == Cell::Empty {
                let mut test = *self;
                test.cells[pos] = opponent_piece;
                if test.winner() == Some(opponent_piece.to_player().unwrap()) {
                    winning_moves.insert(pos);
                }
            }
        }

        winning_moves.len() >= 2
    }

    /// Check if current player has an immediate winning move available
    pub fn has_immediate_win(&self) -> bool {
        let current_piece = self.to_move.to_cell();

        for pos in 0..9 {
            if self.cells[pos] == Cell::Empty {
                let mut test = *self;
                test.cells[pos] = current_piece;
                if test.winner() == Some(self.to_move) {
                    return true;
                }
            }
        }

        false
    }

    /// Check if a player has won
    pub fn has_won(&self, player: Player) -> bool {
        super::lines::LineAnalyzer::has_won(&self.cells, player)
    }

    /// Check if the game is over (win or draw)
    pub fn is_terminal(&self) -> bool {
        self.has_won(Player::X) || self.has_won(Player::O) || self.empty_positions().is_empty()
    }

    /// Check if the position is a draw (all cells filled, no winner)
    pub fn is_draw(&self) -> bool {
        !self.cells.contains(&Cell::Empty) && self.winner().is_none()
    }

    /// Get the winner if there is one
    pub fn winner(&self) -> Option<Player> {
        if self.has_won(Player::X) {
            Some(Player::X)
        } else if self.has_won(Player::O) {
            Some(Player::O)
        } else {
            None
        }
    }

    /// Find the position where two board states differ (for inferring moves)
    ///
    /// Returns the first position where the cells differ, or None if identical.
    pub fn find_changed_position(&self, other: &BoardState) -> Option<usize> {
        self.cells
            .iter()
            .zip(other.cells.iter())
            .position(|(a, b)| a != b)
    }

    /// Get a canonical string representation for use as a key
    pub fn encode(&self) -> String {
        format!(
            "{}_{}",
            self.cells.iter().map(|&c| c.to_char()).collect::<String>(),
            match self.to_move {
                Player::X => 'X',
                Player::O => 'O',
            }
        )
    }

    /// Helper: Find the canonical form by searching through all D4 transforms.
    ///
    /// Returns the canonical state, the transform to reach it, and its encoding.
    fn find_canonical_form(&self) -> (BoardState, D4Transform, String) {
        let mut best_state = *self;
        let mut best_transform = D4Transform::identity();
        let mut best_encoding = self.encode();

        for transform in D4Transform::all() {
            let transformed = self.transform(&transform);
            let encoding = transformed.encode();
            if encoding < best_encoding {
                best_encoding = encoding;
                best_state = transformed;
                best_transform = transform;
            }
        }

        (best_state, best_transform, best_encoding)
    }

    /// Create a canonical context that caches the expensive canonicalization computation.
    ///
    /// Use this when you need to perform multiple operations (like mapping moves
    /// to/from canonical coordinates) to avoid redundant canonicalization searches.
    ///
    /// # Example
    ///
    /// ```
    /// use menace::tictactoe::BoardState;
    ///
    /// let state = BoardState::new();
    /// let ctx = state.canonical_context();
    /// let canonical_move = ctx.map_move_to_canonical(0);
    /// let original_move = ctx.map_canonical_to_original(canonical_move);
    /// assert_eq!(original_move, 0);
    /// ```
    pub fn canonical_context(&self) -> CanonicalContext {
        let (state, transform, encoding) = self.find_canonical_form();

        CanonicalContext {
            state,
            transform,
            encoding,
        }
    }

    /// Get both the canonical context and label in one call.
    ///
    /// This is a convenience method that combines [`canonical_context`] and
    /// [`CanonicalLabel::from`] to avoid the common pattern of creating both.
    ///
    /// # Example
    ///
    /// ```
    /// use menace::tictactoe::BoardState;
    ///
    /// let state = BoardState::new();
    /// let (ctx, label) = state.canonical_context_and_label();
    ///
    /// // Use both context and label
    /// let canonical_move = ctx.map_move_to_canonical(0);
    /// // ... use label for lookups
    /// ```
    ///
    /// [`canonical_context`]: Self::canonical_context
    /// [`CanonicalLabel::from`]: crate::types::CanonicalLabel::from
    pub fn canonical_context_and_label(&self) -> (CanonicalContext, crate::types::CanonicalLabel) {
        let ctx = self.canonical_context();
        let label = crate::types::CanonicalLabel::from(&ctx);
        (ctx, label)
    }
}

impl Default for BoardState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BoardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, &cell) in self.cells.iter().enumerate() {
            write!(f, "{}", cell.to_char())?;
            if (i + 1).is_multiple_of(3) && i < 8 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

impl Cell {
    fn to_player(self) -> Option<Player> {
        match self {
            Cell::X => Some(Player::X),
            Cell::O => Some(Player::O),
            Cell::Empty => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_board() {
        let board = BoardState::new();
        assert_eq!(board.to_move, Player::X);
        for i in 0..9 {
            assert_eq!(board.cells[i], Cell::Empty);
        }
    }

    #[test]
    fn test_make_move() {
        let board = BoardState::new();

        // Valid move
        let result = board.make_move(4);
        assert!(result.is_ok());
        let new_board = result.unwrap();
        assert_eq!(new_board.cells[4], Cell::X);
        assert_eq!(new_board.to_move, Player::O);

        // Move on occupied cell
        let result2 = new_board.make_move(4);
        assert!(result2.is_err());
        assert!(result2.unwrap_err().to_string().contains("occupied"));
    }

    #[test]
    fn test_legal_moves() {
        let mut board = BoardState::new();
        assert_eq!(board.legal_moves().len(), 9);

        board = board.make_move(0).unwrap();
        assert_eq!(board.legal_moves().len(), 8);
        assert!(!board.legal_moves().contains(&0));

        board = board.make_move(4).unwrap();
        assert_eq!(board.legal_moves().len(), 7);
        assert!(!board.legal_moves().contains(&4));
    }

    #[test]
    fn test_win_detection_horizontal() {
        let mut board = BoardState::new();
        // X wins on top row
        board = board.make_move(0).unwrap(); // X
        board = board.make_move(3).unwrap(); // O
        board = board.make_move(1).unwrap(); // X
        board = board.make_move(4).unwrap(); // O
        board = board.make_move(2).unwrap(); // X

        assert!(board.is_terminal());
        assert_eq!(board.winner(), Some(Player::X));
    }

    #[test]
    fn test_win_detection_vertical() {
        let mut board = BoardState::new();
        // O wins on middle column (1, 4, 7)
        board = board.make_move(0).unwrap(); // X
        board = board.make_move(1).unwrap(); // O
        board = board.make_move(2).unwrap(); // X
        board = board.make_move(4).unwrap(); // O
        board = board.make_move(5).unwrap(); // X
        board = board.make_move(7).unwrap(); // O

        assert!(board.is_terminal());
        assert_eq!(board.winner(), Some(Player::O));
    }

    #[test]
    fn test_win_detection_diagonal() {
        let mut board = BoardState::new();
        // X wins on main diagonal
        board = board.make_move(0).unwrap(); // X
        board = board.make_move(1).unwrap(); // O
        board = board.make_move(4).unwrap(); // X
        board = board.make_move(2).unwrap(); // O
        board = board.make_move(8).unwrap(); // X

        assert!(board.is_terminal());
        assert_eq!(board.winner(), Some(Player::X));
    }

    #[test]
    fn test_draw_detection() {
        let mut board = BoardState::new();
        // Classic draw game
        board = board.make_move(0).unwrap(); // X
        board = board.make_move(1).unwrap(); // O
        board = board.make_move(2).unwrap(); // X
        board = board.make_move(4).unwrap(); // O
        board = board.make_move(3).unwrap(); // X
        board = board.make_move(6).unwrap(); // O
        board = board.make_move(5).unwrap(); // X
        board = board.make_move(8).unwrap(); // O
        board = board.make_move(7).unwrap(); // X

        assert!(board.is_terminal());
        assert_eq!(board.winner(), None);
    }

    #[test]
    fn test_immediate_win_detection() {
        let mut board = BoardState::new();
        board = board.make_move(0).unwrap(); // X
        board = board.make_move(3).unwrap(); // O
        board = board.make_move(1).unwrap(); // X
        board = board.make_move(4).unwrap(); // O
        // X can win at position 2

        assert!(board.has_immediate_win());
    }

    #[test]
    fn test_canonical_form() {
        let board = BoardState::new().make_move(0).unwrap();
        let canonical = board.canonical();

        // Test that canonical form is consistent
        let board2 = BoardState::new().make_move(2).unwrap();
        let canonical2 = board2.canonical();

        // Both should produce the same canonical form (corner moves are equivalent)
        assert_eq!(canonical, canonical2);
    }

    #[test]
    fn test_from_string() {
        let board = BoardState::from_string("XOX......").unwrap();
        assert_eq!(board.cells[0], Cell::X);
        assert_eq!(board.cells[1], Cell::O);
        assert_eq!(board.cells[2], Cell::X);
        // to_move is calculated based on piece count
        assert_eq!(board.to_move, Player::O);

        // Invalid string length
        let result = BoardState::from_string("XO");
        assert!(result.is_err());

        // Invalid character
        let result = BoardState::from_string("XOZ......");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_string_with_turn_suffix() {
        let board = BoardState::from_string("........._O").unwrap();
        assert_eq!(board.to_move, Player::O);

        let board_with_o_first_move = BoardState::from_string("O........_X").unwrap();
        assert_eq!(board_with_o_first_move.to_move, Player::X);
    }

    #[test]
    fn test_from_string_rejects_inconsistent_suffix() {
        let err = BoardState::from_string("O........_O").unwrap_err();
        assert!(
            err.to_string().contains("inconsistent with O to move"),
            "expected inconsistency error, got {err}"
        );
    }

    #[test]
    fn test_from_label_allows_o_first_root() {
        let board = BoardState::from_label("........._O").unwrap();
        assert_eq!(board.to_move, Player::O);
        assert_eq!(board.occupied_count(), 0);
    }

    #[test]
    fn test_from_label_rejects_large_piece_difference() {
        let result = BoardState::from_label("XXXX....._X");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_label_rejects_turn_mismatch() {
        let result = BoardState::from_label("O........_O");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_label_rejects_conflicting_winners() {
        let result = BoardState::from_label("XXXOOOXXO_O");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_label_allows_o_first_x_win() {
        let board = BoardState::from_label("XXXOO.O.._O").unwrap();
        assert!(board.has_won(Player::X));
        assert_eq!(board.to_move, Player::O);
    }

    #[test]
    fn test_encode() {
        let board = BoardState::from_string("XO.......").unwrap();
        // encode appends the player to move
        // With 1 X and 1 O, it's X's turn (equal counts means X goes)
        assert_eq!(board.encode(), "XO......._X");

        let empty = BoardState::new();
        assert_eq!(empty.encode(), "........._X");
    }

    #[test]
    fn o_win_roundtrip_when_x_opens() {
        let mut board = BoardState::new();
        board = board.make_move(4).unwrap(); // X
        board = board.make_move(0).unwrap(); // O
        board = board.make_move(8).unwrap(); // X
        board = board.make_move(1).unwrap(); // O
        board = board.make_move(3).unwrap(); // X
        board = board.make_move(2).unwrap(); // O wins top row

        assert!(board.has_won(Player::O));
        let encoded = board.encode();
        let parsed = BoardState::from_label(&encoded).expect("roundtrip should succeed");
        assert_eq!(parsed, board);
    }

    #[test]
    fn o_win_roundtrip_when_o_opens() {
        let mut board = BoardState::new_with_player(Player::O);
        board = board.make_move(0).unwrap(); // O
        board = board.make_move(4).unwrap(); // X
        board = board.make_move(1).unwrap(); // O
        board = board.make_move(3).unwrap(); // X
        board = board.make_move(2).unwrap(); // O wins top row

        assert!(board.has_won(Player::O));
        let encoded = board.encode();
        let parsed = BoardState::from_label(&encoded).expect("roundtrip should succeed");
        assert_eq!(parsed, board);
    }

    #[test]
    fn test_display() {
        let board = BoardState::from_string("XOX.O.X..").unwrap();
        let display = format!("{board}");
        // The Display implementation outputs the board with '.' for empty cells
        assert!(display.contains("XOX"));
        assert!(display.contains(".O."));
        assert!(display.contains("X.."));
    }

    #[test]
    fn test_empty_positions() {
        let board = BoardState::new();
        assert_eq!(board.empty_positions().len(), 9);

        let board = board.make_move(4).unwrap();
        let empty = board.empty_positions();
        assert_eq!(empty.len(), 8);
        assert!(!empty.contains(&4));
        assert!(empty.contains(&0));
    }

    #[test]
    fn test_blocking_moves() {
        let mut board = BoardState::new();
        // Set up a position where O needs to block X from winning
        board = board.make_move(0).unwrap(); // X
        board = board.make_move(4).unwrap(); // O
        board = board.make_move(1).unwrap(); // X
        // O must block at position 2

        let blocks = winning_moves_for(&board, Player::X);
        assert!(blocks.contains(&2));
    }

    #[test]
    fn test_player_alternation() {
        let mut board = BoardState::new();
        assert_eq!(board.to_move, Player::X);

        board = board.make_move(0).unwrap();
        assert_eq!(board.to_move, Player::O);

        board = board.make_move(1).unwrap();
        assert_eq!(board.to_move, Player::X);

        board = board.make_move(2).unwrap();
        assert_eq!(board.to_move, Player::O);
    }

    // Helper function (needs to be imported from game_tree module in real usage)
    fn winning_moves_for(state: &BoardState, player: Player) -> Vec<usize> {
        let mut wins = Vec::new();
        let piece = player.to_cell();

        for pos in 0..9 {
            if state.cells[pos] == Cell::Empty {
                let mut test = *state;
                test.cells[pos] = piece;
                if test.winner() == Some(player) {
                    wins.push(pos);
                }
            }
        }

        wins
    }
}
