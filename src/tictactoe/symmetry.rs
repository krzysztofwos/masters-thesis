//! D4 symmetry group operations for board canonicalization

use serde::{Deserialize, Serialize};

use super::board::{BoardState, Cell};

/// D4 symmetry transformation (dihedral group of the square)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct D4Transform {
    /// Rotation in degrees (0, 90, 180, 270)
    pub rotation: u16,
    /// Whether to apply reflection
    pub reflection: bool,
}

impl D4Transform {
    /// Create identity transform
    pub fn identity() -> Self {
        D4Transform {
            rotation: 0,
            reflection: false,
        }
    }

    /// Get all 8 D4 transforms
    pub fn all() -> Vec<D4Transform> {
        let mut transforms = Vec::with_capacity(8);
        for rotation in [0, 90, 180, 270] {
            transforms.push(D4Transform {
                rotation,
                reflection: false,
            });
            transforms.push(D4Transform {
                rotation,
                reflection: true,
            });
        }
        transforms
    }

    /// Apply transform to a position (0-8)
    pub fn transform_position(&self, pos: usize) -> usize {
        let (mut row, mut col) = (pos / 3, pos % 3);

        // Apply left-right reflection first (mirror across the vertical axis) so the
        // core library always reflects before rotating. Example helpers may use the
        // opposite order but enumerate the same D4 set.
        if self.reflection {
            col = 2 - col;
        }

        // Apply rotation (clockwise)
        for _ in 0..(self.rotation / 90) {
            let new_row = col;
            let new_col = 2 - row;
            row = new_row;
            col = new_col;
        }

        row * 3 + col
    }

    /// Apply the inverse transform to a position
    pub fn apply_inverse_to_pos(&self, pos: usize) -> usize {
        self.inverse().transform_position(pos)
    }

    /// Apply transform to an array of cells
    pub fn apply_to_cells(&self, cells: &[Cell; 9]) -> [Cell; 9] {
        let mut transformed = [Cell::Empty; 9];
        for idx in 0..9 {
            transformed[self.transform_position(idx)] = cells[idx];
        }
        transformed
    }

    /// Apply the inverse transform to an array of cells
    pub fn apply_inverse_to_cells(&self, cells: &[Cell; 9]) -> [Cell; 9] {
        let mut transformed = [Cell::Empty; 9];
        let inverse = self.inverse();
        for idx in 0..9 {
            transformed[inverse.transform_position(idx)] = cells[idx];
        }
        transformed
    }

    /// Get the inverse transform
    pub fn inverse(&self) -> D4Transform {
        if self.reflection {
            // Reflections combined with rotations have specific inverses:
            // - reflection + 0° is self-inverse
            // - reflection + 90° is self-inverse
            // - reflection + 180° is self-inverse
            // - reflection + 270° is self-inverse
            // This is because in our implementation order (reflect then rotate),
            // these operations are involutions
            *self
        } else {
            // Pure rotation: inverse is opposite rotation
            D4Transform {
                rotation: (360 - self.rotation) % 360,
                reflection: false,
            }
        }
    }
}

impl BoardState {
    /// Apply a D4 transform to the board
    pub fn transform(&self, t: &D4Transform) -> Self {
        let mut cells = [Cell::Empty; 9];
        for i in 0..9 {
            cells[t.transform_position(i)] = self.cells[i];
        }
        BoardState {
            cells,
            to_move: self.to_move,
        }
    }

    /// Get the canonical (lexicographically minimal) form under D4 symmetry
    ///
    /// **Performance Warning**: This method recomputes the canonicalization on every call.
    /// If you need multiple operations (e.g., getting both the canonical state and transform,
    /// or mapping moves to/from canonical coordinates), use `canonical_context()` once
    /// and cache the result instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::tictactoe::BoardState;
    ///
    /// // Single operation - use canonical()
    /// let board = BoardState::new();
    /// let canonical = board.canonical();
    ///
    /// // Multiple operations - use canonical_context() to avoid recomputation
    /// let ctx = board.canonical_context();
    /// let canonical_state = &ctx.state;
    /// let encoding = &ctx.encoding;
    /// let canonical_move = ctx.map_move_to_canonical(4);
    /// ```
    pub fn canonical(&self) -> Self {
        self.canonical_context().state
    }
}
