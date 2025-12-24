//! Newtype wrappers for improved type safety and domain modeling.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A position on the game board (0-8 for Tic-Tac-Toe).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Position(usize);

impl Position {
    /// Create a new position, validating it's within board bounds.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidPosition`] if the position is >= 9.
    pub fn new(value: usize) -> Result<Self, crate::Error> {
        if value < 9 {
            Ok(Position(value))
        } else {
            Err(crate::Error::InvalidPosition { position: value })
        }
    }

    /// Get the inner value.
    pub fn value(&self) -> usize {
        self.0
    }
}

impl From<Position> for usize {
    fn from(pos: Position) -> Self {
        pos.0
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A weight or probability value (non-negative).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Weight(f64);

impl Weight {
    /// Create a new weight, validating it's non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidWeight`] if the weight is negative or not finite.
    pub fn new(value: f64) -> Result<Self, crate::Error> {
        if value >= 0.0 && value.is_finite() {
            Ok(Weight(value))
        } else {
            Err(crate::Error::InvalidWeight { value })
        }
    }

    /// Create a weight with a default value of zero if the value is invalid.
    pub fn new_or_zero(value: f64) -> Self {
        Self::new(value).unwrap_or(Self::zero())
    }

    /// Create a zero weight.
    pub const fn zero() -> Self {
        Weight(0.0)
    }

    /// Create a weight from a raw value without validation.
    ///
    /// # Safety
    /// This is const and doesn't validate. Only use with known-good constant values.
    pub const fn from_raw(value: f64) -> Self {
        Weight(value)
    }

    /// Get the inner value.
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if weight is zero.
    pub fn is_zero(&self) -> bool {
        self.0 == 0.0
    }

    /// Add to this weight.
    pub fn add(&mut self, amount: f64) {
        self.0 += amount;
        debug_assert!(self.0 >= 0.0, "Weight became negative after addition");
    }

    /// Subtract from this weight, saturating at zero.
    pub fn subtract(&mut self, amount: f64) {
        self.0 = (self.0 - amount).max(0.0);
    }
}

impl From<Weight> for f64 {
    fn from(weight: Weight) -> Self {
        weight.0
    }
}

impl fmt::Display for Weight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}", self.0)
    }
}

/// A collection of weights for moves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoveWeights(pub Vec<(Position, Weight)>);

impl MoveWeights {
    /// Create new move weights.
    pub fn new(weights: Vec<(Position, Weight)>) -> Self {
        MoveWeights(weights)
    }

    /// Get the total weight.
    pub fn total(&self) -> f64 {
        self.0.iter().map(|(_, w)| w.value()).sum()
    }

    /// Check if all weights are zero.
    pub fn all_zero(&self) -> bool {
        self.0.iter().all(|(_, w)| w.is_zero())
    }

    /// Find the position with maximum weight.
    pub fn max_weight_position(&self) -> Option<Position> {
        self.0
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(pos, _)| *pos)
    }

    /// Iterate over the (position, weight) pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(Position, Weight)> {
        self.0.iter()
    }

    /// Get the number of move options.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if there are no move options.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Entropy value (non-negative).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Entropy(f64);

impl Entropy {
    /// Create a new entropy value.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::InvalidEntropy`] if the value is negative or not finite.
    pub fn new(value: f64) -> Result<Self, crate::Error> {
        if value >= 0.0 && value.is_finite() {
            Ok(Entropy(value))
        } else {
            Err(crate::Error::InvalidEntropy { value })
        }
    }

    /// Calculate entropy from a probability distribution.
    pub fn from_distribution(probs: &[f64]) -> Self {
        Entropy(crate::utils::shannon_entropy(probs.iter().copied()))
    }

    /// Calculate entropy from weights (normalizes first).
    ///
    /// This is a convenience method that normalizes weights to probabilities
    /// and then calculates Shannon entropy. Returns zero entropy if weights
    /// cannot be normalized (e.g., all zeros).
    pub fn from_weights<I>(weights: I) -> Self
    where
        I: IntoIterator<Item = f64>,
    {
        Entropy(crate::utils::entropy_from_weights(weights))
    }

    /// Get the inner value.
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl fmt::Display for Entropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

/// A sampled move with position, label, and weighted alternatives.
#[derive(Debug, Clone, PartialEq)]
pub struct SampledMove {
    pub position: Position,
    pub label: crate::identifiers::MoveId,
    pub weights: MoveWeights,
}

impl SampledMove {
    /// Create a new sampled move.
    pub fn new(
        position: Position,
        label: crate::identifiers::MoveId,
        weights: MoveWeights,
    ) -> Self {
        SampledMove {
            position,
            label,
            weights,
        }
    }
}

/// A validated canonical board state label.
///
/// This newtype ensures that only valid canonical labels are used throughout
/// the system, providing compile-time guarantees that prevent invalid state
/// labels from propagating through the codebase.
///
/// # Examples
///
/// ```
/// use menace::tictactoe::BoardState;
/// use menace::types::CanonicalLabel;
///
/// let state = BoardState::new();
/// let ctx = state.canonical_context();
///
/// // Create from canonical context (safe, no validation needed)
/// let label = CanonicalLabel::from(&ctx);
///
/// // Parse from string (validates the format)
/// let label = CanonicalLabel::parse("........._X").unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CanonicalLabel(String);

impl CanonicalLabel {
    /// Parse and validate a canonical label from a string.
    ///
    /// This validates that the string represents a valid board state encoding.
    ///
    /// # Errors
    ///
    /// Returns an error if the label is not a valid board state encoding.
    pub fn parse(s: &str) -> Result<Self, crate::Error> {
        // Validate by attempting to parse as a board state
        crate::tictactoe::BoardState::from_label(s)?;
        Ok(CanonicalLabel(s.to_string()))
    }

    /// Create from a canonical context (unchecked, for internal use).
    ///
    /// This is safe because the context is known to contain a valid canonical encoding.
    pub(crate) fn from_context(ctx: &crate::tictactoe::board::CanonicalContext) -> Self {
        CanonicalLabel(ctx.encoding.clone())
    }

    /// Get the label as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert into the underlying String.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl AsRef<str> for CanonicalLabel {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CanonicalLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&crate::tictactoe::board::CanonicalContext> for CanonicalLabel {
    fn from(ctx: &crate::tictactoe::board::CanonicalContext) -> Self {
        Self::from_context(ctx)
    }
}

/// Board size constant for Tic-Tac-Toe.
pub const BOARD_SIZE: usize = 9;

/// Initial bead count for MENACE matchboxes.
pub const INITIAL_BEAD_COUNT: usize = 4;

/// Default reinforcement values.
pub mod reinforcement {
    use super::Weight;

    /// Default weight adjustment for a win.
    pub const WIN: Weight = Weight::from_raw(3.0);

    /// Default weight adjustment for a draw.
    pub const DRAW: Weight = Weight::from_raw(1.0);

    /// Default weight adjustment for a loss.
    pub const LOSS: Weight = Weight::from_raw(-1.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_validation() {
        assert!(Position::new(0).is_ok());
        assert!(Position::new(8).is_ok());
        assert!(Position::new(9).is_err());
        assert!(Position::new(100).is_err());
    }

    #[test]
    fn test_weight_validation() {
        assert!(Weight::new(0.0).is_ok());
        assert!(Weight::new(1.5).is_ok());
        assert!(Weight::new(-1.0).is_err());
        assert!(Weight::new(f64::NAN).is_err());
        assert!(Weight::new(f64::INFINITY).is_err());
    }

    #[test]
    fn test_weight_operations() {
        let mut weight = Weight::new(5.0).unwrap();
        weight.add(3.0);
        assert_eq!(weight.value(), 8.0);

        weight.subtract(10.0);
        assert_eq!(weight.value(), 0.0); // Saturates at zero
    }

    #[test]
    fn test_move_weights() {
        let weights = MoveWeights::new(vec![
            (Position::new(0).unwrap(), Weight::new(2.0).unwrap()),
            (Position::new(1).unwrap(), Weight::new(3.0).unwrap()),
            (Position::new(2).unwrap(), Weight::new(1.0).unwrap()),
        ]);

        assert_eq!(weights.total(), 6.0);
        assert!(!weights.all_zero());
        assert_eq!(
            weights.max_weight_position(),
            Some(Position::new(1).unwrap())
        );
    }

    #[test]
    fn test_entropy_calculation() {
        let probs = vec![0.5, 0.5];
        let entropy = Entropy::from_distribution(&probs);
        assert!((entropy.value() - std::f64::consts::LN_2).abs() < 0.001);
    }
}
