//! Domain identifier types for MENACE learning states and moves.
//!
//! These types provide type-safe wrappers around string identifiers used throughout
//! the MENACE decision-making system.

use std::{borrow::Borrow, fmt};

use serde::{Deserialize, Serialize};

/// Unique identifier for a board state in the decision tree.
///
/// StateIds are used to uniquely identify board positions in the game tree.
/// They are typically canonical string representations of board states.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StateId(String);

impl StateId {
    /// Create a new state identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::identifiers::StateId;
    ///
    /// let state = StateId::new("X........_O");
    /// assert_eq!(state.as_str(), "X........_O");
    /// ```
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Get the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert the identifier into its inner String.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq<&str> for StateId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<StateId> for &str {
    fn eq(&self, other: &StateId) -> bool {
        *self == other.as_str()
    }
}

impl Borrow<str> for StateId {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<String> for StateId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for StateId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl AsRef<str> for StateId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

/// Unique identifier for a move (state transition) in the decision tree.
///
/// MoveIds uniquely identify transitions between states, typically encoding
/// both the source state and the move position.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MoveId(String);

impl MoveId {
    /// Create a new move identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use menace::identifiers::MoveId;
    ///
    /// let move_id = MoveId::new("X........_O_4");
    /// assert_eq!(move_id.as_str(), "X........_O_4");
    /// ```
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Get the identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert the identifier into its inner String.
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl fmt::Display for MoveId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq<&str> for MoveId {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<MoveId> for &str {
    fn eq(&self, other: &MoveId) -> bool {
        *self == other.as_str()
    }
}

impl Borrow<str> for MoveId {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<String> for MoveId {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for MoveId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl AsRef<str> for MoveId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
