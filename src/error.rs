//! Error types for the MENACE crate

use thiserror::Error;

/// Main error type for the MENACE crate
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    #[error("invalid move: position {position} is already occupied")]
    InvalidMove { position: usize },

    #[error("game already over")]
    GameOver,

    #[error("no valid moves available")]
    NoValidMoves,

    #[error("board string too short: expected {expected} cells, got {got} in '{context}'")]
    InvalidBoardLength {
        expected: usize,
        got: usize,
        context: String,
    },

    #[error("invalid character '{character}' at position {position} in '{context}'")]
    InvalidCellCharacter {
        character: char,
        position: usize,
        context: String,
    },

    #[error("invalid piece counts: X={x_count}, O={o_count} (must be equal or X ahead by 1)")]
    InvalidPieceCounts { x_count: usize, o_count: usize },

    #[error("invalid label format '{label}' (expected format: '{expected}')")]
    InvalidLabelFormat { label: String, expected: String },

    #[error("missing {part} in label '{label}'")]
    MissingLabelPart { part: String, label: String },

    #[error("invalid player '{player}' in label '{label}' (expected 'X' or 'O')")]
    InvalidPlayerString { player: String, label: String },

    #[error("invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("trajectory has no states")]
    EmptyTrajectory,

    #[error("failed to {operation}: {source}")]
    Io {
        operation: String,
        #[source]
        source: std::io::Error,
    },

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("failed to {operation}: {message}")]
    SerializationContext { operation: String, message: String },

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("position {position} is out of bounds (must be 0-8)")]
    InvalidPosition { position: usize },

    #[error("weight {value} must be non-negative and finite")]
    InvalidWeight { value: f64 },

    #[error("entropy {value} must be non-negative and finite")]
    InvalidEntropy { value: f64 },

    #[error("illegal move generated at position {position} in game tree: {context}")]
    IllegalGeneratedMove { position: usize, context: String },

    #[error("legal move from legal_moves() failed unexpectedly: {message}")]
    LegalMoveFailed { message: String },

    #[error("progress bar template error: {message}")]
    ProgressBarTemplate { message: String },

    #[error("invalid state filter '{input}'. Expected one of: {expected}")]
    ParseStateFilter { input: String, expected: String },

    #[error("invalid restock mode '{input}'. Expected one of: {expected}")]
    ParseRestockMode { input: String, expected: String },

    #[error("invalid bead schedule '{input}': {reason}")]
    ParseBeadSchedule { input: String, reason: String },

    #[error("matchbox for state '{label}' is depleted and restock mode is 'none'")]
    DepletedMatchbox { label: String },

    #[error("internal consistency error: move '{move_id}' has weight but no source state")]
    MissingMoveSource { move_id: String },

    #[error("non-terminal state '{state}' has no available actions")]
    NoActionsAvailable { state: String },

    #[error("opponent model '{opponent}' failed to provide required data: {context}")]
    OpponentModelError { opponent: String, context: String },
}

/// Convenience type alias for Results using the crate's Error type
pub type Result<T> = std::result::Result<T, Error>;

impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Error::Io {
            operation: "IO operation".to_string(),
            source,
        }
    }
}
