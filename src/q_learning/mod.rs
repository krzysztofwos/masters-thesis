//! Q-learning and SARSA temporal difference learning
//!
//! This module implements temporal difference (TD) learning algorithms for
//! game-playing agents. TD methods bootstrap value estimates from successor
//! states, enabling faster learning than Monte Carlo methods.
//!
//! ## Algorithms
//!
//! - **Q-learning**: Off-policy TD control that learns optimal Q* values
//! - **SARSA**: On-policy TD control that learns Q^π for the followed policy
//!
//! ## Key Differences
//!
//! | Aspect | Q-learning | SARSA |
//! |--------|------------|-------|
//! | Policy | Off-policy (learns Q*) | On-policy (learns Q^π) |
//! | Update | Uses max_a Q(s',a') | Uses actual Q(s',a') |
//! | Exploration | Can be reckless | More conservative |
//! | Convergence | To optimal policy | To followed policy |
//!
//! ## Usage Example
//!
//! ```no_run
//! use menace::q_learning::{QLearningAgent, SarsaAgent};
//!
//! // Create Q-learning agent
//! let q_agent = QLearningAgent::new(
//!     0.5,   // learning_rate
//!     0.99,  // discount_factor
//!     0.5,   // epsilon (exploration)
//!     0.995, // epsilon_decay
//!     0.01,  // min_epsilon
//!     0.0,   // q_init
//! );
//!
//! // Create SARSA agent
//! let sarsa_agent = SarsaAgent::new(
//!     0.5,   // learning_rate
//!     0.99,  // discount_factor
//!     0.5,   // epsilon (exploration)
//!     0.995, // epsilon_decay
//!     0.01,  // min_epsilon
//!     0.0,   // q_init
//! );
//! ```

pub mod agent;
pub mod q_table;
pub mod serialization;

// Public re-exports
pub use agent::{QLearningAgent, SarsaAgent};
pub use q_table::QTable;
pub use serialization::{SavedTdAgent, TdAlgorithm};
