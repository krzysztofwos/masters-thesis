//! Ports (trait boundaries) for external dependencies.
//!
//! This module defines the interfaces between the domain layer and infrastructure.
//! Following hexagonal architecture, these traits are owned by the domain and
//! implemented by adapters in the infrastructure layer.

pub mod learner;
pub mod observer;
pub mod repository;

pub use learner::Learner;
pub use observer::Observer;
pub use repository::WorkspaceRepository;
