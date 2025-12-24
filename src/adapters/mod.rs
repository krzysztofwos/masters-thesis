//! Adapters implementing domain ports.
//!
//! This module contains infrastructure implementations of the traits defined
//! in the ports module. Following hexagonal architecture, adapters depend on
//! domain ports, not the other way around.

pub mod in_memory_repository;
pub mod msgpack_repository;

pub use in_memory_repository::InMemoryRepository;
pub use msgpack_repository::MsgPackRepository;
