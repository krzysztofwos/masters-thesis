//! Application layer with dependency injection container.
//!
//! This module provides the dependency injection infrastructure for the MENACE
//! application, following hexagonal architecture principles. The container
//! owns infrastructure dependencies and provides factory methods for creating
//! domain objects.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │           Application Layer (app)           │
//! │  ┌──────────────────────────────────────┐   │
//! │  │       AppContainer (DI Container)    │   │
//! │  └──────────────┬───────────────────────┘   │
//! │                 │ owns                       │
//! │                 ▼                            │
//! │  ┌──────────────────────────────────────┐   │
//! │  │  Infrastructure (adapters)           │   │
//! │  │  - MsgPackRepository                 │   │
//! │  │  - InMemoryRepository (testing)      │   │
//! │  └──────────────┬───────────────────────┘   │
//! │                 │ implements                 │
//! │                 ▼                            │
//! │  ┌──────────────────────────────────────┐   │
//! │  │  Domain Ports (ports)                │   │
//! │  │  - WorkspaceRepository trait         │   │
//! │  └──────────────┬───────────────────────┘   │
//! │                 │ used by                    │
//! │                 ▼                            │
//! │  ┌──────────────────────────────────────┐   │
//! │  │  Domain Logic                        │   │
//! │  │  - MenaceAgent                       │   │
//! │  │  - MenaceWorkspace                   │   │
//! │  └──────────────────────────────────────┘   │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ## Production
//!
//! ```
//! use menace::app::{App, AgentConfig};
//! use menace::StateFilter;
//!
//! let app = App::new();
//! let config = AgentConfig::new(StateFilter::Michie).with_seed(42);
//! let agent = app.create_agent(config)?;
//! # Ok::<(), menace::Error>(())
//! ```
//!
//! ## Testing
//!
//! ```
//! use menace::app::App;
//! use menace::adapters::InMemoryRepository;
//!
//! let app = App::for_testing()
//!     .with_repository(InMemoryRepository::new())
//!     .with_default_seed(42)
//!     .build();
//! ```

pub mod config;
pub mod container;

pub use config::AgentConfig;
pub use container::{App, AppBuilder};
