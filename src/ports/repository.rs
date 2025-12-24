//! Repository port for workspace persistence.
//!
//! This module defines the trait boundary between the domain and infrastructure
//! layers for workspace storage and retrieval.

use std::path::Path;

use crate::{Result, workspace::MenaceWorkspace};

/// Port for persisting and loading MENACE workspaces.
///
/// This trait abstracts the storage mechanism, allowing different implementations
/// (MessagePack, JSON, database, etc.) without coupling the domain logic to
/// specific serialization formats.
///
/// # Examples
///
/// ```no_run
/// use menace::ports::WorkspaceRepository;
/// use menace::MenaceWorkspace;
/// use std::path::Path;
///
/// fn save_workspace<R: WorkspaceRepository>(
///     repo: &R,
///     workspace: &MenaceWorkspace,
///     path: &Path,
/// ) -> menace::Result<()> {
///     repo.save(workspace, path)
/// }
/// ```
pub trait WorkspaceRepository {
    /// Save a workspace to persistent storage.
    ///
    /// # Arguments
    ///
    /// * `workspace` - The workspace to save
    /// * `path` - The location where the workspace should be saved
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The path cannot be created or written to
    /// - Serialization fails
    /// - I/O errors occur during writing
    fn save(&self, workspace: &MenaceWorkspace, path: &Path) -> Result<()>;

    /// Load a workspace from persistent storage.
    ///
    /// # Arguments
    ///
    /// * `path` - The location from which to load the workspace
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file does not exist or cannot be read
    /// - The file format is invalid or corrupted
    /// - Deserialization fails
    fn load(&self, path: &Path) -> Result<MenaceWorkspace>;
}
