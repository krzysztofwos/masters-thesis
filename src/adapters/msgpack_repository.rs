//! MessagePack implementation of workspace repository.
//!
//! This adapter implements the WorkspaceRepository port using rmp_serde for
//! compact binary serialization.

use std::{fs::File, path::Path};

use crate::{Result, error::Error, ports::WorkspaceRepository, workspace::MenaceWorkspace};

/// MessagePack-based workspace repository.
///
/// Provides persistent storage using the MessagePack binary format via rmp_serde.
/// This format offers good compression and fast serialization/deserialization.
///
/// # Examples
///
/// ```no_run
/// use menace::adapters::MsgPackRepository;
/// use menace::ports::WorkspaceRepository;
/// use menace::{MenaceWorkspace, StateFilter};
/// use std::path::Path;
///
/// let repo = MsgPackRepository;
/// let workspace = MenaceWorkspace::new(StateFilter::Michie)?;
///
/// // Save workspace
/// repo.save(&workspace, Path::new("trained.msgpack"))?;
///
/// // Load workspace
/// let loaded = repo.load(Path::new("trained.msgpack"))?;
/// # Ok::<(), menace::Error>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MsgPackRepository;

impl MsgPackRepository {
    /// Create a new MessagePack repository.
    pub fn new() -> Self {
        Self
    }
}

impl WorkspaceRepository for MsgPackRepository {
    fn save(&self, workspace: &MenaceWorkspace, path: &Path) -> Result<()> {
        let mut file = File::create(path).map_err(|source| Error::Io {
            operation: format!("create file {path:?}"),
            source,
        })?;

        rmp_serde::encode::write(&mut file, workspace).map_err(|e| {
            Error::SerializationContext {
                operation: "serialize workspace to MessagePack".to_string(),
                message: e.to_string(),
            }
        })?;

        Ok(())
    }

    fn load(&self, path: &Path) -> Result<MenaceWorkspace> {
        let file = File::open(path).map_err(|source| Error::Io {
            operation: format!("open file {path:?}"),
            source,
        })?;

        let workspace =
            rmp_serde::decode::from_read(&file).map_err(|e| Error::SerializationContext {
                operation: "deserialize workspace from MessagePack".to_string(),
                message: e.to_string(),
            })?;

        Ok(workspace)
    }
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::StateFilter;

    #[test]
    fn test_msgpack_roundtrip() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test_workspace.msgpack");

        let repo = MsgPackRepository::new();
        let workspace =
            MenaceWorkspace::new(StateFilter::Michie).expect("Failed to create workspace");

        repo.save(&workspace, &file_path).expect("Failed to save");
        let loaded = repo.load(&file_path).expect("Failed to load");

        assert_eq!(
            workspace.decision_labels().count(),
            loaded.decision_labels().count()
        );
    }

    #[test]
    fn test_load_nonexistent_returns_error() {
        let repo = MsgPackRepository::new();
        let result = repo.load(Path::new("/tmp/nonexistent_12345.msgpack"));
        assert!(result.is_err());
    }

    #[test]
    fn test_save_to_invalid_path_returns_error() {
        let repo = MsgPackRepository::new();
        let workspace =
            MenaceWorkspace::new(StateFilter::Michie).expect("Failed to create workspace");
        let result = repo.save(&workspace, Path::new("/invalid_dir_12345/file.msgpack"));
        assert!(result.is_err());
    }
}
