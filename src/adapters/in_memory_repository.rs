//! In-memory workspace repository for testing.
//!
//! This adapter provides a pure in-memory implementation of WorkspaceRepository,
//! enabling fast tests without any file system I/O.

use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
};

use crate::{Result, error::Error, ports::WorkspaceRepository, workspace::MenaceWorkspace};

/// In-memory repository for testing.
///
/// Stores workspaces in memory using a shared HashMap, avoiding file system I/O
/// entirely. Perfect for fast, isolated tests.
///
/// # Examples
///
/// ```
/// use menace::adapters::InMemoryRepository;
/// use menace::ports::WorkspaceRepository;
/// use menace::{MenaceWorkspace, StateFilter};
/// use std::path::Path;
///
/// let repo = InMemoryRepository::new();
/// let workspace = MenaceWorkspace::new(StateFilter::Michie)?;
///
/// // Save to "memory" (not disk)
/// repo.save(&workspace, Path::new("test_workspace"))?;
///
/// // Load from "memory"
/// let loaded = repo.load(Path::new("test_workspace"))?;
/// # Ok::<(), menace::Error>(())
/// ```
///
/// # Thread Safety
///
/// This repository is thread-safe and can be safely cloned and shared across
/// threads. All clones share the same underlying storage.
#[derive(Clone)]
pub struct InMemoryRepository {
    storage: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

impl InMemoryRepository {
    /// Create a new empty in-memory repository.
    pub fn new() -> Self {
        Self {
            storage: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get the number of workspaces currently stored.
    ///
    /// Useful for testing to verify save operations occurred.
    pub fn count(&self) -> usize {
        self.storage.lock().unwrap().len()
    }

    /// Clear all stored workspaces.
    ///
    /// Useful for resetting state between tests.
    pub fn clear(&self) {
        self.storage.lock().unwrap().clear();
    }

    /// Check if a workspace exists at the given path.
    pub fn contains(&self, path: &Path) -> bool {
        let key = path.to_string_lossy().to_string();
        self.storage.lock().unwrap().contains_key(&key)
    }
}

impl Default for InMemoryRepository {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkspaceRepository for InMemoryRepository {
    fn save(&self, workspace: &MenaceWorkspace, path: &Path) -> Result<()> {
        let key = path.to_string_lossy().to_string();

        let bytes = rmp_serde::to_vec(workspace).map_err(|e| Error::SerializationContext {
            operation: "serialize workspace for in-memory storage".to_string(),
            message: e.to_string(),
        })?;

        self.storage.lock().unwrap().insert(key, bytes);
        Ok(())
    }

    fn load(&self, path: &Path) -> Result<MenaceWorkspace> {
        let key = path.to_string_lossy().to_string();
        let storage = self.storage.lock().unwrap();

        let bytes = storage.get(&key).ok_or_else(|| Error::Io {
            operation: format!("load workspace from in-memory storage at {path:?}"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "key not found in memory"),
        })?;

        rmp_serde::from_slice(bytes).map_err(|e| Error::SerializationContext {
            operation: "deserialize workspace from in-memory storage".to_string(),
            message: e.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StateFilter;

    #[test]
    fn test_in_memory_save_and_load() {
        let repo = InMemoryRepository::new();
        let workspace = MenaceWorkspace::new(StateFilter::Michie).unwrap();

        let path = Path::new("test_workspace");

        // Initially empty
        assert_eq!(repo.count(), 0);
        assert!(!repo.contains(path));

        // Save
        repo.save(&workspace, path).unwrap();
        assert_eq!(repo.count(), 1);
        assert!(repo.contains(path));

        // Load
        let loaded = repo.load(path).unwrap();
        assert_eq!(
            workspace.decision_labels().count(),
            loaded.decision_labels().count()
        );
    }

    #[test]
    fn test_load_nonexistent_returns_error() {
        let repo = InMemoryRepository::new();
        let result = repo.load(Path::new("nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_clear_removes_all() {
        let repo = InMemoryRepository::new();
        let workspace = MenaceWorkspace::new(StateFilter::Michie).unwrap();

        repo.save(&workspace, Path::new("ws1")).unwrap();
        repo.save(&workspace, Path::new("ws2")).unwrap();
        assert_eq!(repo.count(), 2);

        repo.clear();
        assert_eq!(repo.count(), 0);
    }

    #[test]
    fn test_clone_shares_storage() {
        let repo1 = InMemoryRepository::new();
        let repo2 = repo1.clone();

        let workspace = MenaceWorkspace::new(StateFilter::Michie).unwrap();
        let path = Path::new("shared");

        // Save via repo1
        repo1.save(&workspace, path).unwrap();

        // Load via repo2 (should see the same data)
        let loaded = repo2.load(path).unwrap();
        assert_eq!(
            workspace.decision_labels().count(),
            loaded.decision_labels().count()
        );

        // Both should report same count
        assert_eq!(repo1.count(), 1);
        assert_eq!(repo2.count(), 1);
    }
}
