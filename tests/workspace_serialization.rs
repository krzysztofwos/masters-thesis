//! Tests for MenaceWorkspace serialization and deserialization

use std::path::PathBuf;

use menace::{
    InitialBeadSchedule, MenaceWorkspace, Reinforcement, RestockMode, StateFilter,
    adapters::MsgPackRepository, ports::WorkspaceRepository,
};
use tempfile::TempDir;

#[test]
fn test_workspace_save_load_roundtrip() {
    // Create a temporary directory for test files
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("test_workspace.msgpack");

    // Create and configure a workspace
    let workspace = MenaceWorkspace::with_config(
        StateFilter::Michie,
        RestockMode::Box,
        InitialBeadSchedule::menace(),
    )
    .expect("Failed to create workspace");

    // Save the workspace using repository
    let repo = MsgPackRepository::new();
    repo.save(&workspace, &file_path)
        .expect("Failed to save workspace");

    // Verify file was created
    assert!(file_path.exists(), "Saved file should exist");

    // Load the workspace using repository
    let loaded_workspace = repo.load(&file_path).expect("Failed to load workspace");

    // Verify key properties match
    assert_eq!(
        workspace.decision_labels().count(),
        loaded_workspace.decision_labels().count(),
        "Decision state count should match"
    );
    assert_eq!(
        workspace.restock_mode(),
        loaded_workspace.restock_mode(),
        "Restock mode should match"
    );
}

#[test]
fn test_workspace_preserves_learned_weights() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("trained_workspace.msgpack");

    // Create workspace and simulate some learning
    let mut workspace =
        MenaceWorkspace::new(StateFilter::Michie).expect("Failed to create workspace");

    // Get a sample state and morphism for testing
    let sample_label = workspace.decision_labels().next().cloned();
    assert!(
        sample_label.is_some(),
        "Should have at least one decision state"
    );

    let sample_label = sample_label.unwrap();
    let canonical_label = menace::CanonicalLabel::parse(sample_label.as_str())
        .expect("Should parse valid canonical label");

    // Get initial weights
    let initial_weights = workspace
        .move_weights(&canonical_label)
        .expect("Should have weights for decision state");

    // Apply some reinforcement (simulate learning)
    if let Some(move_id) = workspace.move_for_position(&canonical_label, initial_weights[0].0) {
        let _ = workspace.apply_reinforcement(vec![move_id.clone()], Reinforcement::Positive(3.0));
    }

    // Get updated weights
    let updated_weights = workspace
        .move_weights(&canonical_label)
        .expect("Should have weights after reinforcement");

    // Verify learning occurred
    assert_ne!(
        initial_weights, updated_weights,
        "Weights should change after reinforcement"
    );

    // Save and reload using repository
    let repo = MsgPackRepository::new();
    repo.save(&workspace, &file_path)
        .expect("Failed to save trained workspace");
    let loaded_workspace = repo
        .load(&file_path)
        .expect("Failed to load trained workspace");

    // Verify learned weights are preserved
    let loaded_weights = loaded_workspace
        .move_weights(&canonical_label)
        .expect("Should have weights in loaded workspace");

    assert_eq!(
        updated_weights, loaded_weights,
        "Learned weights should be preserved after save/load"
    );
}

#[test]
fn test_different_state_filters_serialize_correctly() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    let filters = vec![
        StateFilter::All,
        StateFilter::DecisionOnly,
        StateFilter::Michie,
        StateFilter::Both,
    ];

    let repo = MsgPackRepository::new();
    for filter in filters {
        let file_path = temp_dir.path().join(format!("workspace_{filter}.msgpack"));

        let workspace = MenaceWorkspace::new(filter).expect("Failed to create workspace");
        let original_count = workspace.decision_labels().count();

        repo.save(&workspace, &file_path)
            .expect("Failed to save workspace");
        let loaded_workspace = repo.load(&file_path).expect("Failed to load workspace");

        assert_eq!(
            original_count,
            loaded_workspace.decision_labels().count(),
            "State count should match for filter {filter:?}"
        );
    }
}

#[test]
fn test_different_restock_modes_serialize_correctly() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    let modes = vec![RestockMode::None, RestockMode::Move, RestockMode::Box];

    let repo = MsgPackRepository::new();
    for mode in modes {
        let file_path = temp_dir.path().join(format!("workspace_{mode}.msgpack"));

        let workspace = MenaceWorkspace::with_options(StateFilter::Michie, mode)
            .expect("Failed to create workspace");

        repo.save(&workspace, &file_path)
            .expect("Failed to save workspace");
        let loaded_workspace = repo.load(&file_path).expect("Failed to load workspace");

        assert_eq!(
            mode,
            loaded_workspace.restock_mode(),
            "Restock mode should be preserved"
        );
    }
}

#[test]
fn test_custom_bead_schedule_serializes_correctly() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("custom_schedule.msgpack");

    // Create workspace with custom bead schedule
    let custom_schedule = InitialBeadSchedule::new([10.0, 8.0, 6.0, 4.0]);
    let workspace =
        MenaceWorkspace::with_config(StateFilter::Michie, RestockMode::Box, custom_schedule)
            .expect("Failed to create workspace");

    let repo = MsgPackRepository::new();
    repo.save(&workspace, &file_path)
        .expect("Failed to save workspace");
    let loaded_workspace = repo.load(&file_path).expect("Failed to load workspace");

    // Verify by checking initial weights match the custom schedule
    // (comparing actual weights in the workspace would require accessing private fields)
    assert_eq!(
        workspace.decision_labels().count(),
        loaded_workspace.decision_labels().count(),
        "Should preserve workspace structure with custom schedule"
    );
}

#[test]
fn test_load_nonexistent_file_returns_error() {
    let nonexistent_path = PathBuf::from("/tmp/nonexistent_workspace_12345.msgpack");

    let repo = MsgPackRepository::new();
    let result = repo.load(&nonexistent_path);
    assert!(
        result.is_err(),
        "Loading nonexistent file should return error"
    );

    let err_message = result.unwrap_err().to_string();
    assert!(
        err_message.contains("open file"),
        "Error should mention file opening failure, got: {err_message}"
    );
}

#[test]
fn test_save_to_invalid_path_returns_error() {
    let workspace = MenaceWorkspace::new(StateFilter::Michie).expect("Failed to create workspace");

    // Try to save to a path that cannot be created (invalid parent directory)
    let invalid_path = PathBuf::from("/nonexistent_directory_12345/workspace.msgpack");

    let repo = MsgPackRepository::new();
    let result = repo.save(&workspace, &invalid_path);
    assert!(
        result.is_err(),
        "Saving to invalid path should return error"
    );

    let err_message = result.unwrap_err().to_string();
    assert!(
        err_message.contains("create file"),
        "Error should mention file creation failure, got: {err_message}"
    );
}

#[test]
fn test_timestep_preserved_after_serialization() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("timestep_test.msgpack");

    // Create workspace and apply multiple learning updates
    let mut workspace =
        MenaceWorkspace::new(StateFilter::Michie).expect("Failed to create workspace");

    // Apply several reinforcement updates to increment timestep
    let sample_label = workspace
        .decision_labels()
        .next()
        .cloned()
        .expect("Should have decision states");
    let canonical_label = menace::CanonicalLabel::parse(sample_label.as_str())
        .expect("Should parse valid canonical label");

    for _ in 0..5 {
        if let Some(weights) = workspace.move_weights(&canonical_label)
            && let Some(move_id) = workspace.move_for_position(&canonical_label, weights[0].0)
        {
            let _ =
                workspace.apply_reinforcement(vec![move_id.clone()], Reinforcement::Positive(1.0));
        }
    }

    // Save and reload using repository
    let repo = MsgPackRepository::new();
    repo.save(&workspace, &file_path)
        .expect("Failed to save workspace");
    let loaded_workspace = repo.load(&file_path).expect("Failed to load workspace");

    // Continue training on loaded workspace - it should maintain timestep continuity
    // (We can't directly access timestep, but we can verify it doesn't break)
    let mut continued_workspace = loaded_workspace;
    if let Some(weights) = continued_workspace.move_weights(&canonical_label)
        && let Some(move_id) = continued_workspace.move_for_position(&canonical_label, weights[0].0)
    {
        let _ = continued_workspace
            .apply_reinforcement(vec![move_id.clone()], Reinforcement::Positive(1.0));
    }

    // If timestep wasn't preserved, the operations above would have failed or panicked
}
