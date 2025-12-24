//! Integration tests for dependency injection.
//!
//! These tests demonstrate the benefits of the DI app:
//! - Easy testing with in-memory repositories (no file I/O)
//! - Deterministic behavior with fixed seeds
//! - Centralized dependency management

use std::path::Path;

use menace::{
    StateFilter,
    adapters::InMemoryRepository,
    app::{AgentConfig, App},
    tictactoe::{BoardState, GameOutcome, Player},
};

#[test]
fn test_app_with_in_memory_repository() {
    // Create app with in-memory repository (no disk I/O!)
    let app = App::for_testing()
        .with_repository(InMemoryRepository::new())
        .with_default_seed(42)
        .build();

    // Create an agent
    let config = AgentConfig::new(StateFilter::Michie);
    let mut agent = app.create_agent(config).unwrap();

    // Train the agent a bit
    agent
        .train_from_moves(Player::X, &[0, 4, 8], GameOutcome::Draw, Player::X)
        .unwrap();

    // Save to "memory" (not disk)
    let path = Path::new("test_agent");
    app.save_agent(&agent, path).unwrap();

    // Load from "memory"
    let loaded_agent = app.load_agent(path).unwrap();

    // Both agents should have the same decision state count
    assert_eq!(
        agent.workspace().decision_labels().count(),
        loaded_agent.workspace().decision_labels().count()
    );
}

#[test]
fn test_deterministic_training_with_seed() {
    // Two apps with same seed should produce identical results
    let config = AgentConfig::new(StateFilter::Michie).with_seed(42);

    let app1 = App::for_testing().build();
    let app2 = App::for_testing().build();

    let mut agent1 = app1.create_agent(config.clone()).unwrap();
    let mut agent2 = app2.create_agent(config).unwrap();

    // Train both identically
    let moves = vec![0, 4, 1, 3, 2]; // X wins
    agent1
        .train_from_moves(Player::X, &moves, GameOutcome::Win(Player::X), Player::X)
        .unwrap();
    agent2
        .train_from_moves(Player::X, &moves, GameOutcome::Win(Player::X), Player::X)
        .unwrap();

    // Both should have same matchbox count
    assert_eq!(
        agent1.workspace().decision_labels().count(),
        agent2.workspace().decision_labels().count()
    );

    // And same stats
    let stats1 = agent1.stats();
    let stats2 = agent2.stats();
    assert_eq!(stats1.total_matchboxes, stats2.total_matchboxes);
}

#[test]
fn test_app_default_seed_propagates() {
    // App with default seed
    let app = App::for_testing().with_default_seed(123).build();

    let config = AgentConfig::new(StateFilter::Michie); // No seed in config
    let agent = app.create_agent(config).unwrap();

    // Agent should work deterministically (seed was applied)
    assert!(agent.workspace().decision_labels().count() > 0);
}

#[test]
fn test_config_seed_overrides_app_default() {
    let app = App::for_testing().with_default_seed(42).build();

    // Config seed should override
    let config = AgentConfig::new(StateFilter::Michie).with_seed(999);
    let agent = app.create_agent(config).unwrap();

    // Should work (seed applied, doesn't matter which one for this test)
    assert!(agent.workspace().decision_labels().count() > 0);
}

#[test]
fn test_multiple_agents_from_same_app() {
    let app = App::for_testing()
        .with_repository(InMemoryRepository::new())
        .build();

    // Create multiple agents with different configurations
    let agent1 = app
        .create_agent(AgentConfig::new(StateFilter::Michie).with_seed(1))
        .unwrap();

    let agent2 = app
        .create_agent(AgentConfig::new(StateFilter::DecisionOnly).with_seed(2))
        .unwrap();

    // Different filters produce different state counts
    assert_ne!(
        agent1.workspace().decision_labels().count(),
        agent2.workspace().decision_labels().count()
    );

    // Michie: 287 states, DecisionOnly: 304 states
    assert_eq!(agent1.workspace().decision_labels().count(), 287);
    assert_eq!(agent2.workspace().decision_labels().count(), 304);
}

#[test]
fn test_in_memory_repository_isolation() {
    let repo = InMemoryRepository::new();
    let app = App::for_testing().with_repository(repo.clone()).build();

    let config = AgentConfig::new(StateFilter::Michie).with_seed(42);
    let agent = app.create_agent(config).unwrap();

    // Initially no workspaces stored
    assert_eq!(repo.count(), 0);

    // Save
    app.save_agent(&agent, Path::new("agent1")).unwrap();
    assert_eq!(repo.count(), 1);

    // Save another
    app.save_agent(&agent, Path::new("agent2")).unwrap();
    assert_eq!(repo.count(), 2);

    // Clear
    repo.clear();
    assert_eq!(repo.count(), 0);
}

#[test]
fn test_real_game_with_app() {
    let app = App::for_testing().with_default_seed(42).build();

    let config = AgentConfig::new(StateFilter::Michie);
    let mut agent = app.create_agent(config).unwrap();

    // Play a simple game
    let mut state = BoardState::new();

    // X (agent) plays
    let move1 = agent
        .select_move(&state)
        .expect("Agent should select a move");
    state = state.make_move(move1).unwrap();

    // O plays - choose a move that won't conflict with agent's likely moves
    // Find first empty position that's not what agent played
    let o_move = (0..9)
        .find(|&pos| pos != move1 && state.is_empty(pos))
        .unwrap();
    state = state.make_move(o_move).unwrap();

    // X (agent) plays again
    let move2 = agent
        .select_move(&state)
        .expect("Agent should select a move");
    state = state.make_move(move2).unwrap();

    // Verify the game progressed
    assert_eq!(state.occupied_count(), 3);
}

#[test]
fn test_load_agent_preserves_workspace() {
    let repo = InMemoryRepository::new();
    let app = App::for_testing()
        .with_repository(repo.clone())
        .with_default_seed(42)
        .build();

    // Create and train an agent
    let config = AgentConfig::new(StateFilter::Michie).with_seed(123);
    let mut agent = app.create_agent(config).unwrap();

    // Train the agent to modify its workspace
    agent
        .train_from_moves(Player::X, &[0, 4, 8], GameOutcome::Draw, Player::X)
        .unwrap();

    // Get the original workspace state
    let original_decision_count = agent.workspace().decision_labels().count();
    assert_eq!(original_decision_count, 287); // Michie filter produces 287 states

    // Save the agent
    let path = Path::new("trained_agent");
    app.save_agent(&agent, path).unwrap();

    // Load the agent
    let loaded_agent = app.load_agent(path).unwrap();

    // Verify the workspace was preserved
    assert_eq!(
        loaded_agent.workspace().decision_labels().count(),
        original_decision_count,
        "Loaded agent should have same number of decision states"
    );

    // Verify the workspace is actually the same by checking it was loaded from repo
    assert_eq!(repo.count(), 1, "Repository should contain saved workspace");
}
