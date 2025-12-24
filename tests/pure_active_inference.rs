//! Tests for Pure Active Inference implementation

use menace::{
    menace::MenaceAgent,
    pipeline::{MenaceLearner, RandomLearner, TrainingConfig, TrainingPipeline},
    tictactoe::Player,
};

#[test]
fn test_pure_aif_construction() {
    // Test that Pure AIF agent can be constructed
    let agent = MenaceAgent::builder()
        .pure_active_inference_uniform(0.5)
        .seed(42)
        .build();

    assert!(agent.is_ok(), "Pure AIF agent construction should succeed");

    let agent = agent.unwrap();
    assert_eq!(agent.algorithm_name(), "Pure Active Inference (Uniform)");
}

#[test]
fn test_pure_aif_learns_from_outcomes() {
    // Test that Pure AIF updates beliefs after observing outcomes
    let agent = MenaceAgent::builder()
        .pure_active_inference_uniform(0.5)
        .seed(7)
        .build()
        .expect("agent construction should succeed");

    let mut learner = MenaceLearner::new(agent, "PureAIF".to_string());
    let mut opponent = RandomLearner::new("Random".to_string());

    // Train for a few games
    let config = TrainingConfig {
        num_games: 10,
        seed: Some(42),
        agent_player: Player::X,
        first_player: Player::X,
    };

    let mut pipeline = TrainingPipeline::new(config);
    let result = pipeline.run(&mut learner, &mut opponent);

    assert!(result.is_ok(), "Training should complete without errors");

    let result = result.unwrap();
    assert_eq!(result.total_games, 10, "Should have played 10 games");

    // Agent should have some wins or draws after 10 games
    assert!(
        result.wins + result.draws > 0,
        "Pure AIF should achieve at least some wins or draws vs random play"
    );
}

#[test]
fn test_pure_aif_vs_oracle_difference() {
    // Test that Pure AIF (learning) differs from Oracle (perfect knowledge)
    let pure_agent = MenaceAgent::builder()
        .pure_active_inference_uniform(0.5)
        .seed(42)
        .build()
        .expect("pure agent construction should succeed");

    let oracle_agent = MenaceAgent::builder()
        .oracle_active_inference_uniform(0.5)
        .seed(42)
        .build()
        .expect("oracle agent construction should succeed");

    // Both should construct successfully
    assert_eq!(
        pure_agent.algorithm_name(),
        "Pure Active Inference (Uniform)"
    );
    assert_eq!(
        oracle_agent.algorithm_name(),
        "Oracle Active Inference (Uniform)"
    );

    // They should use different stats
    let pure_stats = pure_agent.algorithm_stats();
    let oracle_stats = oracle_agent.algorithm_stats();

    // Pure AIF tracks action-outcome pairs
    assert!(
        pure_stats.contains_key("tracked_action_pairs"),
        "Pure AIF should track action-outcome pairs"
    );

    // Oracle is marked as oracle
    assert!(
        oracle_stats.contains_key("is_oracle"),
        "Oracle should be marked as is_oracle"
    );
}

#[test]
fn test_pure_aif_adversarial() {
    // Test that Pure AIF works with adversarial opponent model
    let agent = MenaceAgent::builder()
        .pure_active_inference_adversarial(0.7)
        .seed(123)
        .build();

    assert!(
        agent.is_ok(),
        "Pure AIF with adversarial opponent should construct"
    );

    let agent = agent.unwrap();
    assert_eq!(
        agent.algorithm_name(),
        "Pure Active Inference (Adversarial)"
    );
}

#[test]
fn test_pure_aif_minimax() {
    // Test that Pure AIF works with minimax opponent model
    let agent = MenaceAgent::builder()
        .pure_active_inference_minimax()
        .seed(456)
        .build();

    assert!(
        agent.is_ok(),
        "Pure AIF with minimax opponent should construct"
    );

    let agent = agent.unwrap();
    assert_eq!(agent.algorithm_name(), "Pure Active Inference (Minimax)");
}

#[test]
fn test_pure_aif_custom_preferences() {
    use menace::active_inference::{OpponentKind, PreferenceModel};

    // Test that Pure AIF works with custom preferences
    let prefs = PreferenceModel::from_probabilities(0.9, 0.5, 0.1)
        .with_epistemic_weight(0.3)
        .with_policy_lambda(2.0);

    let agent = MenaceAgent::builder()
        .pure_active_inference_custom(OpponentKind::Uniform, prefs)
        .seed(789)
        .build();

    assert!(
        agent.is_ok(),
        "Pure AIF with custom preferences should construct"
    );
}
