use std::collections::HashMap;

use menace::{
    active_inference::OpponentKind,
    menace::{active::PureActiveInference, learning::LearningAlgorithm},
    tictactoe::{BoardState, GameOutcome, Player, symmetry::D4Transform},
    workspace::{InitialBeadSchedule, MenaceWorkspace, RestockMode, StateFilter},
};

fn distribution_for_state(
    workspace: &MenaceWorkspace,
    label: &menace::types::CanonicalLabel,
) -> HashMap<usize, f64> {
    workspace
        .move_distribution(label)
        .expect("state should have a move distribution")
        .into_iter()
        .collect()
}

#[test]
fn pure_aif_default_o_prior_prefers_agent_wins() {
    let mut agent = PureActiveInference::new_for_player(OpponentKind::Uniform, 0.0, Player::O);
    let mut workspace = MenaceWorkspace::with_config(
        StateFilter::Both,
        RestockMode::default(),
        InitialBeadSchedule::default(),
    )
    .expect("workspace should construct");

    let mut state = BoardState::new();
    // Opponent (X) plays in the corner; agent (O) is to move.
    state = state.make_move(0).expect("X should play first move");

    let (ctx, label) = state.canonical_context_and_label();
    let canonical_label = label.clone();

    let good_move = 4; // center
    let bad_move = 1; // edge

    let good_states = vec![state];
    let good_moves = vec![good_move];
    for _ in 0..5 {
        let _ = agent.train_from_game(
            &mut workspace,
            &good_states,
            &good_moves,
            GameOutcome::Win(Player::O),
            Player::O,
        );
    }

    let bad_states = vec![state];
    let bad_moves = vec![bad_move];
    for _ in 0..5 {
        let _ = agent.train_from_game(
            &mut workspace,
            &bad_states,
            &bad_moves,
            GameOutcome::Win(Player::X),
            Player::O,
        );
    }

    let dist = distribution_for_state(&workspace, &canonical_label);
    let canonical_good = ctx.map_move_to_canonical(good_move);
    let canonical_bad = ctx.map_move_to_canonical(bad_move);

    let prob_good = dist
        .get(&canonical_good)
        .copied()
        .expect("good move should be present");
    let prob_bad = dist
        .get(&canonical_bad)
        .copied()
        .expect("bad move should be present");

    assert!(
        prob_good > prob_bad,
        "O-agent should prefer winning actions (good: {prob_good}, bad: {prob_bad})"
    );
}

#[test]
fn pure_aif_canonical_mapping_handles_rotations() {
    let mut agent = PureActiveInference::new_for_player(OpponentKind::Uniform, 0.0, Player::X);
    let mut workspace = MenaceWorkspace::with_config(
        StateFilter::Michie,
        RestockMode::default(),
        InitialBeadSchedule::default(),
    )
    .expect("workspace should construct");

    let mut state = BoardState::new();
    state = state.make_move(2).expect("X plays top-right");
    state = state.make_move(4).expect("O plays center");

    let (ctx, label) = state.canonical_context_and_label();
    assert!(
        ctx.transform != D4Transform::identity(),
        "test requires a non-identity canonical transform"
    );
    let canonical_label = label.clone();

    let good_move_original = 0;
    let bad_move_original = 3;

    let canonical_good = ctx.map_move_to_canonical(good_move_original);
    let canonical_bad = ctx.map_move_to_canonical(bad_move_original);

    let good_states = vec![state];
    let good_moves = vec![good_move_original];
    for _ in 0..5 {
        let _ = agent.train_from_game(
            &mut workspace,
            &good_states,
            &good_moves,
            GameOutcome::Win(Player::X),
            Player::X,
        );
    }

    let bad_states = vec![state];
    let bad_moves = vec![bad_move_original];
    for _ in 0..5 {
        let _ = agent.train_from_game(
            &mut workspace,
            &bad_states,
            &bad_moves,
            GameOutcome::Win(Player::O),
            Player::X,
        );
    }

    let dist = distribution_for_state(&workspace, &canonical_label);
    let prob_good = dist
        .get(&canonical_good)
        .copied()
        .expect("good move should be present");
    let prob_bad = dist
        .get(&canonical_bad)
        .copied()
        .expect("bad move should be present");

    assert!(
        prob_good > prob_bad,
        "EFE update should favour canonical good move even under rotations (good: {prob_good}, bad: {prob_bad})"
    );
}
