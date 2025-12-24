use std::collections::HashMap;

use menace::{
    Error,
    types::CanonicalLabel,
    workspace::{MenaceWorkspace, RestockMode, StateFilter},
};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn sampling_restocks_empty_box_in_box_mode() {
    // Build a workspace in Box restock mode
    let mut ws = MenaceWorkspace::with_options(StateFilter::Michie, RestockMode::Box).unwrap();

    // Pick a decision state
    let lbl_str = ws
        .decision_labels()
        .next()
        .expect("at least one decision state")
        .clone();
    let lbl = CanonicalLabel::parse(&lbl_str).unwrap();
    let state = ws.state(&lbl).unwrap();

    // Zero out all move weights for this state
    let mut zeros = HashMap::new();
    for mv in state.legal_moves() {
        zeros.insert(mv, 0.0_f64);
    }
    ws.set_move_weights(&lbl, &zeros);

    // Now sampling should restock the box and succeed
    let mut rng = StdRng::seed_from_u64(42);
    let sampled = ws
        .sample_move(&lbl, &mut rng)
        .expect("sampling should succeed after auto-restock");
    assert!(
        !sampled.weights.is_empty(),
        "restocked matchbox should yield a distribution"
    );
}

#[test]
fn sampling_errors_when_box_depleted_in_none_mode() {
    let mut ws = MenaceWorkspace::with_options(StateFilter::Michie, RestockMode::None).unwrap();

    let lbl_str = ws
        .decision_labels()
        .next()
        .expect("at least one decision state")
        .clone();
    let lbl = CanonicalLabel::parse(&lbl_str).unwrap();
    let state = ws.state(&lbl).unwrap();

    let mut zeros = HashMap::new();
    for mv in state.legal_moves() {
        zeros.insert(mv, 0.0_f64);
    }
    ws.set_move_weights(&lbl, &zeros);

    let mut rng = StdRng::seed_from_u64(7);
    let err = ws
        .sample_move(&lbl, &mut rng)
        .expect_err("depleted matchbox should produce an error");

    assert!(
        matches!(err, Error::DepletedMatchbox { .. }),
        "expected DepletedMatchbox error, got {err:?}"
    );
}
