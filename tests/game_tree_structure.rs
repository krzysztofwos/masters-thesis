use menace::{identifiers::StateId, tictactoe::build_reduced_game_tree};

#[test]
fn reduced_game_tree_includes_moves_for_available_targets() {
    let configs = [(false, false), (true, false), (false, true), (true, true)];

    for (x_only, menace_rules) in configs {
        let tree = build_reduced_game_tree(x_only, menace_rules);

        for state_id in tree.states.iter() {
            let state_str = state_id.as_str();
            let Some(state) = tree.canonical_states.get(state_str) else {
                panic!(
                    "State {state_str} in tree.states but missing from canonical_states (x_only={x_only}, menace_rules={menace_rules})"
                );
            };

            if state.is_terminal() {
                continue;
            }

            for mv in state.legal_moves() {
                let next_state = state
                    .make_move(mv)
                    .expect("expected legal move when enumerating game tree")
                    .canonical();
                let next_label = next_state.encode();
                let next_state_id = StateId::from(next_label.as_str());

                if !tree.states.contains(&next_state_id) {
                    continue;
                }

                // Check if this move exists in the game tree
                if let Some(moves) = tree.moves.get(state_id) {
                    let found = moves.iter().any(|(move_id, target)| {
                        target == &next_state_id && move_id.as_str().ends_with(&format!("_{mv}"))
                    });

                    assert!(
                        found,
                        "Missing move from {state_str} to {next_label} (position {mv}) in config x_only={x_only}, menace_rules={menace_rules}"
                    );
                }
            }
        }
    }
}
