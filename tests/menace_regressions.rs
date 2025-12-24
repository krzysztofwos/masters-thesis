use std::{cell::RefCell, cmp::Ordering, rc::Rc};

use menace::{
    ActiveAdversarialOpponent, ActiveBeliefs, ActiveEFEMode, ActiveGenerativeModel,
    ActiveMinimaxOpponent, ActivePreferenceModel, ActiveUniformOpponent, InitialBeadSchedule,
    MenaceWorkspace, Reinforcement, RestockMode, StateFilter, efe,
    menace::{
        Matchbox, MenaceAgent, TrainingConfig, TrainingSession,
        training::{OpponentType, TrainingBlockConfig},
    },
    tictactoe::{BoardState, Player},
};
use rand::{SeedableRng, rngs::StdRng};

mod common;

fn approx_eq(a: f64, b: f64) -> bool {
    approx_eq_tol(a, b, 1e-9)
}

fn approx_eq_tol(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn menace_state_filters_match_historical_counts() {
    let workspace_all =
        MenaceWorkspace::new(StateFilter::All).expect("workspace construction should succeed");
    assert_eq!(workspace_all.decision_labels().count(), 338);
    for label_str in workspace_all.decision_labels() {
        let label = menace::CanonicalLabel::parse(label_str).expect("valid canonical label");
        let state = workspace_all
            .state(&label)
            .expect("state should be present for each label");
        assert_eq!(state.to_move, Player::X);
    }

    let workspace_decision = MenaceWorkspace::new(StateFilter::DecisionOnly)
        .expect("workspace construction should succeed");
    assert_eq!(workspace_decision.decision_labels().count(), 304);

    let workspace_michie =
        MenaceWorkspace::new(StateFilter::Michie).expect("workspace construction should succeed");
    assert_eq!(workspace_michie.decision_labels().count(), 287);
}

#[test]
fn state_filter_both_excludes_forced_and_double_threat_states() {
    let workspace = MenaceWorkspace::with_config(
        StateFilter::Both,
        RestockMode::default(),
        InitialBeadSchedule::default(),
    )
    .expect("workspace construction should succeed");

    for label_str in workspace.decision_labels() {
        let label = menace::CanonicalLabel::parse(label_str).expect("valid canonical label");
        let state = workspace
            .state(&label)
            .expect("workspace must contain state for each decision label");

        assert!(
            !state.has_forced_move(),
            "forced state included for label {label_str}"
        );
        assert!(
            !state.opponent_has_double_threat(),
            "double-threat state included for label {label_str}"
        );
    }
}

#[test]
fn state_filter_both_includes_o_start_matchbox() {
    let workspace = MenaceWorkspace::with_config(
        StateFilter::Both,
        RestockMode::default(),
        InitialBeadSchedule::default(),
    )
    .expect("workspace construction should succeed");

    let label = menace::CanonicalLabel::parse("........._O").expect("label should parse");
    assert!(
        workspace.has_matchbox(&label),
        "expected O-first root to have a matchbox when using StateFilter::Both"
    );
}

#[test]
fn o_states_receive_initial_bead_schedule() {
    let state = BoardState::new()
        .make_move_force(4)
        .expect("center move should be legal");
    assert_eq!(state.to_move, Player::O);

    let (_ctx, label) = state.canonical_context_and_label();

    let workspace = MenaceWorkspace::with_config(
        StateFilter::Both,
        RestockMode::default(),
        InitialBeadSchedule::default(),
    )
    .expect("workspace construction should succeed");

    assert!(
        workspace.has_matchbox(&label),
        "expected O-to-move state to produce a matchbox"
    );

    let weights = workspace
        .move_weights(&label)
        .expect("weights should exist for non-filtered O states");
    for (_, weight) in weights {
        assert!(
            approx_eq_tol(weight, 4.0, 1e-9),
            "expected weight 4.0 for early O state, got {weight}"
        );
    }
}

#[test]
fn matchbox_restock_occurs_only_when_box_empty() {
    let mut matchbox = Matchbox::new("root".to_string(), vec![0, 4], 6);
    assert_eq!(matchbox.base_beads(), 1);
    assert_eq!(matchbox.total_beads(), 2);

    matchbox.reinforce(0, -1);
    assert_eq!(matchbox.bead_count(0).unwrap(), 0);
    assert_eq!(matchbox.bead_count(4).unwrap(), 1);
    assert_eq!(matchbox.total_beads(), 1);

    matchbox.reinforce(4, -1);
    let num_positions = matchbox.all_beads().count();
    let expected_total = matchbox.base_beads() * num_positions as u32;
    assert_eq!(matchbox.total_beads(), expected_total);
    for (_pos, count) in matchbox.all_beads() {
        assert_eq!(count, matchbox.base_beads());
    }
}

#[test]
fn restock_modes_behave_as_configured() {
    let (_root_ctx, root_label) = BoardState::new().canonical_context_and_label();

    // RestockMode::None should leave depleted move weights at zero.
    let mut no_restock = MenaceWorkspace::with_options(StateFilter::Michie, RestockMode::None)
        .expect("workspace construction should succeed");
    let move_id = no_restock
        .move_for_position(&root_label, 0)
        .cloned()
        .expect("root should include move 0");
    let _ = no_restock.apply_reinforcement(vec![move_id.clone()], Reinforcement::Negative(10.0));
    let weight_none = no_restock
        .move_weights(&root_label)
        .and_then(|weights| weights.into_iter().find(|(mv, _)| *mv == 0))
        .map(|(_, weight)| weight)
        .expect("weight should be recorded");
    assert!(approx_eq_tol(weight_none, 0.0, 1e-9));

    // RestockMode::Move should immediately restore the depleted move to its base weight.
    let mut move_restock = MenaceWorkspace::with_options(StateFilter::Michie, RestockMode::Move)
        .expect("workspace construction should succeed");
    let move_id = move_restock
        .move_for_position(&root_label, 0)
        .cloned()
        .expect("root should include move 0");
    let _ = move_restock.apply_reinforcement(vec![move_id], Reinforcement::Negative(10.0));
    let weight_move = move_restock
        .move_weights(&root_label)
        .and_then(|weights| weights.into_iter().find(|(mv, _)| *mv == 0))
        .map(|(_, weight)| weight)
        .expect("weight should be recorded");
    assert!(approx_eq_tol(weight_move, 4.0, 1e-9));

    // RestockMode::Box should restore all moves once the entire box is exhausted.
    let mut box_restock = MenaceWorkspace::with_options(StateFilter::Michie, RestockMode::Box)
        .expect("workspace construction should succeed");
    let move_ids: Vec<_> = box_restock
        .move_weights(&root_label)
        .expect("root weights should be available")
        .into_iter()
        .map(|(mv, _)| {
            box_restock
                .move_for_position(&root_label, mv)
                .cloned()
                .expect("move should exist for position")
        })
        .collect();
    for move_id in move_ids {
        let _ = box_restock.apply_reinforcement(vec![move_id], Reinforcement::Negative(10.0));
    }
    let weights_after = box_restock
        .move_weights(&root_label)
        .expect("weights should exist after restock");
    for (_, weight) in weights_after {
        assert!(approx_eq_tol(weight, 4.0, 1e-9));
    }

    // Ensure new move labels align with underscore suffix format.
    let label = no_restock
        .move_for_position(&root_label, 4)
        .expect("move must exist for position 4");
    assert!(label.as_str().contains('_'));
    assert!(!label.as_str().contains("::"));
}

#[test]
fn adversarial_and_minimax_opponents_both_run() {
    // This test verifies that both adversarial and minimax opponents produce valid evaluations.
    // Note: With KL divergence risk, adversarial (maximize risk) and minimax (optimal play)
    // produce DIFFERENT values because game-theoretic optimality is not the same as
    // maximizing KL divergence from preferred distribution.
    let model = ActiveGenerativeModel::new();
    let root = model.root().to_string();

    let preferences = ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01)
        .with_epistemic_weight(0.0)
        .with_efe_mode(ActiveEFEMode::Approx);

    let adversarial = ActiveAdversarialOpponent;
    let minimax_opponent = ActiveMinimaxOpponent;
    let beliefs = ActiveBeliefs::symmetric(1.0);

    let adversarial_actions =
        model.evaluate_actions(&root, &preferences, &adversarial, &beliefs, Player::X);
    let minimax_actions =
        model.evaluate_actions(&root, &preferences, &minimax_opponent, &beliefs, Player::X);

    // Both should produce the same number of actions
    assert_eq!(adversarial_actions.len(), minimax_actions.len());

    // Both should produce valid (finite) free energy values
    for action in &adversarial_actions {
        assert!(
            action.free_energy.is_finite(),
            "adversarial free energy should be finite"
        );
        assert!(action.risk.is_finite(), "adversarial risk should be finite");
    }

    for action in &minimax_actions {
        assert!(
            action.free_energy.is_finite(),
            "minimax free energy should be finite"
        );
        assert!(action.risk.is_finite(), "minimax risk should be finite");
    }
}

#[test]
fn minimax_opponent_runs_without_panic() {
    let agent = MenaceAgent::new(Some(7)).expect("agent construction should succeed");
    let config = TrainingConfig {
        num_games: 1,
        opponent: OpponentType::Minimax,
        logging: false,
        seed: Some(11),
        restock: None,
        curriculum: None,
        agent_player: Player::X,
        first_player: Player::X,
    };

    let mut session = TrainingSession::new(agent, config);
    session
        .train()
        .expect("minimax opponent should complete training");

    assert_eq!(session.games_played, 1);
    let total = session.results.wins + session.results.draws + session.results.losses;
    assert_eq!(total, 1);
}

#[test]
fn menace_vs_menace_training_executes() {
    use menace::workspace::StateFilter;

    // Use StateFilter::Both for player-agnostic agents
    let agent = MenaceAgent::builder()
        .seed(101)
        .filter(StateFilter::Both)
        .build()
        .expect("agent construction should succeed");
    let opponent_agent = MenaceAgent::builder()
        .seed(202)
        .filter(StateFilter::Both)
        .build()
        .expect("agent construction should succeed");

    let config = TrainingConfig {
        num_games: 5,
        opponent: OpponentType::AnotherMenace(Rc::new(RefCell::new(opponent_agent))),
        logging: false,
        seed: Some(303),
        restock: None,
        curriculum: None,
        agent_player: Player::X,
        first_player: Player::X,
    };

    let mut session = TrainingSession::new(agent, config);
    session
        .train()
        .expect("menace vs menace training should complete");

    assert_eq!(session.games_played, 5);
    let total = session.results.wins + session.results.draws + session.results.losses;
    assert_eq!(total, 5);
}

#[test]
fn curriculum_blocks_sum_games() {
    let agent = MenaceAgent::new(Some(404)).expect("agent construction should succeed");
    let curriculum = vec![
        TrainingBlockConfig::new(OpponentType::Random, 3),
        TrainingBlockConfig::new(OpponentType::Minimax, 2),
    ];
    let config = TrainingConfig {
        num_games: 0,
        opponent: OpponentType::Random,
        logging: false,
        seed: Some(505),
        restock: None,
        curriculum: Some(curriculum),
        agent_player: Player::X,
        first_player: Player::X,
    };

    let mut session = TrainingSession::new(agent, config);
    session
        .train()
        .expect("curriculum-based training should complete");

    assert_eq!(session.games_played, 5);
    let total = session.results.wins + session.results.draws + session.results.losses;
    assert_eq!(total, 5);
}

#[test]
fn exact_policy_selects_same_argmin_as_approx_free_energy() {
    let model = ActiveGenerativeModel::new();
    let root = model.root().to_string();
    let beta = 0.5;

    let approx_preferences = ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01)
        .with_epistemic_weight(beta)
        .with_efe_mode(ActiveEFEMode::Approx);
    let opponent = ActiveAdversarialOpponent;
    let beliefs = ActiveBeliefs::symmetric(1.0);

    let approx_actions =
        model.evaluate_actions(&root, &approx_preferences, &opponent, &beliefs, Player::X);

    let mut min_score = f64::INFINITY;
    for action in &approx_actions {
        min_score = min_score.min(action.risk - beta * action.epistemic);
    }
    let argmin: Vec<usize> = approx_actions
        .iter()
        .filter_map(|action| {
            let score = action.risk - beta * action.epistemic;
            if approx_eq(score, min_score) {
                Some(action.action)
            } else {
                None
            }
        })
        .collect();
    assert!(
        !argmin.is_empty(),
        "expected at least one approximate argmin action"
    );

    let exact_preferences = ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01)
        .with_epistemic_weight(beta)
        .with_efe_mode(ActiveEFEMode::Exact)
        .with_policy_lambda(1e-6);
    let exact_summary =
        model.exact_state_summary(&root, &exact_preferences, &opponent, &beliefs, Player::X);

    let (max_index, _) = exact_summary
        .policy
        .q
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .expect("policy should contain at least one action");

    let chosen_action = exact_summary.actions[max_index].action;
    assert!(
        argmin.contains(&chosen_action),
        "exact policy chose action {chosen_action} not present in approx argmin set {argmin:?}"
    );
}

#[test]
fn center_move_preferred_under_uniform_play_with_beta_half() {
    let model = ActiveGenerativeModel::new();
    let root = model.root().to_string();
    let beta = 0.5;
    let preferences = ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01)
        .with_epistemic_weight(beta)
        .with_efe_mode(ActiveEFEMode::Approx);
    let opponent = ActiveUniformOpponent;
    let beliefs = ActiveBeliefs::symmetric(1.0);
    let actions = model.evaluate_actions(&root, &preferences, &opponent, &beliefs, Player::X);
    assert!(
        !actions.is_empty(),
        "root state should have available actions"
    );

    let mut best_score = f64::INFINITY;
    let mut best_moves = Vec::new();
    for action in &actions {
        let score = action.risk - beta * action.epistemic;
        match score.partial_cmp(&best_score).unwrap_or(Ordering::Equal) {
            Ordering::Less => {
                best_score = score;
                best_moves.clear();
                best_moves.push(action.action);
            }
            Ordering::Equal => best_moves.push(action.action),
            Ordering::Greater => {}
        }
    }

    assert!(
        best_moves.contains(&4),
        "expected center move (4) in best move set, got {best_moves:?}"
    );
}

#[test]
fn training_with_seed_is_deterministic() {
    let config = TrainingConfig {
        num_games: 200,
        opponent: OpponentType::Random,
        logging: true,
        seed: Some(777),
        restock: None,
        curriculum: None,
        agent_player: Player::X,
        first_player: Player::X,
    };

    let agent_one = MenaceAgent::new(Some(2024)).expect("agent construction should succeed");
    let mut session_one = TrainingSession::new(agent_one, config.clone());
    session_one
        .train()
        .expect("first training run should succeed");

    let agent_two = MenaceAgent::new(Some(2024)).expect("agent construction should succeed");
    let mut session_two = TrainingSession::new(agent_two, config);
    session_two
        .train()
        .expect("second training run should succeed");

    assert_eq!(session_one.results.wins, session_two.results.wins);
    assert_eq!(session_one.results.draws, session_two.results.draws);
    assert_eq!(session_one.results.losses, session_two.results.losses);
    assert_eq!(
        session_one.results.win_rate_history,
        session_two.results.win_rate_history
    );
    assert_eq!(
        session_one.results.kl_history.len(),
        session_two.results.kl_history.len()
    );
    for (a, b) in session_one
        .results
        .kl_history
        .iter()
        .zip(session_two.results.kl_history.iter())
    {
        if (a.is_nan() && b.is_nan()) || (a.is_infinite() && b.is_infinite()) {
            continue;
        }
        let diff = (*a - *b).abs();
        assert!(diff < 1e-6, "kl divergence mismatch: diff={diff}");
    }

    let (_root_ctx, root_label) = BoardState::new().canonical_context_and_label();
    let dist_one = session_one
        .agent
        .workspace()
        .move_distribution(&root_label)
        .expect("root distribution should exist after training");
    let dist_two = session_two
        .agent
        .workspace()
        .move_distribution(&root_label)
        .expect("root distribution should exist after training");

    assert_eq!(dist_one.len(), dist_two.len());
    for ((mv_a, w_a), (mv_b, w_b)) in dist_one.iter().zip(dist_two.iter()) {
        assert_eq!(mv_a, mv_b);
        assert!(approx_eq_tol(*w_a, *w_b, 1e-8));
    }
}

#[test]
fn dirichlet_information_gain_matches_monte_carlo_estimate() {
    let scenarios: &[&[f64]] = &[&[1.0, 1.0, 1.0], &[0.7, 2.3, 4.5, 1.1], &[0.3, 0.3, 0.3]];
    let mut rng = StdRng::seed_from_u64(0x5eed_1234);

    for alpha in scenarios {
        let exact = efe::dirichlet_categorical_mi(alpha);
        let predictive: Vec<f64> = {
            let total: f64 = alpha.iter().sum();
            alpha.iter().map(|a| a / total).collect()
        };

        let trials = 100_000;
        let mut estimate = 0.0;
        for _ in 0..trials {
            let theta = common::sample_dirichlet(alpha, &mut rng);
            let obs = common::sample_categorical(&theta, &mut rng);
            estimate += (theta[obs] / predictive[obs]).ln();
        }
        estimate /= trials as f64;

        assert!(
            (exact - estimate).abs() < 5e-3,
            "dirichlet MI mismatch: exact={exact}, estimate={estimate}, alpha={alpha:?}"
        );
    }
}
