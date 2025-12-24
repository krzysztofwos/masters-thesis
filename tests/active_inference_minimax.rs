use menace::{
    ActiveAdversarialOpponent, ActiveBeliefs, ActiveEFEMode, ActiveGenerativeModel,
    ActiveMinimaxOpponent, ActivePolicyPrior, ActivePreferenceModel, ActiveUniformOpponent,
    efe::dirichlet_categorical_mi, tictactoe::Player,
};

#[test]
fn beta_zero_with_minimax_opponent_is_flat_at_root() {
    let prefs =
        ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01).with_epistemic_weight(0.0);
    let model = ActiveGenerativeModel::new();

    let opponent = ActiveMinimaxOpponent;
    let beliefs = ActiveBeliefs::symmetric(1.0);

    let evals = model.evaluate_actions(model.root(), &prefs, &opponent, &beliefs, Player::X);
    assert!(!evals.is_empty());

    let min_f = evals.first().unwrap().free_energy;
    let max_f = evals.last().unwrap().free_energy;

    assert!(
        (max_f - min_f).abs() < 1e-9,
        "Expected flat free-energy landscape at root for beta=0 with minimax opponent. Got min={min_f}, max={max_f}"
    );
}

#[test]
fn minimax_opponent_returns_pure_risk_even_with_beta() {
    let prefs =
        ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01).with_epistemic_weight(1.5);
    let model = ActiveGenerativeModel::new();

    let opponent = ActiveMinimaxOpponent;
    let beliefs = ActiveBeliefs::symmetric(1.0);

    let evals = model.evaluate_actions(model.root(), &prefs, &opponent, &beliefs, Player::X);
    assert!(!evals.is_empty());

    for eval in evals {
        assert!(
            (eval.free_energy - eval.risk).abs() < 1e-9,
            "Free energy should equal risk under minimax; got F={} vs risk={} for action {}",
            eval.free_energy,
            eval.risk,
            eval.action
        );
        assert!(eval.epistemic.abs() < 1e-9);
        assert!(eval.ambiguity.abs() < 1e-9);
    }
}

#[test]
fn policy_lambda_small_concentrates_on_argmin() {
    let prefs = ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01)
        .with_epistemic_weight(0.5)
        .with_efe_mode(ActiveEFEMode::Exact)
        .with_policy_lambda(1e-6);
    let model = ActiveGenerativeModel::new();

    let opponent = ActiveUniformOpponent;
    let beliefs = ActiveBeliefs::symmetric(1.0);

    let summary = model.exact_state_summary(model.root(), &prefs, &opponent, &beliefs, Player::X);
    assert!(
        !summary.actions.is_empty(),
        "Root state should have legal actions"
    );

    let costs: Vec<f64> = summary
        .actions
        .iter()
        .map(|action| action.risk - prefs.epistemic_weight * action.epistemic)
        .collect();
    let (min_index, _min_cost) = costs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .expect("Expected at least one action");

    let q = &summary.policy.q;
    let (max_prob_index, max_prob) = q
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, prob)| (idx, *prob))
        .expect("Policy should contain probabilities");

    assert_eq!(
        max_prob_index, min_index,
        "Dominant policy mass should align with minimum cost action"
    );
    assert!(
        max_prob > 1.0 - 1e-6,
        "Small lambda should concentrate nearly all mass on the best action; got {max_prob}"
    );

    let residual: f64 = q
        .iter()
        .enumerate()
        .filter_map(|(idx, prob)| (idx != min_index).then_some(*prob))
        .sum();
    assert!(
        residual < 1e-6,
        "Remaining probability mass {residual} should be negligible when lambda is tiny"
    );
}

#[test]
fn policy_lambda_large_matches_prior() {
    let prefs = ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01)
        .with_epistemic_weight(0.5)
        .with_efe_mode(ActiveEFEMode::Exact)
        .with_policy_lambda(1e6);
    let model = ActiveGenerativeModel::new();

    let opponent = ActiveUniformOpponent;
    let beliefs = ActiveBeliefs::symmetric(1.0);

    let summary = model.exact_state_summary(model.root(), &prefs, &opponent, &beliefs, Player::X);
    assert!(
        !summary.actions.is_empty(),
        "Root state should have legal actions"
    );

    for (q_value, prior_value) in summary.policy.q.iter().zip(summary.policy.prior.iter()) {
        assert!(
            (q_value - prior_value).abs() < 1e-6,
            "Large lambda should recover the prior; got q={q_value} vs prior={prior_value}"
        );
    }
}

#[test]
fn opponent_eig_is_zero_in_adversarial_mode() {
    let prefs =
        ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01).with_epistemic_weight(0.5);
    let model = ActiveGenerativeModel::new();

    let opponent = ActiveAdversarialOpponent;
    let beliefs = ActiveBeliefs::symmetric(1.0);

    let evals = model.evaluate_actions(model.root(), &prefs, &opponent, &beliefs, Player::X);
    assert!(!evals.is_empty());

    for eval in evals {
        assert!(
            eval.opponent_eig.abs() < 1e-9,
            "Adversarial opponent should produce zero opponent_eig; got {}",
            eval.opponent_eig
        );
    }
}

#[test]
fn menace_positional_prior_prefers_center_move() {
    let prefs = ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01)
        .with_policy_prior(ActivePolicyPrior::MenacePositional);
    let model = ActiveGenerativeModel::new();

    let opponent = ActiveUniformOpponent;
    let beliefs = ActiveBeliefs::symmetric(1.0);

    let evals = model.evaluate_actions(model.root(), &prefs, &opponent, &beliefs, Player::X);
    assert!(!evals.is_empty());

    let center = evals
        .iter()
        .find(|eval| eval.action == 4)
        .map(|eval| eval.policy_prior)
        .expect("Center move should be available at root");
    let corner = evals
        .iter()
        .find(|eval| matches!(eval.action, 0 | 2 | 6 | 8))
        .map(|eval| eval.policy_prior)
        .expect("Corner move should be available at root");
    let edge = evals
        .iter()
        .find(|eval| matches!(eval.action, 1 | 3 | 5 | 7))
        .map(|eval| eval.policy_prior)
        .expect("Edge move should be available at root");

    assert!(
        center > corner && corner > edge,
        "Expected prior weights center > corner > edge, got center={center}, corner={corner}, edge={edge}"
    );
}

#[test]
fn dirichlet_mi_is_monotone_decreasing_in_alpha() {
    let alphas = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
    let ks = [2usize, 3, 5, 9];
    for &k in &ks {
        let mut previous = f64::INFINITY;
        for &alpha in &alphas {
            let params = vec![alpha; k];
            let mi = dirichlet_categorical_mi(&params);
            assert!(
                mi <= previous + 1e-9,
                "Dirichlet MI should decrease as alpha grows; k={k}, alpha={alpha}, prev={previous}, current={mi}"
            );
            previous = mi;
        }
    }
}

#[test]
fn opponent_state_summary_aligns_with_dirichlet_mi() {
    let base_prefs =
        ActivePreferenceModel::from_probabilities(0.9, 0.09, 0.01).with_epistemic_weight(0.5);
    let model = ActiveGenerativeModel::new();

    let uniform = ActiveUniformOpponent;
    let root_beliefs = ActiveBeliefs::symmetric(1.0);

    let root_actions = model.evaluate_actions(
        model.root(),
        &base_prefs,
        &uniform,
        &root_beliefs,
        Player::X,
    );
    let next_state = root_actions
        .iter()
        .find(|action| action.action == 4)
        .or_else(|| root_actions.first())
        .expect("Root should have at least one action");
    let state_label = next_state.next_state.clone();

    let beliefs_lo = ActiveBeliefs::symmetric(0.5);
    let summary_lo =
        model.opponent_state_summary(&state_label, &base_prefs, &uniform, &beliefs_lo, Player::X);
    let alpha_lo = vec![0.5; summary_lo.actions.len()];
    let expected_lo = dirichlet_categorical_mi(&alpha_lo);
    let eig_lo = summary_lo.information_gain;
    assert!(
        (eig_lo - expected_lo).abs() < 1e-9,
        "Expected EIG {eig_lo} to match closed form {expected_lo}"
    );
    let total_weight_lo: f64 = summary_lo
        .actions
        .iter()
        .map(|action| action.predictive_weight)
        .sum();
    assert!(
        (total_weight_lo - 1.0).abs() < 1e-9,
        "Predictive weights should sum to 1 (got {total_weight_lo})"
    );

    let beliefs_hi = ActiveBeliefs::symmetric(5.0);
    let summary_hi =
        model.opponent_state_summary(&state_label, &base_prefs, &uniform, &beliefs_hi, Player::X);
    let alpha_hi = vec![5.0; summary_hi.actions.len()];
    let expected_hi = dirichlet_categorical_mi(&alpha_hi);
    let eig_hi = summary_hi.information_gain;
    assert!(
        (eig_hi - expected_hi).abs() < 1e-9,
        "Expected EIG {eig_hi} to match closed form {expected_hi}"
    );
    assert!(
        eig_lo > eig_hi + 1e-9,
        "Lower alpha should yield higher information gain ({eig_lo} vs {eig_hi})"
    );
}
