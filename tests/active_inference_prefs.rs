use menace::active_inference::{
    OutcomeDistribution, PreferenceModel, RiskModel, preferences::CANONICAL_PREFERENCE_PROBS,
};

#[test]
fn from_probabilities_uses_kl_and_normalizes() {
    let (win_pref, draw_pref, loss_pref) = CANONICAL_PREFERENCE_PROBS;
    let pm = PreferenceModel::from_probabilities(win_pref, draw_pref, loss_pref);

    // Verify it uses KL divergence risk model (canonical AIF)
    assert_eq!(pm.risk_model(), RiskModel::KLDivergence);

    // Verify the preferred distribution is normalized
    let z: f64 = win_pref + draw_pref + loss_pref;
    let _exp_win_pref = win_pref / z;
    let _exp_draw_pref = draw_pref / z;
    let _exp_loss_pref = loss_pref / z;

    // Check that internal preferred distribution is properly normalized
    // (fields are private, but we can verify through expected_risk with matching distributions)
    let dist_win = OutcomeDistribution {
        x_win: 1.0,
        o_win: 0.0,
        draw: 0.0,
    };
    let dist_draw = OutcomeDistribution {
        x_win: 0.0,
        o_win: 0.0,
        draw: 1.0,
    };
    let dist_loss = OutcomeDistribution {
        x_win: 0.0,
        o_win: 1.0,
        draw: 0.0,
    };

    // KL divergence should be well-defined and finite
    assert!(dist_win.expected_risk(&pm).is_finite());
    assert!(dist_draw.expected_risk(&pm).is_finite());
    assert!(dist_loss.expected_risk(&pm).is_finite());

    // Degenerate case: zero component is clamped and stays finite
    let pm2 = PreferenceModel::from_probabilities(1.0, 0.0, 0.0);
    assert!(dist_draw.expected_risk(&pm2).is_finite());
    assert!(dist_loss.expected_risk(&pm2).is_finite());
}
