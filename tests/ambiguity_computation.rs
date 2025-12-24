//! Tests for ambiguity (outcome uncertainty) computation in Active Inference
//!
//! This test suite validates that the ambiguity term H[Q(o|π)] is correctly
//! computed as the Shannon entropy of outcome distributions.

use menace::{
    active_inference::{OutcomeDistribution, PreferenceModel},
    tictactoe::Player,
};

#[test]
fn test_ambiguity_deterministic_outcomes() {
    // Deterministic outcomes should have zero ambiguity
    let certain_win = OutcomeDistribution::terminal(Some(Player::X));
    assert_eq!(
        certain_win.ambiguity(),
        0.0,
        "Certain win should have zero ambiguity"
    );

    let certain_draw = OutcomeDistribution::terminal(None);
    assert_eq!(
        certain_draw.ambiguity(),
        0.0,
        "Certain draw should have zero ambiguity"
    );

    let certain_loss = OutcomeDistribution::terminal(Some(Player::O));
    assert_eq!(
        certain_loss.ambiguity(),
        0.0,
        "Certain loss should have zero ambiguity"
    );
}

#[test]
fn test_ambiguity_uncertain_outcomes() {
    // Uniform distribution should have maximum ambiguity for 3 outcomes
    let uniform = OutcomeDistribution {
        x_win: 1.0 / 3.0,
        draw: 1.0 / 3.0,
        o_win: 1.0 / 3.0,
    };

    let ambiguity = uniform.ambiguity();
    let max_entropy_3outcomes = 3.0_f64.ln(); // ln(3) ≈ 1.099

    assert!(
        (ambiguity - max_entropy_3outcomes).abs() < 0.01,
        "Uniform distribution should have entropy ~ln(3) = {max_entropy_3outcomes}, got {ambiguity}"
    );
}

#[test]
fn test_ambiguity_partial_uncertainty() {
    // Mix of outcomes should have intermediate ambiguity
    let mixed = OutcomeDistribution {
        x_win: 0.7,
        draw: 0.2,
        o_win: 0.1,
    };

    let ambiguity = mixed.ambiguity();

    // Should be between 0 (deterministic) and ln(3) (uniform)
    assert!(
        ambiguity > 0.0 && ambiguity < 1.1,
        "Mixed distribution should have ambiguity between 0 and ln(3), got {ambiguity}"
    );
}

#[test]
fn test_ambiguity_in_efe_computation() {
    // Test that ambiguity is properly incorporated into EFE
    let prefs = PreferenceModel::from_probabilities(0.9, 0.5, 0.1)
        .with_epistemic_weight(0.5)
        .with_ambiguity_weight(1.0); // Penalize uncertainty

    let certain_dist = OutcomeDistribution::terminal(Some(Player::X));
    let uncertain_dist = OutcomeDistribution {
        x_win: 0.5,
        draw: 0.3,
        o_win: 0.2,
    };

    // Compute risks
    let risk_certain = certain_dist.expected_risk(&prefs);
    let risk_uncertain = uncertain_dist.expected_risk(&prefs);

    // Compute ambiguities
    let amb_certain = certain_dist.ambiguity();
    let amb_uncertain = uncertain_dist.ambiguity();

    // Compute EFE: G = Risk + β_amb × Ambiguity - β_ep × Epistemic
    // (epistemic = 0 for this test)
    let efe_certain = risk_certain + prefs.ambiguity_weight * amb_certain;
    let efe_uncertain = risk_uncertain + prefs.ambiguity_weight * amb_uncertain;

    // With positive ambiguity weight, uncertain outcomes should have higher EFE
    assert!(
        efe_uncertain > efe_certain,
        "Uncertain outcome should have higher EFE when ambiguity_weight > 0. certain={efe_certain}, uncertain={efe_uncertain}"
    );
}

#[test]
fn test_negative_ambiguity_weight() {
    // Negative ambiguity weight means prefer keeping options open
    let prefs = PreferenceModel::from_probabilities(0.9, 0.5, 0.1)
        .with_epistemic_weight(0.0)
        .with_ambiguity_weight(-0.5); // Reward uncertainty

    let certain_dist = OutcomeDistribution::terminal(Some(Player::X));
    let uncertain_dist = OutcomeDistribution {
        x_win: 0.4,
        draw: 0.3,
        o_win: 0.3,
    };

    let risk_certain = certain_dist.expected_risk(&prefs);
    let risk_uncertain = uncertain_dist.expected_risk(&prefs);

    let amb_certain = certain_dist.ambiguity();
    let amb_uncertain = uncertain_dist.ambiguity();

    let _efe_certain = risk_certain + prefs.ambiguity_weight * amb_certain;
    let _efe_uncertain = risk_uncertain + prefs.ambiguity_weight * amb_uncertain;

    // With negative ambiguity weight, uncertain outcomes might have lower EFE
    // (depends on risk difference, but ambiguity term should decrease EFE)
    let amb_contribution_certain = prefs.ambiguity_weight * amb_certain;
    let amb_contribution_uncertain = prefs.ambiguity_weight * amb_uncertain;

    assert!(
        amb_contribution_uncertain < amb_contribution_certain,
        "Negative ambiguity weight should decrease EFE for uncertain outcomes"
    );
}

#[test]
fn test_ambiguity_properties() {
    // Test mathematical properties of entropy

    // Property 1: Non-negativity
    let distributions = [
        OutcomeDistribution {
            x_win: 1.0,
            draw: 0.0,
            o_win: 0.0,
        },
        OutcomeDistribution {
            x_win: 0.5,
            draw: 0.5,
            o_win: 0.0,
        },
        OutcomeDistribution {
            x_win: 0.33,
            draw: 0.34,
            o_win: 0.33,
        },
        OutcomeDistribution {
            x_win: 0.7,
            draw: 0.2,
            o_win: 0.1,
        },
    ];

    for dist in &distributions {
        assert!(
            dist.ambiguity() >= 0.0,
            "Entropy must be non-negative, got {} for distribution {:?}",
            dist.ambiguity(),
            dist
        );
    }

    // Property 2: Maximum entropy for uniform distribution
    let uniform = OutcomeDistribution {
        x_win: 1.0 / 3.0,
        draw: 1.0 / 3.0,
        o_win: 1.0 / 3.0,
    };
    let uniform_entropy = uniform.ambiguity();

    for dist in &distributions {
        assert!(
            dist.ambiguity() <= uniform_entropy + 0.01,
            "Uniform distribution should have maximum entropy"
        );
    }
}

#[test]
fn test_builder_with_ambiguity_weight() {
    // Test that PreferenceModel builder methods work correctly
    let prefs = PreferenceModel::from_probabilities(0.9, 0.5, 0.1).with_ambiguity_weight(0.75);

    assert_eq!(prefs.ambiguity_weight, 0.75);

    // Test zero (default)
    let prefs_default = PreferenceModel::from_probabilities(0.9, 0.5, 0.1);
    assert_eq!(prefs_default.ambiguity_weight, 0.0);

    // Test negative
    let prefs_negative =
        PreferenceModel::from_probabilities(0.9, 0.5, 0.1).with_ambiguity_weight(-0.5);
    assert_eq!(prefs_negative.ambiguity_weight, -0.5);
}
