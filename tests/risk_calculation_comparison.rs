//! Tests for KL divergence risk calculation
//!
//! This test validates the canonical Active Inference KL divergence risk model,
//! verifying that:
//! 1. KL divergence is computed correctly
//! 2. Risk values are in expected ranges
//! 3. KL properties (non-negativity, zero at perfect alignment) hold

use menace::active_inference::{
    OutcomeDistribution, PreferenceModel, preferences::CANONICAL_PREFERENCE_PROBS,
};

fn canonical_preferences() -> PreferenceModel {
    let (win_pref, draw_pref, loss_pref) = CANONICAL_PREFERENCE_PROBS;
    PreferenceModel::from_probabilities(win_pref, draw_pref, loss_pref)
}

#[test]
fn test_kl_certain_win() {
    // Create preferences: moderately prefer win (0.6), lean toward draws (0.35), avoid loss (0.05)
    let prefs = canonical_preferences();

    // Test case 1: Certain win
    let dist_certain_win = OutcomeDistribution {
        x_win: 1.0,
        o_win: 0.0,
        draw: 0.0,
    };

    let risk = dist_certain_win.expected_risk(&prefs);

    println!("Certain win:");
    println!("  KL risk:     {risk:.6}");

    // For KL: Q=[1,0,0] vs P=[0.6,0.35,0.05]
    // KL = 1*ln(1/0.6) = ln(1.666...) ≈ 0.511
    assert!(
        risk > 0.0 && risk < 1.0,
        "KL should be positive and moderate for certain win"
    );
}

#[test]
fn test_kl_certain_draw() {
    let prefs = canonical_preferences();

    // Test case 2: Certain draw (moderate risk)
    let dist_certain_draw = OutcomeDistribution {
        x_win: 0.0,
        o_win: 0.0,
        draw: 1.0,
    };

    let risk = dist_certain_draw.expected_risk(&prefs);

    println!("\nCertain draw:");
    println!("  KL risk:     {risk:.6}");

    // For KL: Q=[0,1,0] vs P=[0.6,0.35,0.05]
    // KL = 1*ln(1/0.35) = ln(2.857...) ≈ 1.050
    assert!(
        risk > 1.0 && risk < 1.5,
        "KL should be ~1.1 for certain draw"
    );
}

#[test]
fn test_kl_certain_loss() {
    let prefs = canonical_preferences();

    // Test case 3: Certain loss (high risk)
    let dist_certain_loss = OutcomeDistribution {
        x_win: 0.0,
        o_win: 1.0,
        draw: 0.0,
    };

    let risk = dist_certain_loss.expected_risk(&prefs);

    println!("\nCertain loss:");
    println!("  KL risk:     {risk:.6}");

    // For KL: Q=[0,0,1] vs P=[0.6,0.35,0.05]
    // KL = 1*ln(1/0.05) = ln(20.0) ≈ 2.996
    assert!(
        (2.9..3.1).contains(&risk),
        "KL should be ~3.0 for certain loss"
    );
}

#[test]
fn test_kl_mixed_distribution() {
    let prefs = canonical_preferences();

    // Test case 4: Mixed distribution (realistic game state)
    // e.g., 20% win, 60% draw, 20% loss
    let dist_mixed = OutcomeDistribution {
        x_win: 0.2,
        o_win: 0.2,
        draw: 0.6,
    };

    let risk = dist_mixed.expected_risk(&prefs);

    println!("\nMixed distribution (20% win, 60% draw, 20% loss):");
    println!("  KL risk:     {risk:.6}");

    // KL divergence for mixed distribution should be positive and finite
    assert!(
        risk > 0.0 && risk.is_finite(),
        "KL should be positive and finite for mixed distribution"
    );
}

#[test]
fn test_kl_perfect_alignment_zero() {
    // When predicted distribution exactly matches preferred distribution,
    // KL divergence should be zero (no divergence)
    let prefs_kl = canonical_preferences();

    // Create distribution matching the canonical preferences [0.6, 0.35, 0.05]
    let dist_aligned = OutcomeDistribution {
        x_win: 0.6,
        o_win: 0.05,
        draw: 0.35,
    };

    let risk_kl = dist_aligned.expected_risk(&prefs_kl);

    println!("\nPerfect alignment with preferences:");
    println!("  KL risk: {risk_kl:.6}");

    // KL divergence should be very close to 0 when distributions match
    assert!(
        risk_kl.abs() < 0.01,
        "KL should be ~0 for perfectly aligned distributions"
    );
}

#[test]
fn test_kl_properties() {
    let prefs_kl = canonical_preferences();

    // Property 1: KL divergence is always non-negative
    let distributions = [
        OutcomeDistribution {
            x_win: 1.0,
            o_win: 0.0,
            draw: 0.0,
        },
        OutcomeDistribution {
            x_win: 0.0,
            o_win: 1.0,
            draw: 0.0,
        },
        OutcomeDistribution {
            x_win: 0.0,
            o_win: 0.0,
            draw: 1.0,
        },
        OutcomeDistribution {
            x_win: 0.33,
            o_win: 0.33,
            draw: 0.34,
        },
        OutcomeDistribution {
            x_win: 0.5,
            o_win: 0.3,
            draw: 0.2,
        },
    ];

    println!("\nKL divergence non-negativity test:");
    for (i, dist) in distributions.iter().enumerate() {
        let kl = dist.expected_risk(&prefs_kl);
        println!("  Distribution {i}: KL = {kl:.6}");
        assert!(kl >= 0.0, "KL divergence must be non-negative");
    }
}

#[test]
fn test_kl_magnitude_ranges() {
    // Test that KL divergence produces values in expected ranges
    // This is important for comparing with epistemic values (which are also in nats)
    let prefs = canonical_preferences();

    let test_distributions = vec![
        (
            "Certain win",
            OutcomeDistribution {
                x_win: 1.0,
                o_win: 0.0,
                draw: 0.0,
            },
        ),
        (
            "Certain draw",
            OutcomeDistribution {
                x_win: 0.0,
                o_win: 0.0,
                draw: 1.0,
            },
        ),
        (
            "Certain loss",
            OutcomeDistribution {
                x_win: 0.0,
                o_win: 1.0,
                draw: 0.0,
            },
        ),
        (
            "Uniform",
            OutcomeDistribution {
                x_win: 0.33,
                o_win: 0.33,
                draw: 0.34,
            },
        ),
        (
            "Win-biased",
            OutcomeDistribution {
                x_win: 0.7,
                o_win: 0.1,
                draw: 0.2,
            },
        ),
        (
            "Draw-biased",
            OutcomeDistribution {
                x_win: 0.1,
                o_win: 0.1,
                draw: 0.8,
            },
        ),
    ];

    println!("\nKL divergence across different distributions:");
    println!("{:<15} {:>12}", "Distribution", "KL (nats)");
    println!("{}", "-".repeat(30));

    let mut risks = Vec::new();

    for (name, dist) in &test_distributions {
        let risk = dist.expected_risk(&prefs);
        println!("{name:<15} {risk:>12.6}");
        risks.push(risk);
    }

    // Calculate mean and range
    let mean: f64 = risks.iter().sum::<f64>() / risks.len() as f64;
    let min = risks.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = risks.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("\nStatistics:");
    println!("  KL: mean={mean:.3}, range=[{min:.3}, {max:.3}]");

    // Verify all risks are non-negative and finite
    for risk in &risks {
        assert!(risk >= &0.0, "KL divergence must be non-negative");
        assert!(risk.is_finite(), "KL divergence must be finite");
    }
}

#[test]
fn test_kl_builder_methods() {
    // Test that builder methods work with KL model
    let prefs = canonical_preferences()
        .with_epistemic_weight(1.5)
        .with_policy_lambda(2.0);

    assert_eq!(prefs.epistemic_weight, 1.5);
    assert_eq!(prefs.policy_lambda, 2.0);

    use menace::active_inference::RiskModel;
    assert_eq!(prefs.risk_model(), RiskModel::KLDivergence);
}
