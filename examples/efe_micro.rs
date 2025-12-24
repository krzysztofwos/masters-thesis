//! Worked example: one matchbox as a Dirichlet node.
//!
//! This example reproduces the numerical Active Inference terms from the
//! "micro-MENACE" walkthrough described in the thesis: two legal actions, three
//! terminal outcomes, and Dirichlet beliefs over each action's outcomes.

use menace::{active_inference::PreferenceModel, efe::dirichlet_categorical_mi, tictactoe::Player};
use statrs::function::gamma::digamma;

const UTILITIES: [f64; 3] = [3.0, 1.0, -1.0];
const LAMBDA_VALUES: [f64; 4] = [0.0, 0.25, 0.5, 1.0];
const EPS: f64 = 1e-12;

fn main() {
    println!("Micro MENACE example: one matchbox with two actions and three outcomes\n");

    let alpha_a1 = vec![2.0, 1.0, 1.0];
    let alpha_a2 = vec![1.0, 1.0, 2.0];

    let preferences = PreferenceModel::from_utilities(UTILITIES[0], UTILITIES[1], UTILITIES[2])
        .for_player(Player::X);
    let p_c = vec![
        preferences.agent_win_preference(),
        preferences.draw_preference(),
        preferences.agent_loss_preference(),
    ];
    let ln_p_c: Vec<f64> = p_c.iter().map(|&p| p.ln()).collect();

    println!(
        "Softmax preferences p(o|C) over (win, draw, loss): {}",
        tuple_to_string(&p_c, 4)
    );
    println!(
        "Log-preferences ln p(o|C): {}\n",
        tuple_to_string(&ln_p_c, 4)
    );

    let stats_a1 = compute_stats(&alpha_a1, &ln_p_c);
    let stats_a2 = compute_stats(&alpha_a2, &ln_p_c);

    println!("Posterior predictive outcome distributions q(o|a):");
    println!(
        "  a1 -> {}   risk = {:.4}   H_pred = {:.4}   H_ale = {:.4}   I(θ;o) = {:.4}",
        tuple_to_string(&stats_a1.predictive, 2),
        stats_a1.risk,
        stats_a1.h_pred,
        stats_a1.h_ale,
        stats_a1.mi
    );
    println!(
        "  a2 -> {}   risk = {:.4}   H_pred = {:.4}   H_ale = {:.4}   I(θ;o) = {:.4}\n",
        tuple_to_string(&stats_a2.predictive, 2),
        stats_a2.risk,
        stats_a2.h_pred,
        stats_a2.h_ale,
        stats_a2.mi
    );

    println!("Expected free energy G_λ(a) = risk - λ · I(θ;o):");
    println!("  λ       G_λ(a1)   G_λ(a2)   preferred action");
    for &lambda in &LAMBDA_VALUES {
        let g1 = stats_a1.risk - lambda * stats_a1.mi;
        let g2 = stats_a2.risk - lambda * stats_a2.mi;
        let preferred = if g1 <= g2 { "a1" } else { "a2" };
        println!("  {lambda:>4.2}   {g1:>8.4}   {g2:>8.4}   {preferred}");
    }

    println!("\nOne-step learning updates for action a1:");
    println!(
        "{:<22} {:<22} {:>8} {:>10} {:>10} {:>10}",
        "α (win,draw,loss)", "q(o|a1)", "risk", "H_pred", "H_ale", "I(θ;o)"
    );
    for (label, alpha) in [
        ("(2,1,1) before", alpha_a1.clone()),
        ("(3,1,1) after WIN", add_observation(&alpha_a1, 0)),
        ("(2,2,1) after DRAW", add_observation(&alpha_a1, 1)),
        ("(2,1,2) after LOSS", add_observation(&alpha_a1, 2)),
    ] {
        let stats = compute_stats(&alpha, &ln_p_c);
        println!(
            "{:<22} {:<22} {:>8.4} {:>10.4} {:>10.4} {:>10.4}",
            label,
            tuple_to_string(&stats.predictive, 2),
            stats.risk,
            stats.h_pred,
            stats.h_ale,
            stats.mi
        );
    }
}

#[derive(Debug)]
struct ActionStats {
    predictive: Vec<f64>,
    risk: f64,
    h_pred: f64,
    h_ale: f64,
    mi: f64,
}

fn compute_stats(alpha: &[f64], ln_p_c: &[f64]) -> ActionStats {
    assert_eq!(alpha.len(), ln_p_c.len());
    let predictive = predictive_distribution(alpha);
    let risk = risk_from_ln_preferences(&predictive, ln_p_c);
    let h_pred = entropy(&predictive);
    let mi = dirichlet_categorical_mi(alpha);
    let h_ale = aleatoric_entropy(alpha);
    debug_assert!((h_pred - h_ale - mi).abs() < 1e-6);

    ActionStats {
        predictive,
        risk,
        h_pred,
        h_ale,
        mi,
    }
}

fn predictive_distribution(alpha: &[f64]) -> Vec<f64> {
    let total: f64 = alpha.iter().sum();
    assert!(total > 0.0, "Dirichlet counts must be positive");
    alpha.iter().map(|&a| (a / total).max(EPS)).collect()
}

fn risk_from_ln_preferences(pred: &[f64], ln_p_c: &[f64]) -> f64 {
    pred.iter()
        .zip(ln_p_c.iter())
        .map(|(&q, &ln_pref)| -q * ln_pref)
        .sum()
}

fn entropy(dist: &[f64]) -> f64 {
    -dist
        .iter()
        .map(|&p| {
            let clamped = p.max(EPS);
            clamped * clamped.ln()
        })
        .sum::<f64>()
}

fn aleatoric_entropy(alpha: &[f64]) -> f64 {
    let total: f64 = alpha.iter().sum();
    let psi_total = digamma(total + 1.0);
    let weighted = alpha
        .iter()
        .map(|&ai| (ai / total) * digamma(ai + 1.0))
        .sum::<f64>();
    psi_total - weighted
}

fn add_observation(alpha: &[f64], outcome_idx: usize) -> Vec<f64> {
    let mut updated = alpha.to_vec();
    updated[outcome_idx] += 1.0;
    updated
}

fn tuple_to_string(values: &[f64], decimals: usize) -> String {
    let parts: Vec<String> = values.iter().map(|v| format!("{v:.decimals$}")).collect();
    format!("({})", parts.join(", "))
}
