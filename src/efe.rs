//! Exact expected-free-energy utilities for MENACE.
//!
//! This module provides building blocks for the Active Inference baseline:
//! - Dirichlet entropy and one-step expected information gain
//! - KL-regularised optimal policy computation
//! - Convenience helpers for reporting exact state-level variational free energy
//! - Decomposition helpers for arbitrary policies (e.g., MENACE bead distributions)

use statrs::function::gamma::{digamma, ln_gamma};

use crate::utils::{NormalizationFallback, normalize_weights_with_options};

const EPS: f64 = 1e-12;

/// Dirichlet distribution over a fixed set of categories.
#[derive(Clone, Debug)]
pub struct Dirichlet {
    alpha: Vec<f64>,
}

impl Dirichlet {
    /// Construct from concentration parameters. All entries must be strictly positive.
    pub fn new(alpha: Vec<f64>) -> Self {
        assert!(
            !alpha.is_empty(),
            "Dirichlet requires at least one category"
        );
        assert!(
            alpha.iter().all(|&a| a > 0.0),
            "Dirichlet parameters must be > 0"
        );
        Self { alpha }
    }

    /// Symmetric Dirichlet with identical concentration per category.
    pub fn symmetric(k: usize, concentration: f64) -> Self {
        assert!(k > 0, "Dirichlet requires positive dimension");
        assert!(concentration > 0.0, "Dirichlet concentration must be > 0");
        Self {
            alpha: vec![concentration; k],
        }
    }

    /// Access raw concentration parameters.
    pub fn alpha(&self) -> &[f64] {
        &self.alpha
    }

    /// Number of categories.
    pub fn k(&self) -> usize {
        self.alpha.len()
    }

    /// Sum of concentration parameters.
    pub fn alpha0(&self) -> f64 {
        self.alpha.iter().sum()
    }

    /// Predictive categorical distribution (mean of the Dirichlet).
    pub fn predictive(&self) -> Vec<f64> {
        let total = self.alpha0();
        self.alpha.iter().map(|&a| (a / total).max(EPS)).collect()
    }

    /// Log multivariate beta function: ln B(alpha) = sum ln Γ(alpha_i) - ln Γ(alpha_0).
    fn ln_beta(&self) -> f64 {
        let sum_ln_gamma: f64 = self.alpha.iter().map(|&a| ln_gamma(a)).sum();
        let ln_gamma_sum = ln_gamma(self.alpha0());
        sum_ln_gamma - ln_gamma_sum
    }

    /// Entropy of the Dirichlet distribution.
    pub fn entropy(&self) -> f64 {
        let alpha0 = self.alpha0();
        let term_norm = (alpha0 - self.k() as f64) * digamma(alpha0);
        let term_components: f64 = self.alpha.iter().map(|&a| (a - 1.0) * digamma(a)).sum();
        self.ln_beta() + term_norm - term_components
    }

    /// Expected Dirichlet entropy after observing a single categorical outcome.
    pub fn expected_entropy_after_one_observation(&self) -> f64 {
        let predictive = self.predictive();
        predictive
            .iter()
            .enumerate()
            .map(|(idx, &p)| {
                let mut updated = self.alpha.clone();
                updated[idx] += 1.0;
                let posterior = Dirichlet::new(updated);
                p * posterior.entropy()
            })
            .sum()
    }

    /// Reduction in Dirichlet **differential** entropy after one observation.
    ///
    /// Note: This is *not* the mutual information between θ and o. For epistemic
    /// value in Active Inference, use [`dirichlet_categorical_mi`].
    pub fn entropy_drop_one_observation(&self) -> f64 {
        (self.entropy() - self.expected_entropy_after_one_observation()).max(0.0)
    }
}

/// Returns the Dirichlet–Categorical mutual information for a single observation: I(θ; o).
/// I(θ; o) = H[predictive] - E_θ[H(θ)] for a single categorical draw under Dirichlet(α).
pub fn dirichlet_categorical_mi(alpha: &[f64]) -> f64 {
    if alpha.len() < 2 {
        return 0.0;
    }
    if alpha.iter().any(|&a| !a.is_finite() || a <= 0.0) {
        return 0.0;
    }
    let a0: f64 = alpha.iter().sum();
    if !a0.is_finite() || a0 <= 0.0 {
        return 0.0;
    }
    let mut h_predictive = 0.0;
    for &ai in alpha {
        let pi = ai / a0;
        if pi > 0.0 {
            h_predictive -= pi * pi.ln();
        }
    }
    let psi_a0p1 = digamma(a0 + 1.0);
    let mut expected_entropy_of_theta = psi_a0p1;
    for &ai in alpha {
        let pi = ai / a0;
        if pi > 0.0 {
            expected_entropy_of_theta -= pi * digamma(ai + 1.0);
        }
    }
    (h_predictive - expected_entropy_of_theta).max(0.0)
}

/// Numerically stable log-sum-exp.
fn logsumexp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    let sum = values.iter().map(|v| (v - max).exp()).sum::<f64>();
    max + sum.ln()
}

/// Compute the KL-regularised optimal policy: q*(a) ∝ p(a) exp(-cost[a] / lambda).
pub fn optimal_policy_with_kl(cost: &[f64], prior: &[f64], lambda: f64) -> Vec<f64> {
    assert!(!cost.is_empty(), "policy requires costs");
    assert_eq!(cost.len(), prior.len(), "cost/prior length mismatch");
    assert!(
        lambda.is_finite() && lambda > 0.0,
        "lambda must be positive"
    );

    let log_weights: Vec<f64> = cost
        .iter()
        .zip(prior.iter())
        .map(|(&c, &p)| {
            let log_p = p.max(EPS).ln();
            log_p + (-c / lambda)
        })
        .collect();

    let normaliser = logsumexp(&log_weights);
    log_weights
        .into_iter()
        .map(|lw| (lw - normaliser).exp())
        .collect()
}

/// KL divergence between categorical distributions.
pub fn kl_divergence_categorical(q: &[f64], p: &[f64]) -> f64 {
    assert_eq!(q.len(), p.len(), "categorical length mismatch");
    q.iter()
        .zip(p.iter())
        .map(|(&qi, &pi)| {
            let qi = qi.max(EPS);
            let pi = pi.max(EPS);
            qi * (qi / pi).ln()
        })
        .sum()
}

/// Normalize a distribution with uniform fallback and epsilon clamping.
///
/// This is a convenience wrapper around `normalize_weights_with_options` that:
/// - Falls back to uniform distribution when normalization fails
/// - Clamps all probabilities to at least EPS to prevent numerical issues
/// - Panics if the input is empty (distributions must have at least one entry)
///
/// # Panics
///
/// Panics if `values` is empty.
pub fn normalize_distribution(values: &[f64]) -> Vec<f64> {
    assert!(!values.is_empty(), "distribution requires entries");

    normalize_weights_with_options(
        values.iter().copied(),
        NormalizationFallback::Uniform,
        Some(EPS),
    )
    .expect("Uniform fallback should always succeed for non-empty input")
}

/// Compute the free-energy decomposition for an arbitrary policy.
pub fn decompose_policy(
    policy: &[f64],
    risk: &[f64],
    epistemic: &[f64],
    beta: f64,
    lambda: f64,
    prior: &[f64],
) -> PolicyDecomposition {
    assert_eq!(policy.len(), risk.len(), "policy/risk length mismatch");
    assert_eq!(
        risk.len(),
        epistemic.len(),
        "risk/epistemic length mismatch"
    );
    assert_eq!(policy.len(), prior.len(), "policy/prior length mismatch");

    let policy_norm = normalize_distribution(policy);
    let prior_norm = normalize_distribution(prior);
    let expected_risk: f64 = policy_norm
        .iter()
        .zip(risk.iter())
        .map(|(&q, &r)| q * r)
        .sum();
    let expected_epistemic: f64 = policy_norm
        .iter()
        .zip(epistemic.iter())
        .map(|(&q, &e)| q * e)
        .sum();
    let policy_kl = kl_divergence_categorical(&policy_norm, &prior_norm);
    let free_energy = expected_risk - beta * expected_epistemic + lambda * policy_kl;

    PolicyDecomposition {
        policy: policy_norm,
        expected_risk,
        expected_epistemic,
        policy_kl,
        free_energy,
    }
}

/// Summary of the KL-regularised optimal policy for a state.
#[derive(Clone, Debug)]
pub struct ExactPolicySummary {
    pub q: Vec<f64>,
    pub policy_kl: f64,
    pub f_exact: f64,
    pub prior: Vec<f64>,
    pub expected_risk: f64,
    pub expected_epistemic: f64,
}

#[derive(Clone, Debug)]
pub struct PolicyDecomposition {
    pub policy: Vec<f64>,
    pub expected_risk: f64,
    pub expected_epistemic: f64,
    pub policy_kl: f64,
    pub free_energy: f64,
}

/// Build an exact policy summary from per-action risk and epistemic values.
pub fn exact_policy_from_risk_ep(
    risk: &[f64],
    epistemic: &[f64],
    beta: f64,
    lambda: f64,
    prior: &[f64],
) -> ExactPolicySummary {
    assert_eq!(
        risk.len(),
        epistemic.len(),
        "risk/epistemic length mismatch"
    );
    assert_eq!(risk.len(), prior.len(), "risk/prior length mismatch");

    let costs: Vec<f64> = risk
        .iter()
        .zip(epistemic.iter())
        .map(|(&r, &e)| r - beta * e)
        .collect();
    let q = optimal_policy_with_kl(&costs, prior, lambda);
    let prior_norm = normalize_distribution(prior);
    let decomposition = decompose_policy(&q, risk, epistemic, beta, lambda, &prior_norm);

    ExactPolicySummary {
        q: decomposition.policy,
        policy_kl: decomposition.policy_kl,
        f_exact: decomposition.free_energy,
        prior: prior_norm,
        expected_risk: decomposition.expected_risk,
        expected_epistemic: decomposition.expected_epistemic,
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};
    use rand_distr::{Distribution, Gamma};

    use super::*;

    #[test]
    fn eig_decreases_with_concentration() {
        let broad = Dirichlet::symmetric(3, 1.0);
        let sharp = Dirichlet::symmetric(3, 10.0);
        assert!(dirichlet_categorical_mi(broad.alpha()) > dirichlet_categorical_mi(sharp.alpha()));
    }

    #[test]
    fn dirichlet_mi_matches_monte_carlo_estimate() {
        let alpha = [1.3, 2.1, 3.4];
        let analytic = dirichlet_categorical_mi(&alpha);
        let alpha0: f64 = alpha.iter().sum();
        let predictive: Vec<f64> = alpha.iter().map(|&a| a / alpha0).collect();

        let mut rng = StdRng::seed_from_u64(42);
        let gamma: Vec<_> = alpha
            .iter()
            .map(|&a| Gamma::new(a, 1.0).expect("gamma parameters valid"))
            .collect();

        let samples = 50_000;
        let mut mc = 0.0;
        for _ in 0..samples {
            let mut draw: Vec<f64> = gamma.iter().map(|dist| dist.sample(&mut rng)).collect();
            let sum: f64 = draw.iter().sum();
            for value in draw.iter_mut() {
                *value /= sum;
            }
            for (i, &theta_i) in draw.iter().enumerate() {
                if theta_i > 0.0 && predictive[i] > 0.0 {
                    mc += theta_i * (theta_i / predictive[i]).ln();
                }
            }
        }
        mc /= samples as f64;

        assert!(
            (mc - analytic).abs() < 5e-3,
            "Monte Carlo MI ({mc}) should match analytic value ({analytic})"
        );
    }

    #[test]
    fn optimal_policy_reduces_to_prior_when_lambda_large() {
        let risk = [1.0, 0.5];
        let epistemic = [0.0, 0.0];
        let prior = [0.8, 0.2];
        let summary = exact_policy_from_risk_ep(&risk, &epistemic, 0.0, 1e6, &prior);
        for (qi, pi) in summary.q.iter().zip(summary.prior.iter()) {
            assert!((qi - pi).abs() < 1e-6);
        }
        assert!(summary.policy_kl < 1e-8);
    }

    #[test]
    fn optimal_policy_concentrates_when_lambda_tiny() {
        let costs = [0.0, 0.25, 0.5];
        let prior = [0.6, 0.3, 0.1];
        let policy = optimal_policy_with_kl(&costs, &prior, 1e-9);
        assert_eq!(policy.len(), costs.len());
        let total: f64 = policy.iter().sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(policy[0] > 1.0 - 1e-6);
        assert!(policy[1] < 1e-6 && policy[2] < 1e-6);
    }

    #[test]
    fn decompose_policy_handles_epsilon_prior_entries() {
        let policy = [1.0, 0.0, 0.0];
        let risk = [0.0, 1.0, 1.0];
        let epistemic = [0.5, 0.0, 0.0];
        let prior = [1e-15, 0.7, 0.3];
        let decomposition = decompose_policy(&policy, &risk, &epistemic, 0.5, 1.0, &prior);
        assert!((decomposition.policy[0] - 1.0).abs() < 1e-9);
        assert!(decomposition.policy.iter().all(|p| p.is_finite()));
        assert!(decomposition.policy_kl.is_finite());
        assert!(decomposition.free_energy.is_finite());
    }

    #[test]
    fn decompose_policy_matches_exact_summary() {
        let risk = [0.2, 0.4, 0.1];
        let epistemic = [0.5, 0.25, 0.0];
        let prior = [1.0, 2.0, 1.0];
        let beta = 0.7;
        let lambda = 1.2;
        let summary = exact_policy_from_risk_ep(&risk, &epistemic, beta, lambda, &prior);
        let decomp = decompose_policy(&summary.q, &risk, &epistemic, beta, lambda, &summary.prior);

        assert!((decomp.free_energy - summary.f_exact).abs() < 1e-9);
        assert!((decomp.policy_kl - summary.policy_kl).abs() < 1e-9);
        assert!((decomp.expected_risk - summary.expected_risk).abs() < 1e-9);
        assert!((decomp.expected_epistemic - summary.expected_epistemic).abs() < 1e-9);
    }
}
