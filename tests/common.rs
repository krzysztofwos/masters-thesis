//! Common test utilities for the menace test suite.
//!
//! This module provides statistical sampling functions used across multiple tests.

use std::f64::consts::PI;

use rand::{Rng, distr::StandardUniform, rngs::StdRng};

/// Sample from a Dirichlet distribution using the Gamma-Dirichlet relationship.
///
/// # Arguments
///
/// * `alpha` - Concentration parameters for the Dirichlet distribution
/// * `rng` - Random number generator
///
/// # Returns
///
/// A vector of samples from Dir(alpha) that sum to 1.0
pub fn sample_dirichlet(alpha: &[f64], rng: &mut StdRng) -> Vec<f64> {
    let mut draws = Vec::with_capacity(alpha.len());
    let mut total = 0.0;
    for &a in alpha {
        let value = sample_gamma(a, rng);
        draws.push(value);
        total += value;
    }
    draws.iter_mut().for_each(|value| *value /= total);
    draws
}

/// Sample from a categorical distribution given a probability vector.
///
/// # Arguments
///
/// * `weights` - Probability weights (need not sum to 1, will be normalized)
/// * `rng` - Random number generator
///
/// # Returns
///
/// An index sampled according to the categorical distribution
pub fn sample_categorical(weights: &[f64], rng: &mut StdRng) -> usize {
    debug_assert!(!weights.is_empty());
    let ticket: f64 = rng.sample(StandardUniform);
    let mut cumulative = 0.0;
    for (idx, weight) in weights.iter().enumerate() {
        cumulative += weight;
        if ticket <= cumulative {
            return idx;
        }
    }
    weights.len() - 1
}

/// Sample from a standard normal distribution using Box-Muller transform.
fn sample_standard_normal(rng: &mut StdRng) -> f64 {
    let mut u1: f64 = rng.sample(StandardUniform);
    u1 = u1.max(1e-12);
    let u2: f64 = rng.sample(StandardUniform);
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = 2.0 * PI * u2;
    radius * angle.cos()
}

/// Sample from a Gamma distribution using Marsaglia and Tsang's method.
///
/// # Arguments
///
/// * `shape` - Shape parameter (must be > 0)
/// * `rng` - Random number generator
///
/// # Returns
///
/// A sample from Gamma(shape, 1.0)
///
/// # Panics
///
/// Panics if shape <= 0
fn sample_gamma(shape: f64, rng: &mut StdRng) -> f64 {
    assert!(shape > 0.0, "gamma shape must be positive");
    if shape < 1.0 {
        let u: f64 = rng.sample(StandardUniform);
        let u = u.max(1e-12);
        return sample_gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }

    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = sample_standard_normal(rng);
        let v_candidate = 1.0 + c * x;
        if v_candidate <= 0.0 {
            continue;
        }
        let v = v_candidate * v_candidate * v_candidate;
        let u: f64 = rng.sample(StandardUniform);

        if u < 1.0 - 0.0331 * x.powi(4) {
            return d * v;
        }

        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}
