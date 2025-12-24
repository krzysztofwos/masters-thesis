//! Utility functions for the MENACE crate

use rand::{Rng, distr::StandardUniform, prelude::IndexedRandom};

/// Calculate Shannon entropy from a probability distribution.
///
/// The Shannon entropy is calculated as: H = -Σ(p * ln(p)) for p > 0
///
/// # Arguments
///
/// * `probabilities` - Iterator of probability values that should sum to 1.0
///
/// # Returns
///
/// The Shannon entropy value (always non-negative).
///
/// # Examples
///
/// ```
/// use menace::utils::shannon_entropy;
///
/// // Uniform distribution over 2 outcomes
/// let entropy = shannon_entropy(vec![0.5, 0.5]);
/// assert!((entropy - std::f64::consts::LN_2).abs() < 0.001);
///
/// // Deterministic distribution (zero entropy)
/// let entropy = shannon_entropy(vec![1.0, 0.0, 0.0]);
/// assert!(entropy.abs() < 0.001);
/// ```
pub fn shannon_entropy<I>(probabilities: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    probabilities
        .into_iter()
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}

/// Calculate entropy from weights (normalizes first).
///
/// This is a convenience function that normalizes weights to probabilities
/// and then calculates Shannon entropy.
///
/// # Arguments
///
/// * `weights` - Iterator of non-negative weight values
///
/// # Returns
///
/// The Shannon entropy value, or 0.0 if total weight is zero or negative.
///
/// # Examples
///
/// ```
/// use menace::utils::entropy_from_weights;
///
/// // Entropy from equal weights
/// let entropy = entropy_from_weights(vec![1.0, 1.0]);
/// assert!((entropy - std::f64::consts::LN_2).abs() < 0.001);
///
/// // Zero total weight returns zero entropy
/// let entropy = entropy_from_weights(vec![0.0, 0.0]);
/// assert_eq!(entropy, 0.0);
/// ```
pub fn entropy_from_weights<I>(weights: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    normalize_weights(weights)
        .map(shannon_entropy)
        .unwrap_or(0.0)
}

/// Fallback behavior when weight normalization fails (zero or negative total).
#[derive(Debug, Clone, Copy)]
pub enum NormalizationFallback {
    /// Return None if normalization fails
    None,
    /// Fall back to uniform distribution
    Uniform,
}

/// Normalize weights to probabilities that sum to 1.0 with configurable fallback.
///
/// This utility converts arbitrary non-negative weights into normalized probabilities.
/// When the total weight is zero or non-finite, different fallback strategies can be used.
///
/// # Arguments
///
/// * `weights` - Iterator of weight values
/// * `fallback` - Strategy to use when normalization fails
/// * `epsilon` - Minimum value to clamp probabilities to (prevents exact zeros)
///
/// # Returns
///
/// - `Some(Vec<f64>)` containing normalized probabilities
/// - `None` if normalization fails and `NormalizationFallback::None` is used
///
/// # Examples
///
/// ```
/// use menace::utils::{normalize_weights_with_options, NormalizationFallback};
///
/// // Basic normalization
/// let normalized = normalize_weights_with_options(
///     vec![1.0, 2.0, 1.0],
///     NormalizationFallback::None,
///     None,
/// ).unwrap();
/// assert_eq!(normalized, vec![0.25, 0.5, 0.25]);
///
/// // Fallback to uniform when total is zero
/// let normalized = normalize_weights_with_options(
///     vec![0.0, 0.0, 0.0],
///     NormalizationFallback::Uniform,
///     None,
/// ).unwrap();
/// assert_eq!(normalized, vec![1.0/3.0, 1.0/3.0, 1.0/3.0]);
///
/// // With epsilon clamping (values are clamped, then renormalized)
/// let normalized = normalize_weights_with_options(
///     vec![1.0, 0.0],
///     NormalizationFallback::None,
///     Some(1e-12),
/// ).unwrap();
/// // After renormalization, the second value will be smaller than epsilon
/// assert!(normalized.iter().all(|&x| x > 0.0));
/// assert!((normalized.iter().sum::<f64>() - 1.0).abs() < 1e-10);
/// ```
pub fn normalize_weights_with_options<I>(
    weights: I,
    fallback: NormalizationFallback,
    epsilon: Option<f64>,
) -> Option<Vec<f64>>
where
    I: IntoIterator<Item = f64>,
{
    let weights_vec: Vec<f64> = weights.into_iter().collect();

    if weights_vec.is_empty() {
        return match fallback {
            NormalizationFallback::None => None,
            NormalizationFallback::Uniform => Some(vec![]),
        };
    }

    let eps = epsilon.unwrap_or(0.0);
    let sum: f64 = weights_vec.iter().sum();

    // Check if sum is valid (positive and finite)
    if !sum.is_finite() || sum <= eps {
        return apply_fallback(fallback, weights_vec.len());
    }

    // Normalize and clamp to epsilon if provided
    let mut normalized: Vec<f64> = if eps > 0.0 {
        weights_vec.iter().map(|&w| (w / sum).max(eps)).collect()
    } else {
        weights_vec.iter().map(|&w| w / sum).collect()
    };

    // If epsilon was applied, renormalize to ensure sum = 1.0
    if eps > 0.0 {
        let total: f64 = normalized.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return apply_fallback(fallback, normalized.len());
        }
        for value in normalized.iter_mut() {
            *value /= total;
        }
    }

    Some(normalized)
}

/// Helper: Apply normalization fallback strategy
fn apply_fallback(fallback: NormalizationFallback, len: usize) -> Option<Vec<f64>> {
    match fallback {
        NormalizationFallback::None => None,
        NormalizationFallback::Uniform => {
            let uniform = 1.0 / len as f64;
            Some(vec![uniform; len])
        }
    }
}

/// Normalize weights to probabilities that sum to 1.0.
///
/// This utility converts arbitrary non-negative weights into normalized probabilities.
///
/// # Arguments
///
/// * `weights` - Iterator of non-negative weight values
///
/// # Returns
///
/// - `Some(Vec<f64>)` containing normalized probabilities if total weight is positive
/// - `None` if total weight is zero or negative
///
/// # Examples
///
/// ```
/// use menace::utils::normalize_weights;
///
/// // Normalize weights
/// let normalized = normalize_weights(vec![1.0, 2.0, 1.0]).unwrap();
/// assert_eq!(normalized, vec![0.25, 0.5, 0.25]);
///
/// // Zero total weight returns None
/// let normalized = normalize_weights(vec![0.0, 0.0]);
/// assert_eq!(normalized, None);
/// ```
pub fn normalize_weights<I>(weights: I) -> Option<Vec<f64>>
where
    I: IntoIterator<Item = f64>,
{
    normalize_weights_with_options(weights, NormalizationFallback::None, None)
}

/// Normalize weighted key-value pairs while preserving keys.
///
/// This is a convenience function that takes a vector of (K, f64) pairs,
/// normalizes the weights to sum to 1.0, and returns the keys paired with
/// normalized weights.
///
/// This pattern is common when you have discrete choices with weights
/// (e.g., moves with probabilities) and need to convert them to a proper
/// probability distribution.
///
/// # Type Parameters
///
/// * `K` - Type of the keys (e.g., move positions, state labels)
///
/// # Arguments
///
/// * `weighted_items` - Vector of (key, weight) pairs where weights are non-negative
///
/// # Returns
///
/// - `Some(Vec<(K, f64)>)` with normalized probabilities if total weight is positive
/// - `None` if total weight is zero or negative
///
/// # Examples
///
/// ```
/// use menace::utils::normalize_weighted_pairs;
///
/// // Normalize move weights
/// let moves = vec![(0, 1.0), (1, 2.0), (2, 1.0)];
/// let normalized = normalize_weighted_pairs(moves).unwrap();
/// assert_eq!(normalized, vec![(0, 0.25), (1, 0.5), (2, 0.25)]);
///
/// // Zero total weight returns None
/// let moves = vec![(0, 0.0), (1, 0.0)];
/// assert_eq!(normalize_weighted_pairs(moves), None);
/// ```
pub fn normalize_weighted_pairs<K>(weighted_items: Vec<(K, f64)>) -> Option<Vec<(K, f64)>> {
    // Extract keys and weights
    let (keys, weights): (Vec<_>, Vec<_>) = weighted_items.into_iter().unzip();

    // Normalize weights
    let normalized = normalize_weights(weights)?;

    // Zip keys back with normalized weights
    Some(keys.into_iter().zip(normalized).collect())
}

/// Performs weighted random sampling from a collection of items.
///
/// This utility implements the standard weighted sampling algorithm:
/// 1. Calculate total weight
/// 2. Generate a random threshold in [0, total)
/// 3. Iterate through items, subtracting weights until threshold is reached
/// 4. Return the item where threshold crosses zero
///
/// # Type Parameters
///
/// - `R`: Random number generator implementing `Rng`
/// - `T`: Type of items to sample from (must be `Clone`)
/// - `W`: Type representing weights (must convert to `f64`)
///
/// # Arguments
///
/// - `rng`: Mutable reference to a random number generator
/// - `items`: Slice of (item, weight) tuples to sample from
///
/// # Returns
///
/// - `Some(item)` if sampling succeeds
/// - `None` if the items slice is empty
///
/// # Behavior
///
/// - If all weights are zero or negative, falls back to uniform random selection
/// - If total weight is positive, performs weighted sampling
/// - The last item is returned as a fallback if threshold doesn't cross zero (numerical stability)
///
/// # Examples
///
/// ```
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
/// use menace::utils::weighted_sample;
///
/// let mut rng = StdRng::seed_from_u64(42);
/// let items = vec![("a", 1.0), ("b", 2.0), ("c", 1.0)];
/// let sampled = weighted_sample(&mut rng, &items);
/// assert!(sampled.is_some());
/// ```
pub fn weighted_sample<R, T, W>(rng: &mut R, items: &[(T, W)]) -> Option<T>
where
    R: Rng,
    T: Clone,
    W: Into<f64> + Copy,
{
    if items.is_empty() {
        return None;
    }

    // Calculate total weight
    let total: f64 = items.iter().map(|(_, w)| (*w).into()).sum();

    // If total weight is zero or negative, fall back to uniform sampling
    if total <= 0.0 {
        return items.choose(rng).map(|(item, _)| item.clone());
    }

    // Generate random threshold in [0, total)
    let mut threshold = rng.sample::<f64, _>(StandardUniform) * total;

    // Find the item where threshold crosses zero
    for (item, weight) in items {
        let w = (*weight).into();
        if threshold < w {
            return Some(item.clone());
        }
        threshold -= w;
    }

    // Fallback: return last item (for numerical stability)
    items.last().map(|(item, _)| item.clone())
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::StdRng};

    use super::*;

    #[test]
    fn test_weighted_sample_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        let items: Vec<(i32, f64)> = vec![];
        assert_eq!(weighted_sample(&mut rng, &items), None);
    }

    #[test]
    fn test_weighted_sample_single_item() {
        let mut rng = StdRng::seed_from_u64(42);
        let items = vec![("a", 1.0)];
        assert_eq!(weighted_sample(&mut rng, &items), Some("a"));
    }

    #[test]
    fn test_weighted_sample_zero_weights() {
        let mut rng = StdRng::seed_from_u64(42);
        let items = vec![("a", 0.0), ("b", 0.0), ("c", 0.0)];
        // Should fall back to uniform sampling
        let result = weighted_sample(&mut rng, &items);
        assert!(result.is_some());
    }

    #[test]
    fn test_weighted_sample_distribution() {
        let mut rng = StdRng::seed_from_u64(42);
        let items = vec![("a", 1.0), ("b", 2.0), ("c", 1.0)];

        // Sample many times and check distribution roughly matches weights
        let mut counts = std::collections::HashMap::new();
        for _ in 0..1000 {
            let sample = weighted_sample(&mut rng, &items).unwrap();
            *counts.entry(sample).or_insert(0) += 1;
        }

        // "b" should appear roughly twice as often as "a" or "c"
        let count_a = counts.get(&"a").copied().unwrap_or(0);
        let count_b = counts.get(&"b").copied().unwrap_or(0);
        let count_c = counts.get(&"c").copied().unwrap_or(0);

        assert!(count_b > count_a, "b should appear more than a");
        assert!(count_b > count_c, "b should appear more than c");
        assert!(count_a > 0 && count_c > 0, "all items should appear");
    }

    #[test]
    fn test_weighted_sample_with_u32_weights() {
        let mut rng = StdRng::seed_from_u64(42);
        let items = vec![(0, 1u32), (1, 2u32), (2, 1u32)];
        let result = weighted_sample(&mut rng, &items);
        assert!(result.is_some());
    }

    #[test]
    fn test_weighted_sample_deterministic() {
        // With the same seed, should produce the same result
        let items = vec![("a", 1.0), ("b", 2.0), ("c", 1.0)];

        let mut rng1 = StdRng::seed_from_u64(12345);
        let result1 = weighted_sample(&mut rng1, &items);

        let mut rng2 = StdRng::seed_from_u64(12345);
        let result2 = weighted_sample(&mut rng2, &items);

        assert_eq!(result1, result2);
    }
    #[test]
    fn normalize_weights_returns_none_for_zero_total() {
        let normalized = normalize_weights(vec![0.0, 0.0]);
        assert!(normalized.is_none(), "zero weights should return None");
    }

    #[test]
    fn normalize_weights_uniform_fallback() {
        let normalized = normalize_weights_with_options(
            vec![0.0, 0.0, 0.0],
            NormalizationFallback::Uniform,
            None,
        )
        .expect("uniform fallback should produce probabilities");
        assert_eq!(normalized, vec![1.0 / 3.0; 3]);
    }

    #[test]
    fn normalize_weights_with_epsilon_rebalances() {
        let normalized =
            normalize_weights_with_options(vec![1.0, 0.0], NormalizationFallback::None, Some(1e-6))
                .expect("epsilon-normalized weights should produce probabilities");
        assert!(
            normalized.iter().all(|p| *p > 0.0),
            "epsilon normalization should avoid zeros: {normalized:?}"
        );
        let sum: f64 = normalized.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "normalized weights must sum to 1, got {sum}"
        );
    }

    #[test]
    fn normalize_weighted_pairs_none_when_zero_total() {
        let normalized = normalize_weighted_pairs(vec![(0, 0.0), (1, 0.0)]);
        assert!(
            normalized.is_none(),
            "normalize_weighted_pairs should return None when total weight is zero"
        );
    }
}
