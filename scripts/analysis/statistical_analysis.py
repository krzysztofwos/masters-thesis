"""Statistical analysis utilities for MENACE experiments."""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


def welch_satterthwaite_df(std1: float, n1: int, std2: float, n2: int) -> float:
    """Compute Welch–Satterthwaite degrees of freedom for two independent samples."""
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1_sq = std1**2
    s2_sq = std2**2
    se_sq = s1_sq / n1 + s2_sq / n2
    denom = (s1_sq**2) / (n1**2 * (n1 - 1)) + (s2_sq**2) / (n2**2 * (n2 - 1))
    if denom <= 0.0:
        return float("nan")
    return (se_sq**2) / denom


def compare_algorithms(
    data1: np.ndarray,
    data2: np.ndarray,
    label1: str,
    label2: str,
    alpha: float = 0.05,
    paired: bool = False,
) -> Dict[str, Union[float, str]]:
    """
    Compare two algorithms with proper statistical testing.

    Args:
        data1: Performance metric for algorithm 1 (e.g., games to first win)
        data2: Performance metric for algorithm 2
        label1: Name of algorithm 1
        label2: Name of algorithm 2
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary with statistical results
    """
    # Descriptive statistics
    mean1, std1 = np.mean(data1), np.std(data1, ddof=1)
    mean2, std2 = np.mean(data2), np.std(data2, ddof=1)
    n1, n2 = len(data1), len(data2)

    # Standard errors
    sem1 = std1 / np.sqrt(n1)
    sem2 = std2 / np.sqrt(n2)

    # Hypothesis test
    if paired:
        if n1 != n2:
            raise ValueError(
                f"Paired t-test requires equal sample sizes (got {n1} and {n2})."
            )
        t_stat, p_value = stats.ttest_rel(data1, data2)
        degrees_of_freedom = n1 - 1
        diff = data1 - data2
        diff_mean = float(np.mean(diff))
        diff_std = float(np.std(diff, ddof=1)) if n1 > 1 else 0.0
        cohen_d = diff_mean / diff_std if diff_std > 0 else 0.0  # Cohen's dz
        sem_diff = diff_std / np.sqrt(n1) if n1 > 1 else 0.0
        ci_diff = (
            stats.t.interval(0.95, degrees_of_freedom, loc=diff_mean, scale=sem_diff)
            if n1 > 1
            else (diff_mean, diff_mean)
        )
        test_name = "Paired t-test"
    else:
        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
        degrees_of_freedom = welch_satterthwaite_df(std1, n1, std2, n2)

        # Standardized mean difference (using RMS of group SDs).
        avg_std = np.sqrt((std1**2 + std2**2) / 2)
        cohen_d = (mean1 - mean2) / avg_std if avg_std > 0 else 0.0

        diff_mean = float(mean1 - mean2)
        sem_diff = float(np.sqrt(std1**2 / n1 + std2**2 / n2))
        ci_diff = (
            stats.t.interval(0.95, degrees_of_freedom, loc=diff_mean, scale=sem_diff)
            if np.isfinite(degrees_of_freedom)
            else (np.nan, np.nan)
        )
        test_name = "Welch's t-test"

    # 95% confidence intervals
    ci1 = stats.t.interval(0.95, n1 - 1, loc=mean1, scale=sem1)
    ci2 = stats.t.interval(0.95, n2 - 1, loc=mean2, scale=sem2)

    # Effect size interpretation
    if abs(cohen_d) < 0.2:
        effect_size_interp = "negligible"
    elif abs(cohen_d) < 0.5:
        effect_size_interp = "small"
    elif abs(cohen_d) < 0.8:
        effect_size_interp = "medium"
    else:
        effect_size_interp = "large"

    # Significance interpretation
    if p_value < 0.001:
        significance = "*** (p < 0.001)"
    elif p_value < 0.01:
        significance = "** (p < 0.01)"
    elif p_value < alpha:
        significance = "* (p < 0.05)"
    else:
        significance = "n.s. (not significant)"

    return {
        "algorithm_1": label1,
        "algorithm_2": label2,
        "mean_1": mean1,
        "mean_2": mean2,
        "std_1": std1,
        "std_2": std2,
        "sem_1": sem1,
        "sem_2": sem2,
        "ci_1_lower": ci1[0],
        "ci_1_upper": ci1[1],
        "ci_2_lower": ci2[0],
        "ci_2_upper": ci2[1],
        "t_statistic": t_stat,
        "p_value": p_value,
        "test": test_name,
        "df": degrees_of_freedom,
        "diff_ci_lower": ci_diff[0],
        "diff_ci_upper": ci_diff[1],
        "cohen_d": cohen_d,
        "effect_size": effect_size_interp,
        "significance": significance,
        "n_1": n1,
        "n_2": n2,
    }


def print_comparison(results: Dict[str, Union[float, str]]) -> None:
    """Print formatted comparison results."""
    print(f"\n{'='*70}")
    print(
        f"Statistical Comparison: {results['algorithm_1']} vs {results['algorithm_2']}"
    )
    print(f"{'='*70}")
    print(f"\n{results['algorithm_1']}:")
    print(
        f"  Mean ± SD: {results['mean_1']:.3f} ± {results['std_1']:.3f} (N={results['n_1']})"
    )
    print(f"  95% CI: [{results['ci_1_lower']:.3f}, {results['ci_1_upper']:.3f}]")
    print(f"\n{results['algorithm_2']}:")
    print(
        f"  Mean ± SD: {results['mean_2']:.3f} ± {results['std_2']:.3f} (N={results['n_2']})"
    )
    print(f"  95% CI: [{results['ci_2_lower']:.3f}, {results['ci_2_upper']:.3f}]")
    print(f"\nStatistical Test ({results['test']}):")
    df = results["df"]
    if isinstance(df, float) and np.isfinite(df) and not df.is_integer():
        df_fmt = f"{df:.1f}"
    else:
        df_fmt = str(int(df)) if np.isfinite(df) else "nan"
    print(f"  t({df_fmt}) = {results['t_statistic']:.3f}")
    print(f"  p = {results['p_value']:.4f} {results['significance']}")
    print(f"  Cohen's d = {results['cohen_d']:.3f} ({results['effect_size']} effect)")
    if results["test"] == "Paired t-test":
        print(
            f"  Mean difference 95% CI: [{results['diff_ci_lower']:.3f}, {results['diff_ci_upper']:.3f}]"
        )
    else:
        print(
            f"  Mean difference 95% CI: [{results['diff_ci_lower']:.3f}, {results['diff_ci_upper']:.3f}]"
        )

    # Interpretation
    diff = results["mean_1"] - results["mean_2"]
    if results["p_value"] < 0.05:
        direction = "faster" if diff < 0 else "slower"
        speedup = (
            abs(results["mean_2"] / results["mean_1"])
            if diff < 0
            else abs(results["mean_1"] / results["mean_2"])
        )
        print(f"\nInterpretation:")
        print(
            f"  {results['algorithm_1']} is {direction} than {results['algorithm_2']}"
        )
        print(f"  Speedup: {speedup:.1f}× (statistically significant)")
    else:
        print(f"\nInterpretation:")
        print(f"  No significant difference detected (p = {results['p_value']:.4f})")
    print(f"{'='*70}\n")


def bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> Tuple[List[bool], float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate (default 0.05)

    Returns:
        Tuple of (list of booleans indicating significance, corrected alpha)
    """
    n_comparisons = len(p_values)
    corrected_alpha = alpha / n_comparisons
    significant = [p < corrected_alpha for p in p_values]

    print(f"\nBonferroni Correction:")
    print(f"  Number of comparisons: {n_comparisons}")
    print(f"  Uncorrected α: {alpha}")
    print(f"  Corrected α: {corrected_alpha:.5f}")
    print(f"  Significant tests: {sum(significant)}/{n_comparisons}")

    return significant, corrected_alpha


def pairwise_comparison_table(
    data_dict: Dict[str, np.ndarray],
    metric_name: str = "Games to First Win",
) -> pd.DataFrame:
    """
    Create pairwise comparison table for multiple algorithms.

    Args:
        data_dict: Dictionary mapping algorithm names to performance arrays
        metric_name: Name of the metric being compared

    Returns:
        DataFrame with pairwise comparison results
    """
    algorithms = list(data_dict.keys())
    n_algs = len(algorithms)

    # Create matrix for p-values
    p_matrix = np.zeros((n_algs, n_algs))
    d_matrix = np.zeros((n_algs, n_algs))

    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            if i != j:
                results = compare_algorithms(
                    data_dict[alg1],
                    data_dict[alg2],
                    alg1,
                    alg2,
                )
                p_matrix[i, j] = results["p_value"]
                d_matrix[i, j] = results["cohen_d"]

    # Create DataFrame
    p_df = pd.DataFrame(p_matrix, index=algorithms, columns=algorithms)
    d_df = pd.DataFrame(d_matrix, index=algorithms, columns=algorithms)

    print(f"\n{metric_name}: Pairwise Comparison Matrix")
    print(f"{'='*80}")
    print("\nP-values (Welch's t-tests):")
    print(p_df.to_string(float_format=lambda x: f"{x:.4f}" if x > 0 else "---"))

    print("\n\nEffect Sizes (Cohen's d):")
    print(d_df.to_string(float_format=lambda x: f"{x:.3f}" if x != 0 else "---"))

    # Bonferroni correction - only use upper triangle (unique pairs)
    all_p_values = [p_matrix[i, j] for i in range(n_algs) for j in range(i + 1, n_algs)]
    significant, corrected_alpha = bonferroni_correction(all_p_values)

    print(
        f"\nSignificance Summary (with Bonferroni correction, α = {corrected_alpha:.5f}):"
    )
    sig_idx = 0
    for i, alg1 in enumerate(algorithms):
        for j, alg2 in enumerate(algorithms):
            if i < j:  # Only upper triangle
                if significant[sig_idx]:
                    print(
                        f"  {alg1} vs {alg2}: p = {p_matrix[i,j]:.4f}, d = {d_matrix[i,j]:.3f} ***"
                    )
                sig_idx += 1

    if sum(significant) == 0:
        print("  No significant differences detected after Bonferroni correction")

    return p_df, d_df


def mann_whitney_u_test(
    data1: np.ndarray,
    data2: np.ndarray,
    label1: str,
    label2: str,
) -> Dict[str, float]:
    """
    Non-parametric alternative to t-test (for small samples or non-normal data).

    Args:
        data1: Performance metric for algorithm 1
        data2: Performance metric for algorithm 2
        label1: Name of algorithm 1
        label2: Name of algorithm 2

    Returns:
        Dictionary with test results
    """
    u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative="two-sided")

    # Rank-biserial correlation (effect size for Mann-Whitney U)
    n1, n2 = len(data1), len(data2)
    rank_biserial = 1 - (2 * u_stat) / (n1 * n2)

    median1 = np.median(data1)
    median2 = np.median(data2)

    return {
        "algorithm_1": label1,
        "algorithm_2": label2,
        "median_1": median1,
        "median_2": median2,
        "u_statistic": u_stat,
        "p_value": p_value,
        "rank_biserial_correlation": rank_biserial,
        "n_1": n1,
        "n_2": n2,
    }


def create_summary_table(data_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Create summary statistics table for all algorithms.

    Args:
        data_dict: Dictionary mapping algorithm names to performance arrays

    Returns:
        DataFrame with summary statistics
    """
    summary = []

    for alg_name, data in data_dict.items():
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data, ddof=1)
        sem = std / np.sqrt(len(data))
        ci_lower, ci_upper = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=sem)

        summary.append(
            {
                "Algorithm": alg_name,
                "N": len(data),
                "Mean": mean,
                "Median": median,
                "Std Dev": std,
                "SEM": sem,
                "95% CI Lower": ci_lower,
                "95% CI Upper": ci_upper,
                "Min": np.min(data),
                "Max": np.max(data),
            }
        )

    df = pd.DataFrame(summary)
    return df.set_index("Algorithm")
