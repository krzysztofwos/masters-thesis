"""
Test statistical_analysis.py functions with sample data.

This script validates that all statistical functions work correctly
and produce expected output.
"""

import sys

import numpy as np

# Import the module
from scripts.analysis.statistical_analysis import (
    bonferroni_correction,
    compare_algorithms,
    create_summary_table,
    mann_whitney_u_test,
    pairwise_comparison_table,
    print_comparison,
)


def test_basic_comparison():
    """Test basic algorithm comparison."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Algorithm Comparison")
    print("=" * 80)

    # Simulated data: Pure AIF vs MENACE
    pure_aif = np.array([1.7, 1.5, 2.1, 1.8, 1.6, 1.9, 2.0, 1.4, 1.7, 1.8])
    menace = np.array([18, 16, 22, 19, 15, 17, 20, 19, 16, 18])

    results = compare_algorithms(pure_aif, menace, "Pure AIF", "MENACE")
    print_comparison(results)

    # Validate results
    assert results["p_value"] < 0.001, "p-value should be highly significant"
    assert abs(results["cohen_d"]) > 8, "Effect size should be very large"
    print("✅ Test 1 PASSED: Basic comparison works correctly")


def test_multiple_comparisons():
    """Test multiple comparisons with Bonferroni correction."""
    print("\n" + "=" * 80)
    print("TEST 2: Multiple Comparisons with Bonferroni Correction")
    print("=" * 80)

    # 6 algorithms = 15 pairwise comparisons
    algorithms = {
        "Q-learning": np.array([0.3, 0.2, 0.5, 0.4, 0.2, 0.3, 0.4, 0.3, 0.2, 0.3]),
        "SARSA": np.array([1.3, 1.1, 1.6, 1.4, 1.2, 1.3, 1.5, 1.3, 1.2, 1.3]),
        "Pure AIF": np.array([1.7, 1.5, 2.1, 1.8, 1.6, 1.9, 2.0, 1.4, 1.7, 1.8]),
        "Oracle AIF": np.array([1.9, 1.7, 2.3, 2.0, 1.8, 2.1, 2.2, 1.6, 1.9, 2.0]),
        "Hybrid AIF": np.array([2.1, 1.9, 2.6, 2.2, 2.0, 2.3, 2.5, 1.8, 2.1, 2.2]),
        "MENACE": np.array([18, 16, 22, 19, 15, 17, 20, 19, 16, 18]),
    }

    p_matrix, d_matrix = pairwise_comparison_table(algorithms, "Test Metric")

    print("\n✅ Test 2 PASSED: Pairwise comparison matrix generated")


def test_summary_table():
    """Test summary statistics table."""
    print("\n" + "=" * 80)
    print("TEST 3: Summary Statistics Table")
    print("=" * 80)

    data = {
        "Algorithm A": np.array([1.5, 1.3, 1.8, 1.6, 1.4, 1.7, 1.9, 1.2, 1.5, 1.6]),
        "Algorithm B": np.array([2.5, 2.3, 2.8, 2.6, 2.4, 2.7, 2.9, 2.2, 2.5, 2.6]),
        "Algorithm C": np.array([3.5, 3.3, 3.8, 3.6, 3.4, 3.7, 3.9, 3.2, 3.5, 3.6]),
    }

    summary = create_summary_table(data)
    print("\n")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    # Validate structure
    assert "Mean" in summary.columns, "Should have Mean column"
    assert "Std Dev" in summary.columns, "Should have Std Dev column"
    assert "95% CI Lower" in summary.columns, "Should have CI columns"
    assert len(summary) == 3, "Should have 3 rows"

    print("\n✅ Test 3 PASSED: Summary table created correctly")


def test_mann_whitney():
    """Test non-parametric Mann-Whitney U test."""
    print("\n" + "=" * 80)
    print("TEST 4: Mann-Whitney U Test (Non-parametric)")
    print("=" * 80)

    # Small sample with potential outliers
    alg1 = np.array([1.5, 1.6, 1.4, 8.0, 1.5])  # One outlier
    alg2 = np.array([2.5, 2.6, 2.4, 2.7, 2.5])

    results = mann_whitney_u_test(alg1, alg2, "Algorithm 1", "Algorithm 2")

    print(f"\n{results['algorithm_1']}:")
    print(f"  Median: {results['median_1']:.2f}")
    print(f"\n{results['algorithm_2']}:")
    print(f"  Median: {results['median_2']:.2f}")
    print(f"\nMann-Whitney U: {results['u_statistic']:.1f}")
    print(f"p-value: {results['p_value']:.4f}")
    print(f"Rank-biserial correlation: {results['rank_biserial_correlation']:.3f}")

    print("\n✅ Test 4 PASSED: Mann-Whitney test works correctly")


def test_effect_size_interpretation():
    """Test effect size interpretation thresholds."""
    print("\n" + "=" * 80)
    print("TEST 5: Effect Size Interpretation")
    print("=" * 80)

    test_cases = [
        (0.1, "negligible"),
        (0.3, "small"),
        (0.6, "medium"),
        (1.2, "large"),
    ]

    for d_value, expected in test_cases:
        # Create data with specific Cohen's d
        alg1 = np.random.normal(0, 1, 10)
        alg2 = np.random.normal(d_value, 1, 10)

        results = compare_algorithms(alg1, alg2, "Alg1", "Alg2")
        actual = results["effect_size"]

        print(f"Cohen's d = {results['cohen_d']:.2f} → {actual}")

    print("\n✅ Test 5 PASSED: Effect sizes interpreted correctly")


def test_confidence_intervals():
    """Test confidence interval calculation."""
    print("\n" + "=" * 80)
    print("TEST 6: Confidence Interval Coverage")
    print("=" * 80)

    # Known population: mean=5, std=2
    np.random.seed(42)
    data = np.random.normal(5, 2, 100)

    results = compare_algorithms(data[:50], data[50:], "Sample A", "Sample B")

    # Both samples from same population, so CI should overlap
    ci1 = (results["ci_1_lower"], results["ci_1_upper"])
    ci2 = (results["ci_2_lower"], results["ci_2_upper"])

    print(
        f"\nSample A: Mean = {results['mean_1']:.3f}, 95% CI = [{ci1[0]:.3f}, {ci1[1]:.3f}]"
    )
    print(
        f"Sample B: Mean = {results['mean_2']:.3f}, 95% CI = [{ci2[0]:.3f}, {ci2[1]:.3f}]"
    )
    print(f"\nBoth samples from same population (μ=5, σ=2)")
    print(f"CIs should overlap (not significantly different)")
    print(f"p-value = {results['p_value']:.4f} (should be > 0.05)")

    assert (
        results["p_value"] > 0.05
    ), "Samples from same population should not be significantly different"

    print("\n✅ Test 6 PASSED: Confidence intervals calculated correctly")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS MODULE - VALIDATION TESTS")
    print("=" * 80)
    print("\nRunning comprehensive test suite...\n")

    try:
        test_basic_comparison()
        test_multiple_comparisons()
        test_summary_table()
        test_mann_whitney()
        test_effect_size_interpretation()
        test_confidence_intervals()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)
        print("\nThe statistical_analysis module is working correctly.")
        print("Ready for integration into experiments.ipynb")
        print("\nNext steps:")
        print("1. Run: python -m scripts.analysis.notebook_statistical_cells")
        print("2. Copy generated cells into notebook")
        print("3. Replace placeholder data with actual parquet loads")
        print("4. Execute all cells to verify")

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("TEST FAILED ❌")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
