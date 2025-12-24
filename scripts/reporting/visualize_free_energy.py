#!/usr/bin/env python3
"""
Free Energy visualization script for MENACE analysis.

This script creates visualizations of Free Energy trajectories during training
to validate the hypothesis that reinforcement learning minimizes Free Energy.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe_denominator = denominator.replace(0, pd.NA).astype(float)
    return numerator / safe_denominator


def load_checkpoint_data(checkpoint_file: Path) -> pd.DataFrame:
    """Load Free Energy checkpoint data from JSON file."""
    with open(checkpoint_file, "r") as f:
        checkpoints = json.load(f)

    # Convert to DataFrame
    data = []
    for game_num, fe_components in checkpoints:
        data.append(
            {
                "game_num": game_num,
                "expected_surprise": fe_components["expected_surprise"],
                "kl_divergence": fe_components["kl_divergence"],
                "total_fe": fe_components["total"],
                "num_states": fe_components["num_states"],
            }
        )

    return pd.DataFrame(data)


def plot_free_energy_trajectory(
    df: pd.DataFrame,
    title: str = "MENACE Free Energy Minimization",
    output_path: Path | None = None,
):
    """Plot Free Energy components over training."""
    df = df.copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)

    # Normalize by number of states
    safe_states = df["num_states"].replace(0, pd.NA).astype(float)
    df["surprise_per_state"] = df["expected_surprise"] / safe_states
    df["kl_per_state"] = df["kl_divergence"] / safe_states
    df["fe_per_state"] = df["total_fe"] / safe_states

    # Plot 1: Total Free Energy
    ax = axes[0, 0]
    ax.plot(df["game_num"], df["total_fe"], "b-", linewidth=2, label="Total F(π)")
    ax.set_xlabel("Training Games")
    ax.set_ylabel("Free Energy (nats)")
    ax.set_title("Total Free Energy vs Training")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Components
    ax = axes[0, 1]
    ax.plot(
        df["game_num"],
        df["expected_surprise"],
        "r-",
        linewidth=2,
        label="Expected Surprise",
    )
    ax.plot(
        df["game_num"], df["kl_divergence"], "g-", linewidth=2, label="KL Divergence"
    )
    ax.set_xlabel("Training Games")
    ax.set_ylabel("Free Energy Component (nats)")
    ax.set_title("Free Energy Decomposition")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Normalized Free Energy (per state)
    ax = axes[1, 0]
    ax.plot(df["game_num"], df["fe_per_state"], "purple", linewidth=2)
    ax.set_xlabel("Training Games")
    ax.set_ylabel("Free Energy (nats/state)")
    ax.set_title("Normalized Free Energy (Average per State)")
    ax.grid(True, alpha=0.3)

    # Plot 4: Ratio of components
    ax = axes[1, 1]
    ax.plot(
        df["game_num"],
        df["surprise_per_state"],
        "r-",
        linewidth=2,
        label="Surprise/state",
    )
    ax.plot(df["game_num"], df["kl_per_state"], "g-", linewidth=2, label="KL/state")
    ax.set_xlabel("Training Games")
    ax.set_ylabel("Free Energy Component (nats/state)")
    ax.set_title("Normalized Components")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    return fig


def compare_agents(
    checkpoint_files: Dict[str, Path],
    title: str = "Free Energy Comparison Across Agents",
    output_path: Path | None = None,
):
    """Compare Free Energy trajectories across different agents."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16)

    # Load all data
    dfs = {}
    for agent_name, checkpoint_file in checkpoint_files.items():
        agent_df = load_checkpoint_data(checkpoint_file)
        safe_states = agent_df["num_states"].replace(0, pd.NA).astype(float)
        agent_df["fe_per_state"] = agent_df["total_fe"] / safe_states
        dfs[agent_name] = agent_df

    # Plot 1: Total Free Energy
    ax = axes[0]
    for agent_name, df in dfs.items():
        ax.plot(
            df["game_num"], df["total_fe"], linewidth=2, label=agent_name, marker="o"
        )
    ax.set_xlabel("Training Games")
    ax.set_ylabel("Free Energy (nats)")
    ax.set_title("Total Free Energy Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Normalized Free Energy
    ax = axes[1]
    for agent_name, df in dfs.items():
        ax.plot(
            df["game_num"],
            df["fe_per_state"],
            linewidth=2,
            label=agent_name,
            marker="o",
        )
    ax.set_xlabel("Training Games")
    ax.set_ylabel("Free Energy (nats/state)")
    ax.set_title("Normalized Free Energy Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison to {output_path}")
    else:
        plt.show()

    return fig


def print_summary_stats(df: pd.DataFrame, agent_name: str = "Agent"):
    """Print summary statistics for Free Energy analysis."""
    df = df.copy()
    print(f"\n=== Free Energy Summary: {agent_name} ===")
    initial_fe = df["total_fe"].iloc[0]
    final_fe = df["total_fe"].iloc[-1]
    total_reduction = initial_fe - final_fe

    print(f"Initial Free Energy: {initial_fe:.2f} nats")
    print(f"Final Free Energy: {final_fe:.2f} nats")
    print(f"Total Reduction: {total_reduction:.2f} nats")

    if abs(initial_fe) > 1e-12:
        percent_reduction = 100 * total_reduction / initial_fe
        print(f"Percent Reduction: {percent_reduction:.1f}%")
    else:
        print("Percent Reduction: N/A (initial Free Energy is zero)")

    # Check for monotonic decrease
    is_decreasing = (df["total_fe"].diff().dropna() <= 0).all()
    print(f"Monotonic Decrease: {'Yes' if is_decreasing else 'No'}")

    # Per-state metrics
    df["fe_per_state"] = _safe_divide(df["total_fe"], df["num_states"])
    print("\nNormalized Free Energy:")

    initial_norm = df["fe_per_state"].iloc[0]
    final_norm = df["fe_per_state"].iloc[-1]
    if pd.notna(initial_norm) and pd.notna(final_norm):
        print(f"  Initial: {initial_norm:.4f} nats/state")
        print(f"  Final: {final_norm:.4f} nats/state")
    else:
        print("  N/A (no analyzed decision states)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Free Energy trajectories from MENACE training"
    )
    parser.add_argument(
        "checkpoint_file", type=Path, help="Path to Free Energy checkpoint JSON file"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for visualization (default: show interactively)",
    )
    parser.add_argument(
        "--compare", type=Path, nargs="+", help="Additional checkpoint files to compare"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="MENACE Free Energy Minimization",
        help="Title for the plot",
    )

    args = parser.parse_args()

    if not args.checkpoint_file.exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint_file}")
        return 1

    # Load and visualize
    df = load_checkpoint_data(args.checkpoint_file)
    print_summary_stats(df, args.checkpoint_file.stem)

    if args.compare:
        # Comparison mode
        checkpoint_files = {args.checkpoint_file.stem: args.checkpoint_file}
        for compare_file in args.compare:
            if compare_file.exists():
                checkpoint_files[compare_file.stem] = compare_file
                print_summary_stats(
                    load_checkpoint_data(compare_file), compare_file.stem
                )
            else:
                print(f"Warning: Comparison file not found: {compare_file}")

        compare_agents(checkpoint_files, title=args.title, output_path=args.output)
    else:
        # Single agent mode
        plot_free_energy_trajectory(df, title=args.title, output_path=args.output)

    return 0


if __name__ == "__main__":
    exit(main())
