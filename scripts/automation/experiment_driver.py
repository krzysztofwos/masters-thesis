import argparse
import itertools
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

from scripts.helpers.report_utils import build_report_name

# Note: Statistical features (confidence intervals, t-tests, etc.) should be
# integrated directly here as the default reporting approach, not as a separate module


def _setup_logging(level: int) -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )


def get_project_root() -> Path:
    """Find the project root by searching for Cargo.toml."""
    current_path = Path(__file__).resolve()
    while current_path.parent != current_path:
        if (current_path / "Cargo.toml").exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("Project root with Cargo.toml not found.")


PROJECT_ROOT = get_project_root()


def main() -> None:
    """Main entry point for the MENACE experiment driver."""
    parser = argparse.ArgumentParser(description="Run MENACE experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiments.")
    run_parser.add_argument(
        "--config-file",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments.yaml",
        help="Path to the experiment configuration file.",
    )
    run_parser.add_argument("--name", type=str, help="Name of the experiment to run.")
    run_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (default: 1 = sequential).",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze experiment data.")
    analyze_parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to the experiment run directory.",
    )

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate reports.")
    report_parser.add_argument(
        "--run-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to the experiment run directory.",
    )
    report_parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "menace_reports",
        help="Directory to store generated reports.",
    )

    args = parser.parse_args()

    # Make run-dir paths absolute from project root if they are relative
    if hasattr(args, "run_dir") and args.run_dir:
        args.run_dir = [
            PROJECT_ROOT / d if not d.is_absolute() else d for d in args.run_dir
        ]

    if args.command == "run":
        run_experiments(args)
    elif args.command == "analyze":
        for run_dir in args.run_dir:
            analyze_data(run_dir)
    elif args.command == "report":
        generate_reports(args)


import json
import multiprocessing

from scripts.analysis.analyze_metrics import collate_stats


def _parse_seeds(spec, default: int) -> list[int]:
    """Parse seed specifications from YAML.

    Supported forms:
    - None: uses range(default)
    - int: uses range(int)
    - list: uses the list verbatim (coerced to ints)
    """
    if spec is None:
        return list(range(default))
    if isinstance(spec, int):
        return list(range(spec))
    return [int(seed) for seed in spec]


def run_job(cmd):
    """Worker function to run a single subprocess command."""
    try:
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, encoding="utf-8"
        )
        return None
    except subprocess.CalledProcessError as e:
        return e


def analyze_data(run_dir):
    """Analyzes the data from a single experiment run."""
    print(f"Analyzing data in {run_dir}")

    # Try to analyze training metrics first
    all_metrics = []
    for metric_file in run_dir.glob("**/*.jsonl"):
        parts = metric_file.parts
        condition_name = parts[-3]
        seed = int(parts[-2].split("_")[1])

        df = pd.read_json(metric_file, lines=True)
        df["condition"] = condition_name
        df["seed"] = seed
        all_metrics.append(df)

    if all_metrics:
        full_df = pd.concat(all_metrics, ignore_index=True)
        output_path = run_dir / "analysis.parquet"
        full_df.to_parquet(output_path)
        print(f"Analysis complete for training run. Saved to {output_path}")

    # Now, try to analyze EFE export data
    all_efe = []
    for efe_file in run_dir.glob("*.csv"):
        condition_name = efe_file.stem
        params = dict(item.split("_", 1) for item in condition_name.split("-"))
        df = pd.read_csv(efe_file)
        for key, value in params.items():
            df[key] = value
        all_efe.append(df)

    if all_efe:
        full_efe_df = pd.concat(all_efe, ignore_index=True)
        output_path = run_dir / "efe_analysis.parquet"
        full_efe_df.to_parquet(output_path)
        print(f"Analysis complete for EFE export. Saved to {output_path}")


def generate_reports(args):
    """Generates reports from the analyzed data."""
    print(f"Generating report for: {args.run_dir}")

    all_training_data = []
    all_efe_data = []

    for run_dir in args.run_dir:
        training_analysis_file = run_dir / "analysis.parquet"
        if training_analysis_file.exists():
            df = pd.read_parquet(training_analysis_file)
            df["experiment"] = run_dir.name
            all_training_data.append(df)

        efe_analysis_file = run_dir / "efe_analysis.parquet"
        if efe_analysis_file.exists():
            df = pd.read_parquet(efe_analysis_file)
            df["experiment"] = run_dir.name
            all_efe_data.append(df)

    _generate_basic_reports(all_training_data, all_efe_data, args)


def _generate_basic_reports(all_training_data, all_efe_data, args):
    """Generate basic reports (original implementation)."""

    num_plots = 0
    if all_training_data:
        num_plots += 3
    if all_efe_data:
        num_plots += 1

    if num_plots == 0:
        print(f"No analysis files found in the specified directories.")
        return

    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 8 * num_plots), squeeze=False)
    fig.suptitle(f"Experiment Comparison Report", fontsize=16)
    plot_idx = 0

    if all_training_data:
        df = pd.concat(all_training_data, ignore_index=True)
        # Combine experiment and condition for a unique hue
        df["hue"] = df["experiment"] + "-" + df["condition"]

        df["is_draw"] = (df["outcome"] == "Draw").astype(int)
        df_sorted = df.sort_values("game_num")
        group_cols = ["hue", "seed"]
        df["draw_rate"] = df_sorted.groupby(group_cols)["is_draw"].transform(
            lambda x: x.cumsum() / (pd.Series(range(1, len(x) + 1), index=x.index))
        )
        df["avg_steps"] = df_sorted.groupby(group_cols)["total_moves"].transform(
            lambda x: x.rolling(window=50, min_periods=1).mean()
        )

        final_games_df = df[df["game_num"] > df["game_num"].max() - 100]
        outcome_counts = (
            final_games_df.groupby(["hue", "outcome"]).size().unstack(fill_value=0)
        )
        outcome_percentages = (
            outcome_counts.div(outcome_counts.sum(axis=1), axis=0) * 100
        )

        # Plot 1: Draw Rate over Time (with 95% CI)
        ax = axes[plot_idx, 0]
        sns.lineplot(
            data=df,
            x="game_num",
            y="draw_rate",
            hue="hue",
            ax=ax,
            errorbar=("ci", 95),  # Add 95% confidence intervals
        )
        ax.set_title("Draw Rate vs. Games Played (with 95% CI)")
        ax.set_xlabel("Games Played")
        ax.set_ylabel("Draw Rate (Cumulative Average)")
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # Plot 2: Average Steps per Game (with 95% CI)
        ax = axes[plot_idx, 0]
        sns.lineplot(
            data=df,
            x="game_num",
            y="avg_steps",
            hue="hue",
            ax=ax,
            errorbar=("ci", 95),  # Add 95% confidence intervals
        )
        ax.set_title("Average Steps per Game (Rolling Mean, with 95% CI)")
        ax.set_xlabel("Games Played")
        ax.set_ylabel("Average Steps")
        ax.grid(True, alpha=0.3)
        plot_idx += 1

        # Plot 3: Final Outcome Percentages
        ax = axes[plot_idx, 0]
        outcome_percentages.plot(kind="bar", stacked=True, ax=ax, colormap="viridis")
        ax.set_title("Final Outcome Percentages (Last 100 Games)")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("Condition")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y")
        plot_idx += 1

    if all_efe_data:
        efe_df = pd.concat(all_efe_data, ignore_index=True)
        root_state_df = efe_df[efe_df["state"] == "........._X"]

        ax = axes[plot_idx, 0]
        sns.scatterplot(
            data=root_state_df,
            x="action_risk",
            y="action_epistemic",
            hue="beta",
            style="opponent",
            s=100,
            ax=ax,
            palette="magma",
        )
        ax.set_title("EFE Decomposition for Opening Move (........._X)")
        ax.set_xlabel("Pragmatic Value (Risk)")
        ax.set_ylabel("Epistemic Value (Info Gain)")
        ax.grid(True)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_name = build_report_name(args.run_dir)
    report_path = args.reports_dir / f"{report_name}_comparison_report.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(report_path)
    print(f"Basic report saved to {report_path}")


def run_experiments(args):
    """Parses the config file and runs the experiments in parallel."""
    with open(args.config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.name:
        experiments_to_run = [
            exp for exp in config["experiments"] if exp["name"] == args.name
        ]
        if not experiments_to_run:
            print(f"Error: Experiment '{args.name}' not found in {args.config_file}")
            return
    else:
        print("Running all experiments...")
        experiments_to_run = config["experiments"]

    jobs = []
    for experiment in experiments_to_run:
        print(f"Generating jobs for experiment: {experiment['name']}")
        base_args = experiment.get("base_args", {})
        menace_binary_path = PROJECT_ROOT / config["menace_binary"]
        seeds = _parse_seeds(
            experiment.get("seeds"),
            default=config.get("default_seeds", 1),
        )

        if "sub_experiments" in experiment:
            for sub_exp in experiment["sub_experiments"]:
                agent = sub_exp["agent"]
                sub_base_args = sub_exp.get("base_args", {})
                sub_seeds = (
                    seeds
                    if sub_exp.get("seeds") is None
                    else _parse_seeds(
                        sub_exp.get("seeds"),
                        default=config.get("default_seeds", 1),
                    )
                )
                sweep_params = sub_exp.get("sweep", {})
                param_names = list(sweep_params.keys())
                param_combinations = list(
                    itertools.product(*[sweep_params[key] for key in param_names])
                )

                for combo in param_combinations:
                    job_params = dict(zip(param_names, combo))
                    run_args = {**base_args, **sub_base_args, **job_params}

                    job_name_parts = [
                        f"{key}_{value}" for key, value in job_params.items()
                    ]
                    job_name = f"{agent}-" + "-".join(job_name_parts)
                    output_dir = (
                        PROJECT_ROOT
                        / config["output_data_dir"]
                        / experiment["name"]
                        / job_name
                    )

                    for seed in sub_seeds:
                        seed_dir = output_dir / f"seed_{seed}"
                        seed_dir.mkdir(parents=True, exist_ok=True)

                        cmd = [str(menace_binary_path)]
                        cmd.extend(["train", agent])
                        for key, value in run_args.items():
                            # Use --key=value format for negative numbers to avoid CLI parsing issues
                            if isinstance(value, (int, float)) and value < 0:
                                cmd.append(f"--{key.replace('_', '-')}={value}")
                            else:
                                cmd.append(f"--{key.replace('_', '-')}")
                                cmd.append(str(value))
                        cmd.append("--seed")
                        cmd.append(str(seed))
                        cmd.append("--observations")
                        cmd.append(str(seed_dir / "metrics.jsonl"))
                        cmd.append("--summary")
                        cmd.append(str(seed_dir / "training_summary.json"))
                        cmd.append("--output")
                        cmd.append(str(seed_dir / "agent.msgpack"))
                        jobs.append(cmd)
        else:
            sweep_params = experiment.get("sweep", {})
            param_names = list(sweep_params.keys())
            param_combinations = list(
                itertools.product(*[sweep_params[key] for key in param_names])
            )

            for combo in param_combinations:
                job_params = dict(zip(param_names, combo))
                run_args = {**base_args, **job_params}

                if experiment.get("type") == "export":
                    job_name_parts = [
                        f"{key}_{value}" for key, value in job_params.items()
                    ]
                    job_name = "-".join(job_name_parts)
                    output_dir = (
                        PROJECT_ROOT / config["output_data_dir"] / experiment["name"]
                    )
                    output_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = output_dir / f"{job_name}.csv"

                    cmd = [
                        str(menace_binary_path),
                        base_args["command"],
                        base_args["analyzer"],
                    ]
                    for key, value in run_args.items():
                        if key in ["command", "analyzer"]:
                            continue
                        cmd.append(f"--{key.replace('_', '-')}")
                        cmd.append(str(value))
                    cmd.append("--export")
                    cmd.append(str(csv_path))
                    jobs.append(cmd)
                else:
                    job_name_parts = [
                        f"{key}_{value}" for key, value in job_params.items()
                    ]
                    job_name = "-".join(job_name_parts)
                    output_dir = (
                        PROJECT_ROOT
                        / config["output_data_dir"]
                        / experiment["name"]
                        / job_name
                    )

                    for seed in seeds:
                        seed_dir = output_dir / f"seed_{seed}"
                        seed_dir.mkdir(parents=True, exist_ok=True)

                        cmd = [str(menace_binary_path)]
                        cmd.extend(["train", experiment["agent"]])
                        for key, value in run_args.items():
                            # Use --key=value format for negative numbers to avoid CLI parsing issues
                            if isinstance(value, (int, float)) and value < 0:
                                cmd.append(f"--{key.replace('_', '-')}={value}")
                            else:
                                cmd.append(f"--{key.replace('_', '-')}")
                                cmd.append(str(value))
                        cmd.append("--seed")
                        cmd.append(str(seed))
                        cmd.append("--observations")
                        cmd.append(str(seed_dir / "metrics.jsonl"))
                        cmd.append("--summary")
                        cmd.append(str(seed_dir / "training_summary.json"))
                        cmd.append("--output")
                        cmd.append(str(seed_dir / "agent.msgpack"))
                        jobs.append(cmd)

    worker_count = max(1, getattr(args, "workers", 1))
    if worker_count == 1:
        print(f"\nRunning {len(jobs)} job(s) sequentially...")
        results = [run_job(cmd) for cmd in jobs]
    else:
        print(f"\nRunning {len(jobs)} job(s) with {worker_count} worker(s)...")
        with multiprocessing.Pool(processes=worker_count) as pool:
            results = pool.map(run_job, jobs)

    print("\nRun complete. Summary of failures:")
    failures = 0
    for i, result in enumerate(results):
        if result is not None:
            failures += 1
            logging.error(f"Job failed: {' '.join(jobs[i])}")
            logging.error(f"Stderr: {result.stderr}")

    if failures == 0:
        print("All jobs completed successfully.")
    else:
        print(f"{failures} jobs failed.")

    # Refresh aggregate summaries so downstream reports stay current.
    try:
        runs, aggregates = collate_stats(PROJECT_ROOT / config["output_data_dir"])
        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        training_path = results_dir / "analysis_summary.json"
        payload = {
            "runs": [asdict(r) for r in runs],
            "aggregates": [asdict(a) for a in aggregates],
        }
        with training_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nUpdated training metrics in {training_path}")
    except Exception as err:  # pragma: no cover - best-effort logging
        logging.error("Failed to update training metrics: %s", err)

    # Optional evaluation stage
    evaluation_configs = config.get("evaluation_configs", [])
    for eval_config in evaluation_configs:
        eval_path = Path(eval_config)
        if not eval_path.is_absolute():
            eval_path = (PROJECT_ROOT / eval_path).resolve()
        if not eval_path.exists():
            logging.warning("Evaluation config %s not found; skipping.", eval_path)
            continue
        try:
            print(f"\nRunning evaluations defined in {eval_path}...")
            subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "scripts.automation.run_evaluations",
                    "--config",
                    str(eval_path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as err:  # pragma: no cover - logging only
            logging.error("Evaluation command failed: %s", err)


if __name__ == "__main__":
    main()
