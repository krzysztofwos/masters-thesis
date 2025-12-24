import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

from scripts.helpers.eval_utils import ensure_train_opponent

AGENT_FILENAME = "agent.msgpack"
EVAL_EXPORT_FILENAME = "evaluation.json"


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "Cargo.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate project root (Cargo.toml not found)")


PROJECT_ROOT = get_project_root()
RESULTS_PATH = PROJECT_ROOT / "results" / "evaluation_summary.json"


def resolve_menace_binary(config_dir: Path, config: Dict) -> Path:
    binary_rel = Path(config.get("menace_binary", "target/release/menace"))
    return (PROJECT_ROOT / binary_rel).resolve()


def resolve_condition_dirs(
    base_dir: Path, experiment_cfg: Dict
) -> Iterable[Tuple[str, Path]]:
    if not base_dir.exists():
        return

    include = experiment_cfg.get("include_conditions", True)
    exclude: List[str] = experiment_cfg.get("exclude_conditions", []) or []

    if include is True:
        condition_names = [d.name for d in sorted(base_dir.iterdir()) if d.is_dir()]
    elif isinstance(include, list):
        condition_names = include
    else:
        condition_names = [str(include)]

    for name in condition_names:
        if name in exclude:
            continue
        condition_path = base_dir / name
        if condition_path.is_dir():
            yield name, condition_path


def iter_seed_dirs(condition_dir: Path) -> Iterable[Tuple[int, Path]]:
    for seed_dir in sorted(condition_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        try:
            seed = int(seed_dir.name.split("_")[-1])
        except ValueError:
            continue
        yield seed, seed_dir


def dict_to_cli_args(params: Dict) -> List[str]:
    args: List[str] = []
    for key, value in params.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            else:
                args.extend([flag, "false"])
        elif isinstance(value, list):
            for item in value:
                args.extend([flag, str(item)])
        else:
            args.extend([flag, str(value)])
    return args


def parse_seeds(spec, default: int) -> List[int]:
    if spec is None:
        return list(range(default))
    if isinstance(spec, int):
        return list(range(spec))
    return [int(seed) for seed in spec]


def run_evaluation(
    menace: Path,
    agent_path: Path,
    games: int,
    opponent: str,
    export_path: Path,
    seed: int,
    first_player: str,
) -> Dict:
    validation_seed = seed + 1
    cmd = [
        str(menace),
        "evaluate",
        str(agent_path),
        "--opponent",
        opponent,
        "--games",
        str(games),
        "--seed",
        str(seed),
        "--validation-seed",
        str(validation_seed),
        "--first-player",
        first_player,
        "--export",
        str(export_path),
    ]

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)

    try:
        data = json.loads(export_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise RuntimeError(
            f"Failed to parse evaluation export at {export_path}: {err}"
        ) from err

    return {
        "command": cmd,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "metrics": data.get("evaluation", {}),
        "agent": data.get("agent", {}),
    }


def summarise_runs(runs: List[Dict]) -> List[Dict]:
    aggregates: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for run in runs:
        key = (run["experiment"], run["condition"])
        aggregates[key].append(run)

    summary = []
    for (experiment, condition), items in aggregates.items():

        def collect(metric: str) -> Tuple[float, float]:
            values = [itm[metric] for itm in items]
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            return mean_val, std_val

        win_mean, win_std = collect("win_rate")
        draw_mean, draw_std = collect("draw_rate")
        loss_mean, loss_std = collect("loss_rate")

        summary.append(
            {
                "experiment": experiment,
                "condition": condition,
                "seeds": len(items),
                "games": statistics.mean([itm["games"] for itm in items]),
                "win_rate_mean": win_mean,
                "win_rate_std": win_std,
                "draw_rate_mean": draw_mean,
                "draw_rate_std": draw_std,
                "loss_rate_mean": loss_mean,
                "loss_rate_std": loss_std,
                "opponent": items[0].get("opponent"),
                "first_player": items[0].get("first_player"),
            }
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MENACE evaluations for trained agents"
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Evaluation configuration YAML"
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    menace_binary = resolve_menace_binary(args.config.parent, config)

    print(f"Using MENACE binary: {menace_binary}")

    runs: List[Dict] = []

    default_games = config.get("default_games", 100)
    default_opponent = config.get("default_opponent", "optimal")
    default_first_player = config.get("default_first_player", "x")
    default_seeds = config.get("default_seeds", 1)

    for experiment in config.get("experiments", []):
        mode = experiment.get("mode", "artifact")
        base_args = experiment.get("base_args", {})

        print(f"\n=== Experiment: {experiment['name']} ===")

        if mode == "artifact":
            agent_path = Path(base_args["agent_path"])
            if not agent_path.is_absolute():
                agent_path = (PROJECT_ROOT / agent_path).resolve()

            if not agent_path.exists():
                print(f"  ⚠️  Agent root not found, skipping: {agent_path}")
                continue

            games = base_args.get("games", default_games)
            opponent = base_args.get("opponent", default_opponent)
            first_player = base_args.get("first_player", default_first_player)

            print(f"Agent root: {agent_path}")
            print(
                f"Opponent: {opponent} | Games: {games} | First player: {first_player}"
            )

            for condition_name, condition_dir in resolve_condition_dirs(
                agent_path, experiment
            ):
                for seed, seed_dir in iter_seed_dirs(condition_dir):
                    agent_file = seed_dir / AGENT_FILENAME
                    if not agent_file.exists():
                        print(f"  ⚠️  Missing agent artifact: {agent_file}")
                        continue

                    export_path = seed_dir / EVAL_EXPORT_FILENAME
                    result = run_evaluation(
                        menace_binary,
                        agent_file,
                        games,
                        opponent,
                        export_path,
                        seed,
                        first_player,
                    )

                    metrics = result["metrics"]
                    run_record = {
                        "experiment": experiment["name"],
                        "condition": condition_name,
                        "seed": seed,
                        "agent_path": str(agent_file.relative_to(PROJECT_ROOT)),
                        "games": metrics.get("total_games", games),
                        "wins": metrics.get("wins"),
                        "draws": metrics.get("draws"),
                        "losses": metrics.get("losses"),
                        "win_rate": metrics.get("win_rate"),
                        "draw_rate": metrics.get("draw_rate"),
                        "loss_rate": metrics.get("loss_rate"),
                        "opponent": opponent,
                        "first_player": first_player,
                        "export": str(export_path.relative_to(PROJECT_ROOT)),
                    }
                    runs.append(run_record)
            continue

        if mode != "train_eval":
            raise ValueError(f"Unsupported evaluation mode: {mode}")

        seeds = parse_seeds(experiment.get("seeds"), default_seeds)
        base_train_args = base_args.get("train_args", {})
        validation_games = base_args.get("validation_games", default_games)
        base_opponent = base_args.get("opponent", default_opponent)
        base_first_player = base_args.get("first_player", default_first_player)
        base_learner = base_args.get("learner")

        conditions = experiment.get("conditions")
        if not conditions:
            raise ValueError(
                f"Train-eval experiment '{experiment['name']}' requires conditions."
            )

        output_root = PROJECT_ROOT / "results" / "eval_runs" / experiment["name"]

        for condition in conditions:
            condition_label = condition["label"]
            learner = condition.get("learner", base_learner)
            if learner is None:
                raise ValueError(
                    f"Condition '{condition_label}' missing learner in experiment '{experiment['name']}'."
                )

            train_args = dict(base_train_args)
            train_args.update(condition.get("train_args", {}))
            eval_games = condition.get("validation_games", validation_games)
            eval_opponent = condition.get("opponent", base_opponent)
            eval_first_player = condition.get("first_player", base_first_player)
            condition_seeds = (
                parse_seeds(condition.get("seeds"), len(seeds))
                if condition.get("seeds") is not None
                else seeds
            )

            # Ensure we only pass one --opponent flag to the CLI. If the training
            # configuration does not specify an opponent, reuse the evaluation
            # opponent (which in most cases matches the intended training rival).
            train_args = ensure_train_opponent(train_args, eval_opponent)

            print(
                f"Condition: {condition_label} | Learner: {learner} | Train args: {train_args} | Eval opponent: {eval_opponent}"
            )

            for seed in condition_seeds:
                per_seed_dir = output_root / condition_label / f"seed_{seed}"
                per_seed_dir.mkdir(parents=True, exist_ok=True)
                summary_path = per_seed_dir / EVAL_EXPORT_FILENAME

                cmd = [
                    str(menace_binary),
                    "train",
                    learner,
                    *dict_to_cli_args(train_args),
                    "--seed",
                    str(seed),
                    "--validation-games",
                    str(eval_games),
                    "--first-player",
                    eval_first_player,
                    "--summary",
                    str(summary_path),
                ]

                print("Running:", " ".join(cmd))
                completed = subprocess.run(
                    cmd, check=True, text=True, capture_output=True
                )

                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                validation = summary_data.get("validation")
                if validation is None:
                    raise RuntimeError(
                        f"Validation results missing for {experiment['name']} / {condition_label} seed {seed}"
                    )

                record_opponent = train_args.get("opponent", eval_opponent)

                run_record = {
                    "experiment": experiment["name"],
                    "condition": condition_label,
                    "seed": seed,
                    "agent_path": None,
                    "games": validation.get("total_games", eval_games),
                    "wins": validation.get("wins"),
                    "draws": validation.get("draws"),
                    "losses": validation.get("losses"),
                    "win_rate": validation.get("win_rate"),
                    "draw_rate": validation.get("draw_rate"),
                    "loss_rate": validation.get("loss_rate"),
                    "opponent": record_opponent,
                    "first_player": eval_first_player,
                    "export": str(summary_path.relative_to(PROJECT_ROOT)),
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
                runs.append(run_record)

    if not runs:
        print(
            "\n⚠️  No evaluation runs executed; check agent artifacts or configuration."
        )
        payload = {"runs": [], "aggregates": []}
    else:
        aggregates = summarise_runs(runs)
        payload = {"runs": runs, "aggregates": aggregates}

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nWrote evaluation summary to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
