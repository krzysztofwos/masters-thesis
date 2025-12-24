#!/usr/bin/env python3
"""
Package the minimal subset of menace_data/ needed to reproduce thesis tables/figures.

This script intentionally excludes large binary artifacts (e.g. agent.msgpack) while
including seed-level evaluation outputs (and optionally learning curves / EFE exports).

Outputs a single ZIP that contains:
- selected result files
- MANIFEST.json with per-file sha256 + sizes

Expected repo layout (relative to --repo-root):
- menace_data/   experiment outputs (seed_*/evaluation.json, training_summary.json, ...)
- results/       aggregated JSON summaries produced by the reporting pipeline
- configs/       experiment + evaluation configurations

Usage (repo root):
  python thesis/tools/package_thesis_results.py --out thesis_results_minimal.zip
  python thesis/tools/package_thesis_results.py --include-metrics --include-efe --out thesis_results_with_curves.zip
  python thesis/tools/package_thesis_results.py --include-metrics --include-efe --out thesis_results_full_validation.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

DEFAULT_RUN_DIRS = [
    "Pure_AIF_vs_MENACE_Optimal",
    "AIF_Variants_Showdown",
    "Pure_AIF_Beta_Sweep",
    "QL_SARSA_Baselines",
    "Filter_Effect",
    "Restock_Strategy_Comparison",
    "AIF_EFE_Export",
]

KEEP_EVAL = "evaluation.json"
KEEP_METRICS = "metrics.jsonl"
KEEP_TRAINING_SUMMARY = "training_summary.json"

EXCLUDE_NAMES = {
    "agent.msgpack",
}

KEEP_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".tex"}

ZIP_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def should_skip(path: Path) -> bool:
    return path.name in EXCLUDE_NAMES


def iter_seed_dirs(run_dir: Path) -> Iterable[Path]:
    for seed_dir in sorted(run_dir.rglob("seed_*")):
        if seed_dir.is_dir():
            yield seed_dir


def add_file(files: list[Path], path: Path) -> None:
    if not path.exists() or not path.is_file() or should_skip(path):
        return
    files.append(path)


def write_file(z: ZipFile, src: Path, arcname: str) -> None:
    info = ZipInfo(arcname)
    info.date_time = ZIP_FIXED_DATE_TIME
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with src.open("rb") as handle, z.open(info, "w") as out:
        shutil.copyfileobj(handle, out, length=1024 * 1024)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package minimal thesis artifacts for external audit"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root containing menace_data/ (default: current directory)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override menace_data path (default: <repo-root>/menace_data)",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=DEFAULT_RUN_DIRS,
        help="Subdirs of menace_data to include (default: thesis runs)",
    )
    parser.add_argument(
        "--include-metrics",
        action="store_true",
        help="Include per-seed training traces (metrics.jsonl) for learning-curve verification",
    )
    parser.add_argument(
        "--include-efe",
        action="store_true",
        help="Include AIF_EFE_Export CSVs for EFE decomposition verification",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output ZIP path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or (repo_root / "menace_data")).resolve()
    if not data_dir.exists():
        print(f"ERROR: menace_data not found at {data_dir}", file=sys.stderr)
        return 2

    run_dirs = list(args.run_dirs)
    if args.include_efe and "AIF_EFE_Export" not in run_dirs:
        run_dirs.append("AIF_EFE_Export")

    if args.include_efe:
        efe_dir = data_dir / "AIF_EFE_Export"
        required_exports = [
            efe_dir / "beta_0.0-opponent_uniform.csv",
            efe_dir / "beta_0.5-opponent_uniform.csv",
        ]
        missing_exports = [path for path in required_exports if not path.exists()]
        if missing_exports:
            print(
                "ERROR: --include-efe requested but required EFE exports are missing:",
                file=sys.stderr,
            )
            for path in missing_exports:
                print(f"  - {path}", file=sys.stderr)
            print(
                "Run `make thesis-results-run` (or regenerate menace_data/AIF_EFE_Export) then retry.",
                file=sys.stderr,
            )
            return 2

    files: list[Path] = []

    add_file(files, repo_root / "configs" / "thesis_experiments.yaml")
    add_file(files, repo_root / "configs" / "evaluate_thesis.yml")
    add_file(files, repo_root / "results" / "analysis_summary.json")
    add_file(files, repo_root / "results" / "evaluation_summary.json")
    add_file(files, repo_root / "results" / "performance_report.md")

    for run in run_dirs:
        run_path = data_dir / run
        if not run_path.exists():
            continue

        if run == "AIF_EFE_Export":
            if not args.include_efe:
                continue
            for path in sorted(run_path.rglob("*")):
                if (
                    path.is_file()
                    and path.suffix.lower() in KEEP_SUFFIXES
                    and not should_skip(path)
                ):
                    add_file(files, path)
            continue

        for seed_dir in iter_seed_dirs(run_path):
            add_file(files, seed_dir / KEEP_TRAINING_SUMMARY)
            add_file(files, seed_dir / KEEP_EVAL)
            if args.include_metrics:
                add_file(files, seed_dir / KEEP_METRICS)

    # Deduplicate and sort to keep stable ordering.
    unique_files = sorted({path.resolve() for path in files})
    total_bytes = sum(path.stat().st_size for path in unique_files)

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = []
    with ZipFile(
        out_path,
        "w",
        compression=ZIP_DEFLATED,
        compresslevel=9,
    ) as z:
        for path in unique_files:
            rel = path.relative_to(repo_root).as_posix()
            write_file(z, path, rel)
            manifest.append(
                {"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )

        manifest_payload = {
            "total_bytes": total_bytes,
            "files": manifest,
        }
        info = ZipInfo("MANIFEST.json")
        info.date_time = ZIP_FIXED_DATE_TIME
        info.compress_type = ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        z.writestr(info, json.dumps(manifest_payload, indent=2) + "\n")

    print(f"Wrote: {out_path}")
    print(f"Included files: {len(unique_files)}")
    print(f"Raw size (before zip): {total_bytes / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
