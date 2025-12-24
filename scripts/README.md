# MENACE Experiment Scripts

Automated experiment orchestration, analysis, and reporting for MENACE research.

## Directory Structure

```
scripts/
├── automation/
│   └── experiment_driver.py    # Main experiment orchestration
├── analysis/
│   └── statistical_analysis.py # Statistical utilities
├── reporting/
│   └── generate_performance_report.py
├── helpers/
│   └── *.py                    # Utility functions
└── tools/
    └── generate-xml-context.py # XML context generator for LLM review
```

## Quick Start

```bash
# Build release binary
cargo build --release

# Run experiments (uses configs/experiments.yaml)
make results-run

# Analyze and generate reports
make results-report

# Or run the full pipeline
make results
```

## Experiment Driver

The experiment driver orchestrates training runs with parameter sweeps, automatic data collection, and statistical analysis.

### Commands

```bash
# Run all experiments from config
python -m scripts.automation.experiment_driver run --config-file configs/experiments.yaml

# Run specific experiment
python -m scripts.automation.experiment_driver run --name Restock_Strategy_Comparison

# Analyze results
python -m scripts.automation.experiment_driver analyze --run-dir ./menace_data/Restock_Strategy_Comparison

# Generate report
python -m scripts.automation.experiment_driver report --run-dir ./menace_data/Restock_Strategy_Comparison
```

### Configuration

Experiments are defined in YAML. See `configs/experiments.yaml` and `configs/thesis_experiments.yaml`.

```yaml
project_name: "My_Experiments"
output_data_dir: "./menace_data"
menace_binary: "./target/release/menace"
default_seeds: 10

experiments:
  - name: "Experiment_Name"
    agent: "menace" # or "active-inference"
    base_args:
      filter: "michie"
      games: 500
    sweep:
      restock: ["none", "move", "box"] # Creates 3 conditions
```

### Output Structure

```
menace_data/
└── Experiment_Name/
    ├── condition_a/
    │   ├── seed_0/metrics.jsonl
    │   ├── seed_1/metrics.jsonl
    │   └── ...
    ├── condition_b/
    │   └── seed_*/metrics.jsonl
    └── analysis.parquet  # Created by analyze command
```

## Makefile Targets

| Target                | Description                         |
| --------------------- | ----------------------------------- |
| `make results`        | Run experiments + analyze + report  |
| `make results-run`    | Run experiments only                |
| `make results-report` | Analyze and generate reports        |
| `make thesis-results` | Run thesis experiment suite         |
| `make context`        | Generate XML context for LLM review |

## Dependencies

Python dependencies are managed via `pyproject.toml` and installed with:

```bash
make venv  # Uses uv to sync dependencies
```

Key packages: pandas, matplotlib, seaborn, pyyaml, pyarrow
