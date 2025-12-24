SHELL := /bin/bash
TIMESTAMP := $(shell date +%Y-%m-%dT%H:%M:%S)

VENV_PY := .venv/bin/python
PYTHON ?= $(VENV_PY)
EXPERIMENT_CONFIG ?= configs/experiments.yaml
THESIS_EXPERIMENT_CONFIG ?= configs/thesis_experiments.yaml
RUN_DIRS ?= $(patsubst %/,%,$(wildcard menace_data/*/))
THESIS_RUN_DIRS ?= \
	menace_data/Pure_AIF_vs_MENACE_Optimal \
	menace_data/AIF_Variants_Showdown \
	menace_data/Pure_AIF_Beta_Sweep \
	menace_data/QL_SARSA_Baselines \
	menace_data/Filter_Effect \
	menace_data/Restock_Strategy_Comparison

.PHONY: help
help:
	@echo "Common tasks:"
	@echo "  make check                   # fmt + Clippy + test"
	@echo "  make results                 # run experiments, analyze, and regenerate reports"
	@echo "  make results-run             # run experiments (+ evaluations)"
	@echo "  make results-report          # analyze/report current RUN_DIRS + refresh performance report"
	@echo ""
	@echo "Context generation (for LLM review):"
	@echo "  make context                 # generate XML context with code + thesis + configs"
	@echo "  make context-code            # generate XML context with code only"
	@echo ""
	@echo "Thesis tasks:"
	@echo "  make thesis-results          # run thesis experiments + reports"
	@echo "  make thesis-results-run      # run thesis experiments (+ evaluations)"
	@echo "  make thesis-results-report   # analyze/report THESIS_RUN_DIRS"
	@echo "  make thesis-package-minimal  # ZIP minimal seed-level artifacts"

.PHONY: fmt
fmt:
	cargo fmt --all

.PHONY: venv
venv: $(VENV_PY)

$(VENV_PY): pyproject.toml uv.lock
	uv sync

.PHONY: clippy
clippy:
	cargo clippy --all-targets --all-features -- -D warnings

.PHONY: test
test:
	cargo test

.PHONY: lint
lint: fmt clippy

.PHONY: check
check: lint test

.PHONY: release
release:
	cargo build --release

.PHONY: results-run
results-run: release venv
	$(PYTHON) -m scripts.automation.experiment_driver run --config-file $(EXPERIMENT_CONFIG)

.PHONY: run-dir-check
run-dir-check:
	@if [ -z "$(strip $(RUN_DIRS))" ]; then \
		echo "No RUN_DIRS detected. Set RUN_DIRS=\"menace_data/<run_name> [...]\" or run 'make results-run' first."; \
		exit 1; \
	fi

.PHONY: results-report
results-report: run-dir-check venv
	$(PYTHON) -m scripts.automation.experiment_driver analyze --run-dir $(RUN_DIRS)
	$(PYTHON) -m scripts.automation.experiment_driver report --run-dir $(RUN_DIRS)
	$(PYTHON) -m scripts.reporting.generate_performance_report

.PHONY: results
results: results-run results-report

.PHONY: thesis-results-run
thesis-results-run: release venv
	$(PYTHON) -m scripts.automation.experiment_driver run --config-file $(THESIS_EXPERIMENT_CONFIG)

.PHONY: thesis-results-report
thesis-results-report: venv
	$(MAKE) results-report RUN_DIRS="$(THESIS_RUN_DIRS)"

.PHONY: thesis-results
thesis-results: thesis-results-run thesis-results-report

.PHONY: thesis-package-minimal
thesis-package-minimal: venv
	$(PYTHON) thesis/tools/package_thesis_results.py --out thesis_results_minimal.zip

.PHONY: thesis-package-curves
thesis-package-curves: venv
	$(PYTHON) thesis/tools/package_thesis_results.py --include-metrics --include-efe --out thesis_results_with_curves.zip

# Context generation for LLM review
# Repository-specific exclusions are configured here via flags
.PHONY: context
context: venv
	$(PYTHON) scripts/tools/generate-xml-context.py \
		--language bib \
		--language config \
		--language markdown \
		--language python \
		--language rust \
		--language tex \
		--exclude-dir .claude \
		--exclude-dir .quarto \
		--exclude-dir _site \
		--exclude-dir inbox \
		--exclude-dir menace_data \
		--exclude-dir resources \
		--exclude-dir results \
		--git-tracked \
		--output context-$(TIMESTAMP).xml
	@echo "Generated context-$(TIMESTAMP).xml"
	@grep -c '<file path=' context-$(TIMESTAMP).xml || echo "0 files"

.PHONY: context-code
context-code: venv
	$(PYTHON) scripts/tools/generate-xml-context.py \
		--language python \
		--language rust \
		--git-tracked \
		--output context-code-$(TIMESTAMP).xml
	@echo "Generated context-code-$(TIMESTAMP).xml"
	@grep -c '<file path=' context-code-$(TIMESTAMP).xml || echo "0 files"
