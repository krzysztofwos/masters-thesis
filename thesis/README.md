# Thesis (LaTeX)

This directory contains the LaTeX sources for the thesis.

## Build

```bash
make -C thesis
```

The build requires a TeX toolchain with Japanese language support (see `thesis/cimt/doc/` for the class documentation).

## Experiment Pipeline

Thesis experiments are configured via:

- `configs/thesis_experiments.yaml`
- `configs/evaluate_thesis.yml`

Common workflows:

```bash
make thesis-results          # run experiments + evaluations + reports
make thesis-results-run      # run experiments (+ evaluations)
make thesis-results-report   # analyze/report existing runs
```

## Packaging (External Audit)

```bash
make thesis-package-minimal
make thesis-package-curves
```

These targets call `thesis/tools/package_thesis_results.py`.

## Utilities

- `thesis/tools/generate-thesis-context.sh` — build a repo-wide XML context bundle
