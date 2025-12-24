import json
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = PROJECT_ROOT / "results" / "evaluation_summary.json"
REPORT_PATH = PROJECT_ROOT / "results" / "performance_report.md"


def load_summary() -> Dict[Tuple[str, str], Dict]:
    data = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    aggregates = {}
    for entry in data.get("aggregates", []):
        key = (entry["experiment"], entry["condition"])
        aggregates[key] = entry
    return aggregates


def format_rate(mean: float, std: float) -> str:
    return f"{mean * 100:.1f}% ± {std * 100:.1f}%"


def main() -> None:
    aggregates = load_summary()

    rows = [
        {
            "candidates": [("Pure_AIF_vs_MENACE_Eval", "menace-restock_box")],
            "label": "MENACE (restock box)",
            "note": "Mixed curriculum",
        },
        {
            "candidates": [
                ("AIF_Variants_Showdown_Eval", "active-inference-ai_beta_0.5")
            ],
            "label": "Hybrid AIF (β=0.5)",
            "note": "Train regimen: optimal",
        },
        {
            "candidates": [
                ("AIF_Variants_Showdown_Eval", "pure-active-inference-ai_beta_0.5")
            ],
            "label": "Pure AIF (β=0.5)",
            "note": "Train regimen: optimal",
        },
        {
            "candidates": [("Pure_AIF_Beta_Sweep_Eval", "ai_beta_0.0")],
            "label": "Pure AIF (β=0.0)",
            "note": "Train regimen: optimal",
        },
        {
            "candidates": [("Pure_AIF_Beta_Sweep_Eval", "ai_beta_0.25")],
            "label": "Pure AIF (β=0.25)",
            "note": "Train regimen: optimal",
        },
        {
            "candidates": [
                ("Oracle_AIF_Eval", "oracle-active-inference-ai_beta_0.5"),
                ("AIF_Variants_Showdown_Eval", "oracle-active-inference-ai_beta_0.5"),
            ],
            "label": "Oracle AIF (β=0.5)",
            "note": "Tree-derived policy (cache only)",
        },
        {
            "candidates": [
                ("TD_Learners_Eval", "q-learning-vs-random"),
                ("QL_SARSA_Baselines_Eval", "q-learning-opponent_random"),
            ],
            "label": "Q-learning (random regimen)",
            "note": "Train regimen: random",
        },
        {
            "candidates": [
                ("TD_Learners_Eval", "q-learning-vs-defensive"),
                ("QL_SARSA_Baselines_Eval", "q-learning-opponent_defensive"),
            ],
            "label": "Q-learning (defensive regimen)",
            "note": "Train regimen: defensive",
        },
        {
            "candidates": [
                ("TD_Learners_Eval", "sarsa-vs-random"),
                ("QL_SARSA_Baselines_Eval", "sarsa-opponent_random"),
            ],
            "label": "SARSA (random regimen)",
            "note": "Train regimen: random",
        },
        {
            "candidates": [
                ("TD_Learners_Eval", "sarsa-vs-defensive"),
                ("QL_SARSA_Baselines_Eval", "sarsa-opponent_defensive"),
            ],
            "label": "SARSA (defensive regimen)",
            "note": "Train regimen: defensive",
        },
    ]

    header = (
        "| Algorithm | Eval Opponent | Draw Rate | Loss Rate | Seeds | Notes |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
    )
    lines = [header]

    for row in rows:
        candidates = row["candidates"]
        entry = None
        for key in candidates:
            entry = aggregates.get(key)
            if entry:
                break
        if not entry:
            print(f"⚠️  Missing aggregate for {candidates}")
            continue
        draw = format_rate(entry["draw_rate_mean"], entry["draw_rate_std"])
        loss = format_rate(entry["loss_rate_mean"], entry["loss_rate_std"])
        opponent = entry.get("opponent", "see Notes")
        seeds = entry.get("seeds", "?")
        lines.append(
            f"| {row['label']} | {opponent} | {draw} | {loss} | {seeds} | {row['note']} |\n"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote performance report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
