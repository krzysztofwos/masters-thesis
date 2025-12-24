"""Unit tests for lightweight CLI helpers used in experiment automation."""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.helpers.eval_utils import (  # type: ignore  # noqa: E402
    ensure_train_opponent,
)
from scripts.helpers.report_utils import build_report_name  # type: ignore  # noqa: E402

HEX_SUFFIX = re.compile(r"[0-9a-f]{12}$")


def test_build_report_name_with_few_runs_has_digest_suffix():
    run_dirs = ["foo", "bar", "baz"]
    slug = build_report_name(run_dirs)
    assert slug.startswith("foo_vs_bar_vs_baz_")
    match = HEX_SUFFIX.search(slug)
    assert match, "slug should end with 12 hex chars"
    assert len(match.group(0)) == 12


def test_build_report_name_compresses_long_lists():
    run_dirs = ["exp1", "exp2", "exp3", "exp4", "exp5"]
    slug = build_report_name(run_dirs)
    assert "_vs_plus_2_" in slug, "slug should record how many runs were omitted"
    assert HEX_SUFFIX.search(slug)
    assert len(slug) < 64, "compressed slug should stay short"


def test_build_report_name_caps_extreme_prefix_length():
    long_name = "a" * 90
    run_dirs = [long_name, long_name + "_b", long_name + "_c", "extra"]
    slug = build_report_name(run_dirs)
    assert len(slug) <= 100, "slug should be truncated to avoid filesystem limits"
    assert HEX_SUFFIX.search(slug)


def test_ensure_train_opponent_injects_when_missing():
    base_args = {"games": 10}
    updated = ensure_train_opponent(base_args, "optimal")
    assert "opponent" not in base_args, "helper should not mutate the original dict"
    assert updated["opponent"] == "optimal"


def test_ensure_train_opponent_preserves_existing_value():
    base_args = {"opponent": "random", "games": 10}
    updated = ensure_train_opponent(base_args, "optimal")
    assert updated["opponent"] == "random"
    assert updated is base_args


def test_ensure_train_opponent_ignores_missing_eval_value():
    base_args = {"games": 10}
    updated = ensure_train_opponent(base_args, None)
    assert updated is base_args
