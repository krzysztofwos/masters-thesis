"""Shared helpers for experiment automation/reporting."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def build_report_name(run_dirs: Iterable[str | Path]) -> str:
    """Generate a short, unique slug for multi-run reports."""
    names = [Path(p).name for p in run_dirs]
    digest = hashlib.blake2b(
        "_vs_".join(names).encode("utf-8"), digest_size=6
    ).hexdigest()
    prefix_parts = names[:3]
    if len(names) > 3:
        prefix_parts.append(f"plus_{len(names) - 3}")
    prefix = "_vs_".join(prefix_parts)
    prefix = prefix[:80].rstrip("_")
    return f"{prefix}_{digest}"


__all__ = ["build_report_name"]
