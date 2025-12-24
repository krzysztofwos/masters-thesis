"""Utilities shared across evaluation scripts."""

from __future__ import annotations

from typing import Any, Dict, Optional


def ensure_train_opponent(
    train_args: Dict[str, Any], eval_opponent: Optional[str]
) -> Dict[str, Any]:
    """Inject the evaluation opponent into training args when absent."""
    if "opponent" in train_args or not eval_opponent:
        return train_args
    updated = dict(train_args)
    updated["opponent"] = eval_opponent
    return updated


__all__ = ["ensure_train_opponent"]
