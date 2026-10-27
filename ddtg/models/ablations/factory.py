"""Factories for the ablation variants reported in the paper."""

from __future__ import annotations

from ..DDTGmodel import DDTGmodel, _DDTGCore
from .patch_egat import PatchGATModel


ABLATION_NAMES = (
    "full",
    "geometric_bias",
    "without_green_bias",
    "sparse",
    "patch_egat",
)


def build_ablation(
    name: str,
    hidden_dim: int = 256,
    num_tokens: int = 6,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
):
    """Build one reported ablation with the same core hyperparameters."""

    key = name.strip().lower()
    if key not in ABLATION_NAMES:
        raise ValueError(f"unknown ablation {name!r}; choose from {ABLATION_NAMES}")

    common = dict(
        hidden_dim=hidden_dim,
        num_tokens=num_tokens,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )

    if key == "patch_egat":
        return PatchGATModel(**common)
    if key == "full":
        return DDTGmodel(**common)
    return _DDTGCore(
        **common,
        attention="sparse" if key == "sparse" else "dense",
        bias={
            "geometric_bias": "geometry",
            "without_green_bias": "none",
            "sparse": "green",
        }[key],
    )


__all__ = ["build_ablation"]
