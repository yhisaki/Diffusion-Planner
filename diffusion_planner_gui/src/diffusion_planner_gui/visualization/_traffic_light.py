"""Traffic-light colour and label utilities."""

from __future__ import annotations

import numpy as np

_TL_LEGEND_ORDER = [
    ("Green", "green"),
    ("Yellow", "yellow"),
    ("Red", "red"),
    ("White/Unknown", "purple"),
    ("No TL", "black"),
]


def traffic_light_color(tl: np.ndarray) -> str:
    """Return a colour string for a 5-element traffic-light one-hot vector."""
    if tl[0] == 1:
        return "green"
    if tl[1] == 1:
        return "yellow"
    if tl[2] == 1:
        return "red"
    if tl[3] == 1:
        return "purple"
    if tl[4] == 1:
        return "black"
    return "purple"


def traffic_light_label(tl: np.ndarray) -> str:
    """Return a human-readable label for a 5-element traffic-light one-hot vector."""
    if tl[0] == 1:
        return "Green"
    if tl[1] == 1:
        return "Yellow"
    if tl[2] == 1:
        return "Red"
    if tl[3] == 1:
        return "White/Unknown"
    if tl[4] == 1:
        return "No TL"
    return "White/Unknown"


def traffic_light_legend_order() -> list[tuple[str, str]]:
    """Return the legend entry order ``[(label, color), ...]``."""
    return list(_TL_LEGEND_ORDER)
