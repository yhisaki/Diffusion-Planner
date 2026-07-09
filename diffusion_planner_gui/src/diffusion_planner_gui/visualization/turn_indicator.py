"""Turn indicator time-series plot.

Displays the ``turn_indicators`` input sequence stored in the NPZ data (the raw
Autoware ``TurnIndicatorsReport`` enum over the past window up to the current
step) and, when a model is loaded, the predicted turn-indicator class.

Encoding
--------
Input enum values: ``0`` NONE, ``1`` DISABLE (straight/off), ``2`` ENABLE_LEFT,
``3`` ENABLE_RIGHT. The prediction head additionally uses class ``4`` KEEP,
meaning "no change vs. the previous step". The ground-truth class shown in the
title is derived the same way as ``diffusion_planner.loss.make_turn_indicator_gt``:
if the last two steps are equal the class is ``KEEP``, otherwise it is the raw
value of the current step.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

_INPUT_LABELS = {0: "None", 1: "Straight/Off", 2: "Left", 3: "Right"}
_OUTPUT_LABELS = {0: "None", 1: "Straight/Off", 2: "Left", 3: "Right", 4: "Keep"}
_KEEP_CLASS = 4


def _gt_class(seq: np.ndarray) -> int:
    """Replicate ``make_turn_indicator_gt`` for a single 1-D sequence."""
    if seq.size == 0:
        return 0
    if seq.size == 1:
        return int(seq[-1])
    if int(seq[-1]) == int(seq[-2]):
        return _KEEP_CLASS
    return int(seq[-1])


def plot_turn_indicator(
    data: dict[str, np.ndarray],
    turn_indicator_pred: int | None = None,
) -> go.Figure:
    """Plot the ego turn-indicator input sequence over time.

    Args:
        data: NPZ data dict. Uses ``"turn_indicators"`` of shape ``(INPUT_T + 1,)``.
        turn_indicator_pred: Optional predicted class index (``argmax`` of the
            model's ``turn_indicator_logit``); shown in the title when given.

    Returns:
        Plotly Figure. A stepped orange trace shows the input enum over the past
        window; the final marker (time step 0) is the current state.
    """
    fig = go.Figure()

    ti_raw = data.get("turn_indicators")
    if ti_raw is None:
        fig.update_layout(
            template="plotly_white", title="Turn Indicator (no data)", height=300
        )
        return fig

    ti = np.asarray(ti_raw).reshape(-1).astype(int)
    n = ti.size
    t = np.arange(-n + 1, 1)  # negative for past, 0 for current

    fig.add_trace(
        go.Scatter(
            x=t,
            y=ti,
            mode="lines+markers",
            line=dict(color="orange", width=2, shape="hv"),
            marker=dict(size=5),
            name="turn_indicators (input)",
        )
    )

    gt = _gt_class(ti)
    title = f"Turn Indicator — GT: {_OUTPUT_LABELS.get(gt, gt)}"
    if turn_indicator_pred is not None:
        pred = int(turn_indicator_pred)
        title += f" | Pred: {_OUTPUT_LABELS.get(pred, pred)}"

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Time step",
        yaxis=dict(
            title="turn indicator",
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=[_INPUT_LABELS[v] for v in (0, 1, 2, 3)],
            range=[-0.5, 3.5],
        ),
        height=300,
        margin=dict(l=90, r=20, t=40, b=40),
        legend=dict(font=dict(size=9)),
    )
    return fig
