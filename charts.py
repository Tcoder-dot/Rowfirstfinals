"""Deterministic charts selected from verified engine JSON.

Charts are descriptive only. The SciPy/statsmodels engine remains the sole
source of statistical numbers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAX_CHARTS = 6


def make_charts(
    engine: dict[str, Any],
    output_dir: str | Path,
    max_charts: int = MAX_CHARTS,
) -> list[dict[str, Any]]:
    """Create up to six outcome-matched PNG charts from the engine JSON."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    charts: list[dict[str, Any]] = []
    for index, result in enumerate(_results(engine)):
        if len(charts) >= max_charts:
            break
        path = destination / f"outcome-{index + 1}.png"
        try:
            chart = _render_chart(engine, result, path)
        except Exception:
            chart = None
        if chart is not None and path.exists():
            chart["result_index"] = index
            charts.append(chart)
    return charts


def make_chart(engine: dict[str, Any], output_path: str | Path) -> str | None:
    """Backward-compatible helper that renders the first available chart."""
    results = _results(engine)
    if not results:
        return None
    try:
        chart = _render_chart(engine, results[0], Path(output_path))
    except Exception:
        return None
    return chart["path"] if chart else None


def _render_chart(
    engine: dict[str, Any],
    result: dict[str, Any],
    path: Path,
) -> dict[str, Any] | None:
    test = result.get("test")
    outcome = _outcome_label(result)
    fig = None
    caption = ""

    if test in {"student-t", "welch-t", "one-way anova"}:
        groups = _result_groups(result)
        if len(groups) >= 2:
            fig = _bar_figure(groups, f"{outcome}: group means ± SD")
            caption = f"{outcome} — group means with SD error bars"
    elif test == "two-way anova":
        groups = _two_way_groups(engine, result)
        if len(groups) >= 2:
            fig = _bar_figure(groups, f"{outcome}: factor-combination means ± SD")
            caption = f"{outcome} — factor-combination means with SD error bars"
    elif test == "paired-t":
        pairs = result.get("pairs", [])
        if pairs:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            for before_value, after_value in pairs:
                ax.plot(
                    [0, 1],
                    [before_value, after_value],
                    marker="o",
                    color="#2563eb",
                    alpha=0.65,
                )
            ax.set_xticks([0, 1], [result["before"]["name"], result["after"]["name"]])
            ax.set_ylabel(outcome)
            ax.set_title(f"{outcome}: matched before/after values")
            caption = f"{outcome} — matched before/after line plot"
    elif test in {"pearson", "spearman", "simple linear regression"}:
        pairs = result.get("pairs")
        if pairs:
            x, y = zip(*pairs)
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.scatter(x, y, color="#2563eb", label="Observed values")
            if len(pairs) >= 2 and len(set(x)) > 1:
                line_x = np.linspace(min(x), max(x), 100)
                if test == "simple linear regression":
                    line_y = result["intercept"] + result["slope"] * line_x
                else:
                    slope, intercept = np.polyfit(x, y, 1)
                    line_y = intercept + slope * line_x
                ax.plot(line_x, line_y, color="#dc2626", label="Fitted line")
            ax.set_xlabel(result.get("predictor", result.get("xName", "X")))
            ax.set_ylabel(result.get("outcome", result.get("yName", "Y")))
            ax.set_title(f"{outcome}: scatter with fitted line")
            ax.legend()
            caption = f"{outcome} — scatter plot with fitted line"
    elif test in {"fisher-exact", "chi-square"}:
        matrix = (engine.get("ingested") or {}).get("matrix")
        values = [float(value) for row in (matrix or []) for value in row]
        if 2 <= len(values) <= 6 and sum(values) > 0:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.pie(
                values,
                labels=[f"Category {index + 1}" for index in range(len(values))],
                autopct="%1.1f%%",
            )
            ax.set_title(f"{outcome}: count shares")
            caption = f"{outcome} — count-share pie chart"

    if fig is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=160, format="png")
    finally:
        plt.close(fig)
    return {"path": str(path), "outcome": outcome, "caption": caption}


def _bar_figure(groups: list[dict[str, Any]], title: str) -> Any:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        [str(group["name"]) for group in groups],
        [float(group["mean"]) for group in groups],
        yerr=[float(group.get("sd", 0.0)) for group in groups],
        capsize=5,
        color=["#2563eb", "#0f766e", "#d97706", "#7c3aed", "#dc2626", "#0891b2"][: len(groups)],
        alpha=0.85,
    )
    ax.set_ylabel("Mean")
    ax.set_title(title)
    return fig


def _result_groups(result: dict[str, Any]) -> list[dict[str, Any]]:
    test = result.get("test")
    if test in {"student-t", "welch-t"}:
        return [result.get("group1", {}), result.get("group2", {})]
    return list(result.get("groups", []))


def _two_way_groups(
    engine: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = (engine.get("ingested") or {}).get("rows", [])
    factor_a = result.get("factorA")
    factor_b = result.get("factorB")
    outcome = result.get("outcome")
    buckets: dict[str, list[float]] = {}
    for row in rows:
        try:
            name = f"{row[factor_a]} / {row[factor_b]}"
            buckets.setdefault(name, []).append(float(row[outcome]))
        except (KeyError, TypeError, ValueError):
            continue
    return [
        {
            "name": name,
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for name, values in buckets.items()
    ]


def _outcome_label(result: dict[str, Any]) -> str:
    return str(result.get("parameter") or result.get("outcome") or "Measured outcome")


def _results(engine: dict[str, Any]) -> list[dict[str, Any]]:
    results = engine.get("results") or [engine.get("result")]
    return [result for result in results if result]