#!/usr/bin/env python3
"""
太医院 (TaiYiYuan) — Plot Generation Module

Server-side plot generation for TaiYiYuan modeling outputs.
All functions accept the JSON dict returned by the corresponding modeling function.

Dual-mode:
  - Importable: pass output_path=None → returns matplotlib Figure
  - CLI: python plots.py <subcommand> → saves PNG/PDF

Usage:
    python plots.py bsts --trial_id 1 --output plots/bsts.png
    python plots.py its --metric weight --intervention_date 2026-02-01 --output plots/its.png
    python plots.py forest --trial_id 1 --output plots/forest.pdf
    python plots.py heatmap --days 90 --output plots/correlations.png
    python plots.py trend --metric weight --days 90 --output plots/trend.png
    python plots.py power --metric sleep_quality --output plots/power.png
    python plots.py anomalies --metric hrv --days 90 --output plots/anomalies.png
    python plots.py network --days 90 --output plots/network.png

Requires: matplotlib, numpy, pandas
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style & Palette
# ---------------------------------------------------------------------------

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e5e5",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}

PALETTE = {
    "actual": "#2563EB",
    "predicted": "#DC2626",
    "ci_fill": "#FCA5A5",
    "positive": "#16A34A",
    "negative": "#DC2626",
    "neutral": "#6B7280",
    "highlight": "#F59E0B",
    "significance": "#7C3AED",
}

plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = Path(os.environ.get("LONGEVITY_PLOTS_DIR", str(_PROJECT_ROOT / "plots")))


def _save_or_return(fig: Figure, output_path: Optional[Union[str, Path]], fmt: str = "png") -> Union[Figure, Path]:
    """Save figure to disk or return the Figure object."""
    if output_path is None:
        return fig
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), format=fmt, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def _parse_dates(date_strings: list) -> list:
    """Parse date strings to date objects."""
    parsed = []
    for s in date_strings:
        if isinstance(s, date):
            parsed.append(s)
        elif 'T' in str(s):
            parsed.append(datetime.fromisoformat(str(s).replace('Z', '+00:00')).date())
        else:
            parsed.append(date.fromisoformat(str(s)))
    return parsed


# ===========================================================================
# 1. BSTS Counterfactual Plot
# ===========================================================================

def plot_bsts_counterfactual(
    data: dict,
    output_path: Optional[Union[str, Path]] = None,
    fmt: str = "png",
    pre_series: Optional[list] = None,
    title_suffix: str = "",
    **kwargs,
) -> Union[Figure, Path]:
    """Plot Bayesian Structural Time Series counterfactual analysis.

    Args:
        data: Output from CausalAnalyzer.bayesian_structural_time_series()
        pre_series: Optional list of {date, value} dicts for pre-intervention data
        output_path: None = return Figure; path = save and return Path
    """
    if data.get("error"):
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 5)))
        ax.text(0.5, 0.5, f"Insufficient data: {data['error']}", ha="center", va="center",
                fontsize=14, color=PALETTE["neutral"], transform=ax.transAxes)
        ax.set_title(f"BSTS Counterfactual{title_suffix}")
        return _save_or_return(fig, output_path, fmt)

    counterfactual = data["counterfactual"]
    dates = _parse_dates([c["date"] for c in counterfactual])
    actual = [c["actual"] for c in counterfactual]
    predicted = [c["predicted"] for c in counterfactual]
    ci_lo = [c["ci_95"][0] for c in counterfactual]
    ci_hi = [c["ci_95"][1] for c in counterfactual]

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 5)))

    # Pre-intervention series (if provided)
    if pre_series:
        pre_dates = _parse_dates([p["date"] for p in pre_series])
        pre_vals = [p["value"] for p in pre_series]
        ax.plot(pre_dates, pre_vals, color=PALETTE["actual"], linewidth=1.5, label="Actual")

    # Intervention line
    int_date = dates[0]
    ax.axvline(int_date, color=PALETTE["highlight"], linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(int_date, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else max(actual) * 1.05,
            " Intervention", color=PALETTE["highlight"], fontsize=9, va="bottom")

    # Post-period
    ax.plot(dates, actual, color=PALETTE["actual"], linewidth=1.8, label="Actual")
    ax.plot(dates, predicted, color=PALETTE["predicted"], linewidth=1.5, linestyle="--", label="Counterfactual")
    ax.fill_between(dates, ci_lo, ci_hi, color=PALETTE["ci_fill"], alpha=0.3, label="95% CI")

    # Average effect annotation
    avg_effect = data.get("average_effect", {})
    prob_pos = data.get("posterior_prob_positive", 0)
    prob_neg = data.get("posterior_prob_negative", 0)
    prob = max(prob_pos, prob_neg)
    direction = "positive" if prob_pos > prob_neg else "negative"

    if avg_effect:
        est = avg_effect.get("estimate", 0)
        annotation = f"Avg effect: {est:+.3f}\nP(effect {direction}): {prob:.1%}"
        ax.annotate(annotation, xy=(0.98, 0.02), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ddd"))

    ax.set_xlabel("Date")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Bayesian Structural Time Series — Counterfactual Analysis{title_suffix}")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()

    return _save_or_return(fig, output_path, fmt)


# ===========================================================================
# 2. Interrupted Time Series Plot
# ===========================================================================

def plot_its(
    data: dict,
    output_path: Optional[Union[str, Path]] = None,
    fmt: str = "png",
    metric_series: Optional[list] = None,
    title_suffix: str = "",
    **kwargs,
) -> Union[Figure, Path]:
    """Plot Interrupted Time Series analysis.

    Args:
        data: Output from CausalAnalyzer.interrupted_time_series()
        metric_series: Optional list of {date, value} dicts for raw observations
    """
    if data.get("error"):
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 5)))
        ax.text(0.5, 0.5, f"Error: {data['error']}", ha="center", va="center",
                fontsize=14, color=PALETTE["neutral"], transform=ax.transAxes)
        ax.set_title(f"Interrupted Time Series{title_suffix}")
        return _save_or_return(fig, output_path, fmt)

    int_date = date.fromisoformat(data["intervention_date"])
    n_pre = data["n_pre"]
    n_post = data["n_post"]

    pre_trend = data["pre_trend"]
    post_trend = data["post_trend"]
    level_change = data["level_change"]

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 5)))

    # Scatter of observations
    if metric_series:
        dates_raw = _parse_dates([p["date"] for p in metric_series])
        vals_raw = [p["value"] for p in metric_series]
        pre_pts = [(d, v) for d, v in zip(dates_raw, vals_raw) if d < int_date]
        post_pts = [(d, v) for d, v in zip(dates_raw, vals_raw) if d >= int_date]
        if pre_pts:
            ax.scatter([p[0] for p in pre_pts], [p[1] for p in pre_pts],
                       color=PALETTE["actual"], alpha=0.4, s=20, zorder=3)
        if post_pts:
            ax.scatter([p[0] for p in post_pts], [p[1] for p in post_pts],
                       color=PALETTE["predicted"], alpha=0.4, s=20, zorder=3)

    # Regression lines
    pre_start = int_date - timedelta(days=n_pre)
    pre_x = [pre_start, int_date - timedelta(days=1)]
    pre_y = [pre_trend["intercept"], pre_trend["intercept"] + pre_trend["slope"] * n_pre]
    ax.plot(pre_x, pre_y, color=PALETTE["actual"], linewidth=2.5, label="Pre-trend")

    post_end = int_date + timedelta(days=n_post)
    post_x = [int_date, post_end]
    post_y = [post_trend["intercept"], post_trend["intercept"] + post_trend["slope"] * n_post]
    ax.plot(post_x, post_y, color=PALETTE["predicted"], linewidth=2.5, label="Post-trend")

    # Intervention line
    ax.axvline(int_date, color=PALETTE["highlight"], linestyle="--", linewidth=1.5)

    # Level change annotation
    lc = level_change["estimate"]
    lc_ci = level_change["ci_95"]
    lc_p = level_change["p_value"]
    color = PALETTE["positive"] if lc > 0 else PALETTE["negative"]
    ax.annotate(
        f"Level Δ: {lc:+.3f}\n95% CI: [{lc_ci[0]:.3f}, {lc_ci[1]:.3f}]\np = {lc_p:.4f}",
        xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, linewidth=1.5))

    ax.set_xlabel("Date")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Interrupted Time Series{title_suffix}")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()

    return _save_or_return(fig, output_path, fmt)


# ===========================================================================
# 3. Correlation Heatmap
# ===========================================================================

def plot_correlation_heatmap(
    data: list,
    output_path: Optional[Union[str, Path]] = None,
    fmt: str = "png",
    max_metrics: int = 20,
    title_suffix: str = "",
    **kwargs,
) -> Union[Figure, Path]:
    """Plot correlation matrix heatmap from pairwise correlations.

    Args:
        data: Output from PatternDetector.pairwise_correlations() — list of dicts
        max_metrics: Maximum number of metrics to show (filtered by most connections)
    """
    if not data:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No correlation data", ha="center", va="center",
                fontsize=14, color=PALETTE["neutral"], transform=ax.transAxes)
        return _save_or_return(fig, output_path, fmt)

    # Filter to lag=0 entries
    lag0 = [d for d in data if d.get("lag", 0) == 0]
    if not lag0:
        lag0 = data  # fallback if no lag field

    # Build correlation matrix
    metrics = set()
    for d in lag0:
        metrics.add(d.get("metric1", d.get("variable1", "")))
        metrics.add(d.get("metric2", d.get("variable2", "")))

    # Filter to top N metrics by number of significant connections
    sig_counts = {}
    for d in lag0:
        p = d.get("p_adjusted", d.get("p_value", 1))
        if p < 0.05:
            m1 = d.get("metric1", d.get("variable1", ""))
            m2 = d.get("metric2", d.get("variable2", ""))
            sig_counts[m1] = sig_counts.get(m1, 0) + 1
            sig_counts[m2] = sig_counts.get(m2, 0) + 1

    top_metrics = sorted(sig_counts.keys(), key=lambda m: sig_counts[m], reverse=True)[:max_metrics]
    if not top_metrics:
        top_metrics = sorted(metrics)[:max_metrics]

    n = len(top_metrics)
    matrix = np.full((n, n), np.nan)
    mask = np.ones((n, n), dtype=bool)  # True = masked (insignificant)
    idx = {m: i for i, m in enumerate(top_metrics)}

    for d in lag0:
        m1 = d.get("metric1", d.get("variable1", ""))
        m2 = d.get("metric2", d.get("variable2", ""))
        if m1 in idx and m2 in idx:
            r = d.get("pearson_r", d.get("r", 0))
            p = d.get("p_adjusted", d.get("p_value", 1))
            i, j = idx[m1], idx[m2]
            matrix[i, j] = r
            matrix[j, i] = r
            if p < 0.05:
                mask[i, j] = False
                mask[j, i] = False

    # Fill diagonal
    np.fill_diagonal(matrix, 1.0)
    np.fill_diagonal(mask, False)

    figsize = kwargs.get("figsize", (max(8, n * 0.55), max(6, n * 0.5)))
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    masked_matrix = np.ma.array(matrix, mask=mask)
    im = ax.imshow(masked_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Gray out masked cells
    gray_matrix = np.ma.array(np.ones_like(matrix) * 0.5, mask=~mask)
    ax.imshow(gray_matrix, cmap="Greys", vmin=0, vmax=1, alpha=0.15, aspect="auto")

    # Labels
    short_labels = [m.replace("diet.", "D:").replace("exercise.", "E:").replace("body_", "")
                    for m in top_metrics]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson r")

    ax.set_title(f"Pairwise Correlations (FDR q<0.05){title_suffix}")
    fig.tight_layout()

    return _save_or_return(fig, output_path, fmt)


# ===========================================================================
# 4. Trial Forest Plot
# ===========================================================================

def plot_trial_forest(
    data: Union[dict, list],
    output_path: Optional[Union[str, Path]] = None,
    fmt: str = "png",
    title_suffix: str = "",
    **kwargs,
) -> Union[Figure, Path]:
    """Forest plot of trial effect sizes (Cohen's d with 95% CI).

    Args:
        data: Output from CausalAnalyzer.analyze_trial() — single dict or list of dicts
    """
    if isinstance(data, dict):
        data = [data]

    # Filter to trials with comparison data
    trials = [t for t in data if not t.get("insufficient_data") and "comparison" in t]
    if not trials:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No trial data with sufficient observations",
                ha="center", va="center", fontsize=14, color=PALETTE["neutral"],
                transform=ax.transAxes)
        return _save_or_return(fig, output_path, fmt)

    n = len(trials)
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, max(4, n * 1.2))))

    y_positions = range(n)
    for i, trial in enumerate(trials):
        comp = trial["comparison"]
        d_val = comp["effect_size_d"]
        ci = comp["ci_95"]
        p = comp["p_value"]
        name = trial.get("trial_name", f"Trial {trial.get('trial_id', i+1)}")

        # Color by significance and direction
        if p < 0.05 and d_val > 0:
            color = PALETTE["positive"]
        elif p < 0.05 and d_val < 0:
            color = PALETTE["negative"]
        else:
            color = PALETTE["neutral"]

        # Error bar
        ax.errorbar(d_val, i, xerr=[[d_val - ci[0]], [ci[1] - d_val]],
                    fmt="s", color=color, markersize=8, capsize=4, capthick=1.5,
                    linewidth=1.5, zorder=3)

        # Right-side labels
        ax.text(max(ci[1], d_val) + 0.15, i,
                f"d={d_val:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] p={p:.3f}",
                va="center", fontsize=9, color=color)

    # Null effect line
    ax.axvline(0, color=PALETTE["neutral"], linestyle="-", linewidth=1, alpha=0.6)

    # Labels
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([t.get("trial_name", f"Trial {t.get('trial_id', i+1)}")
                        for i, t in enumerate(trials)], fontsize=10)
    ax.set_xlabel("Cohen's d (effect size)")
    ax.set_title(f"Trial Effects — Forest Plot{title_suffix}")
    ax.invert_yaxis()
    fig.tight_layout()

    return _save_or_return(fig, output_path, fmt)


# ===========================================================================
# 5. Anomaly Timeline
# ===========================================================================

def plot_anomaly_timeline(
    data: list,
    metric_series: list,
    output_path: Optional[Union[str, Path]] = None,
    fmt: str = "png",
    metric_name: str = "",
    title_suffix: str = "",
    **kwargs,
) -> Union[Figure, Path]:
    """Plot time series with anomaly highlights.

    Args:
        data: Output from ModelingEngine.detect_anomalies() — list of anomaly dicts
        metric_series: List of {date, value} for the full series
    """
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (12, 4)))

    if not metric_series:
        ax.text(0.5, 0.5, "No metric data", ha="center", va="center",
                fontsize=14, color=PALETTE["neutral"], transform=ax.transAxes)
        return _save_or_return(fig, output_path, fmt)

    dates = _parse_dates([p["date"] for p in metric_series])
    values = np.array([p["value"] for p in metric_series], dtype=float)

    # Full series
    ax.plot(dates, values, color=PALETTE["neutral"], linewidth=0.8, alpha=0.7)

    # Rolling mean + ±2SD band
    s = pd.Series(values, index=dates)
    rolling_mean = s.rolling(window=30, min_periods=7).mean()
    rolling_std = s.rolling(window=30, min_periods=7).std()
    ax.plot(dates, rolling_mean.values, color=PALETTE["actual"], linewidth=1.5,
            linestyle="--", label="30-day mean", alpha=0.8)
    ax.fill_between(dates,
                    (rolling_mean - 2 * rolling_std).values,
                    (rolling_mean + 2 * rolling_std).values,
                    color=PALETTE["actual"], alpha=0.08, label="±2 SD")

    # Anomaly markers
    if data:
        for anom in data:
            a_date = date.fromisoformat(str(anom.get("date", "")))
            a_val = anom.get("value", 0)
            z = anom.get("z_score", 0)
            marker = "^" if z > 0 else "v"
            size = min(abs(z) * 40, 200)
            ax.scatter([a_date], [a_val], color=PALETTE["highlight"],
                       marker=marker, s=size, zorder=5, edgecolors="white", linewidth=0.5)

        # Annotate top 3
        sorted_anoms = sorted(data, key=lambda a: abs(a.get("z_score", 0)), reverse=True)[:3]
        for anom in sorted_anoms:
            a_date = date.fromisoformat(str(anom.get("date", "")))
            a_val = anom.get("value", 0)
            z = anom.get("z_score", 0)
            ax.annotate(f"z={z:.1f}", xy=(a_date, a_val), fontsize=8,
                        xytext=(5, 10 if z > 0 else -15), textcoords="offset points",
                        color=PALETTE["highlight"])

    ax.set_xlabel("Date")
    ax.set_ylabel(metric_name or "Value")
    ax.set_title(f"Anomaly Detection — {metric_name}{title_suffix}")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()

    return _save_or_return(fig, output_path, fmt)


# ===========================================================================
# 6. Trend Analysis Plot
# ===========================================================================

def plot_trend(
    data: dict,
    metric_series: list,
    output_path: Optional[Union[str, Path]] = None,
    fmt: str = "png",
    metric_name: str = "",
    lower_is_better: Optional[bool] = None,
    title_suffix: str = "",
    **kwargs,
) -> Union[Figure, Path]:
    """Plot trend analysis with regression line.

    Args:
        data: Output from ModelingEngine.trend_analysis()
        metric_series: List of {date, value} dicts
        lower_is_better: If True, decreasing trend is colored green
    """
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 5)))

    if not metric_series or data.get("error"):
        ax.text(0.5, 0.5, f"No data: {data.get('error', 'empty series')}", ha="center",
                va="center", fontsize=14, color=PALETTE["neutral"], transform=ax.transAxes)
        return _save_or_return(fig, output_path, fmt)

    dates = _parse_dates([p["date"] for p in metric_series])
    values = np.array([p["value"] for p in metric_series], dtype=float)
    x_numeric = np.array([(d - dates[0]).days for d in dates], dtype=float)

    # Determine trend color
    slope = data.get("slope", data.get("slope_per_day", 0))
    direction = data.get("direction", "increasing" if slope > 0 else "decreasing")

    if lower_is_better is not None:
        is_good = (direction == "decreasing" and lower_is_better) or \
                  (direction == "increasing" and not lower_is_better)
        trend_color = PALETTE["positive"] if is_good else PALETTE["negative"]
    else:
        trend_color = PALETTE["actual"]

    # Scatter
    ax.scatter(dates, values, color=PALETTE["neutral"], alpha=0.4, s=20, zorder=2)

    # Regression line
    intercept = data.get("intercept", values[0])
    reg_y = intercept + slope * x_numeric
    ax.plot(dates, reg_y, color=trend_color, linewidth=2.5, zorder=3, label="Trend")

    # CI band (approximate using SE)
    r_sq = data.get("r_squared", 0)
    p_val = data.get("p_value", 1)
    n = len(values)
    se_y = np.std(values - reg_y) if n > 2 else 0
    ci_width = 1.96 * se_y
    ax.fill_between(dates, reg_y - ci_width, reg_y + ci_width,
                    color=trend_color, alpha=0.1)

    # Annotation
    pct_change = data.get("pct_change", slope * n / (intercept if intercept != 0 else 1) * 100)
    annotation = (f"Slope: {slope:.4f}/day\n"
                  f"R² = {r_sq:.3f}, p = {p_val:.4f}\n"
                  f"Change: {pct_change:+.1f}% over {n} days")
    ax.annotate(annotation, xy=(0.02, 0.98), xycoords="axes fraction",
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ddd"))

    ax.set_xlabel("Date")
    ax.set_ylabel(metric_name or "Value")
    ax.set_title(f"Trend Analysis — {metric_name}{title_suffix}")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.autofmt_xdate()
    fig.tight_layout()

    return _save_or_return(fig, output_path, fmt)


# ===========================================================================
# 7. Power Curve
# ===========================================================================

def plot_power_curve(
    data: dict,
    output_path: Optional[Union[str, Path]] = None,
    fmt: str = "png",
    title_suffix: str = "",
    **kwargs,
) -> Union[Figure, Path]:
    """Plot power curve: MDE vs observations per phase.

    Args:
        data: Output from CausalAnalyzer.power_analysis()
    """
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (9, 5)))

    if data.get("error"):
        ax.text(0.5, 0.5, f"Error: {data.get('error', '')}\n{data.get('message', '')}",
                ha="center", va="center", fontsize=12, color=PALETTE["neutral"],
                transform=ax.transAxes)
        ax.set_title(f"Power Analysis{title_suffix}")
        return _save_or_return(fig, output_path, fmt)

    from scipy import stats as scipy_stats

    within_sd = data["within_person_sd"]
    alpha = data.get("alpha", 0.05)
    power = data.get("power", 0.8)
    current_n = data["n_baseline_observations"]
    recommended_n = data.get("recommended_phase_duration", 32)

    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
    z_beta = scipy_stats.norm.ppf(power)

    # Generate curve
    n_range = np.arange(5, 91)
    mde_d = (z_alpha + z_beta) / np.sqrt(n_range)
    mde_abs = mde_d * within_sd

    # Primary axis: Cohen's d
    ax.plot(n_range, mde_d, color=PALETTE["actual"], linewidth=2, label="MDE (Cohen's d)")

    # Reference lines
    for d_ref, label, style in [(0.8, "Large", ":"), (0.5, "Medium", "--"), (0.2, "Small", "-.")]:
        ax.axhline(d_ref, color=PALETTE["neutral"], linestyle=style, alpha=0.5, linewidth=0.8)
        ax.text(88, d_ref + 0.02, label, fontsize=8, color=PALETTE["neutral"], ha="right")

    # Current n marker
    current_mde = (z_alpha + z_beta) / np.sqrt(current_n)
    ax.axvline(current_n, color=PALETTE["highlight"], linestyle="--", alpha=0.7)
    ax.plot(current_n, current_mde, "o", color=PALETTE["highlight"], markersize=10, zorder=5)
    ax.annotate(f"Current: n={current_n}\nMDE={current_mde:.2f}d",
                xy=(current_n, current_mde), xytext=(current_n + 5, current_mde + 0.1),
                fontsize=9, color=PALETTE["highlight"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["highlight"]))

    # Recommended n
    ax.axvline(recommended_n, color=PALETTE["positive"], linestyle="--", alpha=0.5)
    ax.text(recommended_n, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 1 else 1.4,
            f" Recommended: {recommended_n}", fontsize=8, color=PALETTE["positive"])

    # Secondary y-axis: absolute units
    ax2 = ax.twinx()
    ax2.plot(n_range, mde_abs, color=PALETTE["predicted"], linewidth=1.2, alpha=0.5)
    ax2.set_ylabel(f"MDE (absolute, SD={within_sd:.3f})", color=PALETTE["predicted"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["predicted"])

    ax.set_xlabel("Observations per Phase")
    ax.set_ylabel("MDE (Cohen's d)")
    ax.set_title(f"Power Analysis — {data.get('metric', '')}{title_suffix}")
    ax.set_xlim(5, 90)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    return _save_or_return(fig, output_path, fmt)


# ===========================================================================
# 8. Cross-Module Network
# ===========================================================================

def plot_cross_module_network(
    data: list,
    output_path: Optional[Union[str, Path]] = None,
    fmt: str = "png",
    min_r: float = 0.3,
    title_suffix: str = "",
    **kwargs,
) -> Union[Figure, Path]:
    """Plot cross-module correlation network.

    Args:
        data: Output from PatternDetector.cross_module_scan() or pairwise_correlations()
        min_r: Minimum |r| to draw an edge
    """
    # Filter significant edges above threshold
    edges = []
    for d in data:
        r = d.get("pearson_r", d.get("r", 0))
        p = d.get("p_adjusted", d.get("p_value", 1))
        if abs(r) >= min_r and p < 0.05:
            m1 = d.get("metric1", d.get("variable1", ""))
            m2 = d.get("metric2", d.get("variable2", ""))
            edges.append((m1, m2, r))

    if not edges:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.text(0.5, 0.5, f"No significant edges (|r| >= {min_r})",
                ha="center", va="center", fontsize=14, color=PALETTE["neutral"],
                transform=ax.transAxes)
        ax.set_title(f"Cross-Module Network{title_suffix}")
        return _save_or_return(fig, output_path, fmt)

    # Collect nodes
    nodes = sorted(set(n for e in edges for n in (e[0], e[1])))
    n_nodes = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}

    # Module colors
    module_colors = {
        "diet": "#16A34A",
        "exercise": "#F59E0B",
        "sleep": "#7C3AED",
        "body": "#2563EB",
        "weight": "#2563EB",
        "resting": "#2563EB",
        "hrv": "#2563EB",
        "blood": "#DC2626",
        "fasting": "#DC2626",
    }

    def _node_color(name):
        for prefix, color in module_colors.items():
            if prefix in name.lower():
                return color
        return PALETTE["neutral"]

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 4.0
    positions = {n: (radius * np.cos(a), radius * np.sin(a))
                 for n, a in zip(nodes, angles)}

    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 10)))
    ax.set_aspect("equal")

    # Draw edges
    for m1, m2, r in edges:
        x1, y1 = positions[m1]
        x2, y2 = positions[m2]
        color = PALETTE["positive"] if r > 0 else PALETTE["negative"]
        width = abs(r) * 3
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.4, zorder=1)

    # Draw nodes
    for node in nodes:
        x, y = positions[node]
        color = _node_color(node)
        ax.scatter([x], [y], s=200, color=color, edgecolors="white", linewidth=1.5, zorder=3)
        # Short label
        short = node.replace("diet.", "").replace("exercise.", "").replace("body_", "")[:12]
        ax.text(x, y - 0.5, short, ha="center", va="top", fontsize=7, rotation=0)

    # Legend
    legend_items = [
        mpatches.Patch(color=PALETTE["positive"], alpha=0.6, label="Positive r"),
        mpatches.Patch(color=PALETTE["negative"], alpha=0.6, label="Negative r"),
    ]
    ax.legend(handles=legend_items, loc="upper left", framealpha=0.9)

    ax.set_xlim(-radius * 1.5, radius * 1.5)
    ax.set_ylim(-radius * 1.5, radius * 1.5)
    ax.axis("off")
    ax.set_title(f"Cross-Module Correlation Network (|r| ≥ {min_r}){title_suffix}")
    fig.tight_layout()

    return _save_or_return(fig, output_path, fmt)


# ===========================================================================
# CLI
# ===========================================================================

def _get_db_connection():
    """Get database connection using env var or default path."""
    import sqlite3
    db_path = os.environ.get("TAIYIYUAN_DB", str(_PROJECT_ROOT / "data" / "taiyiyuan.db"))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _get_metric_series_from_db(metric_name: str, days: int = 90) -> list:
    """Fetch metric time series from the database.

    Uses the most recent data point's date as reference (not wall-clock time),
    so demo/historical datasets work correctly.
    """
    conn = _get_db_connection()
    # Find the most recent observation date for this metric
    max_date_row = pd.read_sql_query(
        "SELECT MAX(DATE(timestamp)) AS max_date FROM body_metrics WHERE metric_type = ?",
        conn, params=(metric_name,))
    if max_date_row.empty or max_date_row.iloc[0]["max_date"] is None:
        conn.close()
        return []
    max_date = date.fromisoformat(max_date_row.iloc[0]["max_date"])
    cutoff = (max_date - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT DATE(timestamp) AS date, AVG(value) AS value FROM body_metrics "
        "WHERE metric_type = ? AND DATE(timestamp) >= ? GROUP BY DATE(timestamp) ORDER BY date",
        conn, params=(metric_name, cutoff))
    conn.close()
    return [{"date": r["date"], "value": r["value"]} for _, r in df.iterrows()]


def main():
    parser = argparse.ArgumentParser(
        description="TaiYiYuan Plot Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # bsts
    p = sub.add_parser("bsts", help="BSTS counterfactual plot")
    p.add_argument("--trial_id", type=int, required=True)
    p.add_argument("--output", default="plots/bsts.png")
    p.add_argument("--fmt", default="png")

    # its
    p = sub.add_parser("its", help="Interrupted time series plot")
    p.add_argument("--metric", required=True)
    p.add_argument("--intervention_date", required=True)
    p.add_argument("--pre_days", type=int, default=30)
    p.add_argument("--post_days", type=int, default=30)
    p.add_argument("--output", default="plots/its.png")
    p.add_argument("--fmt", default="png")

    # forest
    p = sub.add_parser("forest", help="Trial forest plot")
    p.add_argument("--trial_id", type=int, nargs="+", required=True)
    p.add_argument("--output", default="plots/forest.png")
    p.add_argument("--fmt", default="png")

    # heatmap
    p = sub.add_parser("heatmap", help="Correlation heatmap")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--max_metrics", type=int, default=20)
    p.add_argument("--output", default="plots/correlations.png")
    p.add_argument("--fmt", default="png")

    # trend
    p = sub.add_parser("trend", help="Trend analysis plot")
    p.add_argument("--metric", required=True)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--output", default="plots/trend.png")
    p.add_argument("--fmt", default="png")

    # power
    p = sub.add_parser("power", help="Power analysis curve")
    p.add_argument("--metric", required=True)
    p.add_argument("--baseline_days", type=int, default=60)
    p.add_argument("--output", default="plots/power.png")
    p.add_argument("--fmt", default="png")

    # anomalies
    p = sub.add_parser("anomalies", help="Anomaly timeline")
    p.add_argument("--metric", required=True)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--output", default="plots/anomalies.png")
    p.add_argument("--fmt", default="png")

    # network
    p = sub.add_parser("network", help="Cross-module network")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--min_r", type=float, default=0.3)
    p.add_argument("--output", default="plots/network.png")
    p.add_argument("--fmt", default="png")

    args = parser.parse_args()

    # Execute
    try:
        if args.command == "trend":
            series = _get_metric_series_from_db(args.metric, args.days)
            # Compute trend inline
            if series:
                values = np.array([s["value"] for s in series], dtype=float)
                x = np.arange(len(values))
                from scipy import stats as sp
                slope, intercept, r, p, se = sp.linregress(x, values)
                trend_data = {
                    "slope": slope, "intercept": intercept,
                    "r_squared": r**2, "p_value": p,
                    "direction": "increasing" if slope > 0 else "decreasing",
                    "pct_change": slope * len(values) / (intercept if intercept else 1) * 100,
                }
            else:
                trend_data = {"error": "no_data"}
            result = plot_trend(trend_data, series, output_path=args.output, fmt=args.fmt,
                               metric_name=args.metric)

        elif args.command == "anomalies":
            series = _get_metric_series_from_db(args.metric, args.days)
            # Simple anomaly detection
            anomalies = []
            if series and len(series) > 30:
                values = np.array([s["value"] for s in series], dtype=float)
                rolling_mean = pd.Series(values).rolling(30, min_periods=7).mean()
                rolling_std = pd.Series(values).rolling(30, min_periods=7).std()
                for i, (s, m, sd) in enumerate(zip(series, rolling_mean, rolling_std)):
                    if pd.notna(m) and pd.notna(sd) and sd > 0:
                        z = (s["value"] - m) / sd
                        if abs(z) > args.threshold:
                            anomalies.append({"date": s["date"], "value": s["value"], "z_score": float(z)})
            result = plot_anomaly_timeline(anomalies, series, output_path=args.output,
                                           fmt=args.fmt, metric_name=args.metric)

        elif args.command == "power":
            series = _get_metric_series_from_db(args.metric, args.baseline_days + 30)
            if series and len(series) >= 5:
                from scipy import stats as sp
                values = np.array([s["value"] for s in series], dtype=float)
                diffs = np.diff(values)
                within_sd = float(np.sqrt(np.mean(diffs**2) / 2))
                z_a = sp.norm.ppf(0.975)
                z_b = sp.norm.ppf(0.8)
                n = len(values)
                power_data = {
                    "metric": args.metric,
                    "baseline_mean": float(np.mean(values)),
                    "baseline_sd": float(np.std(values, ddof=1)),
                    "within_person_sd": within_sd,
                    "n_baseline_observations": n,
                    "mde_cohens_d": float((z_a + z_b) / np.sqrt(n)),
                    "recommended_phase_duration": int(np.ceil(((z_a + z_b) / 0.5)**2)),
                    "alpha": 0.05, "power": 0.8,
                }
            else:
                power_data = {"error": "insufficient_data", "metric": args.metric}
            result = plot_power_curve(power_data, output_path=args.output, fmt=args.fmt)

        elif args.command == "forest":
            # Import causal analyzer — connect to DB
            conn = _get_db_connection()
            trials_data = []
            for tid in args.trial_id:
                obs = pd.read_sql_query(
                    "SELECT phase, metric_name, value FROM trial_observations "
                    "WHERE trial_id = ? AND metric_name = (SELECT primary_outcome_metric FROM trials WHERE id = ?)",
                    conn, params=(tid, tid))
                trial_info = pd.read_sql_query("SELECT * FROM trials WHERE id = ?", conn, params=(tid,))
                if obs.empty or trial_info.empty:
                    continue
                baseline = obs[obs["phase"] == "baseline"]["value"].values.astype(float)
                intervention = obs[obs["phase"] == "intervention"]["value"].values.astype(float)
                if len(baseline) < 3 or len(intervention) < 3:
                    continue
                from scipy import stats as sp
                stat, p_val = sp.ttest_ind(intervention, baseline, equal_var=False)
                n1, n2 = len(intervention), len(baseline)
                m1, m2 = np.mean(intervention), np.mean(baseline)
                s1, s2 = np.std(intervention, ddof=1), np.std(baseline, ddof=1)
                pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
                d = (m1 - m2) / pooled if pooled > 0 else 0
                se_d = np.sqrt((n1+n2)/(n1*n2) + d**2/(2*(n1+n2)))
                trials_data.append({
                    "trial_id": tid,
                    "trial_name": trial_info.iloc[0]["name"],
                    "comparison": {
                        "effect_size_d": float(d),
                        "ci_95": [float(d - 1.96*se_d), float(d + 1.96*se_d)],
                        "p_value": float(p_val),
                    }
                })
            conn.close()
            result = plot_trial_forest(trials_data, output_path=args.output, fmt=args.fmt)

        elif args.command == "heatmap":
            # Run pairwise correlations
            sys.path.insert(0, str(_PROJECT_ROOT / "modeling"))
            sys.path.insert(0, str(_PROJECT_ROOT / "data"))
            from patterns import PatternDetector
            import sqlite3
            db_path = os.environ.get("TAIYIYUAN_DB", str(_PROJECT_ROOT / "data" / "taiyiyuan.db"))

            class _MinDB:
                def __init__(self):
                    self.conn = sqlite3.connect(db_path)
                    self.conn.row_factory = sqlite3.Row

            detector = PatternDetector(db=_MinDB())
            corr_data = detector.pairwise_correlations(days=args.days)
            result = plot_correlation_heatmap(corr_data, output_path=args.output,
                                              fmt=args.fmt, max_metrics=args.max_metrics)

        elif args.command == "network":
            sys.path.insert(0, str(_PROJECT_ROOT / "modeling"))
            sys.path.insert(0, str(_PROJECT_ROOT / "data"))
            from patterns import PatternDetector
            import sqlite3
            db_path = os.environ.get("TAIYIYUAN_DB", str(_PROJECT_ROOT / "data" / "taiyiyuan.db"))

            class _MinDB:
                def __init__(self):
                    self.conn = sqlite3.connect(db_path)
                    self.conn.row_factory = sqlite3.Row

            detector = PatternDetector(db=_MinDB())
            corr_data = detector.pairwise_correlations(days=args.days)
            result = plot_cross_module_network(corr_data, output_path=args.output,
                                               fmt=args.fmt, min_r=args.min_r)

        elif args.command == "bsts":
            # Would need full causal analyzer — simplified for CLI
            print(json.dumps({"error": "Use Python API for BSTS plots (requires pre/post series)"}))
            sys.exit(0)

        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)

        if isinstance(result, Path):
            print(f"Plot saved: {result}")
        else:
            print("Figure returned (no output path specified)")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
