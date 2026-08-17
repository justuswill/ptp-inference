"""
Inference algorithm comparison plots.

Usage
-----
    python scripts/plot.py [inference_dir] [--out path/to/fig.png]

Defaults to /extra/ucibdl1/jcwill/ptp/checkpoints/vicuna/inference.

Layout
------
  Row 1: [teacher alignment (prob%)]  [LLM-as-judge quality]
  Row 2: [compression ratio]  [referee lp prob%]  [p-value]  [outlier %]
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval import load_run, run_label, _load_referee_cache, combined_ztest  # noqa: E402


INFERENCE_DIR = Path("/extra/ucibdl1/jcwill/ptp/checkpoints/vicuna/inference")
REFEREE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def compute_point(records: list[dict], y_fn) -> tuple[float, float, float, int]:
    """
    Returns (x_mean, y_mean, y_se, n) where x = calls/token.
    y_fn(metrics_dict) → float | None; None values are skipped.
    """
    xs, ys = [], []
    for rec in records:
        m = rec.get("metrics", {})
        n_gen = m.get("n_generated_tokens", 0)
        num_calls = m.get("num_calls")
        if n_gen <= 0 or num_calls is None:
            continue
        y = y_fn(m)
        if y is None or not math.isfinite(y):
            continue
        # seq-n-ptp-choice-k's num_calls counts every block call (n_blocks
        # per decision); normalize back to decisions so latency reflects
        # calls per committed-token decision, not per block sub-call.
        n_blocks = m.get("n_blocks", 1)
        xs.append((num_calls / n_blocks) / n_gen)
        ys.append(y)

    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    xa, ya = np.array(xs), np.array(ys)
    return float(xa.mean()), float(ya.mean()), float(ya.std() / math.sqrt(n)), n


def _draw_panel(ax, points, ylabel: str, title: str, show_legend: bool = True) -> None:
    """
    Draw one scatter panel.  points = [(label, x, y, ye), ...] already split
    into first_k / thresh-p / other by the caller.
    """
    def _extract_val(label: str, pattern: str) -> float:
        m = re.search(pattern, label)
        return float(m.group(1)) if m else float("inf")

    def _plot_line(pts, color):
        for fast in (False, True):
            sub = [p for p in pts if ("fast" in p[3]) == fast]
            if sub:
                ax.plot([p[0] for p in sub], [p[1] for p in sub],
                        color=color, linewidth=1, alpha=0.5, zorder=1)

    # seq algorithms make 2 forward passes per step (student + teacher); scale x uniformly
    points = [(lbl, x * 2 if "seq" in lbl else x, y, ye) for lbl, x, y, ye in points]

    first_k = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if "first-" in lbl and "seq-ptp" not in lbl],
        key=lambda t: t[0],
    )
    seq_ptp_first_k = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if "seq-ptp-first-" in lbl],
        key=lambda t: _extract_val(t[3], r"seq-ptp-first-([\d.]+(?:[eE][+-]?\d+)?)"),
    )

    def _is_seq_thresh_p(lbl: str) -> bool:
        return "seq-thresh-" in lbl and "ptp" not in lbl

    def _is_seq_ptp_thresh_p(lbl: str) -> bool:
        return "seq-ptp-thresh-" in lbl

    def _is_seq_ptp_conf_p(lbl: str) -> bool:
        return "seq-ptp-conf-" in lbl

    def _is_seq_inv_p(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-inv-', lbl))

    def _is_conf_p(lbl: str) -> bool:
        return "conf-" in lbl and "seq" not in lbl and "ptp" not in lbl and "oldcon" not in lbl

    thresh_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points
         if "thresh-" in lbl and not _is_seq_thresh_p(lbl) and not _is_seq_ptp_thresh_p(lbl)],
        key=lambda t: _extract_val(t[3], r"thresh-([\d.]+(?:[eE][+-]?\d+)?)"),
    )
    seq_thresh_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_seq_thresh_p(lbl)],
        key=lambda t: (_extract_val(t[3], r"seq-thresh-([\d.]+(?:[eE][+-]?\d+)?)"), 0 if "_raw" in t[3] else 1),
    )
    seq_ptp_thresh_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_seq_ptp_thresh_p(lbl)],
        key=lambda t: (_extract_val(t[3], r"seq-ptp-thresh-([\d.]+(?:[eE][+-]?\d+)?)"), 0 if "_raw" in t[3] else 1),
    )
    seq_ptp_conf_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_seq_ptp_conf_p(lbl)],
        key=lambda t: _extract_val(t[3], r"seq-ptp-conf-([\d.]+(?:[eE][+-]?\d+)?)"),
    )
    seq_inv_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_seq_inv_p(lbl)],
        key=lambda t: (_extract_val(t[3], r"seq-inv-(-?[\d.]+(?:[eE][+-]?\d+)?)"), 0 if "_raw" in t[3] else 1),
    )
    conf_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_conf_p(lbl)],
        key=lambda t: _extract_val(t[3], r"conf[_-]([\d.]+(?:[eE][+-]?\d+)?)"),
    )
    def _is_ratio_k(lbl: str) -> bool:
        return "ratio" in lbl and "seq" not in lbl and "ratio_g" not in lbl and "ratio-p" not in lbl and "ratio_p" not in lbl and "new" not in lbl

    def _is_ratio_p(lbl: str) -> bool:
        return ("ratio-p" in lbl or "ratio_p" in lbl) and "seq" not in lbl

    def _is_oldcon_p(lbl: str) -> bool:
        return "oldcon" in lbl

    def _is_top_k(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-top-\d', lbl))

    top_k = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_top_k(lbl)],
        key=lambda t: _extract_val(t[3], r"seq-top-([\d.]+(?:[eE][+-]?\d+)?)"),
    )

    def _is_seq_ptp_top_k(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ptp-top-\d', lbl))

    seq_ptp_top_k = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_seq_ptp_top_k(lbl)],
        key=lambda t: _extract_val(t[3], r"seq-ptp-top-([\d.]+(?:[eE][+-]?\d+)?)"),
    )

    def _is_top_p(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-topp-', lbl))

    top_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_top_p(lbl)],
        key=lambda t: _extract_val(t[3], r"seq-topp-([\d.]+(?:[eE][+-]?\d+)?)"),
    )

    def _is_seq_ptp_top_p(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ptp-topp-', lbl))

    seq_ptp_top_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_seq_ptp_top_p(lbl)],
        key=lambda t: _extract_val(t[3], r"seq-ptp-topp-([\d.]+(?:[eE][+-]?\d+)?)"),
    )

    def _is_seq_ptp_choice_k(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-(?:n-)?ptp-choice-\d', lbl))

    seq_ptp_choice_k = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_seq_ptp_choice_k(lbl)],
        key=lambda t: _extract_val(t[3], r"seq-(?:n-)?ptp-choice-([\d.]+(?:[eE][+-]?\d+)?)"),
    )

    def _is_ar(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?ar$', lbl))

    ar_pts = [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_ar(lbl)]

    def _is_seq_ptp(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ptp$', lbl))

    seq_ptp_pts = [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_seq_ptp(lbl)]

    def _is_seq_ptp_self(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ptp-self$', lbl))

    seq_ptp_self_pts = [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_seq_ptp_self(lbl)]

    def _is_ptp_judge(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?ptp-judge-', lbl))

    # short = quality-first (fewest tokens/call, "slow"), long = length-first
    # ("fast"); mid balances the two. Connect the line in that slow->fast order.
    _PTP_JUDGE_STYLE_ORDER = {"short": 0, "mid": 1, "long": 2}

    def _ptp_judge_style(lbl: str) -> str | None:
        m = re.search(r"-(short|mid|long)$", lbl)
        return m.group(1) if m else None

    ptp_judge = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_ptp_judge(lbl)],
        key=lambda t: _PTP_JUDGE_STYLE_ORDER.get(_ptp_judge_style(t[3]), -1),
    )

    ratio_k = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_ratio_k(lbl)],
        key=lambda t: t[0],
    )
    ratio_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_ratio_p(lbl)],
        key=lambda t: _extract_val(t[3], r"ratio[_-]p[_-]?([\d.]+(?:[eE][+-]?\d+)?)"),
    )
    oldcon_p = sorted(
        [(x, y, ye, lbl) for lbl, x, y, ye in points if _is_oldcon_p(lbl)],
        key=lambda t: _extract_val(t[3], r"oldcon[_-]p[_-]?([\d.]+(?:[eE][+-]?\d+)?)"),
    )
    other = [(lbl, x, y, ye) for lbl, x, y, ye in points
             if "first-" not in lbl and "thresh-" not in lbl
             and not _is_ratio_k(lbl) and not _is_ratio_p(lbl)
             and not _is_oldcon_p(lbl)
             and not _is_seq_thresh_p(lbl) and not _is_seq_ptp_thresh_p(lbl)
             and not _is_conf_p(lbl) and not _is_seq_ptp_conf_p(lbl)
             and not _is_seq_inv_p(lbl) and not _is_top_k(lbl) and not _is_seq_ptp_top_k(lbl)
             and not _is_top_p(lbl) and not _is_seq_ptp_top_p(lbl)
             and not _is_seq_ptp_choice_k(lbl)
             and not _is_ar(lbl) and not _is_seq_ptp(lbl) and not _is_seq_ptp_self(lbl)
             and not _is_ptp_judge(lbl)]

    _plot_line(first_k, "C0")
    for i, (x, y, ye, lbl) in enumerate(first_k):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="C0", alpha=0.8,
                    label="first_k" if i == 0 else "_nolegend_")
        m = re.search(r"first-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            kv = m.group(1)
            kd = str(int(float(kv))) if float(kv).is_integer() else kv
            ax.annotate(f"k={kd}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="C0", alpha=0.9)

    _plot_line(seq_ptp_first_k, "lightblue")
    for i, (x, y, ye, lbl) in enumerate(seq_ptp_first_k):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="lightblue", alpha=0.8,
                    label="seq-ptp-first-k" if i == 0 else "_nolegend_")
        m = re.search(r"seq-ptp-first-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            kv = m.group(1)
            kd = str(int(float(kv))) if float(kv).is_integer() else kv
            ax.annotate(f"k={kd}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="lightblue", alpha=0.9)

    _plot_line(thresh_p, "lightsalmon")
    for i, (x, y, ye, lbl) in enumerate(thresh_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="lightsalmon", alpha=0.8,
                    label="thresh-p" if i == 0 else "_nolegend_")
        m = re.search(r"thresh-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            f = "f" if "fast" in lbl else ""
            ax.annotate(f"p={m.group(1)}{f}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="lightsalmon", alpha=0.9)

    _plot_line(seq_thresh_p, "orange")
    for i, (x, y, ye, lbl) in enumerate(seq_thresh_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="orange", alpha=0.8,
                    label="seq-thresh-p" if i == 0 else "_nolegend_")
        m = re.search(r"seq-thresh-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            r = "r" if "_raw" in lbl else ""
            f = "f" if "fast" in lbl else ""
            ax.annotate(f"p={m.group(1)}{r}{f}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="orange", alpha=0.9)

    _plot_line(seq_ptp_thresh_p, "moccasin")
    for i, (x, y, ye, lbl) in enumerate(seq_ptp_thresh_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="moccasin", alpha=0.8,
                    label="seq-ptp-thresh-p" if i == 0 else "_nolegend_")
        m = re.search(r"seq-ptp-thresh-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            r = "r" if "_raw" in lbl else ""
            f = "f" if "fast" in lbl else ""
            ax.annotate(f"p={m.group(1)}{r}{f}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="moccasin", alpha=0.9)

    _plot_line(conf_p, "purple")
    for i, (x, y, ye, lbl) in enumerate(conf_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="purple", alpha=0.8,
                    label="conf-p" if i == 0 else "_nolegend_")
        m = re.search(r"conf[_-]([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            ax.annotate(f"p={m.group(1)}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="purple", alpha=0.9)

    _plot_line(seq_ptp_conf_p, "plum")
    for i, (x, y, ye, lbl) in enumerate(seq_ptp_conf_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="plum", alpha=0.8,
                    label="seq-ptp-conf-p" if i == 0 else "_nolegend_")
        m = re.search(r"seq-ptp-conf-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            ax.annotate(f"p={m.group(1)}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="plum", alpha=0.9)

    _plot_line(seq_inv_p, "brown")
    for i, (x, y, ye, lbl) in enumerate(seq_inv_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="brown", alpha=0.8,
                    label="seq-inv-p" if i == 0 else "_nolegend_")
        m = re.search(r"seq-inv-(-?[\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            r = "r" if "_raw" in lbl else ""
            ax.annotate(f"p={m.group(1)}{r}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="brown", alpha=0.9)

    _plot_line(top_k, "red")
    for i, (x, y, ye, lbl) in enumerate(top_k):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="red", alpha=0.8,
                    label="seq-top-k" if i == 0 else "_nolegend_")
        m = re.search(r"seq-top-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            kv = m.group(1)
            kd = str(int(float(kv))) if float(kv).is_integer() else kv
            ax.annotate(f"k={kd}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="red", alpha=0.9)

    _plot_line(seq_ptp_top_k, "lightcoral")
    for i, (x, y, ye, lbl) in enumerate(seq_ptp_top_k):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="lightcoral", alpha=0.8,
                    label="seq-ptp-top-k" if i == 0 else "_nolegend_")
        m = re.search(r"seq-ptp-top-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            kv = m.group(1)
            kd = str(int(float(kv))) if float(kv).is_integer() else kv
            ax.annotate(f"k={kd}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="lightcoral", alpha=0.9)

    _plot_line(top_p, "crimson")
    for i, (x, y, ye, lbl) in enumerate(top_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="crimson", alpha=0.8,
                    label="seq-top-p" if i == 0 else "_nolegend_")
        m = re.search(r"seq-topp-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            ax.annotate(f"p={m.group(1)}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="crimson", alpha=0.9)

    _plot_line(seq_ptp_top_p, "pink")
    for i, (x, y, ye, lbl) in enumerate(seq_ptp_top_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="pink", alpha=0.8,
                    label="seq-ptp-top-p" if i == 0 else "_nolegend_")
        m = re.search(r"seq-ptp-topp-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            ax.annotate(f"p={m.group(1)}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="pink", alpha=0.9)

    _plot_line(seq_ptp_choice_k, "teal")
    for i, (x, y, ye, lbl) in enumerate(seq_ptp_choice_k):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="teal", alpha=0.8,
                    label="seq-ptp-choice-k" if i == 0 else "_nolegend_")
        m = re.search(r"seq-(?:n-)?ptp-choice-([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            kv = m.group(1)
            kd = str(int(float(kv))) if float(kv).is_integer() else kv
            n = "n" if "seq-n-ptp-choice-" in lbl else ""
            ax.annotate(f"{n}k={kd}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="teal", alpha=0.9)

    for i, (x, y, ye, lbl) in enumerate(ar_pts):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="black", alpha=0.8,
                    label="ar" if i == 0 else "_nolegend_")

    for i, (x, y, ye, lbl) in enumerate(seq_ptp_self_pts):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="#333333", alpha=0.8,
                    label="seq-ptp-self" if i == 0 else "_nolegend_")

    for i, (x, y, ye, lbl) in enumerate(seq_ptp_pts):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="gray", alpha=0.8,
                    label="seq-ptp" if i == 0 else "_nolegend_")

    _plot_line(ptp_judge, "gray")
    for i, (x, y, ye, lbl) in enumerate(ptp_judge):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="gray", alpha=0.8,
                    label="ptp-judge" if i == 0 else "_nolegend_")
        style = _ptp_judge_style(lbl)
        if style:
            ax.annotate(style, xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="gray", alpha=0.9)

    _plot_line(ratio_k, "C2")
    for i, (x, y, ye, lbl) in enumerate(ratio_k):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="C2", alpha=0.8,
                    label="ratio-k" if i == 0 else "_nolegend_")
        m = re.search(r"ratio[_-]([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            kv = m.group(1)
            kd = str(int(float(kv))) if float(kv).is_integer() else kv
            ax.annotate(f"k={kd}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="C2", alpha=0.9)

    _plot_line(ratio_p, "lightgreen")
    for i, (x, y, ye, lbl) in enumerate(ratio_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="lightgreen", alpha=0.9,
                    label="ratio-p" if i == 0 else "_nolegend_")
        m = re.search(r"ratio[_-]p[_-]?([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            ax.annotate(f"p={m.group(1)}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="lightgreen", alpha=0.9)

    _plot_line(oldcon_p, "thistle")
    for i, (x, y, ye, lbl) in enumerate(oldcon_p):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color="thistle", alpha=0.9,
                    label="oldcon-p" if i == 0 else "_nolegend_")
        m = re.search(r"oldcon[_-]p[_-]?([\d.]+(?:[eE][+-]?\d+)?)", lbl)
        if m:
            ax.annotate(f"p={m.group(1)}", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="thistle", alpha=0.9)

    # seq-ratio sub-group detectors (k-p checked before k to avoid ambiguity; tmp_ prefix tolerated)
    def _is_seq_ratio_kp(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ratio-k-p', lbl))

    def _is_seq_ratio_k(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ratio-k(?:_g)?-[\d]', lbl))

    def _is_seq_ratio_p(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ratio-p(?:_g)?-[\d]', lbl))

    def _is_seq_ratio_base(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ratio(?:_g)?$', lbl))

    def _is_seq_ptp_ratio_kp(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ptp-ratio-k-p', lbl))

    def _is_seq_ptp_ratio_k(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ptp-ratio-k(?:_g)?-[\d]', lbl))

    def _is_seq_ptp_ratio_p(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ptp-ratio-p(?:_g)?-[\d]', lbl))

    def _is_seq_ptp_ratio_base(lbl: str) -> bool:
        return bool(re.match(r'^(?:tmp_)?seq-ptp-ratio(?:_g)?$', lbl))

    def _any_seq_ratio(lbl: str) -> bool:
        return (_is_seq_ratio_kp(lbl) or _is_seq_ratio_k(lbl) or _is_seq_ratio_p(lbl) or _is_seq_ratio_base(lbl)
                or _is_seq_ptp_ratio_kp(lbl) or _is_seq_ptp_ratio_k(lbl) or _is_seq_ptp_ratio_p(lbl) or _is_seq_ptp_ratio_base(lbl))

    seq_ratio_all = [(lbl, x, y, ye) for lbl, x, y, ye in other if _any_seq_ratio(lbl)]
    auto_pts  = [(lbl, x, y, ye) for lbl, x, y, ye in other if not _any_seq_ratio(lbl) and "seq" not in lbl]
    seq_other = [(lbl, x, y, ye) for lbl, x, y, ye in other if "seq" in lbl and not _any_seq_ratio(lbl)]

    for i, (lbl, x, y, ye) in enumerate(auto_pts + seq_other):
        ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                    capsize=4, capthick=1.2, elinewidth=1,
                    color=f"C{i + 3}", alpha=0.8, label=lbl)

    def _draw_seq_ratio_group(pts_k, pts_p, pts_kp, pts_base, base_color, dark_color, family_label):
        def _color(lbl): return dark_color if "_g" in lbl else base_color

        def _k_sort(lbl):
            v = _extract_val(lbl, r'-k(?:_g)?-([\d.]+(?:[eE][+-]?\d+)?)')
            return (1.0 if v == float('inf') else v, 1 if "_raw" in lbl else 0)

        def _p_sort(lbl):
            v = _extract_val(lbl, r'-p(?:_g)?-([\d.]+(?:[eE][+-]?\d+)?)')
            return (0.0 if v == float('inf') else v, 0 if "_raw" in lbl else 1)

        pts_k = sorted(pts_k, key=lambda t: _k_sort(t[0]))
        pts_p = sorted(pts_p, key=lambda t: _p_sort(t[0]))

        # base point connects into both k-line (at k=1) and p-line (at p=0)
        for pts_line, sort_fn in ((pts_k + pts_base, _k_sort), (pts_p + pts_base, _p_sort)):
            for greedy in (False, True):
                sub = sorted(
                    [(x, y, ye, lbl) for lbl, x, y, ye in pts_line if ("_g" in lbl) == greedy],
                    key=lambda t: sort_fn(t[3])[0],
                )
                if len(sub) > 1:
                    c = dark_color if greedy else base_color
                    ax.plot([p[0] for p in sub], [p[1] for p in sub],
                            color=c, linewidth=1, alpha=0.5, zorder=1)

        seen_labels = set()
        for lbl, x, y, ye in pts_k + pts_p + pts_kp + pts_base:
            c = _color(lbl)
            greedy = "_g" in lbl
            legend_key = f"{family_label}_g" if greedy else family_label
            ax.errorbar(x, y, yerr=ye if ye > 0 else None, fmt="o", markersize=3,
                        capsize=4, capthick=1.2, elinewidth=1, color=c, alpha=0.8,
                        label=legend_key if legend_key not in seen_labels else "_nolegend_")
            seen_labels.add(legend_key)
            gp = "g," if "_g" in lbl else ""
            rp = "r" if "_raw" in lbl else ""
            if _is_seq_ratio_k(lbl) or _is_seq_ptp_ratio_k(lbl):
                m = re.search(r'-k(?:_g)?-([\d.]+)', lbl)
                if m:
                    kv = m.group(1)
                    kd = str(int(float(kv))) if float(kv).is_integer() else kv
                    ann = f"{gp}k={kd}{rp}"
                else:
                    ann = gp.rstrip(",")
            elif _is_seq_ratio_p(lbl) or _is_seq_ptp_ratio_p(lbl):
                m = re.search(r'-p(?:_g)?-([\d.]+)', lbl)
                ann = f"{gp}p={m.group(1)}{rp}" if m else gp.rstrip(",")
            elif _is_seq_ratio_kp(lbl) or _is_seq_ptp_ratio_kp(lbl):
                m = re.search(r'-k-p(?:_g)?-([\d.]+)-([\d.]+)', lbl)
                ann = f"{gp}k={m.group(1)},p={m.group(2)}{rp}" if m else gp.rstrip(",")
            else:
                ann = f"{gp}{rp}".rstrip(",") or ""
            if ann:
                ax.annotate(ann, xy=(x, y), xytext=(4, 4),
                            textcoords="offset points", fontsize=7, color=c, alpha=0.9)

    _draw_seq_ratio_group(
        pts_k   =[(lbl, x, y, ye) for lbl, x, y, ye in seq_ratio_all if _is_seq_ratio_k(lbl)],
        pts_p   =[(lbl, x, y, ye) for lbl, x, y, ye in seq_ratio_all if _is_seq_ratio_p(lbl)],
        pts_kp  =[(lbl, x, y, ye) for lbl, x, y, ye in seq_ratio_all if _is_seq_ratio_kp(lbl)],
        pts_base=[(lbl, x, y, ye) for lbl, x, y, ye in seq_ratio_all if _is_seq_ratio_base(lbl)],
        base_color="green", dark_color="darkgreen", family_label="seq-ratio",
    )
    _draw_seq_ratio_group(
        pts_k   =[(lbl, x, y, ye) for lbl, x, y, ye in seq_ratio_all if _is_seq_ptp_ratio_k(lbl)],
        pts_p   =[(lbl, x, y, ye) for lbl, x, y, ye in seq_ratio_all if _is_seq_ptp_ratio_p(lbl)],
        pts_kp  =[(lbl, x, y, ye) for lbl, x, y, ye in seq_ratio_all if _is_seq_ptp_ratio_kp(lbl)],
        pts_base=[(lbl, x, y, ye) for lbl, x, y, ye in seq_ratio_all if _is_seq_ptp_ratio_base(lbl)],
        base_color="#90EE90", dark_color="seagreen", family_label="seq-ptp-ratio",
    )

    ax.set_xlabel("Latency (calls per token)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    if show_legend:
        ax.legend(fontsize=7, loc="best")
    ax.grid(True, linestyle="--", alpha=0.4)


def _y_teacher_prob(m: dict) -> float | None:
    lp = m.get("gen_lp_mean")
    return math.exp(lp) * 100 if lp is not None and math.isfinite(lp) else None

def _y_judge(m: dict) -> float | None:
    return m.get("referee_judge_score")

def _y_compression(m: dict) -> float | None:
    return m.get("compression_ratio")

def _y_referee_lp_prob(m: dict) -> float | None:
    lp = m.get("referee_lp_mean")
    return math.exp(lp) * 100 if lp is not None and math.isfinite(lp) else None

def _y_pvalue(m: dict) -> float | None:
    return m.get("gen_lp_p_value")  # unused for p-value panel; see combined_ztest path

def _y_outlier_perc(m: dict) -> float | None:
    return m.get("gen_lp_outlier_perc")


PANELS = [
    # (y_fn, ylabel, title)
    (_y_teacher_prob,   "teacher likelihood [%]", "Teacher Alignment"),
    (_y_judge,          "LLM as judge, 1–5", "Generation Quality"),
    (_y_compression,    "Compression ratio", "Generation Complexity / Diversity"),
    (_y_referee_lp_prob,"referee likelihood [%]", "Referee Alignment"),
    (_y_pvalue,         "p-value (H0: tokens ~ teacher)", "Are these from the teacher?"),
    (_y_outlier_perc,   "Near-zero teacher likelihood [%]", "Outlier tokens (%)"),
]


def _keep_no_ptp(name: str) -> bool:
    """Keep runs without 'ptp' or 'inv-p' in their name, except the always-on
    'seq-ptp' baseline, which is exempt from the 'ptp' exclusion."""
    if re.match(r'^(?:tmp_)?seq-inv-', name):
        return False
    if "ptp" not in name:
        return True
    return bool(re.match(r'^(?:tmp_)?seq-ptp$', name))


def _keep_choice_only(name: str) -> bool:
    """Keep only the ar / seq-ptp / seq-ptp-choice-k families."""
    return bool(
        re.match(r'^(?:tmp_)?ar$', name)
        or re.match(r'^(?:tmp_)?seq-ptp$', name)
        or re.match(r'^(?:tmp_)?seq-(?:n-)?ptp-choice-\d', name)
    )


def _keep_topp_thresh_only(name: str) -> bool:
    """Keep only ar / seq-ptp, plus any run with 'thresh-' or 'topp' in its name."""
    return bool(
        re.match(r'^(?:tmp_)?ar$', name)
        or re.match(r'^(?:tmp_)?seq-ptp$', name)
        or "thresh-" in name
        or "topp" in name
    )


def main(inference_dir: Path, out: Path | None, thresh_only: bool = False, big: bool = False,
         no_ptp: bool = False, choice_only: bool = False, topp_thresh_only: bool = False) -> None:
    run_dirs = sorted(
        d for d in inference_dir.iterdir()
        if d.is_dir() and (d / "results.jsonl").exists()
        and (not thresh_only or "thresh" in d.name or "seq-ptp" in d.name)
        and (not no_ptp or _keep_no_ptp(d.name))
        and (not choice_only or _keep_choice_only(d.name))
        and (not topp_thresh_only or _keep_topp_thresh_only(d.name))
    )
    if not run_dirs:
        print(f"No results.jsonl found under {inference_dir}")
        raise SystemExit(1)

    # Load all records once
    all_records = {}
    for run_dir in run_dirs:
        _, records = load_run(run_dir)
        all_records[run_dir] = (run_label(run_dir), records)

    # Inject cached referee scores (log-prob + judge) if available
    for run_dir, (_, records) in all_records.items():
        scores = _load_referee_cache(run_dir, REFEREE_MODEL)
        if scores:
            for rec in records:
                entry = scores.get(str(rec["example_id"]), {})
                rec["metrics"]["referee_lp_mean"]     = entry.get("mean",   float("nan"))
                rec["metrics"]["referee_lp_median"]   = entry.get("median", float("nan"))
                rec["metrics"]["referee_judge_score"] = entry.get("judge",  float("nan"))

    # Compute compression ratio in-memory (mirrors eval.py)
    import gzip
    for _, (_, records) in all_records.items():
        for rec in records:
            text = rec.get("generated", "")
            if text and "compression_ratio" not in rec.get("metrics", {}):
                enc = text.encode("utf-8")
                rec.setdefault("metrics", {})["compression_ratio"] = \
                    len(gzip.compress(enc)) / len(enc)

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    scale = 5 if big else 1
    fig = plt.figure(figsize=(16 * scale, 9 * scale), layout="constrained")
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[2, 1])

    axes = [
        fig.add_subplot(gs[0, 0:2]),  # teacher alignment (wide)
        fig.add_subplot(gs[0, 2:4]),  # judge (wide)
        fig.add_subplot(gs[1, 0]),    # compression
        fig.add_subplot(gs[1, 1]),    # referee lp
        fig.add_subplot(gs[1, 2]),    # p-value
        fig.add_subplot(gs[1, 3]),    # outlier %
    ]

    ax_pval = None
    for ax, (y_fn, ylabel, title) in zip(axes, PANELS):
        points = []
        for run_dir, (label, records) in all_records.items():
            if "_new" in label and label != "ptp_new":
                continue
            if "tmp_ide" in label:
                continue
            if y_fn is _y_pvalue:
                x_mean, _, _, _ = compute_point(
                    records,
                    lambda m: m.get("num_calls", 0) / max(m.get("n_generated_tokens", 1), 1)
                )
                _, p = combined_ztest(records)
                if not math.isnan(x_mean) and p is not None and 0 < p < 1:
                    points.append((label, x_mean, -math.log(-math.log(p)), 0.0))
            else:
                x, y, ye, n = compute_point(records, y_fn)
                if not math.isnan(x):
                    points.append((label, x, y, ye))
        if points:
            _draw_panel(ax, points, ylabel, title, show_legend=ax in axes[:2])
            if y_fn is _y_pvalue:
                ax_pval = ax
        else:
            ax.set_title(f"{title}\n(no data)", fontsize=10)
            ax.axis("off")

    # Custom y-axis for p-value panel: show reference p-values as tick labels
    if ax_pval is not None:
        ref_pvals  = [0.95,  0.1,    1e-10,    1e-100]
        ref_labels = ["≈1",  "0.1", "10⁻¹⁰", "10⁻¹⁰⁰"]
        tick_pos   = [-math.log(-math.log(p)) for p in ref_pvals]
        ax_pval.set_yticks(tick_pos)
        ax_pval.set_yticklabels(ref_labels, fontsize=7)
        # Significance line at p = 0.05
        ax_pval.axhline(-math.log(-math.log(0.05)), color="red",
                        linestyle="--", linewidth=0.8, alpha=0.6)
        ax_pval.annotate("p=0.05", xy=(ax_pval.get_xlim()[0], -math.log(-math.log(0.05))),
                         xytext=(3, 3), textcoords="offset points",
                         fontsize=6, color="red", alpha=0.8)

    fig.suptitle("Inference Algorithm Comparison: Vicuna-7B", fontsize=13)

    if out is None:
        out = inference_dir / "plot.png"
    fig.savefig(out, dpi=450, bbox_inches="tight")
    print(f"\nSaved to {out}")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inference_dir", nargs="?", type=Path, default=INFERENCE_DIR,
        help=f"Directory of inference run subdirectories (default: {INFERENCE_DIR})",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument("--thresh", action="store_true", default=False, help="Only include runs with 'thresh' in their name")
    parser.add_argument("--big", action="store_true", default=False, help="5x figsize and dpi for high-res zoom")
    parser.add_argument("--no-ptp", action="store_true", default=False,
                         help="Only include runs without 'ptp' or 'inv-p' in their name (seq-ptp baseline stays)")
    parser.add_argument("--choice", action="store_true", default=False,
                         help="Only include ar, seq-ptp, and seq-ptp-choice-k runs")
    parser.add_argument("--topp-thresh", action="store_true", default=False,
                         help="Only include ar, seq-ptp, top-p, and thresh-p runs")
    args = parser.parse_args()
    main(args.inference_dir, args.out, thresh_only=args.thresh, big=args.big,
         no_ptp=args.no_ptp, choice_only=args.choice, topp_thresh_only=args.topp_thresh)
