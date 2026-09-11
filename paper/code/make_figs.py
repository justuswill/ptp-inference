"""Regenerate every figure in the paper, from ``paper/data/`` only.

    uv run python paper/code/make_figs.py            # all
    uv run python paper/code/make_figs.py --only pareto
    uv run python paper/code/make_figs.py --png DIR  # PNG previews for eyeballing

Colors are the `dataviz` reference palette's categorical slots, used in their
documented order -- that order is what makes the palette CVD-safe on the adjacent
pairlist, so do not reorder. Scatter forms obey the all-pairs cap: at most three
hues, with marker shape plus direct labels carrying identity. `validate_palette.js`
could not be run on this host (no node), so no hue outside the documented,
pre-validated instance is introduced.

Text here is rendered by matplotlib's own mathtext, NOT by LaTeX: use plain
``%`` and ``.`` in labels, never LaTeX escapes like ``\\%`` or ``vs.\\ `` -- those
render literally. Math goes in ``$...$``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import branching as B
import rundata as R

FIG = R.FIG
PNG_DIR: Path | None = None

# dataviz reference palette, categorical slots 1-8, light mode, fixed order.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
BAR = "#cdd5de"  # recessive fill for observed histograms
# Sequential blue, ordinal use: no lighter than step 250 on a light surface.
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

TEXTWIDTH = 5.5  # ICLR \textwidth in inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": INK3,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "grid.color": "#e8e8e4",
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.25,
    "lines.linewidth": 1.4,
})


def _tidy(*axes):
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def _save(fig, name):
    """PDF for LaTeX, SVG for web/slides; PNG only when previewing."""
    if PNG_DIR is not None:
        PNG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PNG_DIR / f"{name}.png", dpi=190)
        plt.close(fig)
        print(f"  preview {name}.png")
        return
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote fig/{name}.pdf, fig/{name}.svg")


def _r2(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


# --------------------------------------------------------------- model helpers


def _obs_pmf(k: int) -> tuple[np.ndarray, int, int]:
    """Observed T distribution for width k: (pmf over T=1..capT, n, n_dropped)."""
    counts, dropped = R.extinction_counts(k)
    capt = R.CAPT_BY_K[k]
    arr = np.array(counts[1 : capt + 1], dtype=float)
    return arr / arr.sum(), int(arr.sum()), dropped


def _beta_mixture_pmf(k, capt, pi0, rho, a, b, n_grid=200) -> np.ndarray:
    """Beta(a,b) population mixture of branching pmfs: int T_pmf(p) Beta(p) dp."""
    from scipy import stats

    grid = (np.arange(n_grid) + 0.5) / n_grid
    w = stats.beta.pdf(grid, a, b)
    w /= w.sum()
    acc = np.zeros(capt)
    for p, wi in zip(grid, w):
        acc += wi * B.t_pmf(k, capt, pi0, p, rho)[1 : capt + 1]
    return acc


# ------------------------------------------------------------------ figure 1/7


def fig_geometric():
    """K=1: a shifted geometric already fits; the branching model earns its keep
    only by extrapolating across widths (see fig_vs_k)."""
    k, capt = 1, R.CAPT_BY_K[1]
    obs, n, _ = _obs_pmf(k)
    t = np.arange(1, capt + 1)

    # Shifted geometric on T: P(T=i) = (1-p) p^{i-1}, MLE p = 1 - 1/E[T].
    mean_t = float((t * obs).sum())
    p_geo = 1 - 1 / mean_t
    geo = (1 - p_geo) * p_geo ** (t - 1)
    geo[-1] = p_geo ** (capt - 1)  # surviving mass piles up at the structural cap

    f = R.fit()
    bran = B.t_pmf(k, capt, f["pi0"], f["p"], f["rho"])[1 : capt + 1]

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.60, 1.95))
    ax.bar(t + 1, obs * 100, width=0.8, color=BAR, label="observed", zorder=1)
    ax.plot(t + 1, geo * 100, color=S2, marker="o", ms=2.6, zorder=3,
            label=f"shifted geometric ($\\hat p$={p_geo:.3f}, $R^2$={_r2(obs, geo):.3f})")
    ax.plot(t + 1, bran * 100, color=S1, marker="s", ms=2.6, zorder=4,
            label=f"branching, joint fit ($R^2$={_r2(obs, bran):.3f})")
    ax.set_xlabel("tokens accepted per call $G$")
    ax.set_ylabel("share of calls (%)")
    ax.set_xlim(1.3, 14.5)
    ax.legend(loc="upper right")
    ax.text(0.99, 0.55, f"mean $G$ = {1 + mean_t:.3f}\n$n$ = {n} calls",
            transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=6.5)
    _tidy(ax)
    _save(fig, "geometric")


# ------------------------------------------------------------------ figure 2/7


def fig_hist_by_k():
    """Observed vs fitted per-call accepted-token distribution, by ensemble width."""
    ks = R.WIDTHS
    f, s1, pq = R.fit(), R.stage1(), R.per_question()

    fig, axes = plt.subplots(1, len(ks), figsize=(TEXTWIDTH, 1.7), sharey=True)
    for ax, k in zip(axes, ks):
        capt = R.CAPT_BY_K[k]
        obs, _, _ = _obs_pmf(k)
        t = np.arange(1, capt + 1)
        pooled = B.t_pmf(k, capt, f["pi0"], f["p"], f["rho"])[1 : capt + 1]
        ds = pq[R.DATASET_BY_K[k]]
        mix = _beta_mixture_pmf(k, capt, s1["pi0"], s1["rho"], ds["a"], ds["b"])

        ax.bar(t + 1, obs * 100, width=0.85, color=BAR, zorder=1)
        ax.plot(t + 1, pooled * 100, color=S1, lw=1.2, zorder=3, label="pooled")
        ax.plot(t + 1, mix * 100, color=S2, lw=1.2, ls="--", zorder=4, label="heterogeneous")
        ax.set_title(f"$K$ = {k}", pad=3)
        ax.set_xlim(1.3, 13)
        # R^2 in each curve's own color, so the two never need disambiguating.
        ax.text(0.97, 0.97, f"{_r2(obs, pooled):.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color=S1)
        ax.text(0.97, 0.83, f"{_r2(obs, mix):.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5, color=S2)
        ax.text(0.97, 0.66, f"$\\bar G$={1 + float((t * obs).sum()):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
        _tidy(ax)
    axes[0].set_ylabel("calls (%)")
    axes[2].set_xlabel("tokens accepted per call $G$")
    axes[0].text(0.97, 0.97, "$R^2$", transform=axes[0].transAxes, ha="right",
                 va="bottom", fontsize=6.5, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    fig.subplots_adjust(wspace=0.12)
    _save(fig, "hist_by_k")


# ------------------------------------------------------------------ figure 3/7


def fig_vs_k():
    """Where extra ensemble width stops paying: a hard wipeout floor, and E[G]
    growing only logarithmically in K."""
    ks = R.WIDTHS
    f = R.fit()
    obs_p1, obs_eg = [], []
    for k in ks:
        obs, _, _ = _obs_pmf(k)
        capt = R.CAPT_BY_K[k]
        obs_p1.append(obs[0])
        obs_eg.append(1 + float((np.arange(1, capt + 1) * obs).sum()))
    mod_p1 = [B.t_pmf(k, R.CAPT_BY_K[k], f["pi0"], f["p"], f["rho"])[1] for k in ks]
    mod_eg = [1 + B.expected_T(k, R.CAPT_BY_K[k], f["pi0"], f["p"], f["rho"]) for k in ks]

    # Phenomenological offset power law on the one-step wipeout rate.
    from scipy.optimize import curve_fit

    law = lambda kk, c, a, b: c + a * np.asarray(kk, float) ** (-b)
    (c, a, b), _ = curve_fit(law, ks, obs_p1, p0=[0.09, 0.25, 0.43], maxfev=20000)
    kk = np.geomspace(1, 1000, 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.05))

    ax1.plot(kk, law(kk, c, a, b) * 100, color=S3, lw=1.1, ls=":",
             label=f"$c+ak^{{-b}}$ ($R^2$={_r2(np.array(obs_p1), law(ks, c, a, b)):.3f})")
    ax1.plot(ks, np.array(mod_p1) * 100, color=S1, marker="s", ms=3.4,
             label="branching model")
    ax1.plot(ks, np.array(obs_p1) * 100, color=INK, marker="o", ms=3.4, ls="none",
             label="observed", zorder=5)
    ax1.axhline(c * 100, color=INK3, lw=0.6, ls="--")
    ax1.annotate(f"irreducible floor $\\hat c$ = {c:.3f}", xy=(1.1, c * 100),
                 xytext=(0, 3), textcoords="offset points", fontsize=6.5, color=INK2)
    ax1.set_xscale("log")
    ax1.set_xlabel("ensemble width $K$")
    ax1.set_ylabel("$P(T{=}1)$, immediate wipeout (%)")
    ax1.set_ylim(8, 46)
    ax1.legend(loc="upper right")

    ax2.plot(ks, mod_eg, color=S1, marker="s", ms=3.4, label="branching model")
    ax2.plot(ks, obs_eg, color=INK, marker="o", ms=3.4, ls="none", label="observed",
             zorder=5)
    ax2.set_xscale("log")
    ax2.set_xlabel("ensemble width $K$")
    ax2.set_ylabel("$E[G] = 1 + E[T]$")
    ax2.legend(loc="lower right")

    _tidy(ax1, ax2)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "vs_k")


# ------------------------------------------------------------------ figure 4/7


def fig_width_profile():
    """The knapsack solution: optimal width profiles, and what they buy over
    spending the same node budget on full-depth strands."""
    f = R.fit()
    capt = 20
    budgets = [20, 60, 120, 300, 1000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.0))

    for color, bud in zip(SEQ, budgets):
        widths, _ = B.optimal_width_profile(bud, capt, f["pi0"], f["p"], f["rho"])
        ax1.step(np.arange(1, capt + 1), widths, where="mid", color=color, lw=1.3,
                 label=f"$C$ = {bud}")
    ax1.set_xlabel("depth $d$")
    ax1.set_ylabel("optimal width $W(d)$")
    ax1.set_yscale("symlog", linthresh=10)
    ax1.set_ylim(0, 200)
    ax1.legend(loc="upper right", ncol=1)

    grid = np.unique(np.geomspace(20, 1200, 26).astype(int))
    opt, uni = [], []
    for bud in grid:
        widths, surv = B.optimal_width_profile(bud, capt, f["pi0"], f["p"], f["rho"])
        opt.append(1 + B.expected_T_profile(surv, widths))
        kk = max(bud // capt, 1)  # `standard` mode: k strands all at full depth
        uni.append(1 + B.expected_T(kk, capt, f["pi0"], f["p"], f["rho"]))
    ax2.plot(grid, opt, color=S1, label="exact width profile (optimal)")
    ax2.plot(grid, uni, color=S2, ls="--", label="uniform full depth (standard)")
    ax2.set_xscale("log")
    ax2.set_xlabel("node budget $C$")
    ax2.set_ylabel("$E[G]$")
    ax2.legend(loc="lower right")

    _tidy(ax1, ax2)
    fig.subplots_adjust(wspace=0.3)
    _save(fig, "width_profile")


# ------------------------------------------------------------------ figure 5/7

# Acceptance-rule families. Marker shape carries identity because scatter uses
# the all-pairs pairlist, which caps the palette at three hues.
FAMILY = {
    "ar": ("autoregressive", "*", INK),
    "ptp": ("exact", "s", S1),
    "seq-ptp": ("exact", "s", S1),
    "first_k": ("accept-$k$", "v", S2),
    "conf-p": ("student confidence", "^", S2),
    "thresh-p": ("teacher threshold", "<", S2),
    "ratio": ("ratio", "D", S3),
    "ratio-k": ("ratio", "D", S3),
    "ratio-p": ("ratio", "D", S3),
    "seq-ratio": ("ratio", "D", S3),
    "seq-ptp-top-p": ("nucleus", "o", S1),
    "seq-ptp-top-p-choice-k": ("nucleus + ensemble", "P", S1),
    "seq-ptp-self": ("self-verified", "X", S2),
    "seq-ptp-top-p-self": ("self-verified", "X", S2),
    "seq-ptp-top-p-choice-k-self": ("self-verified", "X", S2),
    "ptp_self": ("self-verified", "X", S2),
    "ptp-top-p-choice-k-self": ("self-verified", "X", S2),
}

LEGEND_ORDER = ["autoregressive", "exact", "nucleus", "nucleus + ensemble", "ratio",
                "teacher threshold", "student confidence", "accept-$k$", "self-verified"]


def rows():
    """One record per comparable run, with derived speedup."""
    base = R.speedup_baseline()
    out = []
    for name, rec in R.runs().items():
        if R.is_new(name):
            continue  # re-run variants are never merged into a group
        algo = rec["config"].get("algorithm")
        if algo not in FAMILY:
            continue
        agg = rec["agg"]
        out.append({
            "name": name,
            "algo": algo,
            "family": FAMILY[algo][0],
            "marker": FAMILY[algo][1],
            "color": FAMILY[algo][2],
            "cpc": agg.get("correct_per_call_mean", float("nan")),
            "mspt": agg["ms_per_token_mean"],
            "speedup": base / agg["ms_per_token_mean"],
            "lp": agg.get("gen_lp_mean_prob_mean", float("nan")),
            "ppl": agg.get("gen_lp_mean_ppl_mean", float("nan")),
            "comp": agg.get("compression_ratio_mean", float("nan")),
            "p": rec["config"].get("p"),
            "k": rec["config"].get("k"),
            "mode": rec["config"].get("choice_mode"),
            "z": rec["z"],
            "pval": rec["p"],
        })
    return out


def fig_pareto():
    """Throughput against alignment. Two panels because the two throughput
    measures disagree: tokens per call and wall-clock rank the rules differently.
    """
    rs = rows()
    ar = next(r for r in rs if r["algo"] == "ar")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.15), sharey=True)
    seen = {}
    for ax, xkey, xlabel in ((ax1, "cpc", "accepted tokens per call"),
                             (ax2, "speedup", "wall-clock speedup over AR")):
        for r in rs:
            h = ax.plot(r[xkey], r["lp"], marker=r["marker"], ms=4.2, ls="none",
                        mfc=r["color"], mec="white", mew=0.5, zorder=3)[0]
            seen.setdefault(r["family"], h)
        ax.axhline(ar["lp"], color=INK3, lw=0.6, ls="--", zorder=1)
        ax.set_xlabel(xlabel)
        ax.set_ylim(8, 96)
    ax2.axvline(1.0, color=INK3, lw=0.6, ls=":", zorder=1)
    ax1.set_ylabel("mean token prob. under base model (%)")
    ax1.text(0.98, 0.955, "autoregressive reference", transform=ax1.transAxes,
             fontsize=6.5, color=INK2, ha="right", va="bottom")
    ax2.annotate("no speedup", xy=(1.0, 12), xytext=(3, 0),
                 textcoords="offset points", fontsize=6.5, color=INK2, va="center")

    # Direct-label only the lossless rules -- never a label on every point.
    for r in rs:
        if r["name"] == "seq-ptp":
            ax1.annotate("seq-ptp", (r["cpc"], r["lp"]), textcoords="offset points",
                         xytext=(5, -1), fontsize=6.5, color=INK2)
        if r["name"] == "ptp":
            ax2.annotate("ptp", (r["speedup"], r["lp"]), textcoords="offset points",
                         xytext=(5, -1), fontsize=6.5, color=INK2)
    handles = [(k, seen[k]) for k in LEGEND_ORDER if k in seen]
    fig.legend([h for _, h in handles], [k for k, _ in handles], loc="upper center",
               bbox_to_anchor=(0.5, 1.15), ncol=5, columnspacing=1.0)
    _tidy(ax1, ax2)
    fig.subplots_adjust(wspace=0.07)
    _save(fig, "pareto")


# ------------------------------------------------------------------ figure 6/7


def fig_sweeps():
    """Threshold sweeps. One panel per measure -- never a second y-axis."""
    rs = rows()
    series = [
        ("nucleus", "seq-ptp-top-p", S1, "o"),
        ("teacher probability", "thresh-p", S2, "<"),
        ("student confidence", "conf-p", S3, "^"),
        ("self + nucleus", "seq-ptp-top-p-self", S4, "X"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.95))
    for label, algo, color, marker in series:
        pts = sorted([r for r in rs if r["algo"] == algo], key=lambda r: r["p"])
        if not pts:
            continue
        x = [r["p"] for r in pts]
        for ax, key in zip(axes, ("cpc", "speedup", "lp")):
            ax.plot(x, [r[key] for r in pts], color=color, marker=marker, ms=3.0,
                    label=label)
    ar = next(r for r in rs if r["algo"] == "ar")
    for ax, ylabel, ref in (
        (axes[0], "accepted tokens per call", 1.0),
        (axes[1], "speedup over AR", 1.0),
        (axes[2], "token prob. under base (%)", ar["lp"]),
    ):
        ax.axhline(ref, color=INK3, lw=0.6, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel(r"threshold $\theta$")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4,
               columnspacing=1.2)
    _tidy(*axes)
    fig.subplots_adjust(wspace=0.42)
    _save(fig, "sweeps")


# ------------------------------------------------------------------ figure 7/7


def fig_heterogeneity():
    """One acceptance probability does not fit all prompts."""
    from scipy import stats

    pq = R.per_question()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.9))

    ds = pq["seqptp"]
    ph = np.array(ds["p_hat_i"], float)
    ax1.hist(ph, bins=18, color=BAR, density=True, zorder=1, label="per-question MLE")
    grid = np.linspace(0.01, 0.99, 400)
    ax1.plot(grid, stats.beta.pdf(grid, ds["a"], ds["b"]), color=S1, zorder=3,
             label=f"population Beta({ds['a']:.1f}, {ds['b']:.1f})")
    ax1.axvline(ds["p_pooled"], color=S2, ls="--", lw=1.1, zorder=4,
                label=f"pooled $\\hat p$ = {ds['p_pooled']:.3f}")
    ax1.set_xlabel("per-question $\\hat p_i$   ($K$ = 1)")
    ax1.set_ylabel("density")
    ax1.set_xlim(0.2, 1.0)
    ax1.legend(loc="upper left")
    ax1.text(0.97, 0.45, f"LRT vs. pooled\n$\\chi^2$={ds['lrt_stat']:.0f}, "
                         f"$p$={ds['p_value']:.0e}",
             transform=ax1.transAxes, ha="right", va="top", fontsize=6.5, color=INK2)

    # Does modelling that heterogeneity buy fit quality? Widens with K.
    # R^2 is recomputed from the same predictions the other panels and Table 4
    # use, not read from the upstream stage-2 JSON (whose r2_* were computed
    # against a different pooled prediction).
    ks = R.WIDTHS
    x = np.arange(len(ks))
    f, s1 = R.fit(), R.stage1()
    gain = []
    for k in ks:
        capt = R.CAPT_BY_K[k]
        obs, _, _ = _obs_pmf(k)
        ds_k = pq[R.DATASET_BY_K[k]]
        pooled = B.t_pmf(k, capt, f["pi0"], f["p"], f["rho"])[1 : capt + 1]
        mix = _beta_mixture_pmf(k, capt, s1["pi0"], s1["rho"], ds_k["a"], ds_k["b"])
        # Single series, so no legend box: the axis label names it. Positive
        # means the population model beats one pooled p at that width.
        gain.append(100 * (_r2(obs, mix) - _r2(obs, pooled)))
    ax2.bar(x, gain, width=0.62, color=S1, zorder=2)
    ax2.axhline(0, color=INK3, lw=0.6, zorder=3)
    for xi, g in zip(x, gain):
        ax2.annotate(f"{g:+.2f}", (xi, g), textcoords="offset points",
                     xytext=(0, 2 if g >= 0 else -8), ha="center", fontsize=6,
                     color=INK2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(k) for k in ks])
    ax2.set_xlabel("ensemble width $K$")
    ax2.set_ylabel("$R^2$ gain from heterogeneity\n(percentage points)")
    ax2.margins(y=0.28)

    _tidy(ax1, ax2)
    fig.subplots_adjust(wspace=0.42)
    _save(fig, "heterogeneity")


FIGURES = {
    "geometric": fig_geometric,
    "hist_by_k": fig_hist_by_k,
    "vs_k": fig_vs_k,
    "width_profile": fig_width_profile,
    "pareto": fig_pareto,
    "sweeps": fig_sweeps,
    "heterogeneity": fig_heterogeneity,
}

if __name__ == "__main__":
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    if "--png" in sys.argv:
        PNG_DIR = Path(sys.argv[sys.argv.index("--png") + 1])
    for fname, fn in FIGURES.items():
        if only and fname != only:
            continue
        print(f"{fname}:")
        fn()
