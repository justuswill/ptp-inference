"""Regenerate every table in the paper, from ``paper/data/`` only.

    uv run python paper/code/make_tables.py

Writes ``booktabs`` fragments to ``paper/tab/``, each meant to be ``\\input`` inside
a ``table`` environment in the main document.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

import branching as B
import rundata as R
from make_figs import _beta_mixture_pmf, _r2, rows

TAB = Path(__file__).resolve().parent.parent / "tab"

# Representative runs for the main comparison: one or two settings per rule
# family, chosen to bracket each family's useful range.
MAIN = [
    ("ar", "--"),
    ("ptp", "--"),
    ("seq-ptp", "--"),
    ("tmp_seq-ptp-topp-0.9", r"$\theta=0.9$"),
    ("tmp_seq-ptp-topp-0.98", r"$\theta=0.98$"),
    ("tmp_seq-ptp-topp-choice-5-0.98", r"$\theta=0.98$, $K=5$"),
    ("tmp_seq-ptp-topp-choice-50-0.98", r"$\theta=0.98$, $K=50$"),
    ("ratio", "--"),
    ("ratio-10", r"$\kappa=10$"),
    ("thresh-0.1", r"$\theta=0.1$"),
    ("thresh-0.5", r"$\theta=0.5$"),
    ("conf-p-0.5", r"$\theta=0.5$"),
    ("conf-p-0.9", r"$\theta=0.9$"),
    ("first_2", "$k=2$"),
    ("first_4", "$k=4$"),
    ("seq-ptp-self-tok", "exact"),
    ("tmp_seq-ptp-topp-self-0.8-tok", r"$\theta=0.8$"),
    ("tmp_seq-ptp-topp-choice-50-self-0.8-tok", r"$\theta=0.8$, $K=50$"),
]

PRETTY = {
    "ar": "autoregressive",
    "ptp": "PTP, single call",
    "seq-ptp": "PTP, sequential",
    "tmp_seq-ptp-topp-0.9": "nucleus",
    "tmp_seq-ptp-topp-0.98": "nucleus",
    "tmp_seq-ptp-topp-choice-5-0.98": "nucleus + ensemble",
    "tmp_seq-ptp-topp-choice-50-0.98": "nucleus + ensemble",
    "ratio": "speculative ratio",
    "ratio-10": "speculative ratio",
    "thresh-0.1": "teacher threshold",
    "thresh-0.5": "teacher threshold",
    "conf-p-0.5": "student confidence",
    "conf-p-0.9": "student confidence",
    "first_2": "accept-$k$",
    "first_4": "accept-$k$",
    "seq-ptp-self-tok": "self-verified",
    "tmp_seq-ptp-topp-self-0.8-tok": "self-verified + nucleus",
    "tmp_seq-ptp-topp-choice-50-self-0.8-tok": "self-verified + ensemble",
}

# Rules that preserve the base model's output distribution *by construction*.
# Speculative ratio belongs here in theory, but this implementation measures a
# significant deviation (see the discussion of Table 1), so it is not marked.
LOSSLESS = {"ar", "ptp", "seq-ptp"}


def _write(name: str, body: str, colsep: str | None = None) -> None:
    """Write a booktabs fragment. `colsep` tightens \tabcolsep locally, inside a
    group, so the setting cannot leak into the rest of the document."""
    TAB.mkdir(exist_ok=True)
    if colsep is not None:
        body = f"{{\\setlength{{\\tabcolsep}}{{{colsep}}}%\n{body}\n}}"
    (TAB / f"{name}.tex").write_text(body)
    print(f"  wrote tab/{name}.tex")


def _fmt(v, spec=".2f", dash="--"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return format(v, spec)


def _pval(p):
    if p is None or not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 1e-3 else f"{p:.3f}"


# ------------------------------------------------------------------- main table


def tab_rules():
    by_name = {r["name"]: r for r in rows()}
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"acceptance rule & setting & \multicolumn{1}{c}{tok./call} & "
        r"\multicolumn{1}{c}{ms/tok.} & \multicolumn{1}{c}{speedup} & "
        r"\multicolumn{1}{c}{$\bar q$ (\%)} & \multicolumn{1}{c}{PPL} & "
        r"\multicolumn{1}{c}{$p_{\mathrm{align}}$} \\",
        r"\midrule",
    ]
    for name, setting in MAIN:
        r = by_name.get(name)
        if r is None:
            continue
        label = PRETTY.get(name, name)
        if name in LOSSLESS:
            label += r"$^\dagger$"
        lines.append(
            f"{label} & {setting} & {_fmt(r['cpc'])} & {_fmt(r['mspt'], '.1f')} & "
            f"{_fmt(r['speedup'])}$\\times$ & {_fmt(r['lp'])} & "
            f"{_fmt(r['ppl'], '.2f')} & {_pval(r['pval'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("rules", "\n".join(lines), colsep="4pt")


# -------------------------------------------------------------- ensemble modes


def tab_ensemble():
    by_name = {r["name"]: r for r in rows()}
    groups = [
        ("verified against the base model", r"$\theta=0.98$", [
            ("tmp_seq-ptp-topp-choice-5-0.98", 5, "uniform"),
            ("tmp_seq-ptp-topp-choice-5-0.98-bayes-conjugate", 5, "Bayes, conjugate"),
            ("tmp_seq-ptp-topp-choice-5-0.98-bayes-p", 5, "Bayes, grid"),
            ("tmp_seq-ptp-topp-choice-50-0.98", 50, "uniform"),
            ("tmp_seq-ptp-topp-choice-50-0.98-bayes-conjugate", 50, "Bayes, conjugate"),
            ("tmp_seq-ptp-topp-choice-50-0.98-bayes-p", 50, "Bayes, grid"),
        ]),
        ("self-verified", r"$\theta=0.8$", [
            ("tmp_seq-ptp-topp-choice-50-self-0.8-tok", 50, "uniform"),
            ("tmp_seq-ptp-topp-choice-50-self-0.8-bayes-conjugate-tok", 50,
             "Bayes, conjugate"),
            ("tmp_seq-ptp-topp-choice-50-self-0.8-bayes-p-tok", 50, "Bayes, grid"),
            ("tmp_seq-ptp-topp-choice-50-self-0.8-phead-tok", 50, "learned $p$-head"),
        ]),
    ]
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"verification & allocation & \multicolumn{1}{c}{$K$} & "
        r"\multicolumn{1}{c}{tok./call} & \multicolumn{1}{c}{ms/tok.} & "
        r"\multicolumn{1}{c}{speedup} & \multicolumn{1}{c}{$\bar q$ (\%)} \\",
        r"\midrule",
    ]
    for gi, (gname, setting, entries) in enumerate(groups):
        if gi:
            lines.append(r"\midrule")
        first = True
        for name, k, mode in entries:
            r = by_name.get(name)
            if r is None:
                continue
            head = f"{gname}, {setting}" if first else ""
            first = False
            lines.append(
                f"{head} & {mode} & {k} & {_fmt(r['cpc'])} & "
                f"{_fmt(r['mspt'], '.1f')} & {_fmt(r['speedup'])}$\\times$ & "
                f"{_fmt(r['lp'])} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("ensemble", "\n".join(lines), colsep="4pt")


# ------------------------------------------------------------------ model fits


def _obs_pmf(k):
    counts, _ = R.extinction_counts(k)
    capt = R.CAPT_BY_K[k]
    arr = np.array(counts[1 : capt + 1], float)
    return arr / arr.sum()


def tab_fit():
    """R^2 is recomputed here exactly as in Figure `hist_by_k`, so the table and
    the figure cannot disagree. (The upstream stage-2 JSON also carries r2_pooled
    / r2_bg, but those were computed against a different pooled prediction and
    are not comparable to the joint-MLE model the paper reports.)"""
    f, s1, pq = R.fit(), R.stage1(), R.per_question()
    lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"\multicolumn{1}{c}{$K$} & \multicolumn{1}{c}{calls} & "
        r"\multicolumn{1}{c}{$\bar G$ obs.} & \multicolumn{1}{c}{$E[G]$ model} & "
        r"\multicolumn{1}{c}{$\hat p_{\mathrm{pool}}$} & "
        r"\multicolumn{1}{c}{$R^2_{\mathrm{pool}}$} & "
        r"\multicolumn{1}{c}{$R^2_{\mathrm{het}}$} & "
        r"\multicolumn{1}{c}{$\chi^2_{\mathrm{LRT}}$} \\",
        r"\midrule",
    ]
    for k in R.WIDTHS:
        capt = R.CAPT_BY_K[k]
        obs = _obs_pmf(k)
        ds = pq[R.DATASET_BY_K[k]]
        eg_obs = 1 + float((np.arange(1, capt + 1) * obs).sum())
        eg_mod = 1 + B.expected_T(k, capt, f["pi0"], f["p"], f["rho"])
        pooled = B.t_pmf(k, capt, f["pi0"], f["p"], f["rho"])[1 : capt + 1]
        mix = _beta_mixture_pmf(k, capt, s1["pi0"], s1["rho"], ds["a"], ds["b"])
        lines.append(
            f"{k} & {len(R.percall(k))} & {eg_obs:.3f} & {eg_mod:.3f} & "
            f"{ds['p_pooled']:.3f} & {_r2(obs, pooled):.3f} & {_r2(obs, mix):.3f} & "
            f"{ds['lrt_stat']:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("fit", "\n".join(lines))


# -------------------------------------------------- knapsack worked example


def tab_knapsack():
    """Survival table and the optimal allocation at a concrete node budget."""
    f = R.fit()
    capt = 20
    depths = [1, 2, 3, 4, 5, 6, 8]
    widths_show = [1, 2, 5, 50]
    surv = B.survival_table(max(widths_show), capt, f["pi0"], f["p"], f["rho"])

    lines = [
        r"\begin{tabular}{r" + "r" * len(depths) + r"}",
        r"\toprule",
        r"& \multicolumn{" + str(len(depths)) + r"}{c}{depth $d$} \\",
        r"\cmidrule(lr){2-" + str(len(depths) + 1) + r"}",
        r"$w$ & " + " & ".join(str(d) for d in depths) + r" \\",
        r"\midrule",
    ]
    for w in widths_show:
        lines.append(f"{w} & " + " & ".join(f"{surv[w, d]:.3f}" for d in depths) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("survival", "\n".join(lines))

    # Allocation comparison at a few budgets.
    lines = [
        r"\begin{tabular}{rllrr}",
        r"\toprule",
        r"\multicolumn{1}{c}{budget $\budget$} & allocation & "
        r"\multicolumn{1}{c}{profile $\mathrm{depth}^{\mathrm{count}}$} & "
        r"\multicolumn{1}{c}{strands} & \multicolumn{1}{c}{$E[G]$} \\",
        r"\midrule",
    ]
    for bud in (120, 300, 1000):
        widths, surv_b = B.optimal_width_profile(bud, capt, f["pi0"], f["p"], f["rho"])
        lengths = B.lengths_from_widths(widths, capt)
        eg_opt = 1 + B.expected_T_profile(surv_b, widths)
        kk = max(bud // capt, 1)
        eg_uni = 1 + B.expected_T(kk, capt, f["pi0"], f["p"], f["rho"])
        # Compact description of the strand-length multiset.
        from collections import Counter

        cnt = Counter(lengths)
        desc = "\\,".join(f"{L}^{{{n}}}" for L, n in sorted(cnt.items(), reverse=True))
        lines.append(f"{bud} & exact profile & ${desc}$ & {len(lengths)} & "
                     f"{eg_opt:.3f} \\\\")
        lines.append(f" & uniform full depth & ${capt}^{{{kk}}}$ & {kk} & "
                     f"{eg_uni:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("knapsack", "\n".join(lines), colsep="4pt")


# --------------------------------------------------------------- appendix: all


def tab_all():
    rs = sorted(rows(), key=lambda r: (-r["speedup"]))
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"run & family & \multicolumn{1}{c}{tok./call} & \multicolumn{1}{c}{ms/tok.} & "
        r"\multicolumn{1}{c}{speedup} & \multicolumn{1}{c}{$\bar q$ (\%)} & "
        r"\multicolumn{1}{c}{PPL} & \multicolumn{1}{c}{gzip} \\",
        r"\midrule",
    ]
    for r in rs:
        safe = r["name"].replace("_", r"\_")
        lines.append(
            f"\\texttt{{{safe}}} & {r['family']} & {_fmt(r['cpc'])} & "
            f"{_fmt(r['mspt'], '.1f')} & {_fmt(r['speedup'])}$\\times$ & "
            f"{_fmt(r['lp'])} & {_fmt(r['ppl'], '.2f')} & {_fmt(r['comp'], '.3f')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write("all_runs", "\n".join(lines), colsep="3pt")


TABLES = [tab_rules, tab_ensemble, tab_fit, tab_knapsack, tab_all]

if __name__ == "__main__":
    for fn in TABLES:
        print(f"{fn.__name__}:")
        fn()
