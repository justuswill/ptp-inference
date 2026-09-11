# Paper: *Elucidating Inference in Multi-Token Prediction*

First draft on the ICLR 2026 template. **All numbers are preliminary** —
single-seed runs on 100 Spec-Bench first turns, from the `vicuna_old` checkpoint.

## Layout

```
iclr2026_conference.tex   main document
macros.tex                paper notation (loaded after math_commands.tex)
iclr2026_conference.bib   references
data/                     ← every number the paper uses, as JSON (self-contained)
code/                     ← scripts that build the figures and tables
fig/                      ← generated PDF + SVG plots
fig/slides/               ← figures lifted from the talk deck (PNG)
tab/                      ← generated booktabs fragments (.tex)
```

`code/` reads **only** from `data/`, so figures and tables regenerate without
cluster access. `code/collect_data.py` is the single exception: it is the only
script that touches the cluster and `../scratch/`, and it exists to refresh
`data/`.

## Rebuilding

```bash
# 1. (only when new inference runs land) refresh the data snapshot
uv run python paper/code/collect_data.py --force

# 2. figures and tables
uv run python paper/code/make_figs.py           # --only <name> for one
uv run python paper/code/make_tables.py

# 3. the PDF  (latexmk is not installed on this host)
cd paper
pdflatex iclr2026_conference && bibtex iclr2026_conference
pdflatex iclr2026_conference && pdflatex iclr2026_conference
```

To eyeball figures without opening PDFs:
`uv run python paper/code/make_figs.py --png /tmp/preview`.

## Data files

| file | contents |
|---|---|
| `data/runs.json` | per-run config, aggregated metrics, pooled z-test, for 62 runs |
| `data/percall.json` | raw per-call accepted-token counts `G`, by ensemble width `K` |
| `data/branching_fit.json` | joint-MLE `(pi0, p, rho)` and the stage-1 `(pi0, rho)` |
| `data/per_question.json` | stage-2 per-question and population (Beta) fits |
| `data/meta.json` | provenance: source paths, checkpoint, collection timestamp |

Gotcha: in the upstream `scratch/*_v3_jointM_fit.json`, the key `p0` is **pi0**
(zero-inflation), not the per-strand success probability `p`. `collect_data.py`
renames it on the way in.

## Code

| file | role |
|---|---|
| `code/collect_data.py` | cluster → `data/` snapshot (the only script needing cluster access) |
| `code/rundata.py` | loader over `data/`; conventions (`WIDTHS`, `CAPT_BY_K`, `FLOOR`) |
| `code/branching.py` | the branching model: kernel, survival table, pmf, knapsack DP |
| `code/make_figs.py` | the 7 generated figures → `fig/*.pdf`, `fig/*.svg` |
| `code/make_tables.py` | the 6 table fragments → `tab/*.tex` |

`code/branching.py` mirrors `scripts/inference.py:2193-2310`
(`_branch_transition_matrix`, `_branch_survival_table`, `_exact_optimal_lengths`,
`_solve_width_profile`) as torch-free functions. **Keep the two in sync; do not
refactor `inference.py` to match this copy.**

Figure colors come from the `dataviz` reference palette, used in its documented
slot order. Matplotlib text is rendered by mathtext, not LaTeX — use plain `%`
and `.`, never `\%` or `vs.\ `, or they render literally.

## Known gaps in this draft

- **Main body is 12 pages; ICLR 2026 allows 9.** References start on p. 13,
  appendix on p. 14. Needs ~3 pages of trimming (see below).
- Author list is a placeholder (`TODO(draft)` in the preamble).
- The talk deck's cumulative-gain framing (+18% broader verification, +10% more
  chances, +4% efficient chances, +22% aligned reference, +40–80% PQD) is *not*
  reproduced as a figure; those percentages are from the `vicuna` checkpoint and
  the draft uses `vicuna_old` throughout for internal consistency.
- `R^2` in Fig. 2, Fig. 7 and Table 4 is recomputed from the joint-MLE model, so
  it does **not** match `r2_pooled` / `r2_bg` in the upstream
  `data/per_question.json` (those were fit against a different pooled
  prediction). Don't mix the two sets of numbers.
- The stochastic-ratio rule measures a significant distribution deviation
  (§5.2, "An anomaly worth flagging") despite being lossless in theory —
  unresolved.
- PQD itself is described and connected to the model (§4.6) but not benchmarked;
  no PQD run exists in `data/runs.json`.
- Spec-Bench first turns carry no reference completion, so `ref_lp_*` is NaN and
  all quality measures are relative to the base model, never to ground truth.

### Trimming candidates, in order
1. §3.2 taxonomy — compress the seven rule families into a table.
2. §2 background — the PTP recap can lose a third of its length.
3. Merge Fig. 1 (`geometric`) into Fig. 2 (`hist_by_k`) as a sixth panel.
4. Move Table 3 (`ensemble`) and §5.4 to the appendix.
