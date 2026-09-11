"""Loader for the paper's figures and tables.

Reads **only** from ``paper/data/`` -- the JSON snapshot written by
``collect_data.py``. Nothing here touches the cluster or the repo's ``scratch/``
directory, so every figure is reproducible from the checked-in data alone.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
DATA = PAPER / "data"
FIG = PAPER / "fig"


def _load(name: str):
    path = DATA / name
    if not path.exists():
        raise SystemExit(
            f"missing {path.relative_to(PAPER.parent)} -- run:\n"
            f"    uv run python paper/code/collect_data.py"
        )
    return json.loads(path.read_text())


@lru_cache(maxsize=None)
def runs() -> dict[str, dict]:
    """{run_name: {config, agg, z, p, n}} for every comparable inference run."""
    return _load("runs.json")


@lru_cache(maxsize=None)
def meta() -> dict:
    return _load("meta.json")


@lru_cache(maxsize=None)
def fit() -> dict[str, float]:
    """Joint-MLE (pi0, p, rho): one branching model shared across all widths."""
    return _load("branching_fit.json")["joint_mle"]


@lru_cache(maxsize=None)
def stage1() -> dict[str, float]:
    """Profile-likelihood (pi0, rho), the pair at which the stage-2 Beta(a, b)
    population fits were estimated -- the two must be used together."""
    return _load("branching_fit.json")["stage1_profile"]


@lru_cache(maxsize=None)
def per_question() -> dict[str, dict]:
    """Stage-2 per-question / population fits, keyed by dataset name."""
    return _load("per_question.json")


@lru_cache(maxsize=None)
def _percall() -> dict[str, list[int]]:
    return _load("percall.json")


def percall(k: int) -> list[int]:
    """Raw per-call accepted-token counts G for ensemble width k."""
    return _percall()[str(k)]


# ---------------------------------------------------------------- conventions

WIDTHS = [1, 2, 5, 50, 1000]
DATASET_BY_K = {1: "seqptp", 2: "choice2", 5: "choice5", 50: "choice50", 1000: "seqn1000"}
# Structural proposal-length cap per dataset, matching the fitters.
CAPT_BY_K = {1: 20, 2: 19, 5: 19, 50: 19, 1000: 19}
FLOOR = 2  # G >= FLOOR by construction; T = G - (FLOOR - 1)


def extinction_counts(k: int) -> tuple[list[int], int]:
    """Histogram of extinction time T = G - 1, clipped at the structural cap.

    Returns (counts indexed by T, number of floor violations dropped).
    """
    capt = CAPT_BY_K[k]
    counts = [0] * (capt + 1)
    dropped = 0
    for g in percall(k):
        t = g - (FLOOR - 1)
        if t < 1:
            dropped += 1
            continue
        counts[min(t, capt)] += 1
    return counts, dropped


def speedup_baseline() -> float:
    """AR ms/token, the denominator of every speedup (cf. ``cli/generate.py:202``)."""
    return runs()["ar"]["agg"]["ms_per_token_mean"]


def is_new(name: str) -> bool:
    """Re-run variants. Reported on their own, never merged into a plot group."""
    return "new" in name


if __name__ == "__main__":
    print("checkpoint:", meta()["checkpoint"])
    print("collected: ", meta()["collected_utc"])
    print(f"{len(runs())} runs; AR baseline {speedup_baseline():.2f} ms/token")
    print("joint MLE:", fit())
    print("stage-1:  ", stage1())
    for k in WIDTHS:
        g = percall(k)
        print(f"  K={k:5d}  n={len(g):5d}  mean G={sum(g) / len(g):.3f}")
