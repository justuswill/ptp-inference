"""Collect every number the paper uses into ``paper/data/`` as plain JSON.

This is the ONLY script that touches the cluster filesystem or the repo's
``scratch/`` directory. Run it once (or after new inference runs land):

    uv run python paper/code/collect_data.py [--force]

Everything downstream -- ``rundata.py``, ``make_figs.py``, ``make_tables.py`` --
reads exclusively from ``paper/data/``, so the figures stay reproducible from the
checked-in snapshot alone, with no cluster access.

Outputs
    data/runs.json          per-run config + aggregated metrics + pooled z-test
    data/branching_fit.json joint-MLE (pi0, p, rho) and the stage-1 (pi0, rho)
    data/percall.json       raw per-call accepted-token samples G, by width K
    data/per_question.json  stage-2 per-question / population fits, by width
    data/meta.json          provenance: source paths, checkpoint, timestamps
"""

from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
REPO = PAPER.parent
DATA = PAPER / "data"
SCRATCH = REPO / "scratch"

# ---- sources on the cluster / in the repo -----------------------------------
CKPT = Path("/extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old")
INFERENCE_DIR = CKPT / "inference"
PREFIX = "vicuna_old_"  # scratch/ prefix for this checkpoint's fitted models

PERCALL_FILES = {
    1: f"{PREFIX}choice1_percall_full.json",
    2: f"{PREFIX}choice2_percall_full.json",
    5: f"{PREFIX}choice5_percall_full.json",
    50: f"{PREFIX}choice50_percall_full.json",
    1000: f"{PREFIX}seqn_choice1000_percall_full.json",
}

sys.path.insert(0, str(REPO / "scripts"))


def is_excluded(name: str) -> bool:
    """Superseded or non-comparable run directories.

    ``stale-*`` are marked dead in the repo, ``tmp_ide`` has a single example,
    ``oldcon-p-*`` is the superseded spelling of ``conf-p``.
    """
    return (
        name.startswith("stale-")
        or name == "tmp_ide"
        or name.startswith("oldcon-p-")
        or not name[0].isalnum()
    )


def _compression_ratio(text: str) -> float:
    """gzip repetition proxy, as in ``scripts/eval.py``. Lower = more repetitive."""
    if not text:
        return float("nan")
    enc = text.encode("utf-8")
    return len(gzip.compress(enc)) / len(enc)


def collect_runs() -> dict[str, dict]:
    from eval import aggregate, combined_ztest  # scripts/eval.py

    out: dict[str, dict] = {}
    for run_dir in sorted(INFERENCE_DIR.iterdir()):
        if not run_dir.is_dir() or is_excluded(run_dir.name):
            continue
        results = run_dir / "results.jsonl"
        if not results.exists():
            continue
        records = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
        if not records:
            continue
        for rec in records:
            rec["metrics"]["compression_ratio"] = _compression_ratio(rec.get("generated", ""))
        cfg = run_dir / "run_config.json"
        z, p = combined_ztest(records)
        out[run_dir.name] = {
            "config": json.loads(cfg.read_text()) if cfg.exists() else {},
            "agg": aggregate(records),
            "z": z,
            "p": p,
            "n": len(records),
        }
        print(f"  {run_dir.name} ({len(records)} examples)", flush=True)
    return out


def main(force: bool = False) -> None:
    DATA.mkdir(exist_ok=True)

    runs_path = DATA / "runs.json"
    if runs_path.exists() and not force:
        print(f"{runs_path.name} exists; pass --force to rescan the cluster")
    else:
        print(f"scanning {INFERENCE_DIR}")
        runs = collect_runs()
        runs_path.write_text(json.dumps(runs, indent=1, sort_keys=True))
        print(f"wrote {runs_path.name} ({len(runs)} runs)")

    # Fitted branching model. NOTE: the key `p0` in the v3 file is pi0
    # (zero-inflation), not the per-strand success probability `p`.
    v3 = json.loads((SCRATCH / f"{PREFIX}v3_jointM_fit.json").read_text())
    s1 = json.loads((SCRATCH / f"{PREFIX}joint_per_question_pi0_rho.json").read_text())
    (DATA / "branching_fit.json").write_text(json.dumps({
        "joint_mle": {"pi0": v3["p0"], "p": v3["p"], "rho": v3["rho"], "nll": v3["nll"]},
        "stage1_profile": {"pi0": s1["pi0"], "rho": s1["rho"], "nll": s1["nll"]},
    }, indent=1))
    print("wrote branching_fit.json")

    percall = {str(k): json.loads((SCRATCH / f).read_text()) for k, f in PERCALL_FILES.items()}
    (DATA / "percall.json").write_text(json.dumps(percall))
    print("wrote percall.json  " + "  ".join(
        f"K={k}:n={len(v)}" for k, v in percall.items()))

    pq = json.loads((SCRATCH / f"{PREFIX}joint_per_question_results.json").read_text())
    (DATA / "per_question.json").write_text(json.dumps(pq, indent=1))
    print("wrote per_question.json")

    (DATA / "meta.json").write_text(json.dumps({
        "checkpoint": str(CKPT),
        "inference_dir": str(INFERENCE_DIR),
        "scratch_prefix": PREFIX,
        "percall_sources": {str(k): f"scratch/{f}" for k, f in PERCALL_FILES.items()},
        "collected_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Preliminary single-seed runs on 100 Spec-Bench first turns.",
    }, indent=1))
    print("wrote meta.json")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
