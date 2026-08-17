"""
Aggregate and compare inference runs.

Usage
-----
    python scripts/eval.py /path/to/inference/dir

Each subdirectory under the given path is expected to contain:
    results.jsonl   — one JSON line per example (output of inference.py)
    run_config.json — run metadata

The script:
  1. Loads all runs found in the directory.
  2. Verifies that prompts match across runs (same example_id → same prompt).
  3. Prints a comparison table of mean metrics across all examples.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """Return (run_config, list of records) for one run directory."""
    config_path = run_dir / "run_config.json"
    results_path = run_dir / "results.jsonl"

    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    records = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return config, records


def check_prompt_alignment(runs: list[tuple[str, list[dict]]]) -> bool:
    """
    Verify that all runs cover the same example_ids and that prompts match.
    Returns True if aligned, prints mismatches and returns False otherwise.
    """
    if len(runs) < 2:
        return True

    ref_name, ref_records = runs[0]
    ref_by_id = {r["example_id"]: r["prompt"] for r in ref_records}

    ok = True
    for run_name, records in runs[1:]:
        by_id = {r["example_id"]: r["prompt"] for r in records}

        missing = set(ref_by_id) - set(by_id)
        extra = set(by_id) - set(ref_by_id)
        if missing:
            print(f"  WARNING: run '{run_name}' is missing example_ids: {sorted(missing)[:10]}")
            ok = False
        if extra:
            print(f"  WARNING: run '{run_name}' has extra example_ids: {sorted(extra)[:10]}")
            ok = False

        for eid in set(ref_by_id) & set(by_id):
            if ref_by_id[eid] != by_id[eid]:
                print(f"  WARNING: prompt mismatch at example_id={eid} between '{ref_name}' and '{run_name}'")
                ok = False

    return ok


class QM9RDKitHelper:
    def __init__(self):
        self._chem = self._qed = self._crippen = self._descriptors = None

    def _ensure_rdkit(self):
        if self._chem is None:
            try:
                from rdkit import Chem, RDLogger
                from rdkit.Chem import QED, Crippen, Descriptors
            except ImportError as exc:
                raise RuntimeError("QM9 metrics require `rdkit` — add it to your env.") from exc
            RDLogger.DisableLog("rdApp.*")
            self._chem = Chem
            self._qed = QED
            self._crippen = Crippen
            self._descriptors = Descriptors

    def compute_metrics_from_smiles(self, smiles_list: list[str]) -> dict:
        self._ensure_rdkit()
        canonical, valid_mols = [], []
        for s in smiles_list:
            s = (s or "").strip().replace(" ", "")
            mol = self._chem.MolFromSmiles(s)
            if mol is not None:
                canonical.append(self._chem.MolToSmiles(mol, canonical=True))
                valid_mols.append(mol)
        n_total = len(smiles_list)
        n_valid = len(valid_mols)
        qed_v, logp_v, mw_v = [], [], []
        for mol in valid_mols:
            try:
                qed_v.append(self._qed.qed(mol))
                logp_v.append(self._crippen.MolLogP(mol))
                mw_v.append(self._descriptors.MolWt(mol))
            except Exception:
                pass
        def _ms(v):
            a = np.asarray(v, dtype=np.float64)
            return (float(a.mean()), float(a.std())) if len(a) else (float("nan"), float("nan"))
        qed_m, qed_s = _ms(qed_v)
        logp_m, logp_s = _ms(logp_v)
        mw_m, mw_s = _ms(mw_v)
        return {
            "valid_rate_mean":  n_valid / max(n_total, 1),
            "valid_rate_std":   0.0,
            "unique_rate_mean": len(set(canonical)) / max(n_valid, 1),
            "unique_rate_std":  0.0,
            "qed_mean": qed_m, "qed_std": qed_s,
            "logp_mean": logp_m, "logp_std": logp_s,
            "mw_mean": mw_m, "mw_std": mw_s,
        }


def compute_qm9_metrics(records: list[dict]) -> dict:
    smiles = [r.get("generated", "") for r in records]
    return QM9RDKitHelper().compute_metrics_from_smiles(smiles)


def aggregate(records: list[dict]) -> dict[str, float]:
    """Compute mean and std of every numeric metric field across all records."""
    accum: dict[str, list[float]] = {}
    for rec in records:
        for key, val in rec.get("metrics", {}).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                accum.setdefault(key, []).append(val)

    result = {}
    for key, vals in accum.items():
        # Drop non-finite values (NaN from 0-token outputs, legacy missing fields)
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            continue
        arr = np.array(finite)
        result[f"{key}_mean"] = float(np.mean(arr))
        result[f"{key}_std"] = float(np.std(arr))
        # Probability (%) and perplexity forms for actual log-prob metrics only
        if "_lp_mean" in key or "_lp_median" in key:
            result[f"{key}_prob_mean"] = float(np.mean(np.exp(arr))) * 100
            result[f"{key}_prob_std"] = float(np.std(np.exp(arr))) * 100
            result[f"{key}_ppl_mean"] = float(np.mean(np.exp(-arr)))
            result[f"{key}_ppl_std"] = float(np.std(np.exp(-arr)))
    return result


def combined_ztest(records: list[dict]) -> tuple[float, float] | tuple[None, None]:
    """
    Pool the per-example sufficient statistics to produce one z-statistic and
    p-value for the null hypothesis that all student tokens were sampled from
    the teacher distribution.

    Requires gen_lp_z_num and gen_lp_z_var saved by inference.py (>= current
    version).  Returns (None, None) for legacy results that lack these fields.
    """
    total_num = 0.0
    total_var = 0.0
    n_used = 0
    for rec in records:
        m = rec.get("metrics", {})
        z_num = m.get("gen_lp_z_num")
        z_var = m.get("gen_lp_z_var")
        if z_num is None or z_var is None or not math.isfinite(z_num) or not math.isfinite(z_var):
            continue
        total_num += z_num
        total_var += z_var
        n_used += 1
    if n_used == 0 or total_var <= 0:
        return None, None
    z = total_num / math.sqrt(total_var)
    import math as _math
    p = _math.erfc(abs(z) / _math.sqrt(2))
    return z, p


def _referee_cache_path(run_dir: Path, model_name_or_path: str) -> Path:
    import re
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name_or_path)
    return run_dir / f"referee_{slug}.json"


def _load_referee_cache(run_dir: Path, model_name_or_path: str) -> dict | None:
    """Return cached scores dict {str(example_id): {mean, median}} or None if missing/stale."""
    cache_path = _referee_cache_path(run_dir, model_name_or_path)
    if not cache_path.exists():
        return None
    current_mtime = (run_dir / "results.jsonl").stat().st_mtime
    with open(cache_path) as f:
        cache = json.load(f)
    if cache.get("results_mtime") != current_mtime:
        return None
    return cache["scores"]


def _save_referee_cache(run_dir: Path, model_name_or_path: str, scores: dict) -> None:
    from datetime import datetime
    cache_path = _referee_cache_path(run_dir, model_name_or_path)
    with open(cache_path, "w") as f:
        json.dump({
            "referee_model": model_name_or_path,
            "results_mtime": (run_dir / "results.jsonl").stat().st_mtime,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "scores": scores,
        }, f, indent=2)


def load_referee(model_name_or_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading referee model: {model_name_or_path}")
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    # With >1 GPU, pin the referee entirely to the last device so it never
    # competes for memory with a main model already sitting on cuda:0.
    # With only 1 GPU visible, fall back to Accelerate's best-effort placement.
    if torch.cuda.device_count() > 1:
        device_map = {"": f"cuda:{torch.cuda.device_count() - 1}"}
    else:
        device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map=device_map
    )
    return model.eval(), tok


def referee_log_prob(model, tokenizer, text: str) -> dict:
    import torch
    import torch.nn.functional as F
    if not text.strip():
        return {"mean": float("nan"), "median": float("nan")}
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    if ids.shape[1] < 2:
        return {"mean": float("nan"), "median": float("nan")}
    with torch.inference_mode(), torch.autocast(model.device.type, dtype=torch.bfloat16):
        logits = model(ids).logits.float()
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    per_token = log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    return {"mean": per_token.mean().item(), "median": per_token.median().item()}


_JUDGE_PROMPT = (
    "Rate the quality of the following text on a scale from 1 to 5, "
    "where 1 = completely incoherent or broken, 3 = partially coherent with noticeable errors, "
    "and 5 = fluent and high quality. "
    "Respond with a single digit and nothing else.\n\n"
    "Text:\n{text}\n\nRating:"
)


def referee_judge(model, tokenizer, text: str) -> float:
    import re, torch
    if not text.strip():
        return float("nan")
    prompt = _JUDGE_PROMPT.format(text=text[:1500])
    if getattr(tokenizer, "chat_template", None):
        input_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    else:
        input_text = prompt
    ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(
            ids, max_new_tokens=5, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
    m = re.search(r"[1-5]", response)
    return float(m.group()) if m else float("nan")


def run_label(run_dir: Path) -> str:
    return run_dir.name


def print_table(rows: list[tuple[str, dict]], metrics: list[str], chunk_size: int = 7) -> None:
    import pandas as pd

    data = {}
    for label, agg in rows:
        col = {}
        for metric in metrics:
            mean_key = f"{metric}_mean"
            std_key = f"{metric}_std"
            col[metric] = _fmt(metric, agg[mean_key], agg[std_key]) if mean_key in agg else "—"
        data[label] = col

    df = pd.DataFrame(data)
    df.index.name = "metric"

    labels = list(data.keys())
    for start in range(0, len(labels), chunk_size):
        chunk = labels[start:start + chunk_size]
        print(df[chunk].to_string())
        if start + chunk_size < len(labels):
            print()


DISPLAY_METRICS = [
    "gen_lp_mean_prob",      # exp(gen_lp_mean) in %
    "gen_lp_mean_ppl",       # perplexity = exp(-gen_lp_mean)
    "gen_lp_median_prob",
    "gen_lp_median_ppl",
    "gen_lp_p_value",        # two-sided p-value for H0: tokens ~ teacher
    "gen_lp_outlier_perc",   # % tokens skipped from z-test (near-zero adapted prob)
    "referee_lp_mean_prob",  # exp(referee_lp_mean) in % — quality under referee model
    "referee_lp_mean_ppl",
    "referee_lp_median_prob",
    "referee_lp_median_ppl",
    "referee_judge_score",   # 1–5 rating from referee model
    "compression_ratio",     # gzip(text)/len(text) — lower = more repetitive
    "ms_per_token",
    "correct_per_call",
    "n_generated_tokens",
    "num_calls",
    "n_prompt_tokens",
]

QM9_EXTRA_METRICS = ["valid_rate", "unique_rate", "qed", "logp", "mw"]

def _ppl_decimals(val: float) -> int:
    if not math.isfinite(val) or val <= 0:
        return 2
    return max(0, 3 - int(math.floor(math.log10(val))))


# How to format each metric's cell
def _fmt(metric: str, mean: float, std: float) -> str:
    if metric.endswith("_prob"):
        return f"{mean:5.1f}% ±{std:.1f}%"
    if metric.endswith("_ppl"):
        d = _ppl_decimals(mean)
        return f"{mean:.{d}f} ±{std:.{d}f}"
    if metric == "gen_lp_z_stat":
        return f"{mean:+.3f}"
    if metric == "gen_lp_p_value":
        return f"{mean:.1e}" if mean < 0.01 else f"{mean:.3f}"
    if metric == "gen_lp_outlier_perc":
        return f"{mean:.2f}% ±{std:.2f}%"
    if metric == "referee_judge_score":
        return f"{mean:.2f}/5 ±{std:.2f}"
    if "_lp_" in metric:
        return f"{mean:+.4f} ±{std:.4f}"
    if metric in ("ms_per_token",):
        return f"{mean:6.1f} ±{std:.1f}"
    if metric in ("correct_per_call",):
        return f"{mean:.4f} ±{std:.4f}"
    if metric in ("valid_rate", "unique_rate"):
        return f"{mean*100:5.1f}%"
    if metric == "qed":
        return f"{mean:.3f} ±{std:.3f}"
    if metric == "logp":
        return f"{mean:.2f} ±{std:.2f}"
    if metric == "mw":
        return f"{mean:.1f} ±{std:.1f}"
    return f"{mean:.2f} ±{std:.2f}"


def main(inference_dir: Path, show_all: bool = False, think: bool = False,
         experiment: str = "qwen", referee_model: str | None = None,
         no_thresh: bool = False, no_first: bool = False) -> None:
    if not inference_dir.exists():
        print(f"Error: directory not found: {inference_dir}")
        raise SystemExit(1)

    run_dirs = sorted(
        d for d in inference_dir.iterdir()
        if d.is_dir() and (d / "results.jsonl").exists()
        and (think or "cot" not in d.name)
        and not (no_thresh and "thresh" in d.name)
        and not (no_first and "first" in d.name)
    )
    if not run_dirs:
        print(f"No results.jsonl files found under {inference_dir}")
        raise SystemExit(1)

    print(f"Found {len(run_dirs)} run(s) under {inference_dir}\n")

    loaded: list[tuple[Path, dict, list[dict]]] = []
    for run_dir in run_dirs:
        config, records = load_run(run_dir)
        print(f"  {run_dir.name}: {len(records)} examples  [{config.get('algorithm','?')}, seed={config.get('seed','?')}]")
        loaded.append((run_dir, config, records))

    print()

    # Prompt alignment check
    print("Checking prompt alignment across runs...")
    named_runs = [(run_dir.name, records) for run_dir, _, records in loaded]
    if check_prompt_alignment(named_runs):
        print("  OK — all prompts match.\n")
    else:
        print()

    # Filter by completeness
    complete = [(run_dir, config, records) for run_dir, config, records in loaded if len(records) == 100]
    incomplete = [(run_dir, config, records) for run_dir, config, records in loaded if 0 < len(records) < 100]

    if incomplete:
        for run_dir, _, records in incomplete:
            print(f"  incomplete: {run_dir.name} ({len(records)} examples)")
        if not show_all:
            print(f"  (pass --all to include incomplete runs)\n")
        print()

    to_show = loaded if show_all else complete

    if not to_show:
        print("No runs to show.")
        raise SystemExit(1)

    # Referee scoring: log-prob + judge rating, cached together per run dir.
    # Delete the cache file manually to recompute (e.g. after adding new fields).
    if referee_model:
        all_scores = {run_dir: _load_referee_cache(run_dir, referee_model)
                      for run_dir, _, _ in to_show}
        needs_compute = [(run_dir, records)
                         for run_dir, _, records in to_show
                         if all_scores[run_dir] is None]

        if needs_compute:
            from tqdm import tqdm
            ref_model, ref_tok = load_referee(referee_model)
            for run_dir, records in needs_compute:
                scores = {}
                for rec in tqdm(records, desc=f"referee {run_dir.name}", unit="ex"):
                    text = rec.get("generated", "")
                    entry = referee_log_prob(ref_model, ref_tok, text)
                    entry["judge"] = referee_judge(ref_model, ref_tok, text)
                    scores[str(rec["example_id"])] = entry
                _save_referee_cache(run_dir, referee_model, scores)
                all_scores[run_dir] = scores
        else:
            print("  all referee scores loaded from cache")

        for run_dir, _, records in to_show:
            scores = all_scores[run_dir]
            for rec in records:
                entry = scores.get(str(rec["example_id"]), {})
                rec["metrics"]["referee_lp_mean"]     = entry.get("mean",   float("nan"))
                rec["metrics"]["referee_lp_median"]   = entry.get("median", float("nan"))
                rec["metrics"]["referee_judge_score"] = entry.get("judge",  float("nan"))
        print()

    # Compression ratio — model-free repetition proxy; lower = more repetitive
    import gzip
    for _, _, records in to_show:
        for rec in records:
            text = rec.get("generated", "")
            if text:
                enc = text.encode("utf-8")
                rec["metrics"]["compression_ratio"] = len(gzip.compress(enc)) / len(enc)

    # Aggregate
    rows: list[tuple[str, dict]] = []
    for run_dir, config, records in to_show:
        label = run_label(run_dir) + (f" ({len(records)})" if show_all and len(records) != 100 else "")
        agg = aggregate(records)
        if experiment == "qm9":
            agg.update(compute_qm9_metrics(records))
        # Overwrite per-example averaged z/p with the combined z-test across all tokens
        z, p = combined_ztest(records)
        if z is not None:
            agg["gen_lp_z_stat_mean"] = z
            agg["gen_lp_z_stat_std"] = float("nan")
            agg["gen_lp_p_value_mean"] = p
            agg["gen_lp_p_value_std"] = float("nan")
        rows.append((label, agg))

    # Only show metrics that appear in at least one run
    extra = QM9_EXTRA_METRICS if experiment == "qm9" else []
    all_metrics = DISPLAY_METRICS + extra + [
        m for m in next(iter(rows), (None, {}))[1]
        if m.endswith("_mean") and m[:-5] not in DISPLAY_METRICS and m[:-5] not in extra
    ]
    present = [m for m in all_metrics if any(f"{m}_mean" in agg for _, agg in rows)]

    print_table(rows, present)
    print()

    # Quick interpretation hints
    print("Notes:")
    print("  gen_lp_*        — per-token log-prob of generated text under teacher (higher = better)")
    print("  gen_lp_p_value  — two-sided p-value for H0: tokens ~ teacher (uniform on [0,1] if correct)")
    print("  correct_per_call — PTP tokens accepted per speculative step (higher = better speculation)")
    print("  ms_per_token — wall-clock speed (lower = faster)")
    print("  compression_ratio — gzip size / raw size; lower = more repetitive/degenerate")
    if referee_model:
        print("  referee_lp_*       — per-token log-prob under referee model (independent quality measure)")
        print("  referee_judge_score — 1–5 quality rating from referee model (higher = better)")
    if experiment == "qm9":
        print("  valid_rate  — % generated SMILES parseable by RDKit")
        print("  unique_rate — % unique canonical SMILES among valid")
        print("  qed/logp/mw — drug-likeness, lipophilicity, molecular weight (valid mols only)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("inference_dir", type=Path)
    parser.add_argument("--all", dest="show_all", action="store_true",
                        help="Include incomplete runs (at least 1 example recorded)")
    parser.add_argument("--think", action="store_true",
                        help="Include runs with 'cot' in their folder name (excluded by default)")
    parser.add_argument("--experiment", choices=["qwen", "qm9"], default="qwen",
                        help="Experiment preset; qm9 adds chemistry metrics (default: qwen)")
    parser.add_argument("--referee", metavar="MODEL", default=None,
                        help="HuggingFace model name or path to use as independent referee scorer")
    parser.add_argument("--no-thresh", action="store_true",
                        help="Exclude thresh-* runs from the table")
    parser.add_argument("--no-first", action="store_true",
                        help="Exclude first_* runs from the table")
    args = parser.parse_args()
    main(args.inference_dir, show_all=args.show_all, think=args.think,
         experiment=args.experiment, referee_model=args.referee,
         no_thresh=args.no_thresh, no_first=args.no_first)
