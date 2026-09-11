import sys, json, re
from pathlib import Path

sys.path.insert(0, "/home/jcwill/Projects/ptp/scripts")
sys.path.insert(0, "/home/jcwill/Projects/ptp/src")

import torch
from datasets import load_dataset as hf_load_dataset

from inference import (
    load_model, EXPERIMENT_CONFIGS, iter_spec_bench_pairs,
    SeqPTPTopPSelfInference, _raw_logits_mode,
)

# Usage: collect_seqself_topp_hist.py THRESHOLD [CKPT_NAME] [AR_MODE]
# Per-call G-histogram collection for seq-ptp-top-p-self (single-strand, self-speculative
# teacher, nucleus acceptance at THRESHOLD) -- same role collect_choicek_hist.py plays for
# the choice-k branching-model fits, but for this flat (non-tree) algorithm.
# CKPT_NAME selects /extra/ucibdl1/jcwill/ptp/checkpoints/<CKPT_NAME> and prefixes output
# files; defaults to "vicuna". AR_MODE defaults to "aux" (SeqPTPTopPSelfInference's own
# default), matching the "tmp_seq-ptp-topp-self-{p}" naming (no "-tok" suffix); pass "tok"
# to instead match "tmp_seq-ptp-topp-self-{p}-tok".
threshold = float(sys.argv[1])
ckpt_name = sys.argv[2] if len(sys.argv) > 2 else "vicuna"
ar_mode = sys.argv[3] if len(sys.argv) > 3 else "aux"
ckpt_prefix = "" if ckpt_name == "vicuna" else f"{ckpt_name}_"
tag = f"{ckpt_prefix}topp{threshold}_self" + ("_tok" if ar_mode == "tok" else "")

experiment_dir = Path(f"/extra/ucibdl1/jcwill/ptp/checkpoints/{ckpt_name}")
# Explicit checkpoint for non-default ckpt_names -- keeps load_model's original
# find_best_checkpoint(ckpt_dir) behavior for "vicuna" (checkpoint=None) untouched.
checkpoint = None if ckpt_name == "vicuna" else experiment_dir / "last.ckpt"
exp = EXPERIMENT_CONFIGS["vicuna"]
max_tokens_per_proposal = 20
total_budget = 120
max_new_tokens = 300
seed = 42
n_examples = 100

out_path = Path(f"/home/jcwill/Projects/ptp/scratch/{tag}_percall_full.json")
# Matches the sbatch job's --output path -- reading it back lets a resubmitted run
# resume instead of restarting, as long as the sbatch script uses --open-mode=append.
log_path = Path(f"/home/jcwill/Projects/ptp/scratch/collect_{tag}.log")

resume_rows = []
if log_path.exists():
    pat = re.compile(r"example=(\d+) \(\d+/\d+\) n_calls=(\d+) sum=(\d+)")
    for line in open(log_path):
        m = pat.search(line)
        if m:
            resume_rows.append((int(m.group(1)), int(m.group(2))))

n_resume = len(resume_rows)
if n_resume > 0:
    all_correct = json.load(open(out_path))
    expected = sum(nc for _, nc in resume_rows)
    assert len(all_correct) == expected, (
        f"resume mismatch: {log_path} implies {expected} samples across {n_resume} "
        f"examples, but {out_path} has {len(all_correct)} -- refusing to resume "
        f"(stale/corrupt state); move both files aside to start fresh."
    )
    print(f"Resuming: {n_resume} examples / {len(all_correct)} samples already in "
          f"{out_path}", flush=True)
else:
    all_correct = []

torch.manual_seed(seed)
lit_model, config, autocast_dtype, device, ckpt_path = load_model(experiment_dir, checkpoint, total_budget)

model_cfg = config.get("model", {})
lit_model.top_k = model_cfg.get("top_k", None) or 50
lit_model.top_p = model_cfg.get("top_p", None) or 0.9
lit_model.temperature = model_cfg.get("temperature", None) or 0.7
lit_model.tokens_per_student_call = max_tokens_per_proposal
lit_model.total_token_budget = total_budget

tokenizer = lit_model.model.tokenizer
custom_template = exp.get("chat_template")
if custom_template and not getattr(tokenizer, "chat_template", None):
    tokenizer.chat_template = custom_template

dataset = hf_load_dataset("json", data_files={"train": exp["default_dataset"]}, split="train")

algo = SeqPTPTopPSelfInference(lit_model, device, autocast_dtype, ar_mode=ar_mode, threshold=threshold)
raw_ctx = _raw_logits_mode(lit_model) if algo.raw else __import__("contextlib").nullcontext()
pair_iter = iter_spec_bench_pairs(dataset, tokenizer, 2048)
n_done = n_resume
for i, (example_id, prompt_ids, ref_ids, prompt_str, ref_str) in enumerate(pair_iter):
    if i < n_resume:
        continue  # already collected in a prior (killed/timed-out) run; skip re-processing
    if n_done >= n_examples:
        break
    prompt_ids = prompt_ids.to(device)
    autocast_ctx = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype is not None else __import__("contextlib").nullcontext()
    with raw_ctx, autocast_ctx:
        completion, ptp_metrics = lit_model.generate_seq(
            prompt_ids, max_new_tokens=max_new_tokens,
            needs_teacher=algo.needs_teacher,
            accepted_tokens=algo.accepted_tokens,
            student_forward=algo.student_forward,
            teacher_forward=algo.teacher_forward,
            correct_first_token=algo.correct_first_token,
            shared_kv_cache=algo.shared_kv_cache,
        )
    all_correct.extend(ptp_metrics["correct_all"])
    n_done += 1
    print(f"example={example_id} ({n_done}/{n_examples}) n_calls={len(ptp_metrics['correct_all'])} sum={sum(ptp_metrics['correct_all'])}", flush=True)
    with open(out_path, "w") as f:
        json.dump(all_correct, f)

print(f"DONE: collected {len(all_correct)} per-call samples for threshold={threshold} ar_mode={ar_mode}")
