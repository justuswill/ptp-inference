import sys, json, re, time
from pathlib import Path

sys.path.insert(0, "/home/jcwill/Projects/ptp/scripts")
sys.path.insert(0, "/home/jcwill/Projects/ptp/src")

import torch
from datasets import load_dataset as hf_load_dataset

from inference import (
    load_model, EXPERIMENT_CONFIGS, iter_spec_bench_pairs,
    SeqNPTPChoiceKInference, SeqNPTPTopPChoiceKInference, SeqNPTPTopPChoiceKSelfInference,
)

# Usage: collect_seqn_choicek_hist.py K [N_EXAMPLES] [CKPT_NAME] [TOPP_THRESHOLD] [SELF]
# CKPT_NAME selects /extra/ucibdl1/jcwill/ptp/checkpoints/<CKPT_NAME> and prefixes the
# output files so runs against different checkpoints don't clobber each other; defaults
# to "vicuna" (original behavior, unprefixed output files).
# TOPP_THRESHOLD, if given, switches from SeqNPTPChoiceKInference (exact-match accept) to
# SeqNPTPTopPChoiceKInference (nucleus accept at that threshold). SELF, if the literal
# string "self", further switches to SeqNPTPTopPChoiceKSelfInference (self-speculative
# per-block teacher verification via merged-LoRA instead of a real teacher call); only
# valid together with TOPP_THRESHOLD.
k = int(sys.argv[1])
n_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ckpt_name = sys.argv[3] if len(sys.argv) > 3 else "vicuna"
topp_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else None
self_mode = len(sys.argv) > 5 and sys.argv[5] == "self"
assert not (self_mode and topp_threshold is None), \
    "SELF requires TOPP_THRESHOLD -- no exact-match self-speculative class exists"
ckpt_prefix = "" if ckpt_name == "vicuna" else f"{ckpt_name}_"
variant_prefix = f"topp{topp_threshold}_" if topp_threshold is not None else ""
self_prefix = "self_" if self_mode else ""
prefix = ckpt_prefix + variant_prefix + self_prefix

experiment_dir = Path(f"/extra/ucibdl1/jcwill/ptp/checkpoints/{ckpt_name}")
# Explicit checkpoint for non-default ckpt_names -- keeps load_model's original
# find_best_checkpoint(ckpt_dir) behavior for "vicuna" (checkpoint=None) untouched.
checkpoint = None if ckpt_name == "vicuna" else experiment_dir / "last.ckpt"
exp = EXPERIMENT_CONFIGS["vicuna"]
max_tokens_per_proposal = 20
total_budget = 120
max_new_tokens = 300
seed = 42
block_size = 50

out_path = Path(f"/home/jcwill/Projects/ptp/scratch/{prefix}seqn_choice{k}_percall_full.json")
# Matches the sbatch job's --output path (and what analyze_branching_joint_*.py --prefix
# expects to parse) -- reading it back lets a resubmitted run resume instead of
# restarting, as long as the sbatch script uses --open-mode=append so prior lines survive.
log_path = Path(f"/home/jcwill/Projects/ptp/scratch/collect_{prefix}seqn1000_full.log")

resume_rows = []
if log_path.exists():
    pat = re.compile(r"example=(\d+) \(\d+/\d+\) n_rounds=(\d+) sum=(\d+)")
    for line in open(log_path):
        m = pat.search(line)
        if m:
            resume_rows.append((int(m.group(1)), int(m.group(2))))

n_resume = len(resume_rows)
if n_resume > 0:
    all_correct = json.load(open(out_path))
    expected = sum(nr for _, nr in resume_rows)
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

pair_iter = iter_spec_bench_pairs(dataset, tokenizer, 2048)
n_done = n_resume
t_start = time.time()
for i, (example_id, prompt_ids, ref_ids, prompt_str, ref_str) in enumerate(pair_iter):
    if i < n_resume:
        continue  # already collected in a prior (killed/timed-out) run; skip re-processing
    if n_done >= n_examples:
        break
    prompt_ids = prompt_ids.to(device)
    torch.manual_seed(seed)
    if self_mode:
        algo = SeqNPTPTopPChoiceKSelfInference(lit_model, device, autocast_dtype, k=k,
                                                block_size=block_size, threshold=topp_threshold)
    elif topp_threshold is not None:
        algo = SeqNPTPTopPChoiceKInference(lit_model, device, autocast_dtype, k=k,
                                            block_size=block_size, threshold=topp_threshold)
    else:
        algo = SeqNPTPChoiceKInference(lit_model, device, autocast_dtype, k=k, block_size=block_size)
    autocast_ctx = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype is not None else __import__("contextlib").nullcontext()
    t0 = time.time()
    with autocast_ctx:
        completion, n_calls_raw, correct_all = algo._generate_blocked(prompt_ids, max_new_tokens)
    dt = time.time() - t0
    all_correct.extend(correct_all)
    n_done += 1
    elapsed = time.time() - t_start
    n_new = n_done - n_resume  # elapsed/t_start only cover work done in *this* process
    eta = elapsed / n_new * (n_examples - n_done)
    print(f"k={k} example={example_id} ({n_done}/{n_examples}) n_rounds={len(correct_all)} sum={sum(correct_all)} dt={dt:.1f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    with open(out_path, "w") as f:
        json.dump(all_correct, f)

print(f"DONE: collected {len(all_correct)} per-call samples for k={k} across {n_done} examples", flush=True)
