import sys, json, time
from pathlib import Path

sys.path.insert(0, "/home/jcwill/Projects/ptp/scripts")
sys.path.insert(0, "/home/jcwill/Projects/ptp/src")

import torch
from datasets import load_dataset as hf_load_dataset

from inference import load_model, EXPERIMENT_CONFIGS, iter_spec_bench_pairs, SeqNPTPChoiceKInference

k = int(sys.argv[1])
n_examples = int(sys.argv[2]) if len(sys.argv) > 2 else 100

experiment_dir = Path("/extra/ucibdl1/jcwill/ptp/checkpoints/vicuna")
exp = EXPERIMENT_CONFIGS["vicuna"]
max_tokens_per_proposal = 20
total_budget = 120
max_new_tokens = 300
seed = 42
block_size = 50

out_path = Path(f"/tmp/claude-20015/-home-jcwill-Projects-ptp/d3916985-3338-43af-94c5-20fe11c89503/scratchpad/seqn_choice{k}_percall_full.json")

torch.manual_seed(seed)
lit_model, config, autocast_dtype, device, ckpt_path = load_model(experiment_dir, None, total_budget)

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
all_correct = []
n_done = 0
t_start = time.time()
for example_id, prompt_ids, ref_ids, prompt_str, ref_str in pair_iter:
    if n_done >= n_examples:
        break
    prompt_ids = prompt_ids.to(device)
    torch.manual_seed(seed)
    algo = SeqNPTPChoiceKInference(lit_model, device, autocast_dtype, k=k, block_size=block_size)
    autocast_ctx = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype is not None else __import__("contextlib").nullcontext()
    t0 = time.time()
    with autocast_ctx:
        completion, n_calls_raw, correct_all = algo._generate_blocked(prompt_ids, max_new_tokens)
    dt = time.time() - t0
    all_correct.extend(correct_all)
    n_done += 1
    elapsed = time.time() - t_start
    eta = elapsed / n_done * (n_examples - n_done)
    print(f"k={k} example={example_id} ({n_done}/{n_examples}) n_rounds={len(correct_all)} sum={sum(correct_all)} dt={dt:.1f}s elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    with open(out_path, "w") as f:
        json.dump(all_correct, f)

print(f"DONE: collected {len(all_correct)} per-call samples for k={k} across {n_done} examples", flush=True)
