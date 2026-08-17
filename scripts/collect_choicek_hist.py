import sys, json
from pathlib import Path

sys.path.insert(0, "/home/jcwill/Projects/ptp/scripts")
sys.path.insert(0, "/home/jcwill/Projects/ptp/src")

import torch
from datasets import load_dataset as hf_load_dataset

from inference import load_model, EXPERIMENT_CONFIGS, iter_spec_bench_pairs, SeqPTPChoiceKInference

k = int(sys.argv[1])

experiment_dir = Path("/extra/ucibdl1/jcwill/ptp/checkpoints/vicuna")
exp = EXPERIMENT_CONFIGS["vicuna"]
max_tokens_per_proposal = 20
total_budget = 120
max_new_tokens = 300
seed = 42
n_examples = 100

out_path = Path(f"/home/jcwill/Projects/ptp/scratch/choice{k}_percall_full.json")

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

algo = SeqPTPChoiceKInference(lit_model, device, autocast_dtype, k=k)
pair_iter = iter_spec_bench_pairs(dataset, tokenizer, 2048)
all_correct = []
n_done = 0
for example_id, prompt_ids, ref_ids, prompt_str, ref_str in pair_iter:
    if n_done >= n_examples:
        break
    prompt_ids = prompt_ids.to(device)
    autocast_ctx = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype is not None else __import__("contextlib").nullcontext()
    with autocast_ctx:
        completion, ptp_metrics = lit_model.generate_seq_tree(
            prompt_ids, max_new_tokens=max_new_tokens,
            tree=algo.tree,
            needs_teacher=algo.needs_teacher,
            accepted_tokens=algo.accepted_tokens,
            student_forward=algo.student_forward,
            teacher_forward=algo.teacher_forward,
        )
    all_correct.extend(ptp_metrics["correct_all"])
    n_done += 1
    print(f"k={k} example={example_id} ({n_done}/{n_examples}) n_calls={len(ptp_metrics['correct_all'])} sum={sum(ptp_metrics['correct_all'])}", flush=True)

with open(out_path, "w") as f:
    json.dump(all_correct, f)
print(f"DONE: collected {len(all_correct)} per-call samples for k={k}")
