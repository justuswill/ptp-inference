"""Instrumented debug run: seq-ptp-self-tok on vicuna_old, comparing student_forward
GATED (LoRA off on backlog, on for aux window -- matches training's ar_forward) vs
MERGED (LoRA on everywhere, matching teacher_forward's tok branch) side by side, to
settle empirically whether vicuna_old's student proposal head expects a gated or
merged backlog."""
import sys
from pathlib import Path
import contextlib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import inference as inf  # scripts/inference.py

CKPT_DIR = Path("/extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old")
CKPT = CKPT_DIR / "last.ckpt"

lit_model, config, autocast_dtype, device, ckpt_path = inf.load_model(CKPT_DIR, CKPT, total_token_budget=120)

tokenizer = lit_model.model.tokenizer
prompts = [
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: x+y = 4z, x*y = 4z^2, express x-y in z ASSISTANT:",
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Write a short poem about autumn. ASSISTANT:",
]


def make_student_forward(algo, merged: bool):
    def student_forward(input_ids, auxiliaries, past_key_values):
        if auxiliaries is not None:
            algo._z_student = auxiliaries[0]
        ctx = inf._full_lora_mode(algo.lit_model) if merged else contextlib.nullcontext()
        with ctx:
            return algo.lit_model.model.inference_forward(
                input_ids=input_ids, auxiliaries=auxiliaries,
                past_key_values=past_key_values, use_cache=True,
            )
    return student_forward


def run(merged: bool, prompt: str, max_new_tokens: int = 100):
    algo = inf.SeqPTPSelfInference(lit_model, device, autocast_dtype, ar_mode="tok")
    algo.student_forward = make_student_forward(algo, merged)

    round_log = []
    orig_accepted = algo.accepted_tokens

    def instrumented(student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        result = orig_accepted(student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd)
        st_dec = tokenizer.decode(student_tokens[0, :6])
        ct_dec = tokenizer.decode(correct_tokens[0, :6])
        round_log.append((n_prop, result.shape[1], st_dec, ct_dec))
        return result

    algo.accepted_tokens = instrumented
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        completion, metrics = algo.generate(ids, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(completion[0, ids.shape[1]:])
    return metrics, text, round_log


for i, prompt in enumerate(prompts):
    print(f"\n{'='*24} PROMPT {i} {'='*24}")
    for label, merged in [("GATED (student LoRA off on backlog)", False), ("MERGED (student LoRA everywhere)", True)]:
        print(f"\n--- {label} ---")
        metrics, text, round_log = run(merged, prompt)
        print(f"correct_per_call={metrics['correct_per_call']:.3f} num_calls={metrics['num_calls']}")
        print(text[:400])
        print("per-round (first 8):")
        for r, (n_prop, accepted, st_dec, ct_dec) in enumerate(round_log[:8]):
            print(f"  round {r}: n_prop={n_prop} accepted={accepted}  student={st_dec!r}  teacher={ct_dec!r}")
