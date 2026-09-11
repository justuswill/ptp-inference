"""Compare ptp_self and seq-ptp-self-tok round-by-round on the same vicuna_old
prompts, looking for patterns in why ptp_self's correct_per_call diverges from
seq-ptp-self-tok's (should be roughly seq - 1, per PTP's free bonus-token design)."""
import sys
from pathlib import Path
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import inference as inf  # scripts/inference.py

CKPT_DIR = Path("/extra/ucibdl1/jcwill/ptp/checkpoints/vicuna_old")
CKPT = CKPT_DIR / "last.ckpt"

lit_model, config, autocast_dtype, device, ckpt_path = inf.load_model(CKPT_DIR, CKPT, total_token_budget=120)
tokenizer = lit_model.model.tokenizer

# main() normally sets these after load_model() (scripts/inference.py:3310-3321);
# our debug script bypasses main(), so replicate it here -- matches the real sbatch
# runs' run_config.json (top_k=50, top_p=0.9, temperature=0.7).
model_cfg = config.get("model", {})
lit_model.top_k = model_cfg.get("top_k", None) or 50
lit_model.top_p = model_cfg.get("top_p", None) or 0.9
lit_model.temperature = model_cfg.get("temperature", None) or 0.7
lit_model.tokens_per_student_call = 20
lit_model.total_token_budget = 120

prompts = [
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: x+y = 4z, x*y = 4z^2, express x-y in z ASSISTANT:",
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Write a short poem about autumn. ASSISTANT:",
    "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: What is the capital of France? ASSISTANT:",
]

MAX_NEW = 150


def run_seq_self_tok(prompt):
    algo = inf.SeqPTPSelfInference(lit_model, device, autocast_dtype, ar_mode="tok")
    round_log = []
    orig_accepted = algo.accepted_tokens

    def instrumented(student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        result = orig_accepted(student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd)
        round_log.append(result.shape[1])
        return result

    algo.accepted_tokens = instrumented
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        completion, metrics = algo.generate(ids, max_new_tokens=MAX_NEW)
    return metrics, round_log


def run_ptp_self(prompt):
    algo = inf.FullLoRAPTPInference(lit_model, device, autocast_dtype,
                                     max_tokens_per_proposal=20, total_token_budget=120)
    proposals_log = []
    orig_proposals = lit_model.proposals

    def instrumented(H, num_tokens=None, student_p=None, n_verify=None, double_at=100, metrics=None):
        B = orig_proposals(H, num_tokens=num_tokens, student_p=student_p, n_verify=n_verify,
                            double_at=double_at, metrics=metrics)
        proposals_log.append((len(B), sum(B), max(B) if B else 0))
        return B

    lit_model.proposals = instrumented
    orig_generate = lit_model.generate
    captured = {}

    def generate_capture(*args, **kwargs):
        completion, m = orig_generate(*args, **kwargs)
        captured["correct_all"] = m.get("correct_all", [])
        return completion, m

    lit_model.generate = generate_capture
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    try:
        with torch.no_grad():
            completion, metrics = algo.generate(ids, max_new_tokens=MAX_NEW)
    finally:
        lit_model.proposals = orig_proposals
        lit_model.generate = orig_generate
    round_log = captured.get("correct_all", [])
    return metrics, round_log, proposals_log


for i, prompt in enumerate(prompts):
    print(f"\n{'='*24} PROMPT {i} {'='*24}")

    seq_metrics, seq_rounds = run_seq_self_tok(prompt)
    print(f"\nseq-ptp-self-tok: correct_per_call={seq_metrics['correct_per_call']:.3f} num_calls={seq_metrics['num_calls']}")
    print(f"  per-round accepted (first 20): {seq_rounds[:20]}")
    print(f"  mean={statistics.mean(seq_rounds):.3f} median={statistics.median(seq_rounds)} max={max(seq_rounds)} min={min(seq_rounds)}")

    ptp_metrics, ptp_rounds, prop_log = run_ptp_self(prompt)
    print(f"\nptp_self: correct_per_call={ptp_metrics['correct_per_call']:.3f} num_calls={ptp_metrics['num_calls']}")
    print(f"  per-round accepted (first 20): {ptp_rounds[:20]}")
    print(f"  mean={statistics.mean(ptp_rounds):.3f} median={statistics.median(ptp_rounds)} max={max(ptp_rounds)} min={min(ptp_rounds)}")
    print(f"  proposals() calls (n_buckets, sum_B, max_B), first 10: {prop_log[:10]}")
    sum_Bs = [s for _, s, _ in prop_log]
    print(f"  sum(B) stats: mean={statistics.mean(sum_Bs):.1f} min={min(sum_Bs)} max={max(sum_Bs)}")
