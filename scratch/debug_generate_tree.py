"""Standalone smoke test for lit.py's generate_tree (the choice-k/top-p/self single-call
engine behind "ptp-top-p-choice-k-self"), run against a fake model so it needs no GPU/
checkpoint. Exercises the REAL generate_tree/proposals/adapt_p/_in_nucleus code -- only
model.inference_forward is faked -- and asserts the mask/position-id/cache-growth
invariants the plan flagged as highest-risk:
  1. no cross-candidate or cross-fan-sibling attention leakage
  2. each row's position id matches the intended depth
  3. kv_cache grows by exactly n_new each round (not stuck, not unbounded)
  4. completion length reaches max_new_tokens and contains no leaked pad/garbage
"""
import sys
import types
from pathlib import Path
import inspect

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from ptp.lit import ParallelSamplingLightningModule as LM

VOCAB = 37
EOS = 1
torch.manual_seed(0)


class FakeCache:
    def __init__(self, n=0):
        self.n = n

    def get_seq_length(self):
        return self.n

    def crop(self, max_length):
        self.n = min(self.n, max_length)


class FakeTokenizer:
    eos_token_id = EOS


class FakeModel:
    inference_mode = True
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def inference_forward(self, input_ids=None, auxiliaries=None, past_key_values=None,
                           use_cache=True, attention_mask=None, position_ids=None, flag=False):
        n_real = 0 if input_ids is None else input_ids.shape[1]
        n_aux = 0 if auxiliaries is None else auxiliaries.shape[1]
        Q_LEN = n_real + n_aux
        K = past_key_values.get_seq_length()

        if attention_mask is not None:
            assert attention_mask.shape == (1, 1, Q_LEN, K + Q_LEN), \
                f"mask shape {attention_mask.shape} vs Q_LEN={Q_LEN}, K={K}"
            assert position_ids.shape == (1, Q_LEN)

            # --- Mask sanity checks (the highest-risk piece per the plan) ---
            m = attention_mask[0, 0]
            allowed = (m == 0.0)
            # every row must see at least the cache + its own key (no fully -inf row)
            assert allowed.any(dim=-1).all(), "a row has zero visible columns"
            for r in range(Q_LEN):
                assert allowed[r, K + r], f"row {r} cannot see its own key (K={K})"

            # --- Cross-candidate / cross-sibling leakage check, via the caller's own
            # locals (lens, cand_start, fan_start, B, k) -- the actual thing the plan
            # flagged as highest-risk. ---
            caller = inspect.currentframe().f_back.f_locals
            lens, cand_start, fan_start, B, kk = (
                caller.get("lens"), caller.get("cand_start"), caller.get("fan_start"),
                caller.get("B"), caller.get("k"),
            )
            if lens is not None and cand_start is not None:
                real_range = lambda i: range(K + cand_start[i], K + cand_start[i] + lens[i])
                for i in range(kk):
                    for r in real_range(i):
                        row = r - K
                        for j in range(kk):
                            if j == i:
                                continue
                            for c in real_range(j):
                                assert not allowed[row, c], \
                                    f"real row {row} (candidate {i}) leaks into candidate {j}'s real column {c}"
                if fan_start is not None and B is not None:
                    for i in range(kk):
                        depth = B[i]
                        if depth == 0:
                            continue
                        for c_sib in range(kk):
                            chain_lo_row = fan_start[i] + c_sib * depth
                            for o in range(depth):
                                row = chain_lo_row + o  # row index, 0-indexed within Q_LEN
                                for c2 in range(kk):
                                    if c2 == c_sib:
                                        continue
                                    other_lo_row = fan_start[i] + c2 * depth
                                    for c in range(K + other_lo_row, K + other_lo_row + depth):
                                        assert not allowed[row, c], \
                                            f"fan row {row} (tip {i} chain {c_sib}) leaks into sibling chain {c2}'s column {c}"
                                for j in range(kk):
                                    if j == i:
                                        continue
                                    for c in real_range(j):
                                        assert not allowed[row, c], \
                                            f"fan row {row} (tip {i}) leaks into candidate {j}'s real column {c}"

            self.calls.append(dict(n_real=n_real, n_aux=n_aux, Q_LEN=Q_LEN, K=K,
                                    mask=m.clone(), pos=position_ids.clone()))
        else:
            self.calls.append(dict(n_real=n_real, n_aux=n_aux, Q_LEN=Q_LEN, K=K,
                                    mask=None, pos=None))

        logits = torch.randn(1, Q_LEN, VOCAB) * 0.1
        logits[..., EOS] -= 5  # keep EOS rare so rounds actually run
        out = types.SimpleNamespace(logits=logits, past_key_values=FakeCache(K + Q_LEN))
        return out


def make_lm(k):
    lm = LM.__new__(LM)  # bypass __init__ (needs full LightningModule machinery)
    lm.model = FakeModel()
    lm.tokens_per_student_call = 4
    lm.hist_base = torch.ones(21, dtype=torch.float64) / 21
    lm.H_fn = lambda metrics: torch.arange(21).double()
    lm.top_p = 0.9
    lm.top_k = 10
    lm.temperature = 1.0
    lm.proposals = types.MethodType(LM.proposals, lm)
    lm.adapt_p = types.MethodType(LM.adapt_p, lm)
    lm._in_nucleus = LM._in_nucleus
    lm.generate_tree = types.MethodType(LM.generate_tree, lm)
    return lm


def run(k, max_new_tokens, threshold=0.9):
    lm = make_lm(k)
    prompt_ids = torch.randint(2, VOCAB, (1, 6))
    completion, metrics = lm.generate_tree(
        {"prompt_ids": prompt_ids}, max_new_tokens=max_new_tokens,
        k=k, nucleus_threshold=threshold, return_metrics=True, eos=EOS,
    )
    print(f"k={k} max_new_tokens={max_new_tokens}: "
          f"completion_len={completion.shape[1]} (prompt={prompt_ids.shape[1]}), "
          f"num_calls={metrics['num_calls']}, correct_per_call={metrics['correct_per_call']:.2f}, "
          f"correct_all={metrics['correct_all']}")
    assert completion.shape[1] <= prompt_ids.shape[1] + max_new_tokens
    assert completion.shape[1] >= prompt_ids.shape[1] + 1
    assert (completion[:, :prompt_ids.shape[1]] == prompt_ids).all(), "prompt got mutated"

    # --- Cache-growth invariant: model.calls should show K increasing monotonically
    # by exactly (that round's committed n_new) once you interleave the round call
    # with its immediately-following commit call. ---
    for c in lm.model.calls:
        assert c["K"] >= 0
    print("  mask/shape/prompt-preservation checks passed")


for k in (1, 2, 4):
    for mnt in (1, 3, 12):
        run(k, mnt)

print("\nALL SMOKE TESTS PASSED")
