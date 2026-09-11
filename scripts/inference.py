"""
Batch inference over a chat dataset with swappable inference algorithms.

Usage
-----
    python scripts/inference.py <experiment_dir> [options]

Example
-------
    python scripts/inference.py \\
        /extra/ucibdl1/fdraxler/light-speed-llm-diffusion/saved-ar-ptp-04-25 \\
        --algorithm ptp \\
        --max-tokens-per-proposal 20 \\
        --max-new-tokens 256 \\
        --n-examples 500 \\
        --seed 42

Adding a custom inference algorithm
------------------------------------
Implement the ``InferenceAlgorithm`` protocol and register it:

    class MyAlgorithm:
        name = "my_algo"

        def __init__(self, lit_model, device, autocast_dtype):
            self.lit_model = lit_model
            self.device = device
            self.autocast_dtype = autocast_dtype

        def generate(self, prompt_ids, max_new_tokens):
            # prompt_ids: LongTensor [1, T]
            # Returns: (completion_ids [1, T+N], metrics dict)
            ...

    ALGORITHMS["my_algo"] = MyAlgorithm
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import sys
import termios
import time
import tty
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inference algorithm protocol + built-in implementations
# ---------------------------------------------------------------------------

class InferenceAlgorithm(Protocol):
    name: str

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        prompt_ids : LongTensor [1, T]
        max_new_tokens : int

        Returns
        -------
        completion_ids : LongTensor [1, T+N]   (includes prompt prefix)
        metrics        : dict with at least {"n_generated_tokens": int}
        """
        ...


class SeqInference:
    """Base class for generate_seq-based inference. Subclasses override needs_teacher,
    accepted_tokens, and extra_metrics to define their acceptance logic."""

    needs_teacher = None      # override as instance method to gate teacher calls
    accepted_tokens = None    # override as instance method to customize acceptance
    student_forward = None    # override to replace the student inference_forward call
    teacher_forward = None    # override to replace the teacher inference_forward call
    correct_first_token = True  # override (e.g. False) to change generate_seq's first-token handling
    shared_kv_cache = False   # override (e.g. True) to reuse one KV cache for student+teacher

    def __init__(self, lit_model, device, autocast_dtype, *, raw: bool = False, **kwargs):
        self.lit_model = lit_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.raw = raw

    def extra_metrics(self) -> dict:
        return {}

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        raw_ctx = _raw_logits_mode(self.lit_model) if self.raw else contextlib.nullcontext()
        t0 = time.perf_counter()
        with raw_ctx, autocast_ctx:
            completion, ptp_metrics = self.lit_model.generate_seq(
                prompt_ids, max_new_tokens=max_new_tokens,
                needs_teacher=self.needs_teacher,
                accepted_tokens=self.accepted_tokens,
                student_forward=self.student_forward,
                teacher_forward=self.teacher_forward,
                correct_first_token=self.correct_first_token,
                shared_kv_cache=self.shared_kv_cache,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = completion.shape[1] - prompt_ids.shape[1]
        return completion, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": ptp_metrics.get("num_calls", 0),
            "correct_per_call": float(ptp_metrics.get("correct_per_call", float("nan"))),
            "tokens_per_student_call": self.lit_model.tokens_per_student_call,
            **self.extra_metrics(),
        }


class SeqTreeInference(SeqInference):
    """
    Base class for tree-structured PTP variants (see lit.py's generate_seq_tree):
    each round proposes a masked tree of candidate tokens — built from a
    (node_id, parent_id) list, correct_tokens has the same shape as the proposed tree.
    """

    tree = None               # None -> generate_seq_tree's standard flat-chain fallback; override per subclass

    @staticmethod
    def _depths(parent_list: list[int | None]) -> list[int]:
        depth: list[int] = []
        for p in parent_list:
            depth.append(1 if p is None else depth[p] + 1)
        return depth

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd, parent_list):
        n_nodes = student_tokens.shape[1]
        depth_list = self._depths(parent_list)
        match = (student_tokens[0] == correct_tokens[0]).tolist()

        children: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
        roots: list[int] = []
        for i, p in enumerate(parent_list):
            (roots if p is None else children[p]).append(i)

        def best_path(node: int) -> list[int]:
            if not match[node]:
                return []
            best_child: list[int] = max(
                (best_path(c) for c in children[node]), key=len, default=[],
            )
            return [node] + best_child

        path = max((best_path(r) for r in roots), key=len, default=[])
        if not path:
            # Guaranteed to match via correct_first_token; only reached if the
            # teacher was skipped for this round.
            return correct_tokens[:, roots[0]:roots[0] + 1]

        last = path[-1]
        # Only use a child as the correction if it's genuinely the next depth
        # after `last` — e.g. SeqPTPTreeInference's branch nodes can skip a
        # depth, in which case there's simply no bonus token this round.
        correction = next(
            (c for c in children[last] if depth_list[c] == depth_list[last] + 1), None,
        )
        if correction is not None:
            return torch.cat([student_tokens[:, path], correct_tokens[:, correction:correction + 1]], dim=1)
        return student_tokens[:, path]

    def extra_metrics(self) -> dict:
        return {"n_nodes": self.n_nodes}

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        raw_ctx = _raw_logits_mode(self.lit_model) if self.raw else contextlib.nullcontext()
        t0 = time.perf_counter()
        with raw_ctx, autocast_ctx:
            completion, ptp_metrics = self.lit_model.generate_seq_tree(
                prompt_ids, max_new_tokens=max_new_tokens,
                tree=self.tree,
                needs_teacher=self.needs_teacher,
                accepted_tokens=self.accepted_tokens,
                student_forward=self.student_forward,
                teacher_forward=self.teacher_forward,
                shared_kv_cache=self.shared_kv_cache,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = completion.shape[1] - prompt_ids.shape[1]
        return completion, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": ptp_metrics.get("num_calls", 0),
            "correct_per_call": float(ptp_metrics.get("correct_per_call", float("nan"))),
            "tokens_per_student_call": self.lit_model.tokens_per_student_call,
            **self.extra_metrics(),
        }


class SequentialPTPInference(SeqInference):
    """PTP with a separate teacher model: student proposes via aux, teacher verifies sequentially."""

    name = "seq-ptp"

    def __init__(self, lit_model, device, autocast_dtype, teacher_model=None, **kwargs):
        super().__init__(lit_model, device, autocast_dtype)


class PTPInference:
    """Default PTP speculative decoding using the trained student (LoRA) model."""

    name = "ptp"

    # p for the "geom" partial_mode, fitted over all questions; see scratch/correct.md.
    GEOM_P = 0.698

    # Population Beta(a,b) prior over p for the "beta" partial_mode: the hierarchical
    # Beta-Geometric fit jointly over all questions for the K=1 (1 + Geometric(p)) model.
    # See scratch/joint_per_question_results.json ("seqptp" entry: a, b).
    BETA_PRIOR_A = 12.575047374037903
    BETA_PRIOR_B = 6.030620173416624

    def __init__(self, lit_model, device, autocast_dtype, *, max_tokens_per_proposal: int, total_token_budget: int,
                 partial_mode: str = "count", phead_checkpoint: str | None = None,
                 chead_checkpoint: str | None = None):
        self.lit_model = lit_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.max_tokens_per_proposal = max_tokens_per_proposal
        self.total_token_budget = total_token_budget
        self.partial_mode = partial_mode
        self._example_idx = -1  # incremented at the start of each generate() call
        if partial_mode == "phead":
            assert phead_checkpoint is not None, "--phead-checkpoint is required for --partial-mode phead"
            from ptp.p_head import PHead
            sidecar = torch.load(phead_checkpoint, map_location="cpu", weights_only=False)
            phead = PHead(lit_model.model.model.config.hidden_size)
            phead.load_state_dict(sidecar["p_head_state_dict"])
            phead = phead.to(device).eval()
            lit_model.model.output_hidden_states = True

            def phead_H_fn(metrics):
                with torch.no_grad():
                    p_pred = phead(lit_model._last_context_hidden.float()).item()
                return self.compute_H("phead", metrics={"p_pred": p_pred})

            lit_model.H_fn = phead_H_fn
        elif partial_mode == "chead":
            assert chead_checkpoint is not None, "--chead-checkpoint is required for --partial-mode chead"
            from ptp.p_head import CHead
            sidecar = torch.load(chead_checkpoint, map_location="cpu", weights_only=False)
            chead = CHead(lit_model.model.model.config.hidden_size)
            chead.load_state_dict(sidecar["c_head_state_dict"])
            chead = chead.to(device).eval()
            lit_model.model.output_hidden_states = True

            def chead_H_fn(metrics):
                with torch.no_grad():
                    logits = chead(lit_model._last_context_hidden.float())
                    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                return self.compute_H("chead", metrics={"probs": probs})

            lit_model.H_fn = chead_H_fn
        elif partial_mode == "beta_oracle":
            # Precomputed per-question free MLE p_hat_i from the earlier per-question
            # hierarchical analysis (K=1/seqptp), indexed by example order (matches how
            # iter_spec_bench_pairs is iterated, same as when that data was collected).
            oracle_path = Path(__file__).resolve().parent.parent / "scratch" / "joint_per_question_results.json"
            with open(oracle_path) as f:
                self._oracle_p_hat = json.load(f)["seqptp"]["p_hat_i"]

            def oracle_H_fn(metrics):
                p = self._oracle_p_hat[self._example_idx % len(self._oracle_p_hat)]
                return self.compute_H("beta_oracle", metrics={"p_pred": p})

            lit_model.H_fn = oracle_H_fn
        else:
            lit_model.H_fn = lambda metrics: self.compute_H(partial_mode, lit_model.hist_base, metrics)

    @classmethod
    def compute_H(cls, partial_mode: str, hist_base: torch.Tensor | None = None,
                   metrics: dict | None = None) -> torch.Tensor:
        """
        Reward matrix H: estimated # correct tokens in the next proposal step
        given k proposed tokens (k = 0..20), used by ParallelSamplingLightningModule.proposals().
          - "count": H(k) = k.
          - "hist":  H(k) = E[min(G, k)] under the empirical histogram hist_base.
          - "geom":  H(k) = E[min(G, k)] under the shifted-geometric model
                     G = 1 + Geometric(GEOM_P), clamped to the same 0..20 support.
          - "beta":  Like "geom", but p is the posterior mean of a Beta(BETA_PRIOR_A,
                     BETA_PRIOR_B) prior updated with the #correct-per-call samples seen
                     so far this generation (metrics['correct']): each call contributes
                     one Beta-Geometric trial (correct_i successes, 1 failure).
          - "phead":  p is predicted directly from context by a fine-tuned P-head
                      (see src/ptp/p_head.py); metrics['p_pred'] must be supplied by
                      the caller (PTPInference.__init__'s phead_H_fn).
          - "chead":  Like "hist", but the 21-class categorical distribution over #correct
                      is predicted directly from context by a fine-tuned C-head (non-parametric,
                      no Geometric shape assumption; see src/ptp/p_head.py); metrics['probs']
                      must be supplied by the caller (PTPInference.__init__'s chead_H_fn).
          - "beta_oracle": p is the precomputed per-question free MLE from the earlier
                     per-question hierarchical analysis (scratch/joint_per_question_results.json,
                     "seqptp".p_hat_i), indexed by example order — an oracle upper bound for
                     "beta"'s online per-question estimate, since it uses the whole question's
                     data instead of updating from partial observations.
        """
        arange_21 = torch.arange(21)

        def geom_H(p: float) -> torch.Tensor:
            """H(k) = E[min(G, k)] for G = 1 + Geometric(p), clamped to the 0..20 support."""
            pmf = torch.zeros(21, dtype=torch.float64)
            g = arange_21[1:20].double()
            pmf[1:20] = p * (1 - p) ** (g - 1)
            pmf[20] = 1 - pmf[:20].sum()
            return torch.cumsum(pmf * arange_21, dim=-1)

        if partial_mode == "count":
            return arange_21.double()
        elif partial_mode == "hist":
            assert hist_base is not None, "hist_base must be provided for partial_mode='hist'"
            return torch.cumsum(hist_base * arange_21, dim=-1)
        elif partial_mode == "geom":
            return geom_H(cls.GEOM_P)
        elif partial_mode == "beta":
            correct = metrics["correct"] if metrics is not None else []
            a_post = cls.BETA_PRIOR_A + sum(correct)
            b_post = cls.BETA_PRIOR_B + len(correct)
            return geom_H(a_post / (a_post + b_post))
        elif partial_mode == "phead":
            return geom_H(metrics["p_pred"])
        elif partial_mode == "chead":
            return torch.cumsum(metrics["probs"] * arange_21, dim=-1)
        elif partial_mode == "beta_oracle":
            return geom_H(metrics["p_pred"])
        else:
            raise ValueError(f"Unknown partial_mode: {partial_mode!r}")

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        self._example_idx += 1
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        t0 = time.perf_counter()
        with autocast_ctx:
            completion, ptp_metrics = self.lit_model.generate(
                {"prompt_ids": prompt_ids},
                max_new_tokens=max_new_tokens,
                return_metrics=True,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = completion.shape[1] - prompt_ids.shape[1]
        metrics = {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": ptp_metrics.get("num_calls", 0),
            "correct_per_call": float(ptp_metrics.get("correct_per_call", float("nan"))),
            "tokens_per_student_call": self.max_tokens_per_proposal,
        }
        return completion, metrics


class ARInference:
    """Pure autoregressive teacher (base model, no LoRA adapters)."""

    name = "ar"

    def __init__(self, lit_model, device, autocast_dtype, *, temperature: float | None,
                 top_k: int, top_p: float, teacher_model=None):
        self.lit_model = lit_model
        self.teacher_model = teacher_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=self.temperature is not None,
            use_cache=True,
        )
        if self.temperature is not None:
            gen_kwargs["temperature"] = self.temperature
        if self.top_k:
            gen_kwargs["top_k"] = self.top_k
        if self.top_p < 1.0:
            gen_kwargs["top_p"] = self.top_p

        t0 = time.perf_counter()
        if self.teacher_model is not None:
            with torch.no_grad(), autocast_ctx:
                completion = self.teacher_model.generate(prompt_ids, **gen_kwargs).sequences
        else:
            with torch.no_grad(), autocast_ctx, _teacher_mode(self.lit_model):
                completion = self.lit_model.model.generate(prompt_ids, **gen_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = completion.shape[1] - prompt_ids.shape[1]
        metrics = {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": n_gen,         # one teacher call per token
            "correct_per_call": 1.0,    # trivially, by definition
        }
        return completion, metrics


def _teacher_sample_token(lit_model, logits_1_1_V: torch.Tensor, z: float) -> torch.Tensor:
    """
    Sample one token from the teacher distribution using inverse-CDF, matching
    the exact sampling procedure PTP uses for teacher verification.

    Parameters
    ----------
    logits_1_1_V : [1, 1, V] — raw logits at one position
    z            : scalar float in [0, 1] — pre-drawn uniform random number

    Returns [1, 1] LongTensor.
    """
    if lit_model.temperature is not None and lit_model.temperature != 1.0:
        logits_1_1_V = logits_1_1_V / lit_model.temperature
    p = torch.softmax(logits_1_1_V, dim=-1)            # [1, 1, V]
    tgt_p, tgt_indices = lit_model.adapt_p(p)           # [1, 1, top_k] each
    cdf = tgt_p.cumsum(-1)
    cdf[..., -1] = 1.0
    bin_idx = (cdf > z).max(-1).indices                 # [1, 1]
    return tgt_indices.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)  # [1, 1]


class DeterministicARInference:
    """
    Autoregressive teacher sampling with explicit per-example random seeds.

    At each token position t, uses z[t] (drawn from torch.rand) for inverse-CDF
    sampling, matching exactly what PTP's teacher verification does.  When both
    DeterministicARInference and DeterministicPTPInference are initialised with
    the same base_seed, they produce identical output on the same prompts.
    """

    name = "det_ar"

    def __init__(self, lit_model, device, autocast_dtype, *, base_seed: int):
        self.lit_model = lit_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.base_seed = base_seed
        self._call_count = 0

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        from transformers.cache_utils import DynamicCache

        # Per-example deterministic seed — same scheme as DeterministicPTPInference
        seed = self.base_seed * 100_003 + self._call_count
        self._call_count += 1

        torch.manual_seed(seed)
        # Draw z values upfront, mirroring PTP's z_rnd_all = torch.rand([1, max_new_tokens + ...])
        # Positions 0..max_new_tokens-1 are identical to what PTP draws.
        z = torch.rand(max_new_tokens, device=self.device)

        eos = getattr(self.lit_model.model.tokenizer, "eos_token_id", None)
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )

        completion = prompt_ids.clone()
        kv_cache = DynamicCache()
        t0 = time.perf_counter()

        with torch.inference_mode(), autocast_ctx, _teacher_mode(self.lit_model):
            for t in range(max_new_tokens):
                out = self.lit_model.model.inference_forward(
                    input_ids=completion[:, kv_cache.get_seq_length():],
                    past_key_values=kv_cache,
                    use_cache=True,
                )
                kv_cache = out.past_key_values
                next_token = _teacher_sample_token(self.lit_model, out.logits[:, -1:], z[t].item())
                completion = torch.cat([completion, next_token], dim=1)
                if next_token.item() == eos:
                    break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = completion.shape[1] - prompt_ids.shape[1]
        return completion, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": n_gen,
            "correct_per_call": 1.0,
            "example_seed": seed,
        }


class AcceptFirstKInference(SeqInference):
    """Emits k tokens per step with no teacher verification — always accepts."""

    name = "first-k"

    def __init__(self, lit_model, device, autocast_dtype, *, k: float = 2, **kwargs):
        super().__init__(lit_model, device, autocast_dtype)
        self.k = k

    def needs_teacher(self, student_tokens, student_logits) -> bool:
        return False

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        k_lo = int(math.floor(self.k))
        k_hi = int(math.ceil(self.k))
        frac = self.k - k_lo
        k = k_hi if z_rnd[0] < frac else k_lo
        return student_tokens[:, :k]

    def extra_metrics(self) -> dict:
        return {"accept_first_k": self.k}


class SeqPTPTAcceptFirstKInference(SeqInference):
    """PTP but emits at least k tokens per step."""

    name = "seq-ptp-first-k"

    def __init__(self, lit_model, device, autocast_dtype, *, k: float = 2, **kwargs):
        super().__init__(lit_model, device, autocast_dtype)
        self.k = k

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        k_lo = int(math.floor(self.k))
        k_hi = int(math.ceil(self.k))
        frac = self.k - k_lo
        k = k_hi if z_rnd[0] < frac else k_lo
        matches = student_tokens == correct_tokens[:, :-1]
        first_wrong = max(k - 1, matches.float().argmin() if not matches.all() else student_tokens.shape[1])
        return torch.cat([student_tokens[:, :first_wrong], correct_tokens[:, first_wrong:first_wrong+1]], dim=1)

    def extra_metrics(self) -> dict:
        return {"accept_first_k": self.k}


class SeqThresholdInference(SeqInference):
    """
    Threshold acceptance via generate_seq's accepted_tokens callback.

    At each step the student proposes n tokens; a proposed token is accepted if
    the teacher assigns probability >= threshold to it.  At the first rejection
    (or if all pass), the teacher's distribution is sampled and that token is appended.
    If the sampled token is already the proposed token, keep sampling more positions.
    """

    name = "seq-thresh-p"

    def __init__(self, lit_model, device, autocast_dtype, *, threshold: float = 0.7, fast: bool = False, raw: bool = False, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.threshold = threshold
        self.fast = fast

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        tgt_probs = torch.softmax(tgt_logits.float(), dim=-1)
        tgt_p_proposed = tgt_probs[:, :-1].gather(2, student_tokens.unsqueeze(-1)).squeeze(-1)[0]
        max_tokens = tgt_probs.argmax(dim=-1)
        if not self.fast:
            # fallback to argmax only if model can't find a token
            accept_mask = tgt_p_proposed >= self.threshold
            if not accept_mask[0]:
                max_idx = max_tokens[:, 0]
                if student_tokens[:, 0] == max_idx:
                    accept_mask[0] = True
                else:
                    student_tokens[:, 0] = max_idx
            first_reject = n_prop if accept_mask.all() else int(accept_mask.float().argmin().item())
            return student_tokens[:, :max(1, first_reject)]
        else:
            accept_mask = (tgt_p_proposed >= self.threshold) | (student_tokens[0] == max_tokens[0, :-1])
            first_reject = n_prop if accept_mask.all() else int(accept_mask.float().argmin().item())
            return torch.cat([student_tokens[:, :first_reject], max_tokens[:, first_reject:first_reject + 1]], dim=1)

    def extra_metrics(self) -> dict:
        return {"threshold": self.threshold, "fast": self.fast}


class SeqPTPThresholdInference(SeqInference):
    """
    Threshold acceptance via generate_seq's accepted_tokens callback.

    At each step the student proposes n tokens; a proposed token is accepted if
    the teacher assigns probability >= threshold to it.  At the first rejection
    (or if all pass), the teacher's distribution is sampled and that token is appended.
    If the sampled token is already the proposed token, keep sampling more positions.
    """

    name = "seq-ptp-thresh-p"

    def __init__(self, lit_model, device, autocast_dtype, *, threshold: float = 0.7, raw: bool = False, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.threshold = threshold

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        tgt_probs = torch.softmax(tgt_logits[:, :-1].float(), dim=-1)
        tgt_p_proposed = tgt_probs.gather(2, student_tokens.unsqueeze(-1)).squeeze(-1)[0]
        max_tokens = tgt_probs.argmax(dim=-1)
        # passes teacher threshold (or is highest wrt teacher); or is accepted in PTP
        accept_mask = (tgt_p_proposed >= self.threshold) | (student_tokens == max_tokens) | (student_tokens == correct_tokens[:, :-1])
        first_reject = n_prop if accept_mask.all() else int(accept_mask.float().argmin().item())
        return torch.cat([student_tokens[:, :first_reject], correct_tokens[:, first_reject:first_reject + 1]], dim=1)

    def extra_metrics(self) -> dict:
        return {"threshold": self.threshold}


class EntropyThresholdInference(SeqInference):
    """
    Entropy-adaptive teacher-probability acceptance ("typical acceptance").

    A proposed token x is accepted iff the teacher assigns it

        p(x) > min(epsilon, delta * exp(-H(p)))

    where H(p) is the entropy of the teacher's (already temperature/top-p
    adapted) distribution at that position.  The bound is therefore loose where
    the teacher is uncertain -- many continuations are equally fine -- and tight
    where it is confident that one token is right.  A fixed threshold cannot
    express that; see `seq-thresh-p` for the fixed-threshold sibling.

    Defaults follow Medusa's typical acceptance (epsilon=0.09, delta=0.3).
    Progress is guaranteed by committing the teacher's argmax when nothing at the
    first position clears the bound.

    Note: implemented on the generate_seq driver rather than a fused single-call
    path, because the fused `thresh-p` path is unimplemented.
    """

    name = "entr-p"

    def __init__(self, lit_model, device, autocast_dtype, *, threshold: float = 0.09,
                 delta: float = 0.3, raw: bool = False, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.threshold = threshold
        self.delta = delta

    def _entropy_accept(self, student_tokens, tgt_logits):
        """(accept_mask [n_prop], teacher argmax [1, n_prop]) under the adaptive bound."""
        tgt_probs = torch.softmax(tgt_logits[:, :-1].float(), dim=-1)
        p_proposed = tgt_probs.gather(2, student_tokens.unsqueeze(-1)).squeeze(-1)[0]
        # Filtered tokens have probability exactly 0, and 0 * log(eps) == 0, so
        # the truncated distribution contributes no NaN here.
        entropy = -(tgt_probs * torch.log(tgt_probs + 1e-10)).sum(-1)[0]
        bound = torch.minimum(
            torch.full_like(entropy, self.threshold),
            self.delta * torch.exp(-entropy),
        )
        return p_proposed > bound, tgt_probs.argmax(dim=-1)

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        accept_mask, max_tokens = self._entropy_accept(student_tokens, tgt_logits)
        if not accept_mask[0]:
            # Nothing acceptable at the first position: fall back to the teacher's
            # argmax so the step still commits a token.
            if student_tokens[:, 0] == max_tokens[:, 0]:
                accept_mask[0] = True
            else:
                student_tokens[:, 0] = max_tokens[:, 0]
        first_reject = n_prop if accept_mask.all() else int(accept_mask.float().argmin().item())
        return student_tokens[:, :max(1, first_reject)]

    def extra_metrics(self) -> dict:
        return {"threshold": self.threshold, "delta": self.delta}


class SeqPTPEntropyThresholdInference(EntropyThresholdInference):
    """
    Entropy-adaptive threshold OR exact PTP match, with a PTP correction token.

    Same adaptive bound as `entr-p`, but a token also passes if it is the
    teacher's argmax or if PTP would have accepted it (inverse-CDF match at the
    shared auxiliary), and the first rejected position is filled from the
    teacher's own CDF sample.  The entropy-adaptive analogue of
    `seq-ptp-thresh-p`.
    """

    name = "seq-ptp-entr-p"

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        accept_mask, max_tokens = self._entropy_accept(student_tokens, tgt_logits)
        accept_mask = (
            accept_mask
            | (student_tokens == max_tokens)[0]
            | (student_tokens == correct_tokens[:, :-1])[0]
        )
        first_reject = n_prop if accept_mask.all() else int(accept_mask.float().argmin().item())
        return torch.cat(
            [student_tokens[:, :first_reject], correct_tokens[:, first_reject:first_reject + 1]],
            dim=1,
        )


class ThresholdPTPInference:
    """
    PTP speculative decoding with threshold-based token acceptance.

    Each step: student proposes n tokens, teacher scores them.  A proposed token
    is accepted if the teacher assigns probability >= p to it.  At the first
    rejection (or if all n pass), the teacher's CDF-sampled correction token is
    appended and the step ends.  Ported from generate_old correcting_via='threshold'.
    """

    name = "thresh-p"

    def __init__(self, lit_model, device, autocast_dtype, *,
                 max_tokens_per_proposal: int, total_token_budget: int,
                 threshold: float = 0.7):
        self.lit_model = lit_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.max_tokens_per_proposal = max_tokens_per_proposal
        self.total_token_budget = total_token_budget
        self.threshold = threshold

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError


class SeqConfPInference(SeqInference):
    """
    Student-confidence acceptance via generate_seq.

    Accept proposed tokens while the student's own softmax probability >= threshold.
    Teacher is never called. At least 1 token accepted per step to guarantee progress.
    """

    name = "seq-conf-p"

    def __init__(self, lit_model, device, autocast_dtype, *, threshold: float = 0.7, **kwargs):
        super().__init__(lit_model, device, autocast_dtype)
        self.threshold = threshold

    def needs_teacher(self, student_tokens, student_logits) -> bool:
        return False

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        s_probs = torch.softmax(student_logits.float(), dim=-1)
        s_conf = s_probs.gather(2, student_tokens.unsqueeze(-1)).squeeze(-1)[0]
        accept_mask = s_conf >= self.threshold
        first_reject = n_prop if accept_mask.all() else int(accept_mask.float().argmin().item())
        return student_tokens[:, :max(1, first_reject)]

    def extra_metrics(self) -> dict:
        return {"threshold": self.threshold}


class SeqPTPConfPInference(SeqInference):
    """
    Student-confidence + PTP acceptance via generate_seq.

    Accept proposed tokens if student prob >= threshold or PTP would accept them.
    Teacher is called for PTP verification and the correction token.
    """

    name = "seq-ptp-conf-p"

    def __init__(self, lit_model, device, autocast_dtype, *, threshold: float = 0.7, **kwargs):
        super().__init__(lit_model, device, autocast_dtype)
        self.threshold = threshold

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        s_probs = torch.softmax(student_logits.float(), dim=-1)
        s_conf = s_probs.gather(2, student_tokens.unsqueeze(-1)).squeeze(-1)[0]
        accept_mask = (s_conf >= self.threshold) | (student_tokens == correct_tokens[:, :-1])[0]
        first_reject = n_prop if accept_mask.all() else int(accept_mask.float().argmin().item())
        return torch.cat([student_tokens[:, :first_reject], correct_tokens[:, first_reject:first_reject + 1]], dim=1)

    def extra_metrics(self) -> dict:
        return {"threshold": self.threshold}


class ConfPInference:
    """
    Student-only inference: accept proposed tokens while student softmax prob >= threshold.
    No teacher call, no correction token — just take the confident prefix and continue.
    At least 1 token is accepted per step to guarantee progress.
    """

    name = "conf-p"

    def __init__(self, lit_model, device, autocast_dtype, *,
                 max_tokens_per_proposal: int, total_token_budget: int,
                 threshold: float = 0.7):
        self.lit_model = lit_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.max_tokens_per_proposal = max_tokens_per_proposal
        self.threshold = threshold

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        import numpy as np
        lit_model = self.lit_model
        dev = prompt_ids.device
        eos = getattr(lit_model.model.tokenizer, "eos_token_id", None)

        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None else contextlib.nullcontext()
        )

        tokens = prompt_ids.clone()
        tokens_generated = 0
        num_calls = 0
        metrics_correct: list[int] = []

        z_rnd_all = torch.rand(1, max_new_tokens + self.max_tokens_per_proposal + 2, device=dev)

        t0 = time.perf_counter()

        while tokens_generated < max_new_tokens:
            n_prop = min(self.max_tokens_per_proposal, max_new_tokens - tokens_generated)
            T = tokens.shape[1]
            z_student = z_rnd_all[:, tokens_generated: tokens_generated + n_prop]

            with torch.inference_mode(), autocast_ctx:
                out = lit_model.model.inference_forward(input_ids=tokens, auxiliaries=z_student)
            s_logits = out.logits[:, T: T + n_prop].float()   # [1, n_prop, V]
            s_probs = torch.softmax(s_logits, dim=-1)
            proposed = s_logits.argmax(-1)                     # [1, n_prop]

            s_conf = s_probs[0].gather(dim=-1, index=proposed[0].unsqueeze(-1)).squeeze(-1)
            accept_mask = s_conf >= self.threshold
            first_reject = n_prop if accept_mask.all() else int(accept_mask.float().argmin().item())

            if first_reject > 0:
                accepted = proposed[:, :first_reject]
            else:
                # Student not confident in any token: ask teacher for correction
                with torch.inference_mode(), autocast_ctx, _teacher_mode(lit_model):
                    t_out = lit_model.model.inference_forward(input_ids=tokens)
                tgt_logits = t_out.logits[:, -1:].float()
                if lit_model.temperature is not None and lit_model.temperature != 1.0:
                    tgt_logits = tgt_logits / lit_model.temperature
                tgt_p, tgt_indices = lit_model.adapt_p(torch.softmax(tgt_logits, dim=-1))
                cdf = tgt_p.cumsum(-1); cdf[..., -1] = 1.0
                z_c = z_rnd_all[:, tokens_generated: tokens_generated + 1]
                bin_idx = (cdf > z_c.unsqueeze(-1)).max(-1).indices
                accepted = tgt_indices.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)  # [1, 1]
            tokens = torch.cat([tokens, accepted], dim=1)
            tokens_generated += accepted.shape[1]
            metrics_correct.append(accepted.shape[1])
            num_calls += 1

            if eos is not None and (accepted[0] == eos).any():
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = tokens.shape[1] - prompt_ids.shape[1]
        return tokens, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": num_calls,
            "correct_per_call": float(np.mean(metrics_correct)) if metrics_correct else float("nan"),
            "threshold": self.threshold,
        }


class SeqRatioInference(SeqInference):
    """
    Classical speculative decoding with p/q ratio acceptance criterion via generate_seq.

    Each draft token x at position i is accepted with probability
        min(1, (k_boost * p(x) + p_boost) / q(x))
    where p is the teacher distribution and q is the student distribution.

    greedy=False (default): q(x) = softmax(student_logits)[x_argmax] — actual student prob.
    greedy=True:            q(x) = 1 (deterministic), acceptance = min(1, k_boost * p(x) + p_boost).

    On rejection the teacher's correction token from generate_seq is used.

    k_boost>1 and p_boost>0 artificially inflates acceptance.
    """

    name = "seq-ratio"

    def __init__(self, lit_model, device, autocast_dtype, greedy: bool = False,
                 k_boost: float = 1.0, p_boost: float = 0.0, raw: bool = False, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.greedy = greedy
        self.k_boost = k_boost
        self.p_boost = p_boost

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        tgt_probs = torch.softmax(tgt_logits[:, :-1].float(), dim=-1)
        t_p_proposed = tgt_probs.gather(2, student_tokens.unsqueeze(-1)).squeeze(-1)[0]

        if self.greedy:
            alpha = (self.k_boost * t_p_proposed + self.p_boost).clamp(max=1.0)
        else:
            s_probs = torch.softmax(student_logits.float(), dim=-1)
            q_proposed = s_probs.gather(2, student_tokens.unsqueeze(-1)).squeeze(-1)[0]
            alpha = ((self.k_boost * t_p_proposed + self.p_boost) / q_proposed.clamp(min=1e-15)).clamp(max=1.0)

        a_rnd = torch.rand(n_prop, device=student_tokens.device)
        accept_mask = a_rnd < alpha
        first_reject = n_prop if accept_mask.all() else int(accept_mask.long().argmin().item())

        # Bonus (all accepted): sample from teacher at position n_prop.
        # Rejection: sample from residual normalize(max(0, p - q)).
        p_vec = torch.softmax(tgt_logits[0, first_reject].float(), dim=-1)  # [V]
        if first_reject < n_prop:
            if self.greedy:
                p_vec[student_tokens[0, first_reject]] = 0.0
            else:
                q_vec = torch.softmax(student_logits[0, first_reject].float(), dim=-1)
                p_vec = (p_vec - q_vec).clamp(min=0.0)
            denom = p_vec.sum()
            if denom > 1e-9:
                p_vec = p_vec / denom
            else:
                p_vec = torch.softmax(tgt_logits[0, first_reject].float(), dim=-1)
        cdf = p_vec.cumsum(dim=0)
        cdf[-1] = 1.0
        corr_idx = (cdf > z_rnd[first_reject].item()).nonzero(as_tuple=True)[0][0]
        return torch.cat([student_tokens[:, :first_reject], corr_idx.view(1, 1)], dim=1)

    def extra_metrics(self) -> dict:
        return {"greedy": self.greedy, "k_boost": self.k_boost, "p_boost": self.p_boost}


class SeqPTPRatioInference(SeqInference):
    """
    PTP acceptance extended with the ratio criterion.

    A proposed token is accepted if PTP would accept it (student matches correct_token)
    OR if the ratio test passes: a < min(1, k_boost * p(x) / q(x)).
    The correction/bonus token always comes from the PTP correct_tokens.
    """

    name = "seq-ptp-ratio"

    def __init__(self, lit_model, device, autocast_dtype, *,
                 greedy: bool = False, k_boost: float = 1.0, p_boost: float = 0.0, raw: bool = False, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.greedy = greedy
        self.k_boost = k_boost
        self.p_boost = p_boost

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        tgt_probs = torch.softmax(tgt_logits[:, :-1].float(), dim=-1)
        t_p_proposed = tgt_probs.gather(2, student_tokens.unsqueeze(-1)).squeeze(-1)[0]

        if self.greedy:
            alpha = (self.k_boost * t_p_proposed + self.p_boost).clamp(max=1.0)
        else:
            s_probs = torch.softmax(student_logits.float(), dim=-1)
            q_proposed = s_probs.gather(2, student_tokens.unsqueeze(-1)).squeeze(-1)[0]
            alpha = ((self.k_boost * t_p_proposed + self.p_boost) / q_proposed.clamp(min=1e-15)).clamp(max=1.0)

        a_rnd = torch.rand(n_prop, device=student_tokens.device)
        ptp_accept = (student_tokens == correct_tokens[:, :-1])[0]
        ratio_accept = a_rnd < alpha
        accept_mask = ptp_accept | ratio_accept
        first_reject = n_prop if accept_mask.all() else int(accept_mask.long().argmin().item())
        return torch.cat([student_tokens[:, :first_reject], correct_tokens[:, first_reject:first_reject + 1]], dim=1)

    def extra_metrics(self) -> dict:
        return {"greedy": self.greedy, "k_boost": self.k_boost, "p_boost": self.p_boost}


class RatioInference:
    """
    Speculative decoding using the single joint PTP forward pass (one call per loop).
    Copied from lit.py generate(); only the acceptance criterion (num_correct) is changed.
    """

    name = "ratio"

    def __init__(self, lit_model, device, autocast_dtype, *,
                 max_tokens_per_proposal: int, total_token_budget: int,
                 greedy: bool = False, k_boost: float = 1.0, p_add: float = 0.0):
        self.lit_model = lit_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.max_tokens_per_proposal = max_tokens_per_proposal
        self.greedy = greedy
        self.k_boost = k_boost
        self.p_add = p_add
        lit_model.tokens_per_student_call = max_tokens_per_proposal
        lit_model.total_token_budget = total_token_budget
        lit_model.H_fn = lambda metrics: PTPInference.compute_H("hist", lit_model.hist_base, metrics)

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        import numpy as np
        from transformers.cache_utils import DynamicCache

        lm = self.lit_model
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )

        t0 = time.perf_counter()
        initial_prompt_len = prompt_ids.shape[1]

        with torch.inference_mode(), autocast_ctx:
            assert prompt_ids.shape[0] == 1, "Batch size must be 1"
            assert lm.model.inference_mode, "Call enter_inference_mode() before generate()"
            dev = prompt_ids.device
            tpsc = lm.tokens_per_student_call
            ones_tpsc = torch.ones([1, tpsc], dtype=torch.long, device=dev)
            metrics = {'correct': [], 'N': [], 'Nrel': []}

            fixed_tokens = True
            pad_token = 13
            num_proposed_tokens = lm.total_token_budget or tpsc

            tokens_to_fill = max_new_tokens
            tokens_to_verify = max_new_tokens
            z_rnd_all = torch.rand([prompt_ids.shape[0], max_new_tokens + tpsc + num_proposed_tokens], device=dev, dtype=torch.float32)
            a_rnd_all = torch.rand([1, max_new_tokens + tpsc + num_proposed_tokens], device=dev, dtype=torch.float32)
            n_props = ((num_proposed_tokens // tpsc) * [tpsc] + [(num_proposed_tokens % tpsc)] + tpsc * [0])[:tpsc]
            prev_student_p_max = None   # [1, tpsc] carried from previous step for ratio test
            prev_student_p = None       # [1, tpsc, V] carried for residual (non-greedy only)

            eos = getattr(lm.model.tokenizer, 'eos_token_id', None)
            if eos is None:
                raise ValueError("eos token id must be provided via model.tokenizer.eos_token_id")

            kv_cache = DynamicCache()
            outputs = lm.model.inference_forward(
                input_ids=prompt_ids[:, kv_cache.get_seq_length():-1],
                auxiliaries=z_rnd_all[:, :num_proposed_tokens],
                past_key_values=kv_cache,
                use_cache=True,
            )
            kv_cache = outputs.past_key_values
            kv_cache.crop(prompt_ids.shape[1] - 1)
            prompt_ids = torch.cat([
                prompt_ids,
                pad_token * ones_tpsc[:, :tpsc - 1]
            ], dim=1)
            tokens_to_fill -= tpsc - 1

            while tokens_to_verify > 0:
                n_verify = tokens_to_verify - tokens_to_fill
                seq_len = prompt_ids.shape[1] + sum(n_props)
                pos = prompt_ids.shape[1] - kv_cache.get_seq_length()
                metrics['N'] += [pos + sum(n_props)]

                z_idx = max_new_tokens - tokens_to_verify
                z_rnd = torch.cat([z_rnd_all[:, z_idx + d:z_idx + d + n_prop] for d, n_prop in enumerate(n_props)], dim=1)
                input_ids = prompt_ids[:, kv_cache.get_seq_length():]
                K = kv_cache.get_seq_length()
                P = prompt_ids.shape[1]
                Q_LEN = seq_len - K
                input_position_ids = torch.arange(K, seq_len, device=dev)[None, :]
                midx = pos
                for d, n_prop in enumerate(n_props):
                    input_position_ids[:, midx: midx + n_prop] -= midx - pos + n_verify - d + 1
                    midx += n_prop
                midx = pos
                input_mask = torch.tril(torch.ones(Q_LEN, seq_len, device=dev), diagonal=K)
                for d, n_prop in enumerate(n_props):
                    input_mask[midx: midx + n_prop, P - n_verify + d: K + midx] = 0
                    midx += n_prop
                input_mask = (1 - input_mask[None, None].to(next(lm.parameters()).dtype)) * -1e15

                outputs = lm.model.inference_forward(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    position_ids=input_position_ids,
                    auxiliaries=z_rnd,
                    past_key_values=kv_cache,
                    use_cache=True,
                )

                kv_cache = outputs.past_key_values
                full_logits = outputs.logits
                student_logits = full_logits[:, pos:]
                if lm.temperature is not None and lm.temperature != 1.0:
                    full_logits[:, :pos] = full_logits[:, :pos] / lm.temperature
                full_p = torch.softmax(full_logits, dim=-1)
                student_p = full_p[:, pos:]
                tgt_p, tgt_indices = lm.adapt_p(full_p[:, :pos])
                student_predicted = student_logits.argmax(dim=-1)
                student_p_max = student_p.gather(-1, student_predicted[..., None])[..., 0]

                if n_verify > 0:
                    # --- Ratio acceptance (uses tgt_p / tgt_indices from adapt_p above) ---
                    predict_tokens = prompt_ids[..., -n_verify:]   # [1, n_verify]

                    # Teacher p for each verify token in the adapted top-k distribution
                    match_mask = (tgt_indices[0, :n_verify] == predict_tokens[0].unsqueeze(-1))  # [n_verify, top_k]
                    t_p_verify = (tgt_p[0, :n_verify] * match_mask.float()).sum(-1)              # [n_verify]

                    if self.greedy:
                        alpha = (self.k_boost * t_p_verify + self.p_add).clamp(max=1.0)
                    else:
                        q_verify = (prev_student_p_max[0, :n_verify]
                                    if prev_student_p_max is not None
                                    else torch.ones(n_verify, device=dev))
                        alpha = ((self.k_boost * t_p_verify + self.p_add) / q_verify.clamp(min=1e-9)).clamp(max=1.0)

                    a_idx = max_new_tokens - tokens_to_verify
                    a_accept = a_rnd_all[0, a_idx: a_idx + n_verify]
                    accept_mask = a_accept < alpha
                    num_correct = n_verify if accept_mask.all() else int(accept_mask.long().argmin().item())

                    # Correction / bonus — sample from tgt_p/tgt_indices at position num_correct
                    z_idx = max_new_tokens - tokens_to_verify
                    z_corr = z_rnd_all[0, z_idx + num_correct].item()
                    if num_correct == n_verify:
                        # Bonus: tgt_p/tgt_indices already cover position n_verify
                        p_vec = tgt_p[0, n_verify].clone()          # [top_k]
                        idx_vec = tgt_indices[0, n_verify]           # [top_k]
                    else:
                        # Rejection residual in top-k space
                        p_vec = tgt_p[0, num_correct].clone()        # [top_k]
                        idx_vec = tgt_indices[0, num_correct]        # [top_k]
                        if self.greedy:
                            p_vec[idx_vec == predict_tokens[0, num_correct]] = 0.0
                        else:
                            q_at_topk = (prev_student_p[0, num_correct].gather(-1, idx_vec)
                                         if prev_student_p is not None
                                         else torch.zeros_like(p_vec))
                            p_vec = (p_vec - q_at_topk).clamp(min=0.0)
                        denom = p_vec.sum()
                        if denom <= 1e-9:
                            p_vec = tgt_p[0, num_correct]
                    cdf = p_vec.cumsum(dim=0); cdf[-1] = 1.0
                    k_idx = (cdf > z_corr).nonzero(as_tuple=True)[0][0]
                    corr_token = idx_vec[k_idx].view(1, 1)

                    correct_tokens = torch.zeros(1, n_verify + 1, dtype=torch.long, device=dev)
                    correct_tokens[:, :num_correct] = predict_tokens[:, :num_correct]
                    correct_tokens[:, num_correct] = corr_token[:, 0]
                    num_new = num_correct + 1

                    tokens_to_verify -= num_correct
                    kv_cache.crop(prompt_ids.shape[-1] - (n_verify - num_correct))
                    prev_prop = sum(n_props[:num_correct])
                    ths_student_predicted = student_predicted[:, prev_prop: prev_prop + n_props[num_correct]]
                    ths_student_p = student_p_max[:, prev_prop: prev_prop + n_props[num_correct]]
                    metrics['Nrel'] += [num_correct / n_verify]
                    if fixed_tokens and ths_student_predicted.shape[1] < tpsc:
                        ths_student_predicted = torch.cat([ths_student_predicted, pad_token * ones_tpsc[:, :tpsc - ths_student_predicted.shape[1]]], dim=1)
                        ths_student_p = torch.cat([ths_student_p, ones_tpsc[:, :tpsc - ths_student_p.shape[1]].float()], dim=1)

                    # Carry student probs forward for the next step's ratio test
                    prev_student_p_max = ths_student_p
                    if not self.greedy:
                        sp_slice = student_p[:, prev_prop: prev_prop + n_props[num_correct]]
                        if fixed_tokens and sp_slice.shape[1] < tpsc:
                            pad_len = tpsc - sp_slice.shape[1]
                            sp_slice = torch.cat([
                                sp_slice,
                                torch.full([1, pad_len, sp_slice.shape[2]], 1.0 / sp_slice.shape[2],
                                           device=dev, dtype=sp_slice.dtype),
                            ], dim=1)
                        prev_student_p = sp_slice
                    if ths_student_predicted.shape[1] == 0:
                        match = False
                    else:
                        ths_student_predicted[0, 0] = correct_tokens[0, num_correct]
                        match = ths_student_predicted[0, 0] == correct_tokens[0, num_correct]
                    if not match:
                        prompt_ids = torch.cat([
                            prompt_ids[:, :prompt_ids.shape[1] - n_verify],
                            correct_tokens[:, :num_correct + 1],
                        ], dim=1)
                        tokens_to_verify -= 1
                        tokens_to_fill = tokens_to_verify
                        n_props = lm.proposals(lm.H_fn(metrics), n_verify=0, num_tokens=num_proposed_tokens, metrics=metrics)
                    else:
                        prompt_ids = torch.cat([
                            prompt_ids[:, :prompt_ids.shape[1] - (n_verify - num_correct)],
                            ths_student_predicted,
                        ], dim=1)
                        tokens_to_fill = tokens_to_verify - ths_student_predicted.shape[1]
                        tokens_to_verify -= 1
                        n_props = lm.proposals(lm.H_fn(metrics), student_p=ths_student_p[:, 1:], num_tokens=num_proposed_tokens, metrics=metrics)
                    metrics['correct'] += [num_new]
                    if eos in correct_tokens[:, :num_correct + 1]:
                        break
                else:
                    raise NotImplementedError

            try:
                if ths_student_predicted.shape[1] > 1:
                    prompt_ids = prompt_ids[:, :-ths_student_predicted.shape[1] + 1]
            except NameError:
                pass

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = prompt_ids.shape[1] - initial_prompt_len
        return prompt_ids, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": len(metrics['correct']),
            "correct_per_call": float(np.mean(metrics['correct'])) if metrics['correct'] else float("nan"),
            "greedy": self.greedy,
            "k_boost": self.k_boost,
        }


class RatioKInference(RatioInference):
    name = "ratio-k"

    def __init__(self, lit_model, device, autocast_dtype, *,
                 max_tokens_per_proposal: int, total_token_budget: int,
                 greedy: bool = False, k: float = 2.0):
        super().__init__(
            lit_model, device, autocast_dtype,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_token_budget,
            greedy=greedy,
            k_boost=k,
        )


class RatioPInference(RatioInference):
    """RatioInference with additive acceptance boost: min(1, (p(x) + p_boost) / q(x))."""

    name = "ratio-p"

    def __init__(self, lit_model, device, autocast_dtype, *,
                 max_tokens_per_proposal: int, total_token_budget: int,
                 greedy: bool = False, p: float = 0.1):
        super().__init__(
            lit_model, device, autocast_dtype,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_token_budget,
            greedy=greedy,
            p_add=p,
        )


class SeqInvPInference(SeqInference):
    """
    Inverse-CDF distance acceptance via generate_seq.

    For each draft token x_i with teacher distribution p, compute the CDF interval
    [u_lo, u_hi] = [CDF(x_i - 1), CDF(x_i)] that maps to x_i under inverse-CDF sampling.
    Accept if the signed distance of z_rnd[i] to this interval is <= threshold:

        signed_dist = max(u_lo - z_rnd[i], z_rnd[i] - u_hi)

    signed_dist < 0 means z_rnd is strictly inside the interval; 0 means on the boundary.
    threshold=0 recovers PTP (accept iff teacher would have sampled x_i with z_rnd[i]).
    threshold>0 widens the acceptance band by allowing z_rnd slightly outside.
    Correction always comes from correct_tokens (teacher's inverse-CDF sample with z_rnd).
    """

    name = "seq-inv-p"

    def __init__(self, lit_model, device, autocast_dtype, *, threshold: float = 0.0, raw: bool = False, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.threshold = threshold

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        tgt_probs = torch.softmax(tgt_logits[0, :n_prop], dim=-1)
        cdf = tgt_probs.cumsum(dim=-1)
        cdf[:, -1] = 1.0
        tokens = student_tokens[0].unsqueeze(-1)
        u_hi = cdf.gather(1, tokens).squeeze(-1)
        u_lo = u_hi - tgt_probs.gather(1, tokens).squeeze(-1)
        z = z_rnd[:n_prop]                   # match dtype
        signed_dist = torch.maximum(u_lo - z, z - u_hi)
        accept_mask = signed_dist <= self.threshold
        first_reject = n_prop if accept_mask.all() else int(accept_mask.long().argmin().item())
        return torch.cat([student_tokens[:, :first_reject], correct_tokens[:, first_reject:first_reject + 1]], dim=1)

    def extra_metrics(self) -> dict:
        return {"threshold": self.threshold}


class SeqTopKInference(SeqInference):
    """
    Top-k rank acceptance via generate_seq's accepted_tokens callback.

    A proposed token is accepted iff it is among the teacher's top-k logits
    (by rank, not probability) at that position.  On the first rejection
    (or if all n proposals pass) generation simply continues from the
    accepted prefix — no PTP-style inverse-CDF correction token from
    correct_tokens is appended, since that would tie this criterion to PTP's
    own sampling procedure. At least 1 token is accepted per step to
    guarantee progress.

    Defaults to raw teacher logits (no temperature/top-k/top-p adaptation)
    since rank-in-top-k is the acceptance criterion of interest, not the
    (possibly truncated/rescaled) sampling distribution.
    """

    name = "seq-top-k"

    def __init__(self, lit_model, device, autocast_dtype, *, k: int = 10, raw: bool = True, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.k = k

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        tgt_logits_prop = tgt_logits[:, :-1].float()  # [1, n_prop, V]
        k = min(self.k, tgt_logits_prop.shape[-1])
        topk_idx = tgt_logits_prop.topk(k, dim=-1).indices  # [1, n_prop, k]
        in_topk = (topk_idx == student_tokens.unsqueeze(-1)).any(-1)[0]  # [n_prop]
        first_reject = n_prop if in_topk.all() else int(in_topk.float().argmin().item())
        return student_tokens[:, :max(1, first_reject)]

    def extra_metrics(self) -> dict:
        return {"k": self.k}


class SeqPTPTopKInference(SeqInference):
    """
    PTP acceptance extended with the top-k rank criterion.

    A proposed token is accepted if PTP would accept it (student matches the
    teacher's actual correct_tokens) OR if it ranks within the teacher's
    top-k logits at that position.  The correction/bonus token always comes
    from correct_tokens, same as SeqPTPThresholdInference / SeqPTPRatioInference.

    Defaults to raw teacher logits (--raw), same reasoning as SeqTopKInference.
    """

    name = "seq-ptp-top-k"

    def __init__(self, lit_model, device, autocast_dtype, *, k: int = 10, raw: bool = True, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.k = k

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        tgt_logits_prop = tgt_logits[:, :-1].float()  # [1, n_prop, V]
        k = min(self.k, tgt_logits_prop.shape[-1])
        topk_idx = tgt_logits_prop.topk(k, dim=-1).indices  # [1, n_prop, k]
        in_topk = (topk_idx == student_tokens.unsqueeze(-1)).any(-1)[0]  # [n_prop]
        ptp_accept = (student_tokens == correct_tokens[:, :-1])[0]
        accept_mask = in_topk | ptp_accept
        first_reject = n_prop if accept_mask.all() else int(accept_mask.long().argmin().item())
        return torch.cat([student_tokens[:, :first_reject], correct_tokens[:, first_reject:first_reject + 1]], dim=1)

    def extra_metrics(self) -> dict:
        return {"k": self.k}


def _in_nucleus(tgt_probs: torch.Tensor, tokens: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Standard top-p/nucleus membership test, mirroring lit.py's adapt_p rule:
    a token is kept iff the cumulative probability mass of all *higher*-ranked
    tokens (i.e. excluding itself) is strictly less than ``threshold``.

    tgt_probs : [1, n, V] teacher probabilities at n positions
    tokens    : [1, n]    token ids to test membership for
    Returns   : [n] bool
    """
    sorted_probs, sorted_idx = tgt_probs.sort(dim=-1, descending=True)
    cum_before = sorted_probs.cumsum(dim=-1) - sorted_probs  # exclusive cumsum
    in_nucleus_sorted = cum_before < threshold
    match = sorted_idx == tokens.unsqueeze(-1)
    return (in_nucleus_sorted & match).any(-1)[0]


class SeqTopPInference(SeqInference):
    """
    Nucleus (top-p) mass acceptance via generate_seq's accepted_tokens callback.

    Mirrors standard top-p/nucleus filtering: sort the teacher's distribution
    descending and accept a proposed token iff the cumulative probability mass
    of all higher-ranked tokens is < threshold — i.e. the token falls inside
    the nucleus set defined by cumulative mass, unlike SeqTopKInference's
    fixed rank count.

    Like SeqTopKInference, no PTP-style inverse-CDF correction is appended on
    rejection — generation just continues from the accepted prefix (at least
    1 token per step to guarantee progress), keeping this independent of
    PTP's own sampling procedure.

    Defaults to raw teacher logits (no temperature/top-k/top-p adaptation),
    same reasoning as SeqTopKInference.
    """

    name = "seq-top-p"

    def __init__(self, lit_model, device, autocast_dtype, *, threshold: float = 0.9, raw: bool = True, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.threshold = threshold

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        tgt_probs = torch.softmax(tgt_logits[:, :-1].float(), dim=-1)  # [1, n_prop, V]
        in_nucleus = _in_nucleus(tgt_probs, student_tokens, self.threshold)  # [n_prop]
        first_reject = n_prop if in_nucleus.all() else int(in_nucleus.float().argmin().item())
        return student_tokens[:, :max(1, first_reject)]

    def extra_metrics(self) -> dict:
        return {"threshold": self.threshold}


class SeqPTPTopPInference(SeqInference):
    """
    PTP acceptance extended with the nucleus (top-p) mass criterion.

    A proposed token is accepted if PTP would accept it (student matches the
    teacher's actual correct_tokens) OR if it falls inside the teacher's
    top-p nucleus at that position.  The correction/bonus token always comes
    from correct_tokens, same as SeqPTPTopKInference.

    Defaults to raw teacher logits (--raw), same reasoning as SeqTopPInference.
    """

    name = "seq-ptp-top-p"

    def __init__(self, lit_model, device, autocast_dtype, *, threshold: float = 0.9, raw: bool = True, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, raw=raw)
        self.threshold = threshold

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        tgt_probs = torch.softmax(tgt_logits[:, :-1].float(), dim=-1)  # [1, n_prop, V]
        in_nucleus = _in_nucleus(tgt_probs, student_tokens, self.threshold)  # [n_prop]
        ptp_accept = (student_tokens == correct_tokens[:, :-1])[0]
        accept_mask = in_nucleus | ptp_accept
        first_reject = n_prop if accept_mask.all() else int(accept_mask.long().argmin().item())
        return torch.cat([student_tokens[:, :first_reject], correct_tokens[:, first_reject:first_reject + 1]], dim=1)

    def extra_metrics(self) -> dict:
        return {"threshold": self.threshold}


_CONTROL_CHAR_ESCAPES = {"\n": "</n>", "\t": "</t>", "\r": "</r>"}


def _escape_control_chars(text: str) -> str:
    """Render control characters visibly (e.g. '\\n' -> '</n>') so a decoded
    string never breaks across printed lines or looks blank."""
    for ch, esc in _CONTROL_CHAR_ESCAPES.items():
        text = text.replace(ch, esc)
    return text


def _read_single_key() -> str:
    """Read and echo one raw keypress from stdin without waiting for Enter (Unix only)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    print(ch)
    return ch


class PTPHumanInference(SeqInference):
    """
    Human-in-the-loop acceptance with a guaranteed PTP correction token.

    At every step, prints the context generated so far, then a numbered menu
    of growing candidate continuations built from the student's own proposed
    tokens — option i previews the decoded text if the first i proposals are
    accepted.  Only ``page_size`` options are shown at a time; the final
    entry is "show more", which reveals the next page of proposals (up to
    all n_prop of them).

    Whatever you pick, the teacher's PTP correction token (correct_tokens[i])
    is always appended right after it: choosing "1" yields 1 student token +
    1 teacher correction, "2" yields 2 + 1, etc.  In effect you're marking
    where you think the student's first mistake is, and PTP fixes exactly
    that token — so this always calls the teacher (SeqInference's default
    needs_teacher).

    Intended for interactive single-example exploration (--n-examples 1),
    not unattended batch runs.

    Answers are cached to disk (see set_cache_path) keyed by a hash of the
    prompt tokens, so an interrupted session can be resumed: rerunning with
    the same prompt/seed replays every previously recorded choice instantly
    (no reprompting) and only asks for new ones once the cached history runs
    out.
    """

    name = "ptp-human"

    def __init__(self, lit_model, device, autocast_dtype, *, page_size: int = 5, **kwargs):
        super().__init__(lit_model, device, autocast_dtype)
        self.page_size = page_size
        self._context_ids: torch.Tensor | None = None
        self.cache_path: Path | None = None
        self._cache: dict[str, list[int]] = {}
        self._example_key: str | None = None
        self._replay_idx: int = 0

    def set_cache_path(self, path: Path) -> None:
        """Point this instance at a JSON cache file, loading any existing answers."""
        self.cache_path = path
        self._cache = json.loads(path.read_text()) if path.exists() else {}

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache))

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        self._context_ids = prompt_ids.clone()
        self._example_key = hashlib.sha256(prompt_ids.cpu().numpy().tobytes()).hexdigest()[:16]
        self._cache.setdefault(self._example_key, [])
        self._replay_idx = 0
        return super().generate(prompt_ids, max_new_tokens)

    def _menu_choice(self, student_tokens: torch.Tensor, correct_tokens: torch.Tensor) -> int:
        """
        Return the token count to accept for this step: replayed from the
        cache if this step was already answered in a prior (interrupted) run,
        otherwise prompts interactively and records the new answer.
        """
        recorded = self._cache[self._example_key]
        if self._replay_idx < len(recorded):
            choice = recorded[self._replay_idx]
            self._replay_idx += 1
            return max(1, min(choice, student_tokens.shape[1]))

        choice = self._prompt_menu_choice(student_tokens, correct_tokens)
        recorded.append(choice)
        self._replay_idx += 1
        self._save_cache()
        return choice

    _WINDOW_SIZE = 9  # real options, labeled 0-8; label 9 is always reserved for "more"

    def _prompt_menu_choice(self, student_tokens: torch.Tensor, correct_tokens: torch.Tensor) -> int:
        """
        Print the context generated so far and a sliding-window menu of
        growing candidate continuations built from ``student_tokens`` (label
        L previews the decoded text if the first ``window_start + L``
        proposals are accepted, followed by the PTP correction token that
        would be appended after it).  Up to 9 real options are shown at a
        time (labels 0-8); label 9 is always "show more", which slides the
        window forward by 5 (dropping the first 5 options currently shown
        and revealing 5 new ones).  Blocks on a single keypress (no Enter
        needed) and returns the chosen token count (in [1, n_prop]).
        """
        tokenizer = self.lit_model.model.tokenizer
        n_prop = student_tokens.shape[1]

        # Plain-PTP acceptance count: how many student tokens match correct_tokens
        # before the first mismatch (n_prop if they all match).
        match = student_tokens[0] == correct_tokens[0, :-1]
        ptp_choice = n_prop if match.all() else int(match.float().argmin().item())

        context_ids = self._context_ids[0]
        # Decode candidates jointly with the context (then slice the context back
        # off the resulting string) so word-boundary spaces come out right —
        # decoding student_tokens/correct_tokens in isolation drops the leading
        # space a tokenizer would otherwise insert between context and continuation.
        context_raw = tokenizer.decode(context_ids, skip_special_tokens=False)

        # window_start = accept-count shown at label 0; slide by 5 until the
        # starred PTP choice is visible in the initial window.
        window_start = 1
        while ptp_choice > window_start + self._WINDOW_SIZE - 1 and window_start + self._WINDOW_SIZE - 1 < n_prop:
            window_start += 5

        while True:
            window_end = min(window_start + self._WINDOW_SIZE - 1, n_prop)
            has_more = window_end < n_prop

            print("\033[2J\033[H", end="")  # clear screen, cursor to top: only latest menu visible
            print("\n" + "=" * 80)
            print(_escape_control_chars(context_raw))
            print("-" * 80)
            for count in range(window_start, window_end + 1):
                label = count - window_start
                full_ids = torch.cat([context_ids, student_tokens[0, :count], correct_tokens[0, count:count + 1]], dim=0)
                full_raw = tokenizer.decode(full_ids, skip_special_tokens=False)
                continuation = _escape_control_chars(full_raw[len(context_raw):])
                star = "*" if count == ptp_choice else ""
                print(f"{star}{label}. {continuation}")
            if has_more:
                print("9. show more")

            print("Accept how many tokens? ", end="", flush=True)
            raw = _read_single_key()  # single keypress, no Enter needed
            if not raw.isdigit():
                print("Please press a digit key.")
                continue
            label = int(raw)

            if label == 9:
                if has_more:
                    window_start = min(window_start + 5, max(1, n_prop - self._WINDOW_SIZE + 1))
                    continue
                print("No more options.")
                continue
            count = window_start + label
            if window_start <= count <= window_end:
                return count

            print(f"Enter a digit between 0 and {window_end - window_start}{', or 9 for more' if has_more else ''}.")

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        choice = self._menu_choice(student_tokens, correct_tokens)
        chosen = torch.cat([student_tokens[:, :choice], correct_tokens[:, choice:choice + 1]], dim=1)
        self._context_ids = torch.cat([self._context_ids, chosen], dim=1)
        return chosen

    def extra_metrics(self) -> dict:
        return {}


_PTP_JUDGE_ANSWER_FORMAT = (
    "Respond in exactly two lines and nothing else. Line 1: one brief sentence "
    "naming the specific mistake (grammar, repetition, incoherence, etc.) that "
    "appears if you go past your chosen option — or stating that no such "
    "mistake exists if you pick the last one. Do not restate, list, or quote "
    "the candidates. Line 2, exactly:\nAnswer: <option number>."
)

_PTP_JUDGE_PROMPT_LONG = (
    "You are continuing the text below. Option 1 is the safe answer a strict "
    "verifier already accepts; each option after that accepts one additional "
    "token beyond the previous one. Is there a chance any of these extra tokens "
    "still form a good continuation? Prefer extending as far as plausible, but "
    "the result must still have correct grammar, spelling, and formatting. "
    "Context:\n{context}\n\n"
    "Candidates:\n{options}\n\n"
    f"Reminder: {_PTP_JUDGE_ANSWER_FORMAT}\n"
)

_PTP_JUDGE_PROMPT_MID = (
    "You are continuing the text below. Option 1 is the safe answer a strict "
    "verifier already accepts; each option after that accepts one additional "
    "token beyond the previous one. Is there a chance any of these extra tokens "
    "still form a good continuation? Balance length against quality: only pick "
    "a longer option if it remains grammatically correct, well formatted, and "
    "fits the context. "
    "Context:\n{context}\n\n"
    "Candidates:\n{options}\n\n"
    f"{_PTP_JUDGE_ANSWER_FORMAT}\n"
)

_PTP_JUDGE_PROMPT_SHORT = (
    "You are continuing the text below. Option 1 is the safe answer a strict "
    "verifier already accepts; each option after that accepts one additional "
    "token beyond the previous one. Is there a chance any of these extra tokens "
    "still form a good continuation? Only pick a longer option if you are "
    "confident it is still coherent, with correct grammar, spelling, and "
    "formatting — otherwise stick with option 1. "
    "Context:\n{context}\n\n"
    "Candidates:\n{options}\n\n"
    f"{_PTP_JUDGE_ANSWER_FORMAT}\n"
)

_PTP_JUDGE_PROMPTS = {
    "long": _PTP_JUDGE_PROMPT_LONG,
    "mid": _PTP_JUDGE_PROMPT_MID,
    "short": _PTP_JUDGE_PROMPT_SHORT,
}


class PTPJudgeInference(SeqInference):
    """
    LLM-as-a-judge acceptance with a guaranteed PTP correction token.

    Same mechanic as PTPHumanInference — whatever count "wins" gets the
    teacher's PTP correction token appended right after it (choosing i yields
    i student tokens + 1 correction) — except the choice is made by a
    separate referee LLM instead of a human.

    Unlike PTPHumanInference, the judge is never shown options shorter than
    plain PTP would already accept: option 1 is exactly PTP's own answer
    (the student/teacher matching prefix), and only up to 5 further options
    beyond that are offered, framed as "is there a chance any of these extra
    tokens are still a good continuation" with grammar/formatting emphasized.
    This keeps the judge from ever being more conservative than PTP while
    bounding how deep into the (increasingly unreliable) speculative batch it
    can reach.

    Reuses eval.py's llm-as-a-judge referee-loading path (same model, same
    greedy generation approach), just used to drive acceptance online instead
    of scoring completed runs after the fact.
    """

    name = "ptp-judge"

    def __init__(self, lit_model, device, autocast_dtype, *,
                 referee: str = "Qwen/Qwen2.5-7B-Instruct", judge_prompt: str = "mid",
                 seed: int = 42, **kwargs):
        super().__init__(lit_model, device, autocast_dtype)
        self._context_ids: torch.Tensor | None = None
        self._call_log: list[dict] = []
        self.prompt_template = _PTP_JUDGE_PROMPTS[judge_prompt]
        sys.path.insert(0, str(Path(__file__).parent))
        from eval import load_referee
        self.referee_model, self.referee_tokenizer = load_referee(referee)
        # Loading the referee consumes an unpredictable number of draws from the
        # global RNG (HF's from_pretrained/device placement touches torch's default
        # generator), which would otherwise desync the student's own z_rnd_all
        # sampling from --seed. Reseed so generation is identical to a run with no
        # referee loaded at all.
        torch.manual_seed(seed)

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        self._context_ids = prompt_ids.clone()
        self._call_log = []
        return super().generate(prompt_ids, max_new_tokens)

    def _ask_referee(self, messages: list[dict], max_new_tokens: int) -> str:
        """Run one chat-formatted generate() call against the referee, return its reply text."""
        ref_tok = self.referee_tokenizer
        if getattr(ref_tok, "chat_template", None):
            input_text = ref_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            input_text = "\n".join(m["content"] for m in messages)
        ids = ref_tok(input_text, return_tensors="pt").input_ids.to(self.referee_model.device)
        with torch.inference_mode():
            out = self.referee_model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=ref_tok.eos_token_id,
            )
        return ref_tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def _judge_choice(self, student_tokens: torch.Tensor, correct_tokens: torch.Tensor) -> int:
        tokenizer = self.lit_model.model.tokenizer
        n_prop = student_tokens.shape[1]
        context_ids = self._context_ids[0]
        # Decode candidates jointly with the context (then slice the context back off
        # the result) so word-boundary spaces come out right, same reasoning as
        # PTPHumanInference._prompt_menu_choice.
        context_raw = tokenizer.decode(context_ids, skip_special_tokens=False)

        # Anchor the window to what plain PTP would accept: option 1 is PTP's own
        # answer (the matching prefix against correct_tokens before the first
        # mismatch), and we only ask the judge about up to 5 tokens beyond that —
        # never anything shorter than PTP would already accept.
        match = student_tokens[0] == correct_tokens[0, :-1]
        ptp_choice = n_prop if match.all() else int(match.float().argmin().item())
        lo, hi = ptp_choice, min(ptp_choice + 5, n_prop)

        # Show the referee only a short trailing window of context: 45 "distant"
        # tokens (displayed with a leading "..." if there's more before them, and
        # a trailing "..." to visually bridge into the candidates), plus the last
        # 5 tokens kept separate as a "tail" — that tail gets prepended to every
        # candidate below so each option is readable on its own, without having
        # to cross-reference back to the Context section for local continuity.
        CTX_HEAD, CTX_TAIL = 45, 5
        window = context_ids[-(CTX_HEAD + CTX_TAIL):]
        truncated = context_ids.shape[0] > window.shape[0]
        head = window[:-CTX_TAIL] if window.shape[0] > CTX_TAIL else window[:0]
        # Decode head and the full window jointly (not head/tail in isolation) so
        # word-boundary spaces come out right, same reasoning as
        # PTPHumanInference._prompt_menu_choice.
        head_raw = tokenizer.decode(head, skip_special_tokens=False)
        window_raw = tokenizer.decode(window, skip_special_tokens=False)
        tail_raw = window_raw[len(head_raw):]
        context_display = ("..." if truncated else "") + head_raw + "..."

        options = []
        for i in range(lo, hi + 1):
            full_ids = torch.cat([context_ids, student_tokens[0, :i], correct_tokens[0, i:i + 1]], dim=0)
            full_raw = tokenizer.decode(full_ids, skip_special_tokens=False)
            continuation = _escape_control_chars(tail_raw) + _escape_control_chars(full_raw[len(context_raw):])
            options.append(continuation)

        prompt = self.prompt_template.format(
            context=_escape_control_chars(context_display),
            options="\n".join(f"{i}. {opt}" for i, opt in enumerate(options, start=1)),
        )
        messages = [{"role": "user", "content": prompt}]
        response = self._ask_referee(messages, max_new_tokens=200)

        m = re.search(r"answer\s*:\s*(\d+)", response, re.IGNORECASE)
        if m is not None:
            picked_raw = int(m.group(1))
            answer_type = "normal"
            reasoning = response[:m.start()].strip()
        else:
            # Didn't follow the format (rambled, forgot the Answer line, ...) — give it
            # one more chance, explicitly asking for only the number this time.
            followup_messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": "Answer with only the option number and nothing else."},
            ]
            followup = self._ask_referee(followup_messages, max_new_tokens=5)
            m2 = re.search(r"\d+", followup)
            reasoning = response  # the (malformed) first attempt, kept for inspection
            if m2 is not None:
                picked_raw = int(m2.group())
                answer_type = "fallback"
            else:
                picked_raw = 1  # default to PTP's own answer (option 1)
                answer_type = "none"

        picked = max(1, min(picked_raw, hi - lo + 1))  # option index within the window
        choice = lo + (picked - 1)                      # map back to an actual token count
        self._call_log.append({
            "answer_type": answer_type,
            "reasoning": reasoning,
            "picked_option": picked,
            "extra_tokens": choice - ptp_choice,
        })
        return choice

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        choice = self._judge_choice(student_tokens, correct_tokens)
        chosen = torch.cat([student_tokens[:, :choice], correct_tokens[:, choice:choice + 1]], dim=1)
        self._context_ids = torch.cat([self._context_ids, chosen], dim=1)
        return chosen

    def extra_metrics(self) -> dict:
        n = len(self._call_log)
        extra = [c["extra_tokens"] for c in self._call_log]
        return {
            "extra_tokens_total": sum(extra) if extra else 0,
            "extra_tokens_mean": sum(extra) / n if n else float("nan"),
            "answer_normal_count": sum(1 for c in self._call_log if c["answer_type"] == "normal"),
            "answer_fallback_count": sum(1 for c in self._call_log if c["answer_type"] == "fallback"),
            "answer_none_count": sum(1 for c in self._call_log if c["answer_type"] == "none"),
            "reasoning": [c["reasoning"] for c in self._call_log],
            "picked_option": [c["picked_option"] for c in self._call_log],
        }


class SeqPTPSelfInference(SeqInference):
    """
    Like seq-ptp, but replaces the real teacher/AR verification pass with the
    student model's own gated-LoRA prediction, computed via auxiliary probes
    instead of a plain causal forward.

    Normally teacher_forward is just a plain (ungated) causal pass over
    [backlog context, verify tokens]; row i's logits — from ordinary causal
    attention — give "what comes after position i" for free, using the base
    model (no LoRA). This variant instead appends one auxiliary (z-conditioned,
    LoRA-gated) probe per prefix length 0..T (T = backlog + verify tokens):
    probe i attends only to the first i real tokens and produces the model's
    own ("self", LoRA-adapted) prediction for what comes next — exactly the
    same mechanism the student uses to propose each token, just applied here
    to verify already-chosen tokens instead of proposing new ones.
    generate_seq slices out the trailing n_prop+1 probes (prefixes of length
    backlog..T) itself, exactly matching what it expects from a normal
    teacher_forward call.

    In the student's own proposal forward, the row just before the aux window
    (ungated) and the first aux slot (gated) both predict the same first
    token — that redundancy is why correct_first_token normally overwrites
    slot 0 with the free ungated row. Here every position gets its own gated
    probe, so that shortcut is no longer needed: correct_first_token=False
    keeps the student's own argmax pick for position 0 and verifies it like
    any other position instead of silently overwriting it beforehand.

    Because the prediction at a position is z-dependent (that's the whole
    point of the gated/aux mechanism), verifying with an independently-drawn
    z would compare two different z-conditioned predictions and needn't ever
    agree, even for a "correct" model. student_forward stashes the exact z
    the student used to propose each token (auxiliaries), and teacher_forward
    reuses that same z for the matching verification probe, so the two sides
    are directly comparable — same context, same z, only argmax vs.
    inverse-CDF-sample differs.
    """

    name = "seq-ptp-self"

    def __init__(self, lit_model, device, autocast_dtype, *, ar_mode: str = "aux", **kwargs):
        super().__init__(lit_model, device, autocast_dtype, **kwargs)
        assert ar_mode in ("aux", "tok"), f"ar_mode must be 'aux' or 'tok', got {ar_mode!r}"
        self.ar_mode = ar_mode
        self.correct_first_token = (ar_mode == "tok")
        # Under tok mode both student and teacher compute the real-token prefix via
        # the exact same merged-LoRA forward (no z-conditioning on the retained
        # content), so the two caches are numerically redundant -- share one cache
        # instead of maintaining two identical copies. Not valid for aux mode: the
        # student's gated cache and the teacher's z-probe cache are genuinely
        # different computations there.
        self.shared_kv_cache = (ar_mode == "tok")

    def student_forward(self, input_ids, auxiliaries, past_key_values):
        # auxiliaries is None for the one-off initial KV-cache prefill (generate_seq
        # routes that through this callback too, so both submodes see a consistent
        # student-side cache); real proposal calls always pass it.
        if auxiliaries is not None:
            self._z_student = auxiliaries[0]  # [n_prop], reused by teacher_forward below
        if self.ar_mode == "tok":
            with _full_lora_mode(self.lit_model):
                return self.lit_model.model.inference_forward(
                    input_ids=input_ids,
                    auxiliaries=auxiliaries,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
        return self.lit_model.model.inference_forward(
            input_ids=input_ids,
            auxiliaries=auxiliaries,
            past_key_values=past_key_values,
            use_cache=True,
        )

    def teacher_forward(self, input_ids, past_key_values, use_cache=True):
        if self.ar_mode == "tok":
            # Plain per-position causal verification (one row per real token, like
            # plain seq-ptp), but with LoRA merged in everywhere rather than gated
            # to a window — same trick FullLoRAPTPInference uses for ptp_self. Covers
            # the prefill too (past_key_values is None) so the teacher's own KV
            # cache is merged-LoRA throughout, not a base-only prompt mixed with
            # merged-LoRA verify tokens.
            with _full_lora_mode(self.lit_model):
                return self.lit_model.model.inference_forward(
                    input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
                )

        if past_key_values is None:
            # Initial full-prompt prefill (kv_teacher hasn't been created yet):
            # no verification happening yet, so skip the aux-probe machinery
            # entirely — it would otherwise build one probe per prompt
            # position — and just do a plain forward to fill the KV cache.
            return self.lit_model.model.inference_forward(
                input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache,
            )

        device = input_ids.device
        T = input_ids.shape[1]  # backlog + verify tokens, already real/literal
        n = T + 1               # one probe per prefix length 0..T
        S_kv = past_key_values.get_seq_length()

        # Force the same z the student used for each verify position, so
        # this probe's prediction is directly comparable to the student's
        # own (same context, same z); only the bonus probe (index T, beyond
        # the student's n_prop proposals) and the unused pre-backlog probes
        # (sliced away by the caller) get fresh/independent z.
        z_student = self._z_student
        n_prop = z_student.shape[0]
        backlog = T - n_prop
        z_aux = torch.rand(1, n, device=device, dtype=torch.float32)
        z_aux[0, backlog:backlog + n_prop] = z_student

        S = S_kv + T + n
        mask = torch.full((1, 1, T + n, S), float('-inf'), device=device, dtype=torch.float32)
        for q in range(T):
            mask[0, 0, q, :S_kv + q + 1] = 0.0
        for i in range(n):
            mask[0, 0, T + i, :S_kv + i] = 0.0
            mask[0, 0, T + i, S_kv + T + i] = 0.0  # self

        pos_ids = torch.zeros(1, T + n, dtype=torch.long, device=device)
        for q in range(T):
            pos_ids[0, q] = S_kv + q
        for i in range(n):
            pos_ids[0, T + i] = S_kv + i

        outputs = self.lit_model.model.inference_forward(
            input_ids=input_ids,
            auxiliaries=z_aux,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=mask,
            position_ids=pos_ids,
        )
        # Only the T real tokens should remain cached; drop the n aux probes
        # so generate_seq's subsequent crop(-n_prop) restores correctly.
        past_key_values.crop(past_key_values.get_seq_length() - n)
        return outputs

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        if self.ar_mode == "aux":
            # generate_seq's default correct_tokens comes from inverse-CDF sampling
            # tgt_logits with z_rnd, which disagrees with the student's own argmax
            # even when tgt_logits is identical to student_logits (same context,
            # same shared z) — argmax and "sample at this z" are different
            # operations on the same distribution. adapt_logits (temperature/
            # top-k/top-p) never removes the argmax token, so recomputing
            # correct_tokens via argmax here guarantees agreement whenever the
            # underlying logits truly match. Only applies to aux mode: there,
            # tgt_logits is itself z-conditioned via the same z the student used.
            correct_tokens = tgt_logits.argmax(dim=-1)
        # else (tok): tgt_logits comes from a plain, unconditioned causal forward —
        # no z-conditioning at all, exactly like plain seq-ptp's real AR teacher —
        # so there's no same-z mismatch to work around here. Keep generate_seq's
        # already-computed stochastic correct_tokens (sample_from_logits at z_rnd)
        # instead of forcing argmax: forcing argmax was the actual cause of tok
        # mode's rare repetition-collapse failures — once the self-adapted
        # distribution's argmax favors "repeat the last token/pattern," greedy
        # correction has no stochastic escape and keeps reinforcing the loop.
        matches = student_tokens == correct_tokens[:, :-1]
        first_reject = n_prop if matches.all() else int(matches.float().argmin().item())
        return correct_tokens[:, :first_reject + 1]


class SeqPTPTopPSelfInference(SeqPTPSelfInference):
    """
    SeqPTPSelfInference's self-speculative teacher (aux-probe or tok, per ar_mode)
    combined with SeqPTPTopPInference's nucleus acceptance: a proposed token is
    accepted if it falls inside the (self) teacher's top-p nucleus, not just when
    it exactly matches the teacher's own argmax pick as plain seq-ptp-self
    requires — same widened-acceptance idea as seq-ptp-top-p, just verified
    against the self (LoRA-adapted) teacher instead of the real AR teacher.

    As in SeqPTPSelfInference, this only overrides correct_tokens with the
    teacher's own argmax pick for ar_mode="aux" (where tgt_logits is itself
    z-conditioned via the same z the student used, so argmax-vs-argmax is the
    right comparison); ar_mode="tok" keeps generate_seq's already-computed
    stochastic correct_tokens (sample_from_logits at z_rnd), since tok's
    teacher logits are an unconditioned plain causal forward with no same-z
    issue to work around, and forcing argmax there is a known repetition-loop
    attractor (see SeqPTPSelfInference.accepted_tokens).

    Defaults to raw teacher logits (--raw), same reasoning as SeqPTPTopPInference.
    """

    name = "seq-ptp-top-p-self"

    def __init__(self, lit_model, device, autocast_dtype, *, ar_mode: str = "aux",
                 threshold: float = 0.9, raw: bool = True, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, ar_mode=ar_mode, raw=raw, **kwargs)
        self.threshold = threshold

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd):
        n_prop = student_tokens.shape[1]
        if self.ar_mode == "aux":
            correct_tokens = tgt_logits.argmax(dim=-1)
        tgt_probs = torch.softmax(tgt_logits[:, :-1].float(), dim=-1)  # [1, n_prop, V]
        in_nucleus = _in_nucleus(tgt_probs, student_tokens, self.threshold)  # [n_prop]
        matches = (student_tokens == correct_tokens[:, :-1])[0]  # [n_prop]
        accept_mask = in_nucleus | matches
        first_reject = n_prop if accept_mask.all() else int(accept_mask.long().argmin().item())
        return correct_tokens[:, :first_reject + 1]

    def extra_metrics(self) -> dict:
        return {"threshold": self.threshold}


class SeqPTPTreeInference(SeqTreeInference):
    """
    Tree-structured PTP: proposes a two-strand tree in one masked forward pass.

    Main strand depth = n_prop, the current round's proposal budget
    (min(tokens_per_student_call, tokens remaining), passed in by the caller
    each round). n_nodes = 2 * n_prop (n_prop main strand + n_prop branch
    nodes) when n_branch is unset, so the tree shrinks on a short final round
    instead of always proposing tokens_per_student_call deep.

    Main strand: 1a → 2a → ... → n_prop·a
    Branch strand: 1b (root child), 2b → 1a, 3b → 2a, 4b → 3a, ...

    accepted_tokens (inherited from SeqTreeInference) picks the longest path of
    matching nodes; since branch nodes reattach to the main strand a depth
    ahead (not the immediately-next depth), they're never usable as the bonus
    correction token — only the plain main-strand successor can be.
    """

    name = "seq-ptp-tree"

    def __init__(self, lit_model, device, autocast_dtype, *, n_branch: int | None = None, **kwargs):
        super().__init__(lit_model, device, autocast_dtype)
        self.n_branch = n_branch

    def tree(self, n_prop: int) -> list[int | None]:
        """
        Propose a two-strand tree sized to this round's budget (n_prop).

        Main strand: 0(1a) → 1(2a) → ... → n_prop-1
        Branch strand (n_branch nodes):
            n_prop   (1b): root child (parent=None)
            n_prop+j (j+1 b): parent = j-1 (= j-th main strand node) for j = 1..n_branch-1
        """
        n_branch = self.n_branch if self.n_branch is not None else n_prop
        nodes = []
        for i in range(n_prop):
            nodes.append((i, i - 1 if i > 0 else None))
        nodes.append((n_prop, None))
        for j in range(1, n_branch):
            nodes.append((n_prop + j, j - 1))
        self.n_nodes = len(nodes)
        self.parent_list = [p for _, p in nodes]
        return self.parent_list


class SeqPTPChoiceKInference(SeqTreeInference):
    """
    Choice-k PTP: proposes k independent candidate lines per step (strand 0
    plus k-1 independently-sampled alternatives, each its own self-contained
    chain — see tree()), verified in one masked forward pass.

    accepted_tokens (inherited from SeqTreeInference) walks each strand's own
    chain — each node verified against its own strand's context — and keeps
    whichever strand's matching prefix is longest, appending the teacher
    correction token after it.

    mode="standard" (default): all k strands are n_prop deep, where n_prop is
    the current round's proposal budget (min(tokens_per_student_call, tokens
    remaining), passed in by the caller each round).
    mode="balanced": 1 full-length (n_prop) strand, k//4 half-length
    (n_prop//2) strands, and the remaining k - 1 - k//4 strands at
    quarter-length (n_prop//4) — trades some strands' depth for more
    strands at the same total node budget isn't guaranteed, but keeps one
    long strand as a fallback while spending fewer nodes on the others.
    mode="optimal": strand-length profile chosen to exactly maximize
    E[accepted tokens] for the same total node budget as k strands at full
    depth (k * tokens_per_student_call), under the branching-process
    acceptance model from scratch/correct.md (v3 variant, joint-MLE-fit
    params hard-coded below). Computed once at construction (see
    _exact_optimal_lengths); on a shorter final round each strand's length
    is simply capped at n_prop rather than recomputed, which only ever
    removes trailing nodes.
    mode="optimal-50"/"optimal-25"/"optimal-75": 50% (resp. 25%, resp. 75%)
    of the total node budget is spent on full-length (n_prop-deep) strands,
    guaranteeing a long fallback strand the way mode="balanced" does; the
    rest is handed to the same exact solver (_exact_optimal_lengths) to
    maximize E[accepted tokens] for that remaining budget. Both portions
    are computed once at construction.

    mode="bayes-conjugate"/"bayes-p": like mode="optimal", but (p, rho) [resp.
    just p] are re-estimated online per question instead of held at the fixed
    v3 pooled constants, and the knapsack (_exact_optimal_lengths) is rerun
    every round against the current posterior. "bayes-conjugate" exploits
    exact Beta-Binomial conjugacy: Theta_i ~ Beta(alpha,beta) (population
    prior, reparametrized from _BRANCH_P/_BRANCH_RHO) is conjugate to the
    per-generation (m_d, j_d) Binomial-thinning transitions read directly off
    each round's k-strand verification outcome (see _round_thinning), so
    alpha/beta update in closed form and both p and rho move. pi0 stays fixed
    (a handful of rounds can't identify a rare zero-inflation event). "bayes-p"
    instead starts from the per-question hierarchical fit of p alone
    (scratch/joint_per_question_results.json, nearest fitted ensemble width to
    k; pi0/rho pooled-fixed there too) and, since that kernel has no conjugate
    update, tracks the posterior over p on a discrete grid, reweighted each
    round by the real v3 likelihood (via _branch_survival_table) of this
    round's observed extinction time/censoring.

    Model background (scratch/correct.md secs 1-5): each generation d draws a
    shared frailty Theta_d ~ pi0*delta_0 + (1-pi0)*Beta(alpha,beta) and thins
    however many strands are still both alive and within their own length
    budget, Binomial(·, Theta_d). Because Theta_d applies i.i.d.-conditionally
    to every strand present that round, any fixed subset of an alive
    population is itself Beta-Binomial with the same (alpha,beta) — so the
    count of strands alive-and-in-budget at depth d is distributed exactly as
    the *same* homogeneous model started from W(d) := #{strands with length
    >= d} strands, run for d-1 generations, independent of the other depths.
    That makes E[T] = sum_d P(T_homogeneous(W(d)) >= d) separable across d,
    turning the strand-length choice into a knapsack over "how many strands
    to keep alive at each depth" with cost = sum_d W(d) = sum of strand
    lengths. Every nonincreasing width profile W(1) >= ... >= W(cap_t) >= 0
    is reachable by some multiset of strand lengths (W(d) - W(d+1) strands of
    length d), so this is solved exactly by dynamic programming over depth
    rather than approximated — see _exact_optimal_lengths /
    _solve_width_profile.
    """

    # v3 joint-MLE fit (scratch/correct.md sec 5, scratch/v3_panel_data.json
    # "jointM"): pi0 applies from generation 2 on (generation 1 is pure
    # BetaBinomial), alpha/beta are the (p,rho) reparametrization of the
    # per-generation frailty Beta(alpha,beta). Keyed by checkpoint -- pass
    # branch_fit="vicuna_old" (or add another entry here) to use a different
    # checkpoint's own pooled fit instead of vicuna's original one (the default,
    # kept as-is for backward compatibility with existing vicuna results).
    _BRANCH_FITS = {
        "vicuna": (0.061580373241815055, 0.7129725279317468, 0.858289157908181),
        # scratch/fit_simple_joint.py vicuna_old_ -- same single-(pi0,p,rho)-for-all-K
        # methodology, fit from vicuna_old's own exact-match choice-k collection.
        "vicuna_old": (0.128466, 0.619881, 0.789348),
        # scratch/fit_simple_joint.py vicuna_old_topp0.8_self_ -- same methodology, fit
        # from vicuna_old's self-speculative + nucleus(0.8) collection instead of
        # exact-match; use this one when the generation algorithm itself is a "...-self"
        # variant at that same acceptance criterion, since its branching statistics
        # differ meaningfully from exact-match's (see the corrected k=1000 fix).
        "vicuna_old_self_topp0.8": (0.035939, 0.655973, 0.848974),
    }
    # Kept as class-attribute fallback defaults for _branch_survival_table's classmethod
    # signature; every actual call site below passes explicit self._branch_pi0/p/rho.
    _BRANCH_PI0 = _BRANCH_FITS["vicuna"][0]
    _BRANCH_P = _BRANCH_FITS["vicuna"][1]
    _BRANCH_RHO = _BRANCH_FITS["vicuna"][2]

    name = "seq-ptp-choice-k"

    # Nearest-fitted-K per-question hierarchical fits available for "bayes-p"
    # (scratch/joint_per_question_results.json), keyed by their ensemble width.
    _BAYES_P_FITS = {2: "choice2", 5: "choice5", 50: "choice50", 1000: "seqn1000"}

    # Filename prefix for scratch/{prefix}joint_per_question_results.json, per
    # analyze_branching_joint_per_question.py/_stage2.py's own --prefix convention --
    # keyed the same as _BRANCH_FITS (bayes-p's prior and "oracle" both use this).
    _PER_QUESTION_FIT_PREFIX = {
        "vicuna": "",
        "vicuna_old": "vicuna_old_",
        "vicuna_old_self_topp0.8": "vicuna_old_topp0.8_self_",
    }

    def __init__(self, lit_model, device, autocast_dtype, *, k: int = 4, mode: str = "standard",
                 phead_checkpoint: str | None = None, branch_fit: str = "vicuna", **kwargs):
        super().__init__(lit_model, device, autocast_dtype)
        self.k = k
        self.mode = mode
        self._branch_pi0, self._branch_p, self._branch_rho = self._BRANCH_FITS[branch_fit]
        self._example_idx = -1  # incremented at the start of each generate() call
        self._optimal_lengths: list[int] | None = None
        _optimal_full_frac = {"optimal": 0.0, "optimal-25": 0.25, "optimal-50": 0.5, "optimal-75": 0.75}
        if mode in _optimal_full_frac:
            cap_t = lit_model.tokens_per_student_call
            total_budget = k * cap_t
            n_full = int(total_budget * _optimal_full_frac[mode]) // cap_t
            knapsack_budget = total_budget - n_full * cap_t
            self._optimal_lengths = [cap_t] * n_full + self._exact_optimal_lengths(
                knapsack_budget, cap_t, pi0=self._branch_pi0, p=self._branch_p, rho=self._branch_rho)
        elif mode in ("bayes-conjugate", "bayes-p"):
            self._cap_t = lit_model.tokens_per_student_call
            self._bayes_budget = k * self._cap_t
            nu0 = (1 - self._branch_rho) / self._branch_rho
            self._alpha0, self._beta0 = self._branch_p * nu0, (1 - self._branch_p) * nu0
            if mode == "bayes-p":
                nearest = min(self._BAYES_P_FITS, key=lambda kk: abs(kk - k))
                pq_prefix = self._PER_QUESTION_FIT_PREFIX.get(branch_fit, "")
                fit_path = (Path(__file__).resolve().parent.parent / "scratch" /
                            f"{pq_prefix}joint_per_question_results.json")
                with open(fit_path) as f:
                    fit = json.load(f)[self._BAYES_P_FITS[nearest]]
                self._prior_a, self._prior_b = fit["a"], fit["b"]
                import numpy as np
                self._p_grid = np.linspace(1e-3, 1 - 1e-3, 200)
            self._reset_bayes_posterior()
        elif mode == "phead":
            assert phead_checkpoint is not None, "--phead-checkpoint is required for --choice-mode phead"
            self._cap_t = lit_model.tokens_per_student_call
            self._bayes_budget = k * self._cap_t
            from ptp.p_head import PHead
            sidecar = torch.load(phead_checkpoint, map_location="cpu", weights_only=False)
            phead = PHead(lit_model.model.model.config.hidden_size)
            phead.load_state_dict(sidecar["p_head_state_dict"])
            self._phead = phead.to(device).eval()
            lit_model.model.output_hidden_states = True
        elif mode == "oracle":
            # Per-question free-MLE p_hat_i from the earlier per-question hierarchical
            # analysis (nearest-K fit, same lookup bayes-p uses to seed its prior) --
            # an oracle upper bound for bayes-p/bayes-conjugate/phead, since it uses the
            # whole question's data instead of an online/predicted estimate. Mirrors
            # PTPInference's "beta_oracle" partial_mode.
            self._cap_t = lit_model.tokens_per_student_call
            self._bayes_budget = k * self._cap_t
            nearest = min(self._BAYES_P_FITS, key=lambda kk: abs(kk - k))
            pq_prefix = self._PER_QUESTION_FIT_PREFIX.get(branch_fit, "")
            fit_path = (Path(__file__).resolve().parent.parent / "scratch" /
                        f"{pq_prefix}joint_per_question_results.json")
            with open(fit_path) as f:
                fit = json.load(f)[self._BAYES_P_FITS[nearest]]
            self._oracle_p_hat = fit["p_hat_i"]

    @classmethod
    def _branch_transition_matrix(cls, pi0: float, alpha: float, beta: float, w_max: int):
        """Beta-Binomial (zero-inflated) transition matrix, rows/cols 0..w_max."""
        import numpy as np
        from scipy import special

        m = np.arange(w_max + 1)[:, None]
        j = np.arange(w_max + 1)[None, :]
        with np.errstate(invalid="ignore"):
            log_c = special.gammaln(m + 1) - special.gammaln(j + 1) - special.gammaln(m - j + 1)
            log_b1 = special.gammaln(alpha + j) + special.gammaln(beta + m - j) - special.gammaln(alpha + beta + m)
            log_b0 = special.gammaln(alpha) + special.gammaln(beta) - special.gammaln(alpha + beta)
        mat = np.exp(log_c + log_b1 - log_b0)
        mat[j > m] = 0.0
        mat[0, :] = 0.0
        mat = (1 - pi0) * mat
        mat[:, 0] += pi0
        mat[0, 0] = 1.0
        return mat

    @classmethod
    def _branch_survival_table(cls, w_max: int, cap_t: int, *, pi0: float | None = None,
                                p: float | None = None, rho: float | None = None):
        """S[w, d] = P(T_homogeneous(w) >= d) under the v3 model, d = 1..cap_t.
        pi0/p/rho default to the pooled v3 joint-fit constants; pass overrides
        to evaluate the same model at a posterior/candidate (pi0, p, rho)."""
        import numpy as np

        pi0 = cls._BRANCH_PI0 if pi0 is None else pi0
        p = cls._BRANCH_P if p is None else p
        rho = cls._BRANCH_RHO if rho is None else rho
        nu = (1 - rho) / rho
        alpha, beta = p * nu, (1 - p) * nu
        p0 = cls._branch_transition_matrix(0.0, alpha, beta, w_max)  # generation 1: pi0 exempt
        pc = cls._branch_transition_matrix(pi0, alpha, beta, w_max)  # generations 2..cap_t-1
        surv = np.zeros((w_max + 1, cap_t + 1))
        surv[:, 1] = 1.0
        surv[0, :] = 0.0
        v = np.eye(w_max + 1)
        for d in range(2, cap_t + 1):
            gen = d - 1
            if gen >= cap_t:
                v[:, 0] = 1.0  # forced death at the structural cap
            else:
                v = v @ (p0 if gen == 1 else pc)
            surv[:, d] = 1 - v[:, 0]
        return surv

    @classmethod
    def _exact_optimal_lengths(cls, n_budget: int, cap_t: int, *, pi0: float | None = None,
                                p: float | None = None, rho: float | None = None) -> list[int]:
        """
        Exact strand-length knapsack solver.

        Every nonincreasing integer width profile W(1) >= ... >= W(cap_t) >= 0
        is reachable by some multiset of strand lengths (W(d) - W(d+1)
        strands of length d, via the standard nonincreasing-sequence <->
        partition correspondence), and cost = sum_d W(d) always equals the
        sum of those lengths — so maximizing E[T] = sum_d S[W(d), d] over
        strand-length choices under a node budget is exactly maximizing over
        nonincreasing width profiles under the same budget, solved below by
        DP over depth (see _solve_width_profile). Doubles its width bound
        until the optimum doesn't touch it, so the result is exact regardless
        of n_budget. pi0/p/rho default to the pooled v3 constants; pass
        overrides to solve under a posterior (pi0, p, rho) instead.
        """
        if n_budget <= 0:
            return []

        w_max = max(4 * (n_budget // cap_t) + 4 * cap_t, 1)
        while True:
            surv = cls._branch_survival_table(w_max, cap_t, pi0=pi0, p=p, rho=rho)
            widths = cls._solve_width_profile(surv, n_budget, cap_t, w_max)
            if widths[0] < w_max:
                break
            w_max *= 2

        lengths = []
        for d in range(1, cap_t + 1):
            next_width = widths[d] if d < cap_t else 0
            lengths.extend([d] * (widths[d - 1] - next_width))
        return lengths

    @classmethod
    def _solve_width_profile(cls, surv, n_budget: int, cap_t: int, w_max: int) -> list[int]:
        """
        DP over depth for the width-profile knapsack: with
            V_d[w_cap, b] = max_{0 <= w <= min(w_cap, b)} S[w, d] + V_{d+1}[w, b - w]
        (V_{cap_t+1} == 0), V_d[w_cap, b] is the best achievable
        sum_{d'=d}^{cap_t} S[W(d'), d'] given W(d) <= w_cap and remaining
        budget b. Returns the optimal width profile W(1), ..., W(cap_t).
        """
        import numpy as np

        W, B = w_max + 1, n_budget + 1
        V = {cap_t + 1: np.zeros((W, B))}
        for d in range(cap_t, 0, -1):
            v_next = V[d + 1]
            gain = np.full((W, B), -np.inf)
            for w in range(min(W, B)):
                gain[w, w:] = surv[w, d] + v_next[w, :B - w]
            running_best = np.maximum.accumulate(gain, axis=0)
            v_d = np.empty((W, B))
            for w_cap in range(W):
                b_idx = np.arange(B)
                src_w = np.minimum(w_cap, b_idx)
                v_d[w_cap, :] = running_best[src_w, b_idx]
            V[d] = v_d

        widths, w_cap, b = [], w_max, n_budget
        for d in range(1, cap_t + 1):
            v_next = V[d + 1]
            upper = min(w_cap, b) + 1
            ws = np.arange(upper)
            best_w = int(np.argmax(surv[ws, d] + v_next[ws, b - ws]))
            widths.append(best_w)
            w_cap, b = best_w, b - best_w
        return widths

    def _strand_lengths(self, n_prop: int) -> list[int]:
        k = self.k
        if self.mode in ("bayes-conjugate", "bayes-p"):
            p, rho = self._current_p_rho()
            lengths = self._exact_optimal_lengths(self._bayes_budget, self._cap_t,
                                                   pi0=self._branch_pi0, p=p, rho=rho)
            return [min(length, n_prop) for length in lengths]
        if self.mode == "phead":
            # p predicted per-round from the AR context hidden state (see
            # src/ptp/p_head.py) instead of bayes-p's within-question online update --
            # pooled pi0/rho, same knapsack as "optimal"/bayes-*.
            context = self.lit_model._last_context_hidden
            if context is None:
                # generate_seq_tree's pre-loop sizing probe (tree() called once to
                # measure max_n_nodes before any forward pass has run) -- fall back to
                # the pooled constant, same as "optimal", since no context exists yet.
                p_pred = self._branch_p
            else:
                with torch.no_grad():
                    p_pred = self._phead(context.float()).item()
            self._last_p_pred = p_pred
            lengths = self._exact_optimal_lengths(self._bayes_budget, self._cap_t,
                                                   pi0=self._branch_pi0, p=p_pred, rho=self._branch_rho)
            return [min(length, n_prop) for length in lengths]
        if self.mode == "oracle":
            p = self._oracle_p_hat[self._example_idx % len(self._oracle_p_hat)]
            lengths = self._exact_optimal_lengths(self._bayes_budget, self._cap_t,
                                                   pi0=self._branch_pi0, p=p, rho=self._branch_rho)
            return [min(length, n_prop) for length in lengths]
        if self._optimal_lengths is not None:
            return [min(length, n_prop) for length in self._optimal_lengths]
        if self.mode == "balanced":
            n_half = k // 4
            n_rest = k - 1 - n_half
            return [n_prop] + [max(1, n_prop // 2)] * n_half + [max(1, n_prop // 4)] * n_rest
        return [n_prop] * k

    def _reset_bayes_posterior(self):
        """Reset online posterior state at the start of each question (generate() call)."""
        if self.mode == "bayes-conjugate":
            self._alpha_post, self._beta_post = self._alpha0, self._beta0
        elif self.mode == "bayes-p":
            from scipy import stats
            self._log_post = stats.beta.logpdf(self._p_grid, self._prior_a, self._prior_b)

    def _current_p_rho(self) -> tuple[float, float]:
        if self.mode == "bayes-conjugate":
            ab = self._alpha_post + self._beta_post
            return self._alpha_post / ab, 1.0 / (ab + 1.0)
        import numpy as np
        weights = np.exp(self._log_post - self._log_post.max())
        weights /= weights.sum()
        return float(np.sum(weights * self._p_grid)), self._branch_rho

    def _round_thinning(self, student_tokens, correct_tokens, parent_list):
        """Per-generation Binomial-thinning transitions (m_d, j_d) for this
        round's k-strand ensemble (m_d strands alive entering generation d,
        j_d of them still matching), plus this round's extinction time and
        whether it's right-censored (strands ran out of their own length
        budget while still alive, rather than all dying) -- read directly off
        the real per-node match outcomes, mirroring correct.md's v3 model."""
        match = (student_tokens[0] == correct_tokens[0]).tolist()
        roots = [i for i, par in enumerate(parent_list) if par is None]
        bounds = roots + [len(parent_list)]
        strands = [match[bounds[i]:bounds[i + 1]] for i in range(len(roots))]
        alive = [True] * len(strands)
        transitions = []
        d = 0
        while True:
            at_risk = [i for i, s in enumerate(strands) if alive[i] and d < len(s)]
            if not at_risk:
                return transitions, d, True
            j = sum(1 for i in at_risk if strands[i][d])
            transitions.append((len(at_risk), j))
            if j == 0:
                return transitions, d + 1, False
            for i in at_risk:
                if not strands[i][d]:
                    alive[i] = False
            d += 1

    def _bayes_update(self, student_tokens, correct_tokens, parent_list):
        transitions, T_round, censored = self._round_thinning(student_tokens, correct_tokens, parent_list)
        if self.mode == "bayes-conjugate":
            for m, j in transitions:
                self._alpha_post += j
                self._beta_post += m - j
            return
        import numpy as np
        log_lik = np.empty_like(self._p_grid)
        for i, p in enumerate(self._p_grid):
            surv = self._branch_survival_table(self.k, self._cap_t, pi0=self._branch_pi0, p=p, rho=self._branch_rho)
            s = surv[self.k]
            prob = s[T_round] if censored else (s[T_round] - s[T_round + 1] if T_round < self._cap_t else s[T_round])
            log_lik[i] = np.log(max(prob, 1e-300))
        self._log_post = self._log_post + log_lik

    def tree(self, n_prop: int) -> list[int | None]:
        """
        Propose the full k-strand tree sized to this round's budget (n_prop).

        Strand s's node at depth d attends only to the shared root (prompt)
        context and to its own earlier nodes — never to another strand. E.g.
        for k=2 (standard mode): strand 0 is 1a -> 2a -> 3a and strand 1 is
        1b -> 2b -> 3b, where 3b's only ancestor is 2b (not 3a/2a) — unlike
        SeqPTPTreeInference's branch (which reattaches to the main strand at
        every depth), all k strands here are fully independent chains rooted
        directly at the prompt. In balanced mode, strands may differ in
        depth (see _strand_lengths); node indices are packed strand by
        strand in the same way.

        Node index = running offset across strands in order; parent = None
        if it's a strand's first node, else the previous node in that
        strand.
        """
        nodes = []
        idx = 0
        for length in self._strand_lengths(n_prop):
            for d in range(length):
                nodes.append((idx, None if d == 0 else idx - 1))
                idx += 1
        self.n_nodes = len(nodes)
        self.parent_list = [p for _, p in nodes]
        return self.parent_list

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd, parent_list):
        result = super().accepted_tokens(student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd, parent_list)
        if self.mode in ("bayes-conjugate", "bayes-p"):
            self._bayes_update(student_tokens, correct_tokens, parent_list)
        return result

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        self._example_idx += 1
        if self.mode in ("bayes-conjugate", "bayes-p"):
            self._reset_bayes_posterior()
        return super().generate(prompt_ids, max_new_tokens)

    def extra_metrics(self) -> dict:
        n_strands = len(self._optimal_lengths) if self._optimal_lengths is not None else self.k
        metrics = {"n_nodes": self.n_nodes, "k": self.k, "n_strands": n_strands, "choice_mode": self.mode}
        if self.mode in ("bayes-conjugate", "bayes-p"):
            p, rho = self._current_p_rho()
            metrics["bayes_p"] = p
            metrics["bayes_rho"] = rho
        elif self.mode == "phead" and hasattr(self, "_last_p_pred"):
            metrics["phead_p"] = self._last_p_pred
        elif self.mode == "oracle":
            metrics["oracle_p"] = self._oracle_p_hat[self._example_idx % len(self._oracle_p_hat)]
        return metrics


class SeqPTPTopPChoiceKInference(SeqPTPChoiceKInference):
    """
    SeqPTPChoiceKInference's k-strand tree proposals (k, mode/choice_mode) combined
    with SeqPTPTopPInference's nucleus acceptance criterion: verification uses a real,
    separate teacher call (generate_seq_tree's default tree-masked teacher_forward),
    not the self-speculative LoRA-adapted forward SeqPTPTopPChoiceKSelfInference uses
    -- so there's no ar_mode split (aux vs tok) to make here.

    A tree node is accepted if it exactly matches correct_tokens OR falls inside the
    teacher's top-p nucleus at that node's own parent context (mirrors
    SeqPTPTopPInference's per-position criterion, generalized to the tree: walk each
    strand's matching prefix via SeqTreeInference's best-path logic and append the
    teacher's correction token, same as SeqPTPChoiceKInference's inherited
    accepted_tokens does for exact-match-only acceptance).
    """

    name = "seq-ptp-top-p-choice-k"

    def __init__(self, lit_model, device, autocast_dtype, *, k: int = 4, mode: str = "standard",
                 threshold: float = 0.9, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, k=k, mode=mode, **kwargs)
        self.threshold = threshold

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd, parent_list):
        n_nodes = student_tokens.shape[1]
        depth_list = self._depths(parent_list)
        tgt_probs = torch.softmax(tgt_logits.float(), dim=-1)  # [1, n_nodes, V]
        in_nucleus = _in_nucleus(tgt_probs, student_tokens, self.threshold)  # [n_nodes]
        exact = (student_tokens[0] == correct_tokens[0])
        match = (in_nucleus | exact).tolist()

        children: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
        roots: list[int] = []
        for i, p in enumerate(parent_list):
            (roots if p is None else children[p]).append(i)

        def best_path(node: int) -> list[int]:
            if not match[node]:
                return []
            best_child: list[int] = max(
                (best_path(c) for c in children[node]), key=len, default=[],
            )
            return [node] + best_child

        path = max((best_path(r) for r in roots), key=len, default=[])
        if not path:
            result = correct_tokens[:, roots[0]:roots[0] + 1]
        else:
            last = path[-1]
            correction = next(
                (c for c in children[last] if depth_list[c] == depth_list[last] + 1), None,
            )
            if correction is not None:
                result = torch.cat([student_tokens[:, path], correct_tokens[:, correction:correction + 1]], dim=1)
            else:
                result = student_tokens[:, path]
        if self.mode in ("bayes-conjugate", "bayes-p"):
            self._bayes_update(student_tokens, correct_tokens, parent_list)
        return result

    def extra_metrics(self) -> dict:
        metrics = super().extra_metrics()
        metrics["threshold"] = self.threshold
        return metrics


class SeqPTPTopPChoiceKSelfInference(SeqPTPChoiceKInference):
    """
    SeqPTPChoiceKInference's k-strand tree proposals (k, mode/choice_mode) combined
    with SeqPTPTopPSelfInference's self-speculative + nucleus acceptance: verification
    reuses the model's own LoRA-adapted predictions instead of a separate teacher, and
    a proposed token is accepted if it falls in the (self) teacher's top-p nucleus, not
    just an exact match.

    Only ar_mode="tok" is supported. student_forward/teacher_forward replicate the
    exact tree-masked attention generate_seq_tree's default (student_forward=None /
    teacher_forward=None) path already builds via _build_tree_mask, just wrapped in
    _full_lora_mode so LoRA is merged everywhere (backlog context included) instead of
    gated to the aux/tree-node window -- the same student+teacher KV-cache-consistency
    fix SeqPTPSelfInference's ar_mode="tok" needed, generalized to the tree case.

    ar_mode="aux" isn't implemented: SeqPTPSelfInference's flat aux-probe verifier
    builds one z-conditioned probe per *prefix length* along a single chain: gener-
    alizing that to k independent strands means one z-conditioned probe per *tree
    node*, each masked to attend only to its own strand's ancestor chain -- a new
    masked-attention construction, not just reusing existing pieces, so it's left
    unimplemented for now rather than risking a subtly-wrong probe mask.
    """

    name = "seq-ptp-top-p-choice-k-self"

    def __init__(self, lit_model, device, autocast_dtype, *, k: int = 4, mode: str = "standard",
                 ar_mode: str = "aux", threshold: float = 0.9, **kwargs):
        assert ar_mode == "tok", (
            f"seq-ptp-top-p-choice-k-self only supports ar_mode='tok' for now (got "
            f"{ar_mode!r}) -- a per-node aux-probe verifier for the k-strand tree isn't "
            f"implemented; see this class's docstring for what that would need."
        )
        super().__init__(lit_model, device, autocast_dtype, k=k, mode=mode, **kwargs)
        self.ar_mode = ar_mode
        self.threshold = threshold
        # tok-only, so student and teacher always compute the real-token prefix via
        # the same merged-LoRA forward -- share one cache (see SeqPTPSelfInference).
        self.shared_kv_cache = True

    def student_forward(self, input_ids, auxiliaries, past_key_values, parent_list):
        with _full_lora_mode(self.lit_model):
            if parent_list is None:
                # Initial prompt prefill: no tree yet, plain forward suffices.
                return self.lit_model.model.inference_forward(
                    input_ids=input_ids, auxiliaries=auxiliaries,
                    past_key_values=past_key_values, use_cache=True,
                )
            n_nodes = len(parent_list)
            mask, pos_ids = self.lit_model._build_tree_mask(
                input_ids.shape[1], n_nodes, past_key_values.get_seq_length(),
                input_ids.device, parent_list,
            )
            return self.lit_model.model.inference_forward(
                input_ids=input_ids, auxiliaries=auxiliaries,
                past_key_values=past_key_values, use_cache=True,
                attention_mask=mask, position_ids=pos_ids,
            )

    def teacher_forward(self, input_ids, past_key_values, use_cache, parent_list):
        n_nodes = len(parent_list)
        T_ar = input_ids.shape[1] - n_nodes
        mask, pos_ids = self.lit_model._build_tree_mask(
            T_ar, n_nodes, past_key_values.get_seq_length(), input_ids.device, parent_list,
        )
        with _full_lora_mode(self.lit_model):
            return self.lit_model.model.inference_forward(
                input_ids=input_ids, auxiliaries=None,
                past_key_values=past_key_values, use_cache=use_cache,
                attention_mask=mask, position_ids=pos_ids,
            )

    def accepted_tokens(self, student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd, parent_list):
        # Unlike SeqPTPSelfInference/SeqPTPTopPSelfInference's ar_mode="aux" branch,
        # this class is tok-only: tgt_logits comes from a plain, unconditioned causal
        # forward (no z-conditioning), so there's no same-z mismatch to work around --
        # keep generate_seq_tree's already-computed stochastic correct_tokens
        # (sample_from_logits at z_rnd) rather than forcing argmax, which is a known
        # repetition-loop attractor (see SeqPTPSelfInference.accepted_tokens).
        n_nodes = student_tokens.shape[1]
        depth_list = self._depths(parent_list)
        tgt_probs = torch.softmax(tgt_logits.float(), dim=-1)  # [1, n_nodes, V]
        in_nucleus = _in_nucleus(tgt_probs, student_tokens, self.threshold)  # [n_nodes]
        exact = (student_tokens[0] == correct_tokens[0])
        match = (in_nucleus | exact).tolist()

        children: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
        roots: list[int] = []
        for i, p in enumerate(parent_list):
            (roots if p is None else children[p]).append(i)

        def best_path(node: int) -> list[int]:
            if not match[node]:
                return []
            best_child: list[int] = max(
                (best_path(c) for c in children[node]), key=len, default=[],
            )
            return [node] + best_child

        path = max((best_path(r) for r in roots), key=len, default=[])
        if not path:
            result = correct_tokens[:, roots[0]:roots[0] + 1]
        else:
            last = path[-1]
            correction = next(
                (c for c in children[last] if depth_list[c] == depth_list[last] + 1), None,
            )
            if correction is not None:
                result = torch.cat([student_tokens[:, path], correct_tokens[:, correction:correction + 1]], dim=1)
            else:
                result = student_tokens[:, path]
        if self.mode in ("bayes-conjugate", "bayes-p"):
            self._bayes_update(student_tokens, correct_tokens, parent_list)
        return result

    def extra_metrics(self) -> dict:
        metrics = super().extra_metrics()
        metrics["threshold"] = self.threshold
        return metrics


class SeqNPTPChoiceKInference(SeqPTPChoiceKInference):
    """
    Block-wise choice-k PTP: splits the k candidate strands into
    n_blocks = ceil(k / block_size) blocks (e.g. k=200, block_size=50 ->
    4 blocks of 50). Each block is proposed and verified against the
    teacher with its OWN freshly-drawn z (shared between that block's
    student proposal and teacher check, so the acceptance test stays
    meaningful — see generate() below for why this needs a custom loop
    instead of reusing generate_seq_tree). Nothing is committed until all
    n_blocks have been seen; the first n_blocks-1 calls contribute
    candidates only (0 tokens accepted), and the overall best matching
    path across every strand in every block is picked and committed after
    the nth call. num_calls counts every block call, so this spends
    n_blocks teacher calls to cover the same k strands combined choice-k
    covers in one.

    Bypasses generate_seq_tree entirely: that function's z-window is tied
    to how many tokens have actually been committed, which never advances
    across no-accept blocks, so every block would silently reuse the same
    z (and therefore produce byte-identical proposals) if driven through
    it. Managing the loop here lets each block draw independent z while
    still keeping student and teacher aligned on the SAME z within a
    block (required for the accept/reject check to mean anything).
    """

    name = "seq-n-ptp-choice-k"

    def __init__(self, lit_model, device, autocast_dtype, *, k: int = 200, block_size: int = 50, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, k=k, mode="standard", **kwargs)
        self.block_size = block_size
        self.n_blocks = math.ceil(k / block_size)

    def _block_tree(self, n_strands: int, n_prop: int) -> list[int | None]:
        parent_list: list[int | None] = []
        idx = 0
        for _ in range(n_strands):
            for d in range(n_prop):
                parent_list.append(None if d == 0 else idx - 1)
                idx += 1
        return parent_list

    def _best_path(self, parent_list: list[int | None], match: list[bool]) -> tuple[list[int], int | None]:
        depth_list = self._depths(parent_list)
        children: dict[int, list[int]] = {i: [] for i in range(len(parent_list))}
        roots: list[int] = []
        for i, p in enumerate(parent_list):
            (roots if p is None else children[p]).append(i)

        def best_path(node: int) -> list[int]:
            if not match[node]:
                return []
            best_child: list[int] = max(
                (best_path(c) for c in children[node]), key=len, default=[],
            )
            return [node] + best_child

        path = max((best_path(r) for r in roots), key=len, default=[])
        if not path:
            return [], None
        last = path[-1]
        correction = next(
            (c for c in children[last] if depth_list[c] == depth_list[last] + 1), None,
        )
        return path, correction

    def _match_mask(self, student_tokens, correct_tokens, tgt_logits) -> list[bool]:
        """Per-node accept/reject test, applied to every strand in every block. Exact
        match here; SeqNPTPTopPChoiceKInference overrides this for nucleus acceptance."""
        return (student_tokens[0] == correct_tokens[0]).tolist()

    def _teacher_block_forward(self, teacher_fed, kv_teacher, mask, pos_ids):
        """Per-block teacher verification forward. A real (ungated, unconditioned)
        causal pass here; SeqNPTPTopPChoiceKSelfInference overrides this to verify
        against the model's own merged-LoRA prediction instead (self-speculative)."""
        return self.lit_model.model.inference_forward(
            input_ids=teacher_fed, auxiliaries=None,
            past_key_values=kv_teacher, use_cache=True,
            attention_mask=mask, position_ids=pos_ids,
        )

    def _student_block_forward(self, input_ids_student, z_rnd, kv_student, mask, pos_ids):
        """Per-block student proposal forward: gate_window=n_nodes (LoRA only on the
        aux/proposed tail), so the real-token prefix this call feeds into kv_student is
        computed ungated -- matching this class's ungated _teacher_block_forward, which
        keeps root nodes' tgt_logits_root (from this call) and tgt_logits (from the
        teacher call, same bridge position) on the SAME distribution, as
        correct_first_token-style root matching requires. SeqNPTPTopPChoiceKSelfInference
        overrides this to also merge LoRA everywhere, keeping that same-distribution
        property when its teacher call switches to merged-LoRA too."""
        return self.lit_model.model.inference_forward(
            input_ids=input_ids_student, auxiliaries=z_rnd[None, :],
            past_key_values=kv_student, use_cache=True,
            attention_mask=mask, position_ids=pos_ids,
        )

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        raw_ctx = _raw_logits_mode(self.lit_model) if self.raw else contextlib.nullcontext()
        t0 = time.perf_counter()
        with raw_ctx, autocast_ctx:
            completion, num_calls, correct_all = self._generate_blocked(prompt_ids, max_new_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = completion.shape[1] - prompt_ids.shape[1]
        return completion, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": num_calls,
            "correct_per_call": (sum(correct_all) / num_calls) if num_calls else float("nan"),
            "tokens_per_student_call": self.lit_model.tokens_per_student_call,
            "n_nodes": self.block_size * self.lit_model.tokens_per_student_call,
            "k": self.k,
            "block_size": self.block_size,
            "n_blocks": self.n_blocks,
        }

    @torch.inference_mode()
    def _generate_blocked(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, int, list[int]]:
        lit_model = self.lit_model
        device = prompt_ids.device
        tokens_to_fill = max_new_tokens

        outputs = lit_model.model.inference_forward(
            input_ids=prompt_ids[:, :-1], past_key_values=None, use_cache=True,
        )
        kv_teacher = outputs.past_key_values
        outputs = lit_model.model.inference_forward(
            input_ids=prompt_ids[:, :-1], past_key_values=None, use_cache=True, flag=True,
        )
        kv_student = outputs.past_key_values

        max_n_nodes = self.block_size * lit_model.tokens_per_student_call
        z_rnd_all = torch.rand(max_new_tokens * self.n_blocks, max_n_nodes, device=device, dtype=torch.float32)

        correct_all: list[int] = []
        n_calls = 0
        while tokens_to_fill > 0:
            # Round budget, mirroring generate_seq_tree's n_prop: caps strand
            # depth to what's left to fill on a short final round.
            n_prop = min(lit_model.tokens_per_student_call, tokens_to_fill)
            # Every block this super-round re-processes the SAME unfed real
            # suffix (nothing is committed until the last block), so each
            # block must crop back to this baseline afterward — only the
            # last block is allowed to retain that real-token feed, which is
            # what actually advances the cache once we've picked a winner.
            K_student_start = kv_student.get_seq_length()
            K_teacher_start = kv_teacher.get_seq_length()
            candidates: list[tuple[int, torch.Tensor, torch.Tensor, list[int], int | None]] = []
            for block_idx in range(self.n_blocks):
                start = block_idx * self.block_size
                n_strands = max(0, min(self.block_size, self.k - start))
                parent_list = self._block_tree(n_strands, n_prop)
                n_nodes = len(parent_list)
                z_rnd = z_rnd_all[n_calls, :n_nodes]
                is_last_block = block_idx == self.n_blocks - 1

                input_ids_student = prompt_ids[:, K_student_start:]
                mask, pos_ids = lit_model._build_tree_mask(
                    input_ids_student.shape[1], n_nodes, K_student_start, device, parent_list,
                )
                outputs = self._student_block_forward(input_ids_student, z_rnd, kv_student, mask, pos_ids)
                kv_student.crop(kv_student.get_seq_length() - n_nodes if is_last_block else K_student_start)
                full_logits = outputs.logits
                student_logits = full_logits[:, -n_nodes:]
                student_tokens = student_logits.argmax(dim=2)
                roots = [i for i, p in enumerate(parent_list) if p is None]
                tgt_logits_root = lit_model.adapt_logits(full_logits[:, -n_nodes - 1])
                for r in roots:
                    student_tokens[:, r] = lit_model.sample_from_logits(tgt_logits_root, z_rnd[r])

                input_ids = torch.cat([prompt_ids, student_tokens], dim=1)
                teacher_fed = input_ids[:, K_teacher_start:]
                T_ar = teacher_fed.shape[1] - n_nodes
                mask, pos_ids = lit_model._build_tree_mask(
                    T_ar, n_nodes, K_teacher_start, device, parent_list,
                )
                outputs = self._teacher_block_forward(teacher_fed, kv_teacher, mask, pos_ids)
                kv_teacher.crop(kv_teacher.get_seq_length() - n_nodes if is_last_block else K_teacher_start)
                src_idx = torch.tensor(
                    [T_ar - 1 if p is None else T_ar + p for p in parent_list], device=device,
                )
                tgt_logits = lit_model.adapt_logits(outputs.logits[:, src_idx])
                correct_tokens = lit_model.sample_from_logits(tgt_logits, z_rnd)

                match = self._match_mask(student_tokens, correct_tokens, tgt_logits)
                path, correction = self._best_path(parent_list, match)
                if not path:
                    # Guaranteed to match via correct_first_token.
                    path, correction = [roots[0]], None
                candidates.append((len(path), student_tokens, correct_tokens, path, correction))
                n_calls += 1

            _, best_tokens, best_correct, best_path, best_correction = max(candidates, key=lambda c: c[0])
            if best_correction is not None:
                new_tokens = torch.cat(
                    [best_tokens[:, best_path], best_correct[:, best_correction:best_correction + 1]], dim=1,
                )
            else:
                new_tokens = best_tokens[:, best_path]

            correct_all.append(new_tokens.shape[1])
            prompt_ids = torch.cat([prompt_ids, new_tokens], dim=1)
            tokens_to_fill -= new_tokens.shape[1]
            if lit_model.model.tokenizer.eos_token_id in new_tokens:
                eos_idx = prompt_ids.shape[1] - new_tokens.shape[1] + 1 + \
                          (new_tokens[0] == lit_model.model.tokenizer.eos_token_id).nonzero()[0]
                prompt_ids = prompt_ids[:, :eos_idx]
                break

        return prompt_ids, n_calls, correct_all


class SeqNPTPTopPChoiceKInference(SeqNPTPChoiceKInference):
    """
    SeqNPTPChoiceKInference's block-wise choice-k proposals combined with
    SeqPTPTopPChoiceKInference's nucleus acceptance criterion: a node is accepted if
    it exactly matches correct_tokens OR falls inside the teacher's top-p nucleus at
    that node's own parent context -- same idea as SeqPTPTopPChoiceKInference, just
    plugged into _generate_blocked's per-block _match_mask hook instead of
    generate_seq_tree's accepted_tokens callback (this class bypasses that entirely,
    see SeqNPTPChoiceKInference's docstring).
    """

    name = "seq-n-ptp-top-p-choice-k"

    def __init__(self, lit_model, device, autocast_dtype, *, k: int = 200, block_size: int = 50,
                 threshold: float = 0.9, **kwargs):
        super().__init__(lit_model, device, autocast_dtype, k=k, block_size=block_size, **kwargs)
        self.threshold = threshold

    def _match_mask(self, student_tokens, correct_tokens, tgt_logits) -> list[bool]:
        tgt_probs = torch.softmax(tgt_logits.float(), dim=-1)
        in_nucleus = _in_nucleus(tgt_probs, student_tokens, self.threshold)
        exact = (student_tokens[0] == correct_tokens[0])
        return (in_nucleus | exact).tolist()


class SeqNPTPTopPChoiceKSelfInference(SeqNPTPTopPChoiceKInference):
    """
    SeqNPTPTopPChoiceKInference's block-wise nucleus acceptance, but verified against
    the model's own merged-LoRA prediction (self-speculative, ar_mode="tok" convention)
    instead of a real separate teacher call -- mirrors SeqPTPTopPChoiceKSelfInference's
    student_forward/teacher_forward, applied per block via _student_block_forward /
    _teacher_block_forward.

    Both overrides are required together: merging LoRA into only the teacher call (as
    an earlier version of this class did) makes root nodes' tgt_logits_root (from the
    student call, ungated real-token prefix) and tgt_logits (from the teacher call, now
    LoRA-merged real-token prefix) come from two different distributions at the SAME
    bridge position -- silently breaking the same-z-same-distribution guarantee
    correct_first_token-style root matching relies on, which surfaced as an elevated
    G=1 (immediate root-mismatch) rate specifically at k=1000 relative to the tree-based
    self class (which shares one merged-everywhere cache and never had this asymmetry).

    Unlike the tree-based self class, this keeps _generate_blocked's separate
    kv_student/kv_teacher caches rather than sharing one -- with both now merged-LoRA
    for the real-token prefix, they compute numerically identical results, so this only
    costs the (already-paid-for-every-block) redundant compute, not correctness.
    """

    name = "seq-n-ptp-top-p-choice-k-self"

    def _student_block_forward(self, input_ids_student, z_rnd, kv_student, mask, pos_ids):
        with _full_lora_mode(self.lit_model):
            return self.lit_model.model.inference_forward(
                input_ids=input_ids_student, auxiliaries=z_rnd[None, :],
                past_key_values=kv_student, use_cache=True,
                attention_mask=mask, position_ids=pos_ids,
            )

    def _teacher_block_forward(self, teacher_fed, kv_teacher, mask, pos_ids):
        with _full_lora_mode(self.lit_model):
            return self.lit_model.model.inference_forward(
                input_ids=teacher_fed, auxiliaries=None,
                past_key_values=kv_teacher, use_cache=True,
                attention_mask=mask, position_ids=pos_ids,
            )


class FullLoRAPTPInference:
    """
    PTP speculative decoding with LoRA applied to every token — both proposed
    and verified positions.  Standard PTP gates LoRA to the aux (proposed) tokens
    only; this variant overrides that gating so the student LoRA weights also
    influence the verification (teacher) forward pass.
    """

    name = "ptp_self"

    def __init__(self, lit_model, device, autocast_dtype, *, max_tokens_per_proposal: int, total_token_budget: int):
        self.lit_model = lit_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.max_tokens_per_proposal = max_tokens_per_proposal
        self.total_token_budget = total_token_budget

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        t0 = time.perf_counter()
        with autocast_ctx, _full_lora_mode(self.lit_model):
            completion, ptp_metrics = self.lit_model.generate(
                {"prompt_ids": prompt_ids},
                max_new_tokens=max_new_tokens,
                return_metrics=True,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = completion.shape[1] - prompt_ids.shape[1]
        return completion, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": ptp_metrics.get("num_calls", 0),
            "correct_per_call": float(ptp_metrics.get("correct_per_call", float("nan"))),
            "tokens_per_student_call": self.max_tokens_per_proposal,
        }


class PTPTopPChoiceKSelfInference:
    """
    Single-call analog of SeqPTPTopPChoiceKSelfInference ("seq-ptp-top-p-choice-k-self"),
    the same way FullLoRAPTPInference ("ptp_self") is the single-call analog of
    SeqPTPSelfInference ("seq-ptp-self"). Maintains k independent candidate strands,
    verified and re-proposed together in one masked forward call per round (see
    lit.py's generate_tree), pruned to the single longest-verified-accepted candidate
    every round via a plain trailing KV-cache crop (no non-contiguous cache repack) --
    accepted via exact-match-or-top-p-nucleus against the model's own merged-LoRA
    (self) forward.
    """

    name = "ptp-top-p-choice-k-self"

    def __init__(self, lit_model, device, autocast_dtype, *, k: int = 4,
                 threshold: float = 0.9, max_tokens_per_proposal: int, total_token_budget: int):
        self.lit_model = lit_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.k = k
        self.threshold = threshold
        self.max_tokens_per_proposal = max_tokens_per_proposal
        # Explicit H_fn wiring -- ptp_self's known bug is that it never sets this,
        # silently inheriting lit.py's dummy arange(21) default (see FullLoRAPTPInference
        # above). Same H(k)=k value here (literal partial_mode="count"), but intentional.
        lit_model.H_fn = lambda metrics: PTPInference.compute_H("count", lit_model.hist_base, metrics)

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None
            else contextlib.nullcontext()
        )
        t0 = time.perf_counter()
        with autocast_ctx, _full_lora_mode(self.lit_model):
            completion, ptp_metrics = self.lit_model.generate_tree(
                {"prompt_ids": prompt_ids},
                max_new_tokens=max_new_tokens,
                k=self.k,
                nucleus_threshold=self.threshold,
                return_metrics=True,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = completion.shape[1] - prompt_ids.shape[1]
        return completion, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": ptp_metrics.get("num_calls", 0),
            "correct_per_call": float(ptp_metrics.get("correct_per_call", float("nan"))),
            "tokens_per_student_call": self.max_tokens_per_proposal,
            "k": self.k,
            "threshold": self.threshold,
        }


# ORACLE DEBUG — remove after testing
class OraclePTPInference:
    """PTP with reference tokens forced as accepted tokens each step — isolates proposal quality."""

    name = "oracle_ptp"

    def __init__(self, lit_model, device, autocast_dtype, *, max_tokens_per_proposal: int, total_token_budget: int):
        self.lit_model = lit_model
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.max_tokens_per_proposal = max_tokens_per_proposal
        self.total_token_budget = total_token_budget
        self._ref_ids: torch.Tensor | None = None

    def set_ref(self, ref_ids: torch.Tensor) -> None:
        self._ref_ids = ref_ids

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        if self._ref_ids is None:
            raise RuntimeError("set_ref() must be called before generate()")
        autocast_ctx = (
            torch.autocast(self.device.type, dtype=self.autocast_dtype)
            if self.autocast_dtype is not None else contextlib.nullcontext()
        )
        t0 = time.perf_counter()
        with autocast_ctx:
            completion, ptp_metrics = self.lit_model.generate(
                {"prompt_ids": prompt_ids},
                max_new_tokens=max_new_tokens,
                return_metrics=True,
                oracle_ref_ids=self._ref_ids,
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n_gen = completion.shape[1] - prompt_ids.shape[1]
        return completion, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": elapsed_ms,
            "ms_per_token": elapsed_ms / n_gen if n_gen > 0 else float("nan"),
            "num_calls": ptp_metrics.get("num_calls", 0),
            "correct_per_call": float(ptp_metrics.get("correct_per_call", float("nan"))),
            "tokens_per_student_call": self.max_tokens_per_proposal,
        }
# END ORACLE DEBUG


class ReferenceInference:
    """Returns the reference completion verbatim — useful as a log-prob ceiling."""

    name = "ref"

    def __init__(self, lit_model, device, autocast_dtype, **_kwargs):
        self.device = device
        self._ref_ids: torch.Tensor | None = None

    def set_ref(self, ref_ids: torch.Tensor) -> None:
        self._ref_ids = ref_ids

    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> tuple[torch.Tensor, dict]:
        if self._ref_ids is None:
            raise RuntimeError("set_ref() must be called before generate()")
        completion = torch.cat([prompt_ids, self._ref_ids.to(self.device)], dim=1)
        n_gen = self._ref_ids.shape[1]
        return completion, {
            "n_generated_tokens": n_gen,
            "elapsed_ms": 0.0,
            "ms_per_token": 0.0,
            "num_calls": 0,
            "correct_per_call": float("nan"),
        }


# ---------------------------------------------------------------------------
# Experiment configurations — encode all per-dataset differences in one place
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIGS: dict[str, dict] = {
    "qwen": dict(
        mode="chat",
        default_dataset="allenai/tulu-3-sft-mixture",
        text_column=None,
        max_new_tokens=256,
        english_only=True,
    ),
    "qm9": dict(
        mode="text",
        default_dataset="yairschiff/qm9",
        text_column="canonical_smiles",
        max_new_tokens=32,
        english_only=False,
        teacher_ckpt="/extra/ucibdl1/ptd/qm9/qm9/ar_best_nll.ckpt",
        pdd_repo="/extra/ucibdl1/jcwill/home_old/PycharmProjects/pdd",
    ),
    "vicuna": dict(
        mode="spec_bench",
        default_dataset="/extra/ucibdl1/jcwill/home_old/PycharmProjects/Spec-Bench/data/spec_bench/question.jsonl",
        text_column=None,
        max_new_tokens=1024,
        english_only=False,
        chat_template=(
            "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + ' ' }}"
            "{% set messages = messages[1:] %}{% else %}"
            "{{ 'A chat between a curious user and an artificial intelligence assistant."
            " The assistant gives helpful, detailed, and polite answers to the user\\'s questions. ' }}"
            "{% endif %}{% for message in messages %}"
            "{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}"
            "{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
        ),
    ),
}


# Registry — add custom algorithms here
ALGORITHMS: dict[str, type] = {
    "ptp": PTPInference,
    "seq-ptp": SequentialPTPInference,
    "seq-ptp-self": SeqPTPSelfInference,
    "seq-ptp-top-p-self": SeqPTPTopPSelfInference,
    "ar": ARInference,
    "det_ar": DeterministicARInference,
    "first-k": AcceptFirstKInference,
    "seq-ptp-first-k": SeqPTPTAcceptFirstKInference,
    "seq-thresh-p": SeqThresholdInference,
    "seq-ptp-thresh-p": SeqPTPThresholdInference,
    "seq-conf-p": SeqConfPInference,
    "seq-ptp-conf-p": SeqPTPConfPInference,
    "thresh-p": ThresholdPTPInference,
    "entr-p": EntropyThresholdInference,
    "seq-ptp-entr-p": SeqPTPEntropyThresholdInference,
    "conf-p": ConfPInference,
    "seq-ratio": SeqRatioInference,
    "seq-ratio-k": SeqRatioInference,
    "seq-ratio-p": SeqRatioInference,
    "seq-ratio-k-p": SeqRatioInference,
    "seq-ptp-ratio": SeqPTPRatioInference,
    "seq-ptp-ratio-k": SeqPTPRatioInference,
    "seq-ptp-ratio-p": SeqPTPRatioInference,
    "seq-ptp-ratio-k-p": SeqPTPRatioInference,
    "seq-inv-p": SeqInvPInference,
    "seq-top-k": SeqTopKInference,
    "seq-ptp-top-k": SeqPTPTopKInference,
    "seq-top-p": SeqTopPInference,
    "seq-ptp-top-p": SeqPTPTopPInference,
    "ptp-human": PTPHumanInference,
    "ptp-judge": PTPJudgeInference,
    "seq-ptp-tree": SeqPTPTreeInference,
    "seq-ptp-choice-k": SeqPTPChoiceKInference,
    "seq-ptp-top-p-choice-k": SeqPTPTopPChoiceKInference,
    "seq-ptp-top-p-choice-k-self": SeqPTPTopPChoiceKSelfInference,
    "seq-n-ptp-choice-k": SeqNPTPChoiceKInference,
    "seq-n-ptp-top-p-choice-k": SeqNPTPTopPChoiceKInference,
    "seq-n-ptp-top-p-choice-k-self": SeqNPTPTopPChoiceKSelfInference,
    "ratio": RatioInference,
    "ratio-k": RatioKInference,
    "ratio-p": RatioPInference,
    "ptp_self": FullLoRAPTPInference,
    "ptp-top-p-choice-k-self": PTPTopPChoiceKSelfInference,
    "oracle_ptp": OraclePTPInference,  # ORACLE DEBUG — remove after testing
    "ref": ReferenceInference,
}


# ---------------------------------------------------------------------------
# Teacher-mode context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _raw_logits_mode(lit_model):
    """Temporarily disable temperature, top-k, and top-p so adapt_logits is a no-op."""
    saved = (lit_model.temperature, lit_model.top_k, lit_model.top_p)
    lit_model.temperature = None
    lit_model.top_k = 0
    lit_model.top_p = 1.0
    try:
        yield
    finally:
        lit_model.temperature, lit_model.top_k, lit_model.top_p = saved


@contextlib.contextmanager
def _teacher_mode(lit_model):
    """
    Temporarily set gate_window=0 on all GatedLinearLoraMerged layers so the
    model runs as the base teacher (no LoRA contribution).  Works after
    enter_inference_mode() has fused the PEFT adapters.
    """
    from ptp.transformer import GatedLinearLoraMerged
    modules = [m for m in lit_model.model.model.modules() if isinstance(m, GatedLinearLoraMerged)]
    saved = [m.gate_window for m in modules]
    for m in modules:
        m.gate_window = 0
    try:
        yield
    finally:
        for m, gw in zip(modules, saved):
            m.gate_window = gw


@contextlib.contextmanager
def _full_lora_mode(lit_model):
    """
    Override set_gate_window so LoRA applies to every token regardless of position.
    During PTP generate(), the model normally calls set_gate_window(n_aux) to gate
    only the proposed tokens; this replaces that call with a large sentinel so
    GatedLinearLoraMerged never subtracts LoRA from any token.
    """
    _LARGE = 2 ** 30
    model = lit_model.model
    original_sgw = model.set_gate_window
    model.set_gate_window = lambda _gw, fg: original_sgw(_LARGE)
    try:
        yield
    finally:
        model.set_gate_window = original_sgw

@contextlib.contextmanager
def _gated_mode(lit_model):
    """
    Override set_gate_window so LoRA applies to every token regardless of position, but only for the student.
    If set_gate_window(0) is called, passes 0 through (no gating).
    """
    _LARGE = 2 ** 30
    model = lit_model.model
    original_sgw = model.set_gate_window
    model.set_gate_window = lambda _gw, fg: original_sgw(0 if _gw == 0 and not fg else _LARGE)
    try:
        yield
    finally:
        model.set_gate_window = original_sgw


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def teacher_log_prob(
    lit_model,
    completion_ids: torch.Tensor,
    prompt_length: int,
    autocast_dtype,
    device,
    teacher_model=None,
) -> dict:
    """
    Per-token log p_teacher(token | context) stats over generated tokens only.

    Returns mean, median, and a log-likelihood z-test p-value.  The z-test
    checks H0: student tokens were sampled from the teacher distribution.

    Under H0, sum_t log P_t(s_t) has known mean -sum H(P_t) and known variance
    sum Var_t[log P_t(x)]; the CLT z-statistic is standard normal.

    If teacher_model is provided (QM9 case), uses _forward_backbone_ar instead
    of the gated-LoRA base path.
    """
    autocast_ctx = (
        torch.autocast(device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else contextlib.nullcontext()
    )
    if teacher_model is not None:
        with autocast_ctx:
            logits = teacher_model.model._forward_backbone_ar(completion_ids).float()  # [1, T, V]
    else:
        with autocast_ctx, _teacher_mode(lit_model):
            logits = lit_model.model.inference_forward(input_ids=completion_ids).logits  # [1, T, V]
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)          # [1, T-1, V]
    targets = completion_ids[:, 1:]                             # [1, T-1]
    per_token = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # [T-1]
    gen_lp = per_token[prompt_length - 1:]                     # generated portion only

    # Z = (Σ log P_t(s_t) - Σ E[log P_t]) / sqrt(Σ Var_t)  ~  N(0,1) under H0
    # When top-k/p filtering is active the tokens are drawn from the truncated+renormalized
    # nucleus, not the full vocabulary.  Using full-vocab E/Var biases the numerator
    # positive (nucleus tokens have higher log-prob than the full-distribution mean),
    # producing a spuriously tiny p-value.  Use the adapted distribution instead.
    use_adapted = (
        teacher_model is None
        and hasattr(lit_model, "adapt_p")
        and lit_model.top_k is not None and lit_model.top_k > 0
        and lit_model.top_p is not None
    )
    if use_adapted:
        # Apply the same temperature scaling used during sampling before adapt_p
        temp = getattr(lit_model, "temperature", None)
        scaled_logits = logits[:, :-1] / temp if (temp is not None and temp != 1.0) else logits[:, :-1]
        tgt_p, tgt_indices = lit_model.adapt_p(scaled_logits.softmax(dim=-1))  # [1, T-1, k] each
        log_tgt_p = tgt_p.clamp(min=1e-40).log()                 # [1, T-1, k]
        match = (tgt_indices == targets.unsqueeze(-1))            # [1, T-1, k]
        in_nucleus = match.any(-1)[0]                             # [T-1] bool
        # Nucleus stats
        token_lp  = (tgt_p * match.float()).sum(-1).clamp(min=1e-40).log()[0]  # [T-1]
        neg_ent   = (tgt_p * log_tgt_p).sum(-1)[0]               # [T-1]
        sec_mom   = (tgt_p * log_tgt_p.pow(2)).sum(-1)[0]        # [T-1]
        # Fallback for the rare positions where sampled token is outside our nucleus:
        # use the full temperature-scaled distribution for both the token log-prob and E/Var
        valid = in_nucleus                                         # [T-1]
        if not in_nucleus.all():
            temp_lp = F.log_softmax(scaled_logits, dim=-1)        # [1, T-1, V]
            temp_p  = temp_lp.exp()
            fb_token_lp = temp_lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # [T-1]
            fb_neg_ent  = (temp_p * temp_lp).sum(-1)[0]
            fb_sec_mom  = (temp_p * temp_lp.pow(2)).sum(-1)[0]
            # Use fallback for out-of-nucleus tokens whose full-vocab log-prob is reasonable.
            # Skip tokens with fb_token_lp < -20 nats (~1e-9): these were injected by HF
            # logits processors (forced EOS, repetition penalty, etc.) and are uninformative.
            use_fallback = ~in_nucleus & (fb_token_lp > -20)
            token_lp = torch.where(use_fallback, fb_token_lp, token_lp)
            neg_ent  = torch.where(use_fallback, fb_neg_ent,  neg_ent)
            sec_mom  = torch.where(use_fallback, fb_sec_mom,  sec_mom)
            valid    = in_nucleus | use_fallback
    else:
        temp_lp  = log_probs                                      # already unscaled full vocab
        temp_p   = temp_lp.exp()
        token_lp = temp_lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # [T-1]
        neg_ent  = (temp_p * temp_lp).sum(-1)[0]                 # [T-1]
        sec_mom  = (temp_p * temp_lp.pow(2)).sum(-1)[0]          # [T-1]
        valid    = torch.ones(token_lp.shape[0], dtype=torch.bool, device=token_lp.device)

    var_per_pos  = sec_mom - neg_ent.pow(2)                       # [T-1]
    # Also skip positions where the effective adapted log-prob is near-zero (< -20 nats),
    # which happens when a token lands at the very bottom of the top-k nucleus after
    # temperature renormalization (P_adapted ≈ 0).  These are uninformative for the test.
    valid        = valid & (token_lp > -20)
    gen_valid    = valid[prompt_length - 1:]                      # [T_gen] — False for skipped tokens
    gen_token_lp = token_lp[prompt_length - 1:][gen_valid]
    gen_neg_ent  = neg_ent[prompt_length - 1:][gen_valid]
    gen_var      = var_per_pos[prompt_length - 1:][gen_valid]
    z_num = (gen_token_lp - gen_neg_ent).sum().item()  # sufficient stat: centered log-prob sum
    z_var = gen_var.sum().item()                        # sufficient stat: sum of variances
    if gen_lp.numel() > 0 and z_var > 0:
        z_stat = z_num / math.sqrt(z_var)
        p_value = float(math.erfc(abs(z_stat) / math.sqrt(2)))  # two-sided
    else:
        z_stat = float("nan")
        p_value = float("nan")

    return {
        "mean": gen_lp.mean().item(),
        "median": gen_lp.median().item(),
        "z_num": z_num,   # saved for combined z-test across examples in eval.py
        "z_var": z_var,
        "z_stat": z_stat,
        "p_value": p_value,
        "n_outlier_tokens": int((~gen_valid).sum().item()),
    }


# ---------------------------------------------------------------------------
# Data helpers  (role normalisation reused from ptp.data.chat)
# ---------------------------------------------------------------------------

DEFAULT_USER_ROLES = ["user", "human"]
DEFAULT_ASSISTANT_ROLES = ["assistant", "gpt", "bing", "chatgpt", "bard", "model"]


def _convert_to_chat_format(
    conversation_data,
    user_roles=DEFAULT_USER_ROLES,
    assistant_roles=DEFAULT_ASSISTANT_ROLES,
) -> list[dict]:
    """Normalise a conversation to [{"role": "user"|"assistant", "content": ...}, ...]."""
    conversation = []
    for i, message in enumerate(conversation_data):
        if isinstance(message, dict):
            raw_role = message.get("from") or message.get("role", "")
            content = message.get("value") or message.get("content", "")
            if raw_role in assistant_roles:
                role = "assistant"
            elif raw_role in user_roles:
                role = "user"
            elif raw_role == "system":
                continue
            else:
                raise ValueError(f"Unknown role '{raw_role}'")
            conversation.append({"role": role, "content": content})
        else:
            role = "user" if i % 2 == 0 else "assistant"
            conversation.append({"role": role, "content": message})
    return conversation


def _resolve_conversation_key(item: dict, candidates: list[str]) -> str | None:
    present = [k for k in candidates if k in item and item[k] is not None]
    return present[0] if len(present) == 1 else (present[0] if present else None)


_COT_EMPTY_BLOCK = "<think>\n\n</think>\n\n"


def _is_english(text: str) -> bool:
    if not text.strip():
        return True
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def iter_prompt_completion_pairs(
    dataset,
    tokenizer,
    conversation_keys: list[str],
    user_roles: list[str],
    assistant_roles: list[str],
    max_sequence_length: int,
    only_first_turn: bool = True,
    suppress_cot: bool = True,
    english_only: bool = True,
):
    """
    Yield (example_id, prompt_ids, ref_ids, prompt_str, ref_str) for each
    assistant turn in the dataset.

    ``only_first_turn=True`` yields one pair per conversation (the first
    user→assistant exchange), which is the most common evaluation setup.
    Set to False to yield every turn.

    ``suppress_cot=True`` (default) appends an empty think-block to the prompt
    when the reference starts with one, so the model skips chain-of-thought.
    Pass ``suppress_cot=False`` (--think flag) to let the model reason freely.
    """
    example_id = 0
    for item in dataset:
        key = _resolve_conversation_key(item, conversation_keys)
        if key is None:
            continue
        try:
            conversation = _convert_to_chat_format(item[key], user_roles, assistant_roles)
        except ValueError:
            continue

        if english_only:
            first_user = next((m["content"] for m in conversation if m["role"] == "user"), "")
            if not _is_english(first_user):
                continue

        for msg_idx, msg in enumerate(conversation):
            if msg["role"] != "assistant":
                continue
            prompt_messages = conversation[:msg_idx]
            if not prompt_messages:
                continue

            prompt_str = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full_str = tokenizer.apply_chat_template(
                conversation[: msg_idx + 1], tokenize=False, add_generation_prompt=False
            )
            ref_str = full_str[len(prompt_str):]

            if suppress_cot and ref_str.startswith(_COT_EMPTY_BLOCK):
                prompt_str = prompt_str + _COT_EMPTY_BLOCK
                ref_str = ref_str[len(_COT_EMPTY_BLOCK):]

            prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
            ref_ids = tokenizer(ref_str, return_tensors="pt").input_ids

            # Skip if prompt alone is already too long
            if prompt_ids.shape[1] > max_sequence_length:
                break

            yield example_id, prompt_ids, ref_ids, prompt_str, ref_str
            example_id += 1

            if only_first_turn:
                break


def iter_text_pairs(
    dataset,
    tokenizer,
    text_column: str,
    max_sequence_length: int,
):
    """
    Yield (example_id, prompt_ids, ref_ids, prompt_str, ref_str) for a plain-text dataset.
    Prompt is the BOS token only; ref is the full tokenized text (excluding BOS).
    """
    bos_id = tokenizer.bos_token_id or getattr(tokenizer, "cls_token_id", None)
    eos_id = tokenizer.eos_token_id or getattr(tokenizer, "sep_token_id", None)

    example_id = 0
    for item in dataset:
        text = item.get(text_column, "")
        if not text:
            continue

        full_ids = tokenizer(text, return_tensors="pt").input_ids  # [1, T]

        if bos_id is not None:
            prompt_ids = torch.tensor([[bos_id]], dtype=torch.long)
            # strip leading BOS from full_ids if the tokenizer added one
            ref_start = 1 if full_ids[0, 0].item() == bos_id else 0
            ref_ids = full_ids[:, ref_start:]
        else:
            prompt_ids = full_ids[:, :1]
            ref_ids = full_ids[:, 1:]

        if prompt_ids.shape[1] + ref_ids.shape[1] > max_sequence_length:
            continue

        prompt_str = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
        ref_str = text

        yield example_id, prompt_ids, ref_ids, prompt_str, ref_str
        example_id += 1


def iter_spec_bench_pairs(dataset, tokenizer, max_sequence_length: int):
    """
    Yield (example_id, prompt_ids, ref_ids, prompt_str, ref_str) for the first
    turn of each question in a Spec-Bench JSONL dataset.
    Spec-Bench has no reference answers; ref_ids is empty.
    """
    for item in dataset:
        question_id = item.get("question_id", 0)
        turns = item.get("turns", [])
        if not turns:
            continue
        messages = [{"role": "user", "content": turns[0]}]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(prompt_str, return_tensors="pt").input_ids
        if prompt_ids.shape[1] >= max_sequence_length:
            continue
        ref_ids = torch.zeros(1, 0, dtype=torch.long)
        yield question_id, prompt_ids, ref_ids, prompt_str, ""


# ---------------------------------------------------------------------------
# Model loading  (mirrors generate.py)
# ---------------------------------------------------------------------------

def load_model(experiment_dir: Path, checkpoint: Path | None, total_token_budget: int):
    import yaml
    from omegaconf import DictConfig, OmegaConf
    from ptp.cli.generate import find_best_checkpoint, _load_or_compute_hist_base
    from ptp.utils import instantiate

    with open(experiment_dir / "train.yaml") as f:
        config = DictConfig(yaml.safe_load(f))

    ckpt_dir = Path(config["training"].get("ckpt_dir", experiment_dir))
    if checkpoint is not None and not checkpoint.is_absolute() and not checkpoint.exists():
        checkpoint = ckpt_dir / checkpoint
    ckpt_path = checkpoint or find_best_checkpoint(ckpt_dir)
    print(f"Loading checkpoint: {ckpt_path}")

    OmegaConf.update(config, "model.model.attn_implementation", "sdpa", merge=True)
    lit_model = instantiate(config["model"])
    lit_model.configure_model()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    lit_model.load_state_dict(ckpt["state_dict"])

    gen_cfg = getattr(lit_model.model.model, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.max_length = None
        gen_cfg.temperature = 1.0
        gen_cfg.top_p = 1.0

    precision = config["training"].get("precision", "32-true")
    if "bf16" in str(precision):
        autocast_dtype = torch.bfloat16
    elif "16" in str(precision):
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()

    _load_or_compute_hist_base(lit_model, config, experiment_dir, ckpt_path, ckpt_dir, autocast_dtype, device)

    lit_model.enter_inference_mode(gate_window=total_token_budget)

    return lit_model, config, autocast_dtype, device, ckpt_path


def load_teacher(exp_cfg: dict, device):
    from ptp.data.pdd_teacher import PddARTeacher
    return PddARTeacher(
        ckpt_path=exp_cfg["teacher_ckpt"],
        pdd_repo=exp_cfg["pdd_repo"],
    ).to(device).eval()


@torch.inference_mode()
def generate_qm9_ptp(
    lit_model,
    teacher_model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    n_proposals: int,
    autocast_ctx,
) -> tuple[torch.Tensor, dict]:
    """
    QM9 PTP: one student call proposes via aux mechanism, one teacher call verifies.
    One z value per output position, shared between student aux embedding and teacher acceptance.
    """
    import numpy as np

    dev = prompt_ids.device
    eos = (lit_model.model.tokenizer.eos_token_id
           or getattr(lit_model.model.tokenizer, "sep_token_id", None))
    tokens = prompt_ids.clone()
    tokens_generated = 0
    metrics_correct: list[int] = []

    # One z per output position (+ headroom for the bonus token each step)
    z_rnd_all = torch.rand(1, max_new_tokens + n_proposals + 2, device=dev)

    while tokens_generated < max_new_tokens:
        n_prop = min(n_proposals, max_new_tokens - tokens_generated)
        T = tokens.shape[1]

        # Shared z slice for this step
        z = z_rnd_all[:, tokens_generated : tokens_generated + n_prop + 1]  # [1, n_prop+1]

        # 1. ONE student call: aux z → n_prop proposals (gate_window=n_prop set internally)
        with autocast_ctx:
            out = lit_model.model.inference_forward(input_ids=tokens, auxiliaries=z[:, :n_prop])
        proposed = out.logits[:, T:].argmax(-1)          # [1, n_prop]
        candidate = torch.cat([tokens, proposed], dim=1)

        # 2. ONE teacher call: full forward over prompt + proposals
        with autocast_ctx:
            teacher_logits = teacher_model.model._forward_backbone_ar(candidate).float()

        tgt_logits = teacher_logits[:, T - 1 : T + n_prop]  # [1, n_prop+1, V]
        if lit_model.temperature is not None and lit_model.temperature != 1.0:
            tgt_logits = tgt_logits / lit_model.temperature
        tgt_p, tgt_indices = lit_model.adapt_p(torch.softmax(tgt_logits, dim=-1))

        # 3. Accept/reject via inverse-CDF using the shared z
        right_bin_edges = tgt_p.cumsum(-1)
        right_bin_edges[..., -1] = 1.0
        bin_idx = (right_bin_edges > z.unsqueeze(-1)).max(dim=-1).indices  # [1, n_prop+1]
        correct_tokens = tgt_indices.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)  # [1, n_prop+1]

        matches = proposed == correct_tokens[:, :-1]
        num_correct = int(matches.float().argmin(1)[0])
        if matches.all():
            num_correct = n_prop

        accepted = correct_tokens[:, : num_correct + 1]
        tokens = torch.cat([tokens, accepted], dim=1)
        tokens_generated += num_correct + 1
        metrics_correct.append(num_correct + 1)

        if eos is not None and (accepted[0] == eos).any():
            break

    return tokens, {
        "correct_per_call": float(np.mean(metrics_correct)),
        "correct_all": metrics_correct,
        "num_calls": len(metrics_correct),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    experiment_dir: Path,
    checkpoint: Path | None,
    algorithm: str,
    max_tokens_per_proposal: int,
    total_token_budget: int | None,
    top_k: int,
    top_p: float,
    temperature: float | None,
    max_new_tokens: int,
    n_examples: int | None,
    seed: int,
    output_dir: Path | None,
    save_dir: str | None,
    split: str,
    compile: bool,
    only_first_turn: bool,
    suppress_cot: bool,
    k: float,
    english_only: bool,
    greedy: bool = False,
    p: float = 0.7,
    delta: float = 0.3,
    fast: bool = False,
    raw: bool = False,
    experiment: str = "qwen",
    referee: str = "Qwen/Qwen2.5-7B-Instruct",
    judge_prompt: str = "mid",
    choice_mode: str = "standard",
    partial_mode: str = "count",
    phead_checkpoint: str | None = None,
    chead_checkpoint: str | None = None,
    ar_mode: str = "aux",
    branch_fit: str = "vicuna",
):
    torch.manual_seed(seed)

    total_budget = total_token_budget or max_tokens_per_proposal
    lit_model, config, autocast_dtype, device, ckpt_path = load_model(
        experiment_dir, checkpoint, total_budget
    )

    exp = EXPERIMENT_CONFIGS[experiment]
    teacher_model = load_teacher(exp, device) if "teacher_ckpt" in exp else None

    # Apply sampling params so PTP generate() finds them.
    # Prefer values from train.yaml so inference matches the training distribution;
    # CLI args serve as fallback for models that were trained without these set.
    if experiment == "qm9":
        lit_model.top_k = None
        lit_model.top_p = 1.0
        lit_model.temperature = 1.0
    else:
        model_cfg = config.get("model", {})
        trained_top_k = model_cfg.get("top_k", None)
        trained_top_p = model_cfg.get("top_p", None)
        trained_temperature = model_cfg.get("temperature", None)
        lit_model.top_k = trained_top_k if trained_top_k is not None else top_k
        lit_model.top_p = trained_top_p if trained_top_p is not None else top_p
        if trained_temperature is not None:
            lit_model.temperature = trained_temperature
        elif temperature is not None:
            lit_model.temperature = temperature
    lit_model.tokens_per_student_call = max_tokens_per_proposal
    lit_model.total_token_budget = total_budget

    if compile:
        lit_model.model.model = torch.compile(lit_model.model.model, dynamic=True)

    # Build inference algorithm
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {list(ALGORITHMS)}")
    algo_cls = ALGORITHMS[algorithm]
    if algorithm == "ptp":
        algo = algo_cls(
            lit_model, device, autocast_dtype,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_budget,
            partial_mode=partial_mode,
            phead_checkpoint=phead_checkpoint,
            chead_checkpoint=chead_checkpoint,
        )
    elif algorithm in ("ptp_self", "oracle_ptp"):  # oracle_ptp: ORACLE DEBUG
        algo = algo_cls(
            lit_model, device, autocast_dtype,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_budget,
        )
    elif algorithm == "ptp-top-p-choice-k-self":
        algo = algo_cls(
            lit_model, device, autocast_dtype,
            k=int(k), threshold=p,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_budget,
        )
    elif algorithm == "seq-ptp":
        algo = algo_cls(
            lit_model, device, autocast_dtype, teacher_model,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_budget,
        )
    elif algorithm == "seq-ptp-self":
        algo = algo_cls(lit_model, device, autocast_dtype, ar_mode=ar_mode)
    elif algorithm == "seq-ptp-top-p-self":
        algo = algo_cls(lit_model, device, autocast_dtype, ar_mode=ar_mode, threshold=p)
    elif algorithm == "ar":
        algo = algo_cls(lit_model, device, autocast_dtype,
                        temperature=getattr(lit_model, 'temperature', temperature),
                        top_k=getattr(lit_model, 'top_k', top_k),
                        top_p=getattr(lit_model, 'top_p', top_p),
                        teacher_model=teacher_model)
    elif algorithm == "det_ar":
        algo = algo_cls(lit_model, device, autocast_dtype, base_seed=seed)
    elif algorithm == "det_ptp":
        algo = algo_cls(
            lit_model, device, autocast_dtype,
            base_seed=seed,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_budget,
        )
    elif algorithm in ("first-k", "seq-ptp-first-k"):
        algo = algo_cls(
            lit_model, device, autocast_dtype,
            k=k,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_budget,
        )
    elif algorithm in ("seq-ratio", "seq-ptp-ratio"):
        algo = algo_cls(lit_model, device, autocast_dtype, greedy=greedy, raw=raw)
    elif algorithm in ("seq-ratio-k", "seq-ptp-ratio-k"):
        algo = algo_cls(lit_model, device, autocast_dtype, greedy=greedy, k_boost=k, raw=raw)
    elif algorithm in ("seq-ratio-p", "seq-ptp-ratio-p"):
        algo = algo_cls(lit_model, device, autocast_dtype, greedy=greedy, p_boost=p, raw=raw)
    elif algorithm in ("seq-ratio-k-p", "seq-ptp-ratio-k-p"):
        algo = algo_cls(lit_model, device, autocast_dtype, greedy=greedy, k_boost=k, p_boost=p, raw=raw)
    elif algorithm == "ratio-p":
        algo = algo_cls(
            lit_model, device, autocast_dtype,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_budget,
            greedy=greedy,
            p_add=p,
        )
    elif algorithm in ("ratio", "ratio-k"):
        algo = algo_cls(
            lit_model, device, autocast_dtype,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_budget,
            greedy=greedy,
            k_boost=k,
        )
    elif algorithm == "seq-inv-p":
        algo = algo_cls(lit_model, device, autocast_dtype, threshold=p, raw=raw)
    elif algorithm in ("seq-top-k", "seq-ptp-top-k"):
        algo = algo_cls(lit_model, device, autocast_dtype, k=int(k))
    elif algorithm in ("seq-top-p", "seq-ptp-top-p"):
        algo = algo_cls(lit_model, device, autocast_dtype, threshold=p)
    elif algorithm == "ptp-judge":
        algo = algo_cls(lit_model, device, autocast_dtype, referee=referee, judge_prompt=judge_prompt, seed=seed)
    elif algorithm == "seq-ptp-tree":
        algo = algo_cls(lit_model, device, autocast_dtype)
    elif algorithm == "seq-ptp-choice-k":
        algo = algo_cls(lit_model, device, autocast_dtype, k=int(k), mode=choice_mode,
                        phead_checkpoint=phead_checkpoint, branch_fit=branch_fit)
    elif algorithm == "seq-ptp-top-p-choice-k":
        algo = algo_cls(lit_model, device, autocast_dtype, k=int(k), mode=choice_mode, threshold=p,
                        phead_checkpoint=phead_checkpoint, branch_fit=branch_fit)
    elif algorithm == "seq-ptp-top-p-choice-k-self":
        algo = algo_cls(lit_model, device, autocast_dtype, k=int(k), mode=choice_mode,
                        ar_mode=ar_mode, threshold=p, phead_checkpoint=phead_checkpoint,
                        branch_fit=branch_fit)
    elif algorithm == "seq-n-ptp-choice-k":
        algo = algo_cls(lit_model, device, autocast_dtype, k=int(k))
    elif algorithm == "seq-thresh-p":
        algo = algo_cls(lit_model, device, autocast_dtype, threshold=p, fast=fast, raw=raw)
    elif algorithm in ("seq-ptp-thresh-p",):
        algo = algo_cls(lit_model, device, autocast_dtype, threshold=p, raw=raw)
    elif algorithm in ("entr-p", "seq-ptp-entr-p"):
        algo = algo_cls(lit_model, device, autocast_dtype, threshold=p, delta=delta, raw=raw)
    elif algorithm in ("seq-conf-p", "seq-ptp-conf-p"):
        algo = algo_cls(lit_model, device, autocast_dtype, threshold=p)
    elif algorithm in ("thresh-p", "conf-p"):
        algo = algo_cls(
            lit_model, device, autocast_dtype,
            max_tokens_per_proposal=max_tokens_per_proposal,
            total_token_budget=total_budget,
            threshold=p,
        )
    else:
        # Custom algorithms: pass lit_model, device, autocast_dtype
        algo = algo_cls(lit_model, device, autocast_dtype)

    # Dataset
    from datasets import load_dataset as hf_load_dataset

    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("dataset_name") or exp["default_dataset"]
    conversation_keys = list(data_cfg.get("conversation_keys", ["messages", "conversations", "data"]))
    user_roles = list(data_cfg.get("user_roles", DEFAULT_USER_ROLES))
    assistant_roles = list(data_cfg.get("assistant_roles", DEFAULT_ASSISTANT_ROLES))
    max_sequence_length = int(data_cfg.get("max_sequence_length", 2048))

    print(f"Loading dataset: {dataset_name} (split={split})")
    if Path(dataset_name).exists():
        dataset = hf_load_dataset("json", data_files={"train": dataset_name}, split="train")
    else:
        try:
            dataset = hf_load_dataset(dataset_name, split=split)
        except Exception:
            print(f"  '{split}' split not found, falling back to 'train[:1%]'")
            dataset = hf_load_dataset(dataset_name, split="train[:1%]")

    tokenizer = lit_model.model.tokenizer
    if tokenizer is None and teacher_model is not None:
        tokenizer = teacher_model.tokenizer
        lit_model.model.tokenizer = tokenizer  # needed by generate_qm9_ptp
    custom_template = data_cfg.get("chat_template") or exp.get("chat_template")
    if custom_template and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = custom_template

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        ckpt_dir = Path(config["training"].get("ckpt_dir", experiment_dir))
        if save_dir is not None:
            algo_tag = save_dir
        elif algorithm == "first-k":
            algo_tag = f"first-{k}"
        elif algorithm == "seq-ptp-first-k":
            algo_tag = f"seq-ptp-first-{k}"
        elif algorithm == "ratio-k":
            algo_tag = f"ratio-k{'_g' if greedy else ''}-{k}"
        elif algorithm == "seq-ratio-k":
            algo_tag = f"seq-ratio-k{'_g' if greedy else ''}-{k}"
        elif algorithm == "seq-ratio-p":
            algo_tag = f"seq-ratio-p{'_g' if greedy else ''}-{p}"
        elif algorithm == "seq-ratio-k-p":
            algo_tag = f"seq-ratio-k-p{'_g' if greedy else ''}-{k}-{p}"
        elif algorithm == "seq-ptp-ratio":
            algo_tag = f"seq-ptp-ratio{'_g' if greedy else ''}"
        elif algorithm == "seq-ptp-ratio-k":
            algo_tag = f"seq-ptp-ratio-k{'_g' if greedy else ''}-{k}"
        elif algorithm == "seq-ptp-ratio-p":
            algo_tag = f"seq-ptp-ratio-p{'_g' if greedy else ''}-{p}"
        elif algorithm == "seq-ptp-ratio-k-p":
            algo_tag = f"seq-ptp-ratio-k-p{'_g' if greedy else ''}-{k}-{p}"
        elif algorithm == "ratio-p":
            algo_tag = f"ratio-p{'_g' if greedy else ''}-{p}"
        elif algorithm == "ratio":
            algo_tag = f"ratio{'_g' if greedy else ''}"
        elif algorithm == "seq-ratio":
            algo_tag = f"seq-ratio{'_g' if greedy else ''}"
        elif algorithm == "seq-inv-p":
            algo_tag = f"seq-inv-{p}"
        elif algorithm == "seq-top-k":
            algo_tag = f"seq-top-{int(k)}"
        elif algorithm == "seq-ptp-top-k":
            algo_tag = f"seq-ptp-top-{int(k)}"
        elif algorithm == "seq-top-p":
            algo_tag = f"seq-topp-{p}"
        elif algorithm == "seq-ptp-top-p":
            algo_tag = f"seq-ptp-topp-{p}"
        elif algorithm == "ptp-judge":
            slug = re.sub(r'[^a-zA-Z0-9_-]', '_', referee)
            algo_tag = f"ptp-judge-{slug}-{judge_prompt}"
        elif algorithm == "seq-ptp-tree":
            algo_tag = "seq-ptp-tree"
        elif algorithm == "seq-ptp-choice-k":
            algo_tag = f"seq-ptp-choice-{int(k)}"
            if choice_mode != "standard":
                algo_tag += f"-{choice_mode}"
        elif algorithm == "seq-ptp-top-p-choice-k":
            algo_tag = f"seq-ptp-topp-choice-{int(k)}-{p}"
            if choice_mode != "standard":
                algo_tag += f"-{choice_mode}"
        elif algorithm == "seq-ptp-top-p-choice-k-self":
            algo_tag = f"seq-ptp-topp-choice-{int(k)}-self-{p}"
            if choice_mode != "standard":
                algo_tag += f"-{choice_mode}"
            if ar_mode != "aux":
                algo_tag += f"-{ar_mode}"
        elif algorithm == "seq-n-ptp-choice-k":
            algo_tag = f"seq-n-ptp-choice-{int(k)}"
        elif algorithm == "seq-thresh-p":
            algo_tag = f"seq-thresh-{p}{'_fast' if fast else ''}"
        elif algorithm == "seq-ptp-thresh-p":
            algo_tag = f"seq-ptp-thresh-{p}"
        elif algorithm == "entr-p":
            algo_tag = f"entr-{p}-d{delta}"
        elif algorithm == "seq-ptp-entr-p":
            algo_tag = f"seq-ptp-entr-{p}-d{delta}"
        elif algorithm == "seq-conf-p":
            algo_tag = f"seq-conf-{p}"
        elif algorithm == "seq-ptp-conf-p":
            algo_tag = f"seq-ptp-conf-{p}"
        elif algorithm == "thresh-p":
            algo_tag = f"thresh-{p}"
        elif algorithm == "conf-p":
            algo_tag = f"conf-p-{p}"
        elif algorithm == "ptp":
            algo_tag = "ptp" if partial_mode == "count" else f"ptp-{partial_mode}"
        elif algorithm == "ptp-top-p-choice-k-self":
            algo_tag = f"ptp-topp-choice-{int(k)}-self-{p}"
        elif algorithm == "seq-ptp-self":
            algo_tag = "seq-ptp-self" if ar_mode == "aux" else f"seq-ptp-self-{ar_mode}"
        elif algorithm == "seq-ptp-top-p-self":
            algo_tag = f"seq-ptp-topp-self-{p}" if ar_mode == "aux" else f"seq-ptp-topp-self-{p}-{ar_mode}"
        else:
            algo_tag = algorithm
        _raw_algorithms = {
            "seq-thresh-p", "seq-ptp-thresh-p",
            "entr-p", "seq-ptp-entr-p",
            "seq-ratio", "seq-ratio-k", "seq-ratio-p", "seq-ratio-k-p",
            "seq-ptp-ratio", "seq-ptp-ratio-k", "seq-ptp-ratio-p", "seq-ptp-ratio-k-p",
            "seq-inv-p",
        }
        if raw and algorithm in _raw_algorithms:
            algo_tag += "_raw"
        output_dir = ckpt_dir / "inference" / f"tmp_{algo_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(algo, "set_cache_path"):
        algo.set_cache_path(output_dir / "human_cache.json")

    run_config = {
        "experiment_dir": str(experiment_dir),
        "checkpoint": str(ckpt_path),
        "algorithm": algorithm,
        "max_tokens_per_proposal": max_tokens_per_proposal,
        "total_token_budget": total_budget,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "n_examples": n_examples,
        "seed": seed,
        "split": split,
        "dataset": dataset_name,
        "compile": compile,
        "only_first_turn": only_first_turn,
        "suppress_cot": suppress_cot,
        "k": k,
        "delta": delta,
        "choice_mode": choice_mode,
        "greedy": greedy,
        "p": p,
        "english_only": english_only,
        "timestamp": ts,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Results will be saved to: {output_dir}")

    results_path = output_dir / "results.jsonl"
    if exp["mode"] == "text":
        pair_iter = iter_text_pairs(dataset, tokenizer, exp["text_column"], max_sequence_length)
    elif exp["mode"] == "spec_bench":
        pair_iter = iter_spec_bench_pairs(dataset, tokenizer, max_sequence_length)
    else:
        pair_iter = iter_prompt_completion_pairs(
            dataset, tokenizer, conversation_keys, user_roles, assistant_roles,
            max_sequence_length, only_first_turn=only_first_turn, suppress_cot=suppress_cot,
            english_only=english_only,
        )

    from tqdm import tqdm

    n_done = 0
    # Running averages for postfix stats
    avg_ms: float | None = None
    avg_cpc: float | None = None
    avg_gen_lp: float | None = None
    avg_p_value: float | None = None
    alpha = 0.05  # EMA smoothing factor

    def _ema(prev, val):
        return val if prev is None else (1 - alpha) * prev + alpha * val

    pbar = tqdm(
        total=n_examples,
        desc=f"inference ({algorithm})",
        unit="ex",
        dynamic_ncols=True,
    )

    with open(results_path, "w") as out_f, pbar:
        for example_id, prompt_ids, ref_ids, prompt_str, ref_str in pair_iter:
            if n_examples is not None and n_done >= n_examples:
                break

            prompt_ids = prompt_ids.to(device)
            ref_ids = ref_ids.to(device)

            if hasattr(algo, "set_ref"):
                algo.set_ref(ref_ids)
            completion_ids, gen_metrics = algo.generate(prompt_ids, max_new_tokens)

            gen_lp = teacher_log_prob(
                lit_model, completion_ids, prompt_ids.shape[1], autocast_dtype, device,
                teacher_model=teacher_model,
            )
            ref_full_ids = torch.cat([prompt_ids, ref_ids], dim=1)
            ref_lp = teacher_log_prob(
                lit_model, ref_full_ids, prompt_ids.shape[1], autocast_dtype, device,
                teacher_model=teacher_model,
            )

            generated_str = tokenizer.decode(
                completion_ids[0, prompt_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            record = {
                "example_id": example_id,
                "prompt": prompt_str,
                "reference": ref_str,
                "generated": generated_str,
                "metrics": {
                    **gen_metrics,
                    "gen_lp_mean": gen_lp["mean"],
                    "gen_lp_median": gen_lp["median"],
                    "gen_lp_z_num": gen_lp["z_num"],
                    "gen_lp_z_var": gen_lp["z_var"],
                    "gen_lp_z_stat": gen_lp["z_stat"],
                    "gen_lp_p_value": gen_lp["p_value"],
                    "gen_lp_outlier_perc": gen_lp["n_outlier_tokens"] / max(gen_metrics.get("n_generated_tokens", 1), 1) * 100,
                    "ref_lp_mean": ref_lp["mean"],
                    "ref_lp_median": ref_lp["median"],
                    "n_ref_tokens": ref_ids.shape[1],
                    "n_prompt_tokens": prompt_ids.shape[1],
                },
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            n_done += 1
            ms = gen_metrics.get("ms_per_token", float("nan"))
            cpc = gen_metrics.get("correct_per_call", float("nan"))
            avg_ms = _ema(avg_ms, ms)
            avg_cpc = _ema(avg_cpc, cpc)
            avg_gen_lp = _ema(avg_gen_lp, gen_lp["mean"])
            pv = gen_lp["p_value"]
            if not math.isnan(pv):
                avg_p_value = _ema(avg_p_value, pv)

            pbar.set_postfix(
                ms_tok=f"{avg_ms:.1f}",
                tok_call=f"{avg_cpc:.2f}",
                gen_lp=f"{avg_gen_lp:.2f}",
                p_val=f"{avg_p_value:.3f}" if avg_p_value is not None else "nan",
                refresh=False,
            )
            pbar.update(1)

    print(f"\nDone. {n_done} examples written to {results_path}")


def _parse_args():
    parser = ArgumentParser(
        description="Batch inference over a chat dataset with swappable inference algorithms."
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--algorithm", choices=list(ALGORITHMS), default="ptp",
        help="Inference algorithm (default: ptp)",
    )
    parser.add_argument("--max-tokens-per-proposal", type=int, default=20)
    parser.add_argument("--total-token-budget", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Max tokens to generate (default: from --experiment config)")
    parser.add_argument("--experiment", choices=list(EXPERIMENT_CONFIGS), default="qwen",
                        help="Experiment preset encoding dataset/mode/defaults (default: qwen)")
    parser.add_argument("--n-examples", type=int, default=100, help="Stop after N examples (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-dir", type=str, default=None, help="Override the algo tag in the output path (results go to <ckpt_dir>/inference/tmp_<save-dir>)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: test)")
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument(
        "--all-turns", action="store_true", default=False,
        help="Evaluate every assistant turn (default: only first turn per conversation)",
    )
    parser.add_argument(
        "--think", action="store_true", default=False,
        help="Allow chain-of-thought (default: suppress CoT by injecting empty think block)",
    )
    parser.add_argument(
        "--k", type=float, default=2,
        help="For --algorithm first_k: tokens to accept per step (fractional values round up/down randomly). For --algorithm ratio-k: acceptance boost multiplier applied to p in the ratio. For --algorithm seq-ptp-choice-k: number of independent candidate lines proposed per step (default: 2)",
    )
    parser.add_argument(
        "--greedy", action="store_true", default=False,
        help="For --algorithm ratio / ratio-k: student proposes via argmax (q=1) instead of sampling (default: off)",
    )
    parser.add_argument(
        "--no-english-only", action="store_true", default=False,
        help="Include non-English prompts (default: English-only via ASCII ratio filter)",
    )
    parser.add_argument(
        "--p", type=float, default=0.7,
        help="For --algorithm thresh-p: teacher probability threshold for acceptance (default: 0.7)",
    )
    parser.add_argument(
        "--delta", type=float, default=0.3,
        help="For --algorithm entr-p / seq-ptp-entr-p: coefficient of the "
             "entropy-adaptive bound min(--p, delta * exp(-H)) (default: 0.3). "
             "With --p as epsilon, the defaults (0.09, 0.3) reproduce Medusa's "
             "typical acceptance.",
    )
    parser.add_argument(
        "--fast", action="store_true", default=False,
        help="Enable fast track in seq-thresh variants (skips teacher call)",
    )
    parser.add_argument(
        "--raw", action="store_true", default=False,
        help="Use raw teacher logits (no temperature/top-k/top-p) for acceptance in thresh-p and ratio algorithms",
    )
    parser.add_argument(
        "--referee", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model used as LLM referee for --algorithm ptp-judge (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--prompt", choices=["long", "mid", "short"], default="mid",
        help="Judge prompt style for --algorithm ptp-judge: long (maximize length), "
             "mid (longest while still coherent, default), short (most coherent, not necessarily longest)",
    )
    parser.add_argument(
        "--choice-mode",
        choices=["standard", "balanced", "optimal", "optimal-25", "optimal-50", "optimal-75",
                 "bayes-conjugate", "bayes-p", "phead", "oracle"],
        default="standard",
        help="For --algorithm seq-ptp-choice-k: 'standard' (default) makes all k strands "
             "n_prop deep; 'balanced' makes 1 strand n_prop deep, k//4 strands n_prop//2 deep, "
             "and the rest n_prop//4 deep; 'optimal' picks the strand-length profile that "
             "maximizes expected accepted tokens for the same total node budget "
             "(k * tokens_per_student_call), per the branching-process model in scratch/correct.md; "
             "'optimal-25'/'optimal-50'/'optimal-75' spend that percentage of the budget on full "
             "n_prop-deep strands and hand the rest to the same optimizer; 'bayes-conjugate' and "
             "'bayes-p' are online per-question variants of 'optimal' that re-estimate (p, rho) "
             "[resp. just p] from this question's own observed rounds instead of the fixed pooled "
             "v3 constants -- 'bayes-conjugate' via exact Beta-Binomial conjugacy, 'bayes-p' via a "
             "grid posterior seeded from the per-question hierarchical fit in "
             "scratch/joint_per_question_results.json; 'phead' is like 'optimal' but p is predicted "
             "per-round directly from the AR context hidden state by a fine-tuned P-head (see "
             "src/ptp/p_head.py, same sidecar as --partial-mode phead for --algorithm ptp), pooled "
             "pi0/rho -- requires --phead-checkpoint; 'oracle' uses the precomputed per-question "
             "free-MLE p_hat_i from scratch/joint_per_question_results.json (nearest-K fit, same "
             "lookup 'bayes-p' uses to seed its prior) -- an oracle upper bound for 'bayes-p'/"
             "'bayes-conjugate'/'phead' since it sees the whole question's data instead of an "
             "online or predicted estimate, mirrors --algorithm ptp's 'beta_oracle' partial_mode.",
    )
    parser.add_argument(
        "--partial-mode",
        choices=["count", "hist", "geom", "beta", "phead", "chead", "beta_oracle"],
        default="count",
        help="For --algorithm ptp: how the reward matrix H(k) (estimated # correct tokens "
             "given k proposed tokens) is computed. 'count' (default): H(k) = k. "
             "'hist': H(k) = E[min(G, k)] under the empirical histogram hist_base. "
             "'geom': H(k) = E[min(G, k)] under the shifted-geometric model "
             "G = 1 + Geometric(p), p=0.698 fitted over all questions (see scratch/correct.md). "
             "'beta': like 'geom', but p is the posterior mean of a population Beta prior "
             "(jointly fit over all questions) updated online with the #correct-per-call "
             "samples seen so far this generation. "
             "'phead': p is predicted per-call from context by a fine-tuned P-head "
             "(see scripts/finetune.py); requires --phead-checkpoint. "
             "'chead': like 'hist', but the full 21-class distribution over #correct is "
             "predicted per-call from context by a fine-tuned C-head (non-parametric; see "
             "scripts/finetune.py --head-type c); requires --chead-checkpoint. "
             "'beta_oracle': p is the precomputed per-question free MLE from "
             "scratch/joint_per_question_results.json (an oracle upper bound for 'beta').",
    )
    parser.add_argument(
        "--phead-checkpoint", type=Path, default=None,
        help="Path to a p_head sidecar checkpoint (e.g. <ckpt>_phead_ft.ckpt from "
             "scripts/finetune.py), required for --partial-mode phead.",
    )
    parser.add_argument(
        "--chead-checkpoint", type=Path, default=None,
        help="Path to a c_head sidecar checkpoint (e.g. <ckpt>_chead_ft.ckpt from "
             "scripts/finetune.py --head-type c), required for --partial-mode chead.",
    )
    parser.add_argument(
        "--ar-mode", choices=["aux", "tok"], default="aux",
        help="For --algorithm seq-ptp-self / seq-ptp-top-p-self: 'aux' (default) verifies "
             "with the current gated-LoRA aux-probe mechanism (self-speculative). 'tok' "
             "verifies with a plain ungated causal forward through this same model instead "
             "— the same fallback plain seq-ptp uses (self.model.inference_forward), just "
             "with no separate teacher model.",
    )
    parser.add_argument(
        "--branch-fit", choices=list(SeqPTPChoiceKInference._BRANCH_FITS), default="vicuna",
        help="For --algorithm seq-ptp-choice-k / seq-ptp-top-p-choice-k / "
             "seq-ptp-top-p-choice-k-self: which checkpoint's pooled (pi0, p, rho) v3 "
             "branching fit to use for 'optimal*'/'bayes-*'/'phead'/'oracle' choice-modes "
             "(default: 'vicuna', the original fit). Pass 'vicuna_old' when running "
             "against that checkpoint instead, to avoid a stale-fit mismatch.",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.exists():
        parser.error(f"Experiment directory not found: {experiment_dir}")
    if not (experiment_dir / "train.yaml").exists():
        parser.error(f"train.yaml not found in {experiment_dir}")

    exp = EXPERIMENT_CONFIGS[args.experiment]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else exp["max_new_tokens"]
    english_only = exp["english_only"] if not args.no_english_only else False

    main(
        experiment_dir=experiment_dir,
        checkpoint=args.checkpoint,
        algorithm=args.algorithm,
        max_tokens_per_proposal=args.max_tokens_per_proposal,
        total_token_budget=args.total_token_budget,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=max_new_tokens,
        n_examples=args.n_examples,
        seed=args.seed,
        output_dir=args.output_dir,
        save_dir=args.save_dir,
        split=args.split,
        compile=not args.no_compile,
        only_first_turn=not args.all_turns,
        suppress_cot=not args.think,
        k=args.k,
        greedy=args.greedy,
        english_only=english_only,
        p=args.p,
        delta=args.delta,
        fast=args.fast,
        raw=args.raw,
        experiment=args.experiment,
        referee=args.referee,
        judge_prompt=args.prompt,
        choice_mode=args.choice_mode,
        partial_mode=args.partial_mode,
        phead_checkpoint=args.phead_checkpoint,
        chead_checkpoint=args.chead_checkpoint,
        ar_mode=args.ar_mode,
        branch_fit=args.branch_fit,
    )


if __name__ == "__main__":
    _parse_args()
