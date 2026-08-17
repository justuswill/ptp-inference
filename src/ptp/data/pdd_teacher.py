"""Wrapper around the pdd AR DiT checkpoint for use as a teacher in ptp pregeneration."""
import sys
import types
import torch
from pathlib import Path


class PddARTeacher:
    """
    Loads a pdd AR DiT checkpoint and exposes a generate() interface compatible
    with ptp's pregenerate.py (same contract as HuggingFace's generate with
    output_scores=True, return_dict_in_generate=True).
    """

    def __init__(self, ckpt_path: str, pdd_repo: str, attn_backend: str = 'flex'):
        if str(pdd_repo) not in sys.path:
            sys.path.insert(0, str(pdd_repo))

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'yairschiff/qm9-tokenizer', trust_remote_code=True
        )

        self._patch_tokenizer_modules()
        self._patch_trie()

        import algo as pdd_algo
        import models.dit_flex as _dit_flex
        sys.modules.setdefault('models.dit_sdpa', _dit_flex)

        raw = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
        config = raw['hyper_parameters']['config']
        config.model.attn_backend = attn_backend

        self.model = pdd_algo.AR.load_from_checkpoint(
            str(ckpt_path), map_location='cpu', tokenizer=self.tokenizer, config=config,
            weights_only=False,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    def _patch_tokenizer_modules(self):
        real_key = next(
            (k for k in sys.modules if 'qm9' in k.lower() and k.endswith('.tokenizer')),
            None,
        )
        if real_key is None:
            return
        real_mod = sys.modules[real_key]
        parts = real_key.split('.')
        hash_seg = next(
            (p for p in parts if len(p) >= 8 and all(c in '0123456789abcdef' for c in p)),
            None,
        )
        if hash_seg is None:
            return
        for pkg in ('transformers_modules', 'transformers_modules.yairschiff'):
            if pkg not in sys.modules:
                sys.modules[pkg] = types.ModuleType(pkg)
        hyphen_pkg = types.ModuleType('transformers_modules.yairschiff.qm9-tokenizer')
        hyphen_pkg.__path__ = []
        hyphen_pkg.__package__ = 'transformers_modules.yairschiff.qm9-tokenizer'
        sys.modules['transformers_modules.yairschiff.qm9-tokenizer'] = hyphen_pkg
        hash_key = f'transformers_modules.yairschiff.qm9-tokenizer.{hash_seg}'
        sys.modules[hash_key] = real_mod
        sys.modules[f'{hash_key}.tokenizer'] = real_mod

    @staticmethod
    def _patch_trie():
        import transformers.tokenization_utils as _tu
        if hasattr(_tu, 'Trie'):
            return
        for attr in ('Trie',):
            for mod_name in (
                'transformers.tokenization_utils_python',
                'transformers.tokenization_python',
                'transformers.tokenization_utils_base',
            ):
                try:
                    mod = __import__(mod_name, fromlist=[attr])
                    setattr(_tu, attr, getattr(mod, attr))
                    break
                except (ImportError, AttributeError):
                    pass

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        max_length: int = 32,
        do_sample: bool = True,
        pad_token_id=None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        output_scores: bool = True,
        return_dict_in_generate: bool = True,
        num_return_sequences: int = 1,
        use_cache: bool = True,
        **kwargs,
    ):
        device = input_ids.device
        eos_id = self.tokenizer.eos_token_id or getattr(self.tokenizer, "sep_token_id", None)
        prompt_len = input_ids.shape[1]
        max_new_tokens = max(max_length - prompt_len, 1)

        x = input_ids.expand(num_return_sequences, -1).clone()
        done = torch.zeros(num_return_sequences, dtype=torch.bool, device=device)

        all_tokens: list[torch.Tensor] = []
        all_scores: list[torch.Tensor] = []

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = self.model._forward_backbone_ar(x)[:, -1, :].float()

                if hasattr(self.model, 'mask_index'):
                    logits[:, self.model.mask_index] = -1e6

                if temperature != 1.0:
                    logits = logits / temperature

                if top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, -float('inf'))

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = (cum - torch.softmax(sorted_logits, dim=-1)) > top_p
                    sorted_logits[remove] = -float('inf')
                    logits = logits.scatter(1, sorted_idx, sorted_logits)

                all_scores.append(logits.cpu())

                probs = torch.softmax(logits, dim=-1)
                tokens = torch.multinomial(probs, 1).squeeze(1) if do_sample else logits.argmax(-1)
                tokens[done] = eos_id
                done = done | (tokens == eos_id)
                all_tokens.append(tokens.cpu())
                x = torch.cat([x, tokens.unsqueeze(1)], dim=1)
                if done.all():
                    break

        # Pad remaining steps to max_new_tokens
        vocab = all_scores[0].shape[-1] if all_scores else self.tokenizer.vocab_size
        while len(all_tokens) < max_new_tokens:
            all_tokens.append(torch.full((num_return_sequences,), eos_id))
            all_scores.append(torch.zeros(num_return_sequences, vocab))

        comp_ids = torch.stack(all_tokens, dim=1)                     # [N, T]
        sequences = torch.cat(
            [input_ids.expand(num_return_sequences, -1).cpu(), comp_ids], dim=1
        ).to(device)
        scores = [s.to(device) for s in all_scores]

        class _Out:
            pass

        out = _Out()
        out.sequences = sequences
        out.scores = scores
        return out
