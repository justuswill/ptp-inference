import contextlib
import warnings

from lightning.pytorch import LightningModule
from typing import Literal, List, Mapping, Any

from torch import Tensor

from torch.nn.attention.flex_attention import create_block_mask

from ptp.attention import (
    make_ar_mask_mod,
    make_completion_mask_mod,
)
from ptp.data.collate import IGNORE_INDEX
from ptp.data.utils import predict_bin_edges
from ptp.transformer import TransformerModel, MixedTransformerModel
import torch
import numpy as np
from transformers.cache_utils import DynamicCache, StaticCache

from ptp.utils import instantiate


class ParallelSamplingLightningModule(LightningModule):
    def __init__(self, optim_cfg: dict = None,
                 model_cfg: Mapping[str, Any] | None = None, model: MixedTransformerModel | None = None,
                 completion_loss_weight: float = 1.0,
                 completion_gamma: float = 1.0,
                 pbar_metrics: List[str] | None = None,
                 tokens_per_student_call: int = 20,
                 temperature: float | None = None,
                 top_k: int | None = None,
                 top_p: float | None = None,
                 hist_base: list[float] | None = None,
                 self_mode: bool = False):
        if pbar_metrics is None:
            pbar_metrics = ['correct']
            if completion_loss_weight > 0.0:
                pbar_metrics.append('l_completion')
        super().__init__()
        if (model is None) == (model_cfg is None):
            raise ValueError("Exactly one of model and model_cfg must be provided, got "
                             f"model: {model is None=}, model_cfg: {model_cfg is None=}")
        self.model_cfg = model_cfg
        self.model: MixedTransformerModel | None = model

        self.optim_cfg = optim_cfg
        self.completion_loss_weight = completion_loss_weight
        self.completion_gamma = completion_gamma

        self.pbar_metrics = pbar_metrics

        self.tokens_per_student_call = tokens_per_student_call
        self.total_token_budget: int | None = None
        # Reward matrix used by proposals(); called as H_fn(metrics) before each proposals()
        # call so it can depend on samples seen so far this generation (e.g. partial_mode="beta").
        # H(k) = k by default; override via PTPInference.compute_H.
        self.H_fn = lambda metrics: torch.arange(21).double()
        # If True, run the backbone forward pass (AR + completion) under torch.no_grad()
        # to avoid retaining activations for backward — set when self.model is fully
        # frozen (see PHeadLightningModule.freeze_base()); no effect otherwise.
        self.freeze_backbone_forward: bool = False
        # AR hidden state right before the current call's proposals, captured in generate()
        # when self.model.output_hidden_states is set (see PTPInference partial_mode="phead").
        self._last_context_hidden: torch.Tensor | None = None
        # If True, forward()'s AR/context pass (ar_forward) applies LoRA everywhere too,
        # not just the aux completion window -- self-speculative training, matching
        # inference's _full_lora_mode variants. See PHeadLightningModule/CHeadLightningModule.
        self.self_mode = self_mode
        # If set, forward()'s correct_counts treats a completion token as correct when it
        # falls within the model's own top-p nucleus at that position (not just exact
        # argmax match) -- see compute_sequence_metrics.
        self.nucleus_threshold: float | None = None
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.hist_base = torch.tensor(hist_base, dtype=torch.float64) if hist_base is not None else None
        self._hist_accumulator: list[torch.Tensor] = []
        self.checkpoint_save_mode: str = 'full'

    def configure_model(self) -> None:
        if self.model is None:
            self.model = MixedTransformerModel(**self.model_cfg)

    def enter_inference_mode(self, gate_window: int):
        """Switch to inference mode: fuse LoRA weights into GatedLinearLoraMerged."""
        self.model.enter_inference_mode(gate_window)

    def exit_inference_mode(self):
        """Switch back to training mode: restore GatedLinearLora layers."""
        self.model.exit_inference_mode()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Rename any keys starting with "student." to "model."
        renamed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("student."):
                key = key.replace("student.", "model.", 1)
            if ".u_adapter." in key and ".u_embed." not in key:
                key = key.replace(".u_adapter.", ".u_embed.", 1)
            renamed_state_dict[key] = value

        # Adapter-only checkpoints intentionally omit base-model parameters.
        # Auto-relax strict loading for that case while preserving strict behavior
        # for full checkpoints.
        adapter_only_state = bool(renamed_state_dict) and all(
            self._is_adapter_state_key(key) for key in renamed_state_dict.keys()
        )
        if strict and adapter_only_state:
            warnings.warn(
                "Detected adapter-only state dict; loading with strict=False to allow missing base-model weights.",
                RuntimeWarning,
            )
            strict = False

        if not strict:
            return super().load_state_dict(renamed_state_dict, strict=False, assign=assign)

        try:
            return super().load_state_dict(renamed_state_dict, strict=True, assign=assign)
        except RuntimeError:
            pass

        result = super().load_state_dict(renamed_state_dict, strict=False, assign=assign)
        allowed_missing_prefixes = (
            "model.u_scale_embed.",
            "model.time_embed.",
        )
        disallowed_missing = [
            key for key in result.missing_keys
            if not key.startswith(allowed_missing_prefixes)
            and ".adaLN_modulation." not in key
        ]
        if disallowed_missing or result.unexpected_keys:
            problems = []
            if disallowed_missing:
                problems.append(f"Missing key(s): {disallowed_missing}")
            if result.unexpected_keys:
                problems.append(f"Unexpected key(s): {result.unexpected_keys}")
            raise RuntimeError("Error(s) in loading state_dict for "
                               f"{self.__class__.__name__}: " + "; ".join(problems))
        return result

    @staticmethod
    def _is_adapter_state_key(key: str) -> bool:
        # Keep LoRA parameters and project-specific auxiliary embedding weights.
        return (
            ('lora_' in key)
            or ('.u_embed.' in key)
            or ('.u_scale_embed.' in key)
            or ('.time_embed.' in key)
            or ('.adaLN_modulation.' in key)
        )

    def _adapter_state_dict(self, state_dict: Mapping[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in state_dict.items() if self._is_adapter_state_key(k)}

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        state_dict = checkpoint.get('state_dict')
        if state_dict is None or self.checkpoint_save_mode == 'full':
            return
        if self.checkpoint_save_mode == 'adapter_only':
            adapter_state_dict = self._adapter_state_dict(state_dict)
            checkpoint['state_dict'] = adapter_state_dict

    def configure_optimizers(self):
        config = self.optim_cfg
        optimizer = {}
        active_parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer["optimizer"] = torch.optim.AdamW(
            active_parameters,
            lr=config["lr"],
        )
        if config.get("lr_scheduler", None) is not None:
            optimizer["lr_scheduler"] = instantiate(
                config["lr_scheduler"],
                optimizer=optimizer["optimizer"]
            )
        if config.get("lr_warmup", 0) > 0:
            warmup_steps = config["lr_warmup"]
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps  # Linear warmup
                else:
                    return 1.0  # keep lr constant after warmup

            optimizer["lr_scheduler"] = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer["optimizer"], lr_lambda),
                "interval": "step",
                "frequency": 1,
            }
        return optimizer

    def _log_metrics(self, prefix, metrics, sync_dist=False):
        pbar_metrics = {}
        plot_metrics = {}
        other_metrics = {}
        for key, v in metrics.items():
            prefixed_key = f"{prefix}/{key}"
            if key in self.pbar_metrics:
                pbar_metrics[prefixed_key] = v
            elif (isinstance(v, float) or isinstance(v, int)) or v.shape == torch.Size([]):
                other_metrics[prefixed_key] = v
            else:
                # for idx, vi in enumerate(v):
                #    other_metrics[f"{prefixed_k}_{idx}"] = vi
                # plot_metrics[prefixed_key] = v
                pass
        self.log_dict(pbar_metrics, prog_bar=True, sync_dist=sync_dist)
        for k, v in plot_metrics.items():
            if v.ndim == 1 and self.trainer.logger is not None:
                # Convert to CPU list to prevent GPU memory from being held by wandb
                v_cpu = v.detach().cpu().tolist()
                self.trainer.logger.log_table(key=k, data=list(enumerate(v_cpu)), columns=['position', k])
        self.log_dict(other_metrics, prog_bar=False, sync_dist=sync_dist)

    def training_step(self, batch, batch_idx=None):
        metrics = self.forward(batch, batch_idx)

        loss = metrics['loss']

        metrics_logged = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in metrics.items()}
        metrics_logged['lr'] = self.optimizers().param_groups[0]['lr']
        self._log_metrics('train', metrics_logged)

        return loss

    def on_validation_epoch_start(self):
        self._hist_accumulator = []

    def validation_step(self, batch, batch_idx=None):
        metrics = self.forward(batch, batch_idx, eval=True)
        if 'correct_counts' in metrics:
            self._hist_accumulator.append(metrics['correct_counts'].cpu())
        # Detach all validation metrics (no gradients needed)
        metrics_logged = {k: v.detach() if isinstance(v, torch.Tensor) else v
                         for k, v in metrics.items()}
        self._log_metrics('val', metrics_logged, sync_dist=True)

    def on_validation_epoch_end(self):
        hist = torch.zeros(21, dtype=torch.float64)
        if self._hist_accumulator:
            counts = torch.cat(self._hist_accumulator).clamp(max=20)
            hist = torch.bincount(counts, minlength=21).double()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            hist = hist.to(self.device)
            torch.distributed.all_reduce(hist)
            hist = hist.cpu()
        if hist.sum() > 0:
            self.hist_base = hist / hist.sum()
        self._hist_accumulator = []

    def adapt_logits(self, logits):
        if self.temperature is not None and self.temperature != 1.0:
            logits = logits / self.temperature
        if self.top_k is not None and self.top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, k=self.top_k)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(-1, top_k_indices, top_k_logits)
            logits = mask
        if self.top_p is not None and self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        return logits

    def adapt_p(self, p):
        k = self.top_k if (self.top_k is not None and self.top_k > 0) else p.shape[-1]
        top_k_probs, top_k_indices = torch.topk(p, k=k, dim=-1)
        # remove additional tokens if top_p is more restrictive
        remove = (top_k_probs.cumsum(dim=-1) - top_k_probs) > self.top_p
        top_k_probs = top_k_probs.masked_fill(remove, 0.0)
        # renormalize; sort by token index so CDF bins match vocab-sorted original
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sort_idx = top_k_indices.argsort(dim=-1)
        return top_k_probs.gather(-1, sort_idx), top_k_indices.gather(-1, sort_idx)

    def new_adapt_p(self, p):
        k = self.top_k if (self.top_k is not None and self.top_k > 0) else p.shape[-1]
        top_k_probs, top_k_indices = torch.topk(p, k=k, dim=-1)
        if self.top_p is not None and self.top_p < 1.0:
            remove = (top_k_probs.cumsum(dim=-1) - top_k_probs) > self.top_p
            top_k_probs = top_k_probs.masked_fill(remove, 0.0)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sort_idx = top_k_indices.argsort(dim=-1)
        return top_k_probs.gather(-1, sort_idx), top_k_indices.gather(-1, sort_idx)

    def forward(self, batch, batch_idx=None, eval=False, return_outputs=False):
        # Unmodified student to get kv-cache
        input_ids = batch['input_ids']
        input_mask = batch['input_mask']

        # Build attention mask and lookup targets for arbitrary position completions
        completion_starts:   Tensor = batch['completion_starts']        # (B, N)
        completion_doc_ids:  Tensor = batch.get('completion_doc_ids')   # (B, N) or None
        doc_ids:      Tensor = batch.get('doc_ids')                     # (B, S) or None
        doc_starts:   Tensor = batch.get('doc_starts')                  # (B, D) or None
        doc_lengths:  Tensor = batch.get('doc_lengths')                 # (B, D) or None
        completion_length: int = batch['completion_length']
        left_bin_edges = batch.get("bin_edges_left")
        right_bin_edges = batch.get("bin_edges_right")

        # AR block mask: causal + document-isolated for packed sequences
        if doc_ids is not None:
            ar_block_mask = create_block_mask(
                make_ar_mask_mod(doc_ids),
                B=input_ids.shape[0], H=None,
                Q_LEN=input_ids.shape[1], KV_LEN=input_ids.shape[1],
                device=input_ids.device,
            )
        else:
            ar_block_mask = None  # standard causal

        backbone_ctx = torch.no_grad() if self.freeze_backbone_forward else contextlib.nullcontext()
        with backbone_ctx:
            ar_outputs = None
            if left_bin_edges is None or right_bin_edges is None:
                left_bin_edges, right_bin_edges, ar_outputs = predict_bin_edges(
                    input_ids, input_mask=ar_block_mask,
                    model=lambda input_ids, attention_mask=None: self.model.ar_forward(
                        input_ids, attention_mask, self_mode=self.self_mode),
                    adapt_logits=self.adapt_logits if (self.temperature is not None or self.top_k is not None or self.top_p is not None) else None,
                )
                left_bin_edges = left_bin_edges.detach()
                right_bin_edges = right_bin_edges.detach()
            nested_batch = self.prepare_nested_batch(
                input_ids, input_mask,
                completion_length, completion_starts,
                left_bin_edges, right_bin_edges,
                eval,
                doc_ids=doc_ids,
                completion_doc_ids=completion_doc_ids,
                doc_starts=doc_starts,
                doc_lengths=doc_lengths,
            )
            attention_mask, auxiliaries, completion_ids, position_ids = nested_batch

            batch_size = input_ids.shape[0]
            num_completions = completion_starts.shape[1]
            ar_outputs, completion_outputs = self.model(
                input_ids=input_ids,
                input_mask=input_mask,
                ar_outputs=ar_outputs,
                auxiliaries=auxiliaries.reshape(batch_size, -1),
                auxiliary_position_ids=position_ids.reshape(batch_size, -1),
                auxiliary_mask=attention_mask,
            )

        completion_logits = completion_outputs.logits
        loss_batch_size = batch_size * num_completions * completion_length
        if self.completion_gamma == 1.0:
            completion_loss = torch.nn.functional.cross_entropy(
                completion_logits.reshape(loss_batch_size, -1),
                completion_ids.reshape(loss_batch_size),
                ignore_index=IGNORE_INDEX
            )
        else:
            # Exponential position weights: position k gets weight gamma^k
            gamma_powers = self.completion_gamma ** torch.arange(
                completion_length, device=completion_logits.device, dtype=completion_logits.dtype)
            per_token_loss = torch.nn.functional.cross_entropy(
                completion_logits.reshape(loss_batch_size, -1),
                completion_ids.reshape(loss_batch_size),
                ignore_index=IGNORE_INDEX,
                reduction='none',
            ).reshape(batch_size * num_completions, completion_length)
            valid = completion_ids.reshape(batch_size * num_completions, completion_length) != IGNORE_INDEX
            completion_loss = (per_token_loss * gamma_powers).sum() / \
                (gamma_powers * valid).sum().clamp(min=1e-8)

        # Post-process metrics on flattened completions
        eval_base_shape = (batch_size * num_completions, completion_length)
        completion_logits = completion_logits.reshape(*eval_base_shape, -1)
        metrics = self.compute_sequence_metrics(
            completion_ids.reshape(*eval_base_shape),
            completion_logits.argmax(dim=-1),
            include_outputs=return_outputs,
            student_logits=completion_logits if self.nucleus_threshold is not None else None,
            nucleus_threshold=self.nucleus_threshold,
        )

        loss = 0.0
        if self.completion_loss_weight > 0.0:
            loss = loss + self.completion_loss_weight * completion_loss
        extra = self._compute_extra_losses(ar_outputs, completion_starts, completion_length,
                                            metrics, batch_size, num_completions)
        loss = loss + extra.get('loss', 0.0)
        metrics.update(extra.get('metrics', {}))
        metrics['loss'] = loss
        metrics['l_completion'] = completion_loss
        metrics['num_completions'] = num_completions
        return metrics

    def _compute_extra_losses(self, ar_outputs, completion_starts, completion_length,
                               metrics, batch_size, num_completions) -> dict:
        """Hook for subclasses to add extra loss terms; see PHeadLightningModule."""
        return {}

    def _make_completion_positions(self, completion_starts: Tensor, completion_length: int,
                                   seq_len: int, device) -> tuple[Tensor, Tensor, Tensor]:
        """Compute position_ids, valid_mask, and safe_positions for all completions."""
        starts = completion_starts.to(device)  # (B, N)
        offsets = torch.arange(completion_length, device=device, dtype=torch.long)  # (L,)
        position_ids = starts[:, :, None] + offsets[None, None, :]  # (B, N, L)
        valid_mask = position_ids < seq_len
        safe_positions = position_ids.clamp(max=seq_len - 1)
        return position_ids, valid_mask, safe_positions

    def _make_completion_block_mask(self, starts: Tensor, completion_length: int, seq_len: int,
                                    batch_size: int, device,
                                    doc_ids: Tensor | None, completion_doc_ids: Tensor | None,
                                    doc_starts: Tensor | None, doc_lengths: Tensor | None):
        """Build the flex_attention block mask for the nested completion batch."""
        num_completions = starts.shape[1]
        total_completion_length = num_completions * completion_length
        # KV layout: [prompt S tokens | completion N*L tokens]
        return create_block_mask(
            make_completion_mask_mod(
                completion_starts=starts,
                completion_doc_ids=completion_doc_ids,
                doc_ids=doc_ids,
                doc_starts=doc_starts,
                doc_lengths=doc_lengths,
                seq_len=seq_len,
                completion_length=completion_length,
            ),
            B=batch_size, H=None,
            Q_LEN=total_completion_length,
            KV_LEN=seq_len + total_completion_length,
            device=device,
        )

    def _gather_completion_ids(self, input_ids: Tensor, safe_positions: Tensor,
                               valid_mask: Tensor, num_completions: int,
                               doc_ids: Tensor | None, completion_doc_ids: Tensor | None,
                               starts: Tensor) -> Tensor:
        """Gather target token IDs, masking out-of-bounds and cross-document positions."""
        completion_ids = torch.gather(
            input_ids[:, None, :].expand(-1, num_completions, -1),
            2, safe_positions,
        )
        completion_ids[~valid_mask] = IGNORE_INDEX

        if doc_ids is not None:
            target_doc_ids = torch.gather(
                doc_ids[:, None, :].expand(-1, num_completions, -1),
                2, safe_positions,
            )  # (B, N, L)
            start_doc_ids = completion_doc_ids if completion_doc_ids is not None \
                else torch.gather(doc_ids, 1, starts)  # (B, N)
            cross_doc_mask = target_doc_ids != start_doc_ids[:, :, None]  # (B, N, L)
            first_cross_mask = cross_doc_mask & (cross_doc_mask.cumsum(dim=-1) == 1)
            eos_token_id = getattr(self.model.tokenizer, 'eos_token_id', None)
            completion_ids[first_cross_mask] = eos_token_id if eos_token_id is not None else IGNORE_INDEX
            completion_ids[cross_doc_mask & ~first_cross_mask] = IGNORE_INDEX

        return completion_ids

    def _gather_bin_edges(self, left_bin_edges: Tensor, right_bin_edges: Tensor,
                          safe_positions: Tensor, valid_mask: Tensor,
                          num_completions: int) -> tuple[Tensor, Tensor]:
        """Gather bin edges at completion positions (shifted by -1 for logit alignment)."""
        edge_positions = (safe_positions - 1).clamp(min=0, max=left_bin_edges.shape[1] - 1)
        all_left = torch.gather(left_bin_edges[:, None, :].expand(-1, num_completions, -1), 2, edge_positions)
        all_right = torch.gather(right_bin_edges[:, None, :].expand(-1, num_completions, -1), 2, edge_positions)
        all_left[~valid_mask] = 0.0
        all_right[~valid_mask] = 0.0
        return all_left, all_right

    def prepare_nested_batch(self, input_ids: Tensor, input_mask: Tensor,
                             completion_length: int, completion_starts: Tensor,
                             left_bin_edges: Tensor | Any, right_bin_edges: Tensor | Any,
                             eval: bool,
                             doc_ids: Tensor | None = None,
                             completion_doc_ids: Tensor | None = None,
                             doc_starts: Tensor | None = None,
                             doc_lengths: Tensor | None = None,
                             ):
        if left_bin_edges is None or right_bin_edges is None:
            raise ValueError("left_bin_edges and right_bin_edges must be provided")
        assert left_bin_edges.shape[0] == input_ids.shape[0], \
            f"bin_edges batch dim {left_bin_edges.shape[0]} != input_ids {input_ids.shape[0]}"
        # bin_edges may be (B, S) or (B, S-1) depending on source
        assert left_bin_edges.shape[1] >= input_ids.shape[1] - 1, \
            f"bin_edges length {left_bin_edges.shape[1]} < seq_len-1 {input_ids.shape[1] - 1}"

        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        starts = completion_starts.to(device)  # (B, N)
        num_completions = starts.shape[1]
        assert (starts > 0).all(), "Completion start indices must be > 0 to have bin edges"

        position_ids, valid_mask, safe_positions = self._make_completion_positions(
            starts, completion_length, seq_len, device)
        block_mask = self._make_completion_block_mask(
            starts, completion_length, seq_len, batch_size, device,
            doc_ids, completion_doc_ids, doc_starts, doc_lengths)
        completion_ids = self._gather_completion_ids(
            input_ids, safe_positions, valid_mask, num_completions,
            doc_ids, completion_doc_ids, starts)
        all_left_edges, all_right_edges = self._gather_bin_edges(
            left_bin_edges, right_bin_edges, safe_positions, valid_mask, num_completions)

        position_ids[~valid_mask] = 0
        auxiliaries = self.sample_auxiliaries(all_left_edges, all_right_edges, eval)
        return block_mask, auxiliaries, completion_ids, position_ids

    def sample_auxiliaries(self, left_bin_edges: torch.Tensor, right_bin_edges: torch.Tensor,
                           eval: bool) -> torch.Tensor:
        device = left_bin_edges.device
        if not eval:
            beta_concentration = torch.tensor(0.3, device=device, dtype=torch.float32)
            if device.type == 'mps':
                beta_concentration = beta_concentration.cpu()
            z_rnd = torch.distributions.Beta(beta_concentration, beta_concentration).sample(
                left_bin_edges.shape).to(device)
        else:
            z_rnd = torch.rand(left_bin_edges.shape, device=device, dtype=torch.float32)
        auxiliaries = left_bin_edges + (right_bin_edges - left_bin_edges) * z_rnd
        return auxiliaries

    @torch.no_grad()
    def compute_sequence_metrics(self, completion_ids, student_predicted, include_outputs=False,
                                  student_logits=None, nucleus_threshold=None):
        mask = completion_ids != IGNORE_INDEX
        match = completion_ids == student_predicted
        if nucleus_threshold is not None:
            # A completion token also counts as "correct" if it falls within the model's
            # own top-p nucleus at that position, not just on exact argmax match -- mirrors
            # inference's _in_nucleus (right_bin_edges/adapt_p convention: kept iff the
            # cumulative mass of strictly-higher-ranked tokens is < threshold).
            assert student_logits is not None, "student_logits required when nucleus_threshold is set"
            probs = torch.softmax(student_logits.float(), dim=-1)
            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cum_before = sorted_probs.cumsum(dim=-1) - sorted_probs  # exclusive cumsum
            in_nucleus_sorted = cum_before < nucleus_threshold
            tok_match = sorted_idx == completion_ids.unsqueeze(-1)
            in_nucleus = (in_nucleus_sorted & tok_match).any(-1)
            match = match | in_nucleus
        identical = match & mask
        count_per_length = (mask.long().sum(dim=0) + 1e-8)
        acc_per_position = identical.long().sum(dim=0) / count_per_length
        accuracy = identical.sum() / (mask.sum() + 1e-8)

        metrics = {
            'accuracy': accuracy,
            'acc_per_position': acc_per_position,
            'completion_length': mask.sum(dim=1).float().mean(),
        }

        # Correct count before first error
        correct_counts = torch.where(
            # all tokens correct?
            (match | ~mask).all(dim=1),
            mask.long().sum(1),
            identical.float().argmin(dim=1),
        ).long()
        metrics["correct"] = correct_counts.float().mean()
        metrics["correct_counts"] = correct_counts

        for positional_metric in ['acc_per_position']:
            if positional_metric in metrics:
                short_positional_metric = positional_metric.replace('_per_position', '')
                for position in range(min(10, metrics[positional_metric].shape[0])):
                    metrics[f'{short_positional_metric}_pos_{position}'] = metrics[positional_metric][position]
        if include_outputs:
            metrics['outputs'] = student_predicted
        return metrics

    @staticmethod
    def sample_from_logits(logits, auxiliaries):
        right_bin_edges = torch.softmax(logits, dim=-1).cumsum(dim=-1)
        right_bin_edges[..., -1] = 1
        return (right_bin_edges > auxiliaries[..., None]).max(dim=-1).indices

    @torch.inference_mode()
    def generate_seq(self, prompt_ids, z_rnd_all=None,
                     student_forward=None, teacher_forward=None, shared_kv_cache=False,
                     needs_teacher=None, accepted_tokens=None, correct_first_token=True,
                     max_new_tokens=None, max_length=None, return_metrics=True,
                     ):
        """
        Generate sequences by iteratively calling the student / PTP model and then the verification / teacher model.
        Allows for versatile inference modes.

        Inputs:
        shared_kv - Only use one kv-cache that gets filled on teacher calls. Saves time when using gated LoRA.
        correct_first_token - Use the last non-auxiliary position to correct the first auxiliary position.
                              This is well motivated for gated LoRA.
        """
        assert prompt_ids.shape[0] == 1, "batch size must be 1 for now"
        metrics = {
            'correct': [],
        }
        tokens_prompt = prompt_ids.shape[1]
        tokens_to_fill = float('inf')
        if max_new_tokens is not None:
            tokens_to_fill = min(tokens_to_fill, max_new_tokens)
        if max_length is not None:
            tokens_to_fill = min(tokens_to_fill, max_length - tokens_prompt)
        device = prompt_ids.device
        if z_rnd_all is None:
            z_rnd_all = torch.rand(tokens_to_fill + 1, device=device, dtype=torch.float32)
        assert z_rnd_all.shape[0] >= tokens_to_fill, 'not enough random variables provided'

        # Fill kv caches
        kv_student = None
        kv_teacher = None
        teacher_forward = teacher_forward if teacher_forward is not None else self.model.inference_forward
        outputs = teacher_forward(
            input_ids=prompt_ids[:, :-1],
            past_key_values=kv_teacher,
            use_cache=True
        )
        kv_teacher = outputs.past_key_values
        if not shared_kv_cache:
            # Route through the student_forward callback (if given) so subclasses that
            # need consistent handling across the whole student-side KV cache (e.g. a
            # merged-LoRA mode) see this initial fill too, not just later proposal calls.
            student_prefill = student_forward if student_forward is not None else self.model.inference_forward
            outputs = student_prefill(
                input_ids=prompt_ids[:, :-1],
                auxiliaries=None,
                past_key_values=kv_student,
            )
            kv_student = outputs.past_key_values
        else:
            kv_student = kv_teacher

        while tokens_to_fill > 0:
            # --- Student proposal ---
            n_prop = min(self.tokens_per_student_call, tokens_to_fill)
            z_idx = prompt_ids.shape[1] - tokens_prompt
            z_rnd = z_rnd_all[z_idx:z_idx + n_prop + 1]

            if student_forward is None:
                outputs = self.model.inference_forward(
                    input_ids=prompt_ids[:, kv_student.get_seq_length():],
                    auxiliaries=z_rnd[None, :n_prop],
                    past_key_values=kv_student,
                    use_cache=True
                )
            else:
                outputs = student_forward(
                    input_ids=prompt_ids[:, kv_student.get_seq_length():],
                    auxiliaries=z_rnd[None, :n_prop],
                    past_key_values=kv_student,
                )

            kv_student.crop(kv_student.get_seq_length() - n_prop - (1 if shared_kv_cache else 0))
            full_logits = outputs.logits
            # O-PTP
            student_logits = full_logits[:, -n_prop:]
            student_tokens = student_logits.argmax(dim=2)
            if correct_first_token:
                tgt_logits = self.adapt_logits(full_logits[:, -n_prop - 1])
                correct_token = self.sample_from_logits(tgt_logits, z_rnd[0])
                student_tokens[:, 0] = correct_token
            input_ids = torch.cat([prompt_ids, student_tokens], dim=1)

            # --- Teacher verification ---
            if needs_teacher is None or needs_teacher(student_tokens, student_logits):
                outputs = teacher_forward(
                    input_ids=input_ids[:, kv_teacher.get_seq_length():],
                    past_key_values=kv_teacher,
                    use_cache=True
                )
                # todo: could be less strict, comparing with new_tokens to keep more cache
                kv_teacher.crop(kv_teacher.get_seq_length() - n_prop)
                tgt_logits = self.adapt_logits(outputs.logits[:, -n_prop - 1:])
                correct_tokens = self.sample_from_logits(tgt_logits, z_rnd)
            else:
                tgt_logits = None
                correct_tokens = None

            if accepted_tokens is not None:
                new_tokens = accepted_tokens(student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd)
            else:
                matches = student_tokens == correct_tokens[:, :-1]
                new_tokens = correct_tokens[:, :(matches.float().argmin() if not matches.all() else n_prop) + 1]
            metrics['correct'] += [new_tokens.shape[1]]
            prompt_ids = torch.cat([prompt_ids, new_tokens], dim=1)
            tokens_to_fill -= new_tokens.shape[1]
            if self.model.tokenizer.eos_token_id in new_tokens:
                eos_idx = prompt_ids.shape[1] - new_tokens.shape[1] + 1 + (new_tokens[0] == self.model.tokenizer.eos_token_id).nonzero()[0]
                prompt_ids = prompt_ids[:, :eos_idx]
                break

        if return_metrics:
            metrics = {
                'completion': prompt_ids,
                'correct_per_call': np.mean(metrics['correct']),
                'correct_all': metrics['correct'],
                'num_calls': len(metrics['correct']),
            }
            return prompt_ids, metrics
        return prompt_ids


    @staticmethod
    def _build_tree_mask(T_ar: int, n: int, S_kv: int, device, parent_list):
        """
        Additive attention mask + position ids for a masked tree forward, used
        by generate_seq_tree's default student/teacher forward: T_ar real/AR
        tokens attend causally as usual; each of the n tree nodes attends to
        the AR context plus its own ancestor chain (per parent_list) and
        itself, never to unrelated tree nodes. Nodes at the same depth share a
        position id (they're candidates for the same upcoming slot).
        """
        depth: list[int] = []
        for p in parent_list:
            depth.append(1 if p is None else depth[p] + 1)

        T = T_ar + n
        S = S_kv + T
        mask = torch.full((1, 1, T, S), float('-inf'), device=device, dtype=torch.float32)
        for q in range(T_ar):
            mask[0, 0, q, :S_kv + q + 1] = 0.0
        for k in range(n):
            mask[0, 0, T_ar + k, :S_kv + T_ar] = 0.0
            mask[0, 0, T_ar + k, S_kv + T_ar + k] = 0.0  # self
            p = parent_list[k]
            while p is not None:
                mask[0, 0, T_ar + k, S_kv + T_ar + p] = 0.0
                p = parent_list[p]

        pos_ids = torch.zeros(1, T, dtype=torch.long, device=device)
        for q in range(T_ar):
            pos_ids[0, q] = S_kv + q
        last_ar_pos = S_kv + T_ar - 1
        for k in range(n):
            pos_ids[0, T_ar + k] = last_ar_pos + depth[k]

        return mask, pos_ids

    @staticmethod
    def _in_nucleus(probs: torch.Tensor, tokens: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Standard top-p/nucleus membership test, mirroring adapt_p's rule: a token is
        kept iff the cumulative probability mass of all *higher*-ranked tokens
        (i.e. excluding itself) is strictly less than threshold.

        probs  : [1, n, V] raw (unadapted) softmax probabilities at n positions
        tokens : [1, n]    token ids to test membership for
        Returns: [n] bool
        """
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cum_before = sorted_probs.cumsum(dim=-1) - sorted_probs  # exclusive cumsum
        in_nucleus_sorted = cum_before < threshold
        match = sorted_idx == tokens.unsqueeze(-1)
        return (in_nucleus_sorted & match).any(-1)[0]

    @torch.inference_mode()
    def generate_seq_tree(self, prompt_ids, tree, z_rnd_all=None,
                     student_forward=None, teacher_forward=None, shared_kv_cache=False,
                     needs_teacher=None, accepted_tokens=None, correct_first_token=True,
                     max_new_tokens=None, max_length=None, return_metrics=True,
                     ):
        """
        Tree-structured variant of generate_seq.

        tree(n_prop) -> parent_list: for the current round, returns a length-n_nodes
        list of int|None (parent_list[k] = index of node k's parent within this
        round's n_nodes, or None if k attaches directly to the real/AR context).
        n_nodes is derived from this list and may exceed n_prop, e.g. several
        candidate strands proposed in one round. Defaults to a plain flat chain of
        n_prop nodes (i.e. classic generate_seq behaviour).

        student_forward(input_ids, auxiliaries, past_key_values, parent_list) and
        teacher_forward(input_ids, past_key_values, use_cache, parent_list), if given,
        replace the default tree-masked forward (see _build_tree_mask) — pass None
        (the default) to use the standard tree attention mask, where each node
        attends only to its own ancestors. Unlike generate_seq, verification
        re-checks EVERY node against its own parent's context (not just a flat
        main-strand proposal), so correct_tokens has the same shape as the proposed
        tree: e.g. 2b (parent 1b) can get a different correct token from 2a (parent
        1a) even at the same depth.

        Inputs:
        shared_kv - Only use one kv-cache that gets filled on teacher calls. Saves time when using gated LoRA.
        correct_first_token - Use the last non-auxiliary position to correct every ROOT
                              node (parent=None), e.g. every strand's first token.
                              This is well motivated for gated LoRA.
        """
        assert prompt_ids.shape[0] == 1, "batch size must be 1 for now"
        assert accepted_tokens is not None, "generate_seq_tree requires accepted_tokens"
        metrics = {
            'correct': [],
        }
        tokens_prompt = prompt_ids.shape[1]
        tokens_to_fill = float('inf')
        if max_new_tokens is not None:
            tokens_to_fill = min(tokens_to_fill, max_new_tokens)
        if max_length is not None:
            tokens_to_fill = min(tokens_to_fill, max_length - tokens_prompt)
        device = prompt_ids.device

        if z_rnd_all is None:
            max_n_nodes = len(tree(self.tokens_per_student_call))
            z_rnd_all = torch.rand(int(tokens_to_fill), max_n_nodes, device=device, dtype=torch.float32)

        # Fill kv caches. The teacher prefill always uses the plain model forward
        # (not the possibly tree-aware teacher_forward) since there's no tree yet.
        kv_student = None
        kv_teacher = None
        outputs = self.model.inference_forward(
            input_ids=prompt_ids[:, :-1],
            past_key_values=kv_teacher,
            use_cache=True
        )
        kv_teacher = outputs.past_key_values
        if self.model.output_hidden_states:
            # Context right before round 1's tree begins, same convention as generate()'s
            # pos-1 capture (see PHeadLightningModule / choice-k's "phead" choice_mode).
            self._last_context_hidden = outputs.hidden_states[-1][:, -1]
        if not shared_kv_cache:
            # Route through the student_forward callback (if given), same reasoning
            # as generate_seq: subclasses needing consistent student-side KV cache
            # handling (e.g. a merged-LoRA mode) must see this initial fill too, not
            # just later proposal calls. parent_list=None signals "no tree yet".
            if student_forward is not None:
                outputs = student_forward(
                    input_ids=prompt_ids[:, :-1],
                    auxiliaries=None,
                    past_key_values=kv_student,
                    parent_list=None,
                )
            else:
                outputs = self.model.inference_forward(
                    input_ids=prompt_ids[:, :-1],
                    past_key_values=kv_student,
                    use_cache=True,
                    flag=True,
                )
            kv_student = outputs.past_key_values
        else:
            kv_student = kv_teacher

        while tokens_to_fill > 0:
            # --- Student proposal: a tree of n_nodes candidate tokens ---
            n_prop = min(self.tokens_per_student_call, tokens_to_fill)
            parent_list = tree(n_prop)
            n_nodes = len(parent_list)
            z_idx = prompt_ids.shape[1] - tokens_prompt
            z_rnd = z_rnd_all[z_idx, :n_nodes]

            input_ids_student = prompt_ids[:, kv_student.get_seq_length():]
            if student_forward is None:
                mask, pos_ids = self._build_tree_mask(
                    input_ids_student.shape[1], n_nodes, kv_student.get_seq_length(), device, parent_list,
                )
                outputs = self.model.inference_forward(
                    input_ids=input_ids_student,
                    auxiliaries=z_rnd[None, :],
                    past_key_values=kv_student,
                    use_cache=True,
                    attention_mask=mask,
                    position_ids=pos_ids,
                )
            else:
                outputs = student_forward(
                    input_ids=input_ids_student,
                    auxiliaries=z_rnd[None, :],
                    past_key_values=kv_student,
                    parent_list=parent_list,
                )

            kv_student.crop(kv_student.get_seq_length() - n_nodes - (1 if shared_kv_cache else 0))
            full_logits = outputs.logits
            student_logits = full_logits[:, -n_nodes:]
            student_tokens = student_logits.argmax(dim=2)
            if correct_first_token:
                # Every root shares the same pre-tree context, so they all read
                # from the same non-auxiliary reference row, each with its own z.
                roots = [i for i, p in enumerate(parent_list) if p is None]
                tgt_logits_root = self.adapt_logits(full_logits[:, -n_nodes - 1])
                for r in roots:
                    student_tokens[:, r] = self.sample_from_logits(tgt_logits_root, z_rnd[r])
            input_ids = torch.cat([prompt_ids, student_tokens], dim=1)

            # --- Teacher verification: every node checked against its own parent's context ---
            if needs_teacher is None or needs_teacher(student_tokens, student_logits):
                teacher_fed = input_ids[:, kv_teacher.get_seq_length():]
                T_ar = teacher_fed.shape[1] - n_nodes
                if teacher_forward is None:
                    mask, pos_ids = self._build_tree_mask(
                        T_ar, n_nodes, kv_teacher.get_seq_length(), device, parent_list,
                    )
                    outputs = self.model.inference_forward(
                        input_ids=teacher_fed,
                        auxiliaries=None,
                        past_key_values=kv_teacher,
                        use_cache=True,
                        attention_mask=mask,
                        position_ids=pos_ids,
                    )
                else:
                    outputs = teacher_forward(
                        input_ids=teacher_fed,
                        past_key_values=kv_teacher,
                        use_cache=True,
                        parent_list=parent_list,
                    )
                if self.model.output_hidden_states:
                    # Context right before the NEXT round's tree begins -- read back by
                    # that round's tree(n_prop) call, same lag as generate()'s H_fn.
                    self._last_context_hidden = outputs.hidden_states[-1][:, T_ar - 1]
                kv_teacher.crop(kv_teacher.get_seq_length() - n_nodes)
                src_idx = torch.tensor(
                    [T_ar - 1 if p is None else T_ar + p for p in parent_list], device=device,
                )
                tgt_logits = self.adapt_logits(outputs.logits[:, src_idx])
                correct_tokens = self.sample_from_logits(tgt_logits, z_rnd)
            else:
                tgt_logits = None
                correct_tokens = None

            new_tokens = accepted_tokens(student_tokens, correct_tokens, student_logits, tgt_logits, z_rnd, parent_list)
            metrics['correct'] += [new_tokens.shape[1]]
            prompt_ids = torch.cat([prompt_ids, new_tokens], dim=1)
            tokens_to_fill -= new_tokens.shape[1]
            if self.model.tokenizer.eos_token_id in new_tokens:
                eos_idx = prompt_ids.shape[1] - new_tokens.shape[1] + 1 + \
                          (new_tokens[0] == self.model.tokenizer.eos_token_id).nonzero()[0]
                prompt_ids = prompt_ids[:, :eos_idx]
                break

        if return_metrics:
            metrics = {
                'completion': prompt_ids,
                'correct_per_call': np.mean(metrics['correct']),
                'correct_all': metrics['correct'],
                'num_calls': len(metrics['correct']),
            }
            return prompt_ids, metrics
        return prompt_ids


    def proposals(self, H, num_tokens=None, student_p=None, n_verify=None, double_at=100, metrics=None,
                  A: torch.Tensor | None = None):
        """
        Optimize proposals B wrt overhead adjusted expected # correct tokens
        max_B [sum_i A_i * H(B_i)] / [1 + sum_i B_i / 50)]

        If A_0 = 1 this becomes max_k H(k) / [1 + k / 50] = 14

        H - reward matrix: estimated # correct tokens given k proposed tokens
            (k = 0..20). See PTPInference.compute_H.
        A - optional precomputed per-position acceptance-probability estimate,
            overriding the hist_base/student_p derivation below. Lets a caller pass
            confidences for an arbitrary set of positions (e.g. several independent
            candidates' own tip confidences concatenated together) rather than just
            one chain's per-depth fan -- same joint knapsack either way.
        """
        assert self.hist_base is not None, "hist_base must be provided to use proposals()"
        # Estimated probability of # correct tokens
        if A is not None:
            pass
        elif student_p is None:
            A = self.hist_base
            if n_verify is not None:
                A = torch.cat([A[:n_verify], torch.tensor([A[n_verify:].sum()])])
        else:
            # assert student_p.shape[1] == n_verify
            A = torch.ones(student_p.shape[1] + 1)
            A[1:] = torch.cumprod(student_p[0].cpu(), dim=-1)
            A[:-1] *= 1 - student_p[0].cpu()
        # Reward; Estimated # correct tokens in the next step given k proposed tokens
        # A = A.clip(min=0.05)

        M = self.tokens_per_student_call
        arange_Mp1 = torch.arange(M + 1)
        AH = A[:, None] * H[None, :M + 1]  # [n_pos, M+1]
        inf = torch.tensor(float('inf'))

        if num_tokens is None:
            # Optimize based on per-token cost of 1/<double_at>
            lam = 0
            for _ in range(3):
                B = torch.argmax(AH - lam * arange_Mp1[None, :] / double_at, dim=1)
                lam_pre = lam
                lam = torch.sum(A * H[B]) / (1 + B.sum() / double_at)
                if (lam_pre == lam).all():
                    break
            return B.tolist()
        else:
            num_tokens = min(num_tokens, M * A.shape[0])

            # --- Binary search on lambda --- faster at first
            # lam_lo = 0.0
            # lam_hi = float(A.max())
            # for _ in range(50):
            #     lam = (lam_lo + lam_hi) / 2
            #     B = torch.argmax(AH - lam * self.arange_tpscp1[None, :], dim=1)
            #     total = B.sum().item()
            #     if total > num_tokens:
            #         lam_lo = lam
            #     elif total < num_tokens:
            #         lam_hi = lam
            #     else:
            #         break
            #     if abs(total - num_tokens) <= 5:
            #         break

            # Greedy estimate of lam
            A_idx = torch.argsort(A, descending=True)
            R = num_tokens // M
            r = num_tokens % M
            B = torch.zeros([A.shape[0]], dtype=int)
            B[A_idx[:R]] = M
            if r > 0 and R < A.shape[0]:
                B[A_idx[R]] = r
            if r > 0 and R < A.shape[0]:
                lam = float(A[A_idx[R]] * (H[r] - H[r - 1]))
            else:
                lam = float(A[A_idx[R - 1]] * (H[M] - H[M - 1]))
            B = torch.argmax(AH - lam * arange_Mp1[None, :], dim=1)

            # --- Greedy correction ---
            # B = B.clone()
            total = B.sum().item()

            while total > num_tokens:
                # Decrement the position with the smallest marginal gain of its last token
                # marginal gain of token b_i: A_i * (H[b_i] - H[b_i-1])
                can_dec = B > 0
                gains = torch.where(can_dec, A * (H[B] - H[(B - 1).clamp(min=1)]), inf)
                idx = torch.argmin(gains)
                B[idx] -= 1
                total -= 1

            while total < num_tokens:
                # Increment the position with the highest marginal gain of the next token
                # marginal gain of token b_i+1: A_i * (H[b_i+1] - H[b_i])
                can_inc = B < M
                gains = torch.where(can_inc, (A * (H[(B + 1).clamp(max=M)] - H[B])).clamp(min=1e-10), -inf)
                idx = torch.argmax(gains)
                B[idx] += 1
                total += 1

            return B.tolist()

    @torch.inference_mode()
    def generate(self, batch, max_new_tokens, return_metrics=False, return_past_key_values=False,
                 eos=None, fixed_tokens=True, pad_token=13, past_kv_cache=None, callback=None,
                 oracle_ref_ids=None,  # ORACLE DEBUG — remove after testing
                 **kwargs):
        """
        Partial Quadratic Coding using kv-cached Gated LoRA

        Input:
        ------
        fixed_tokens           - if True, force each transformer call to use the same number of
                                 verifying and proposed tokens, padding if necessary.
        past_kv_cache          - optional (prompt_ids, DynamicCache) from a previous generate()
                                 call for prompt-prefix reuse.
        return_past_key_values - if True, include (prompt_ids, DynamicCache) in the return value
                                 so it can be passed as past_kv_cache on the next call.
        """
        prompt_ids = batch['prompt_ids']
        assert prompt_ids.shape[0] == 1, "Batch size must be 1"
        assert self.model.inference_mode, "Call enter_inference_mode() before generate()"
        # assert self.top_k is not None and self.top_k > 0
        # assert self.top_p is not None and self.top_p < 1.0
        dev = prompt_ids.device
        tpsc = self.tokens_per_student_call
        ones_tpsc = torch.ones([1, tpsc], dtype=torch.long, device=dev)
        metrics = {'correct': [], 'N': [], 'Nrel': []}

        # Number of tokens proposed per speculative decoding step
        num_proposed_tokens = (self.total_token_budget or tpsc) if fixed_tokens else None

        # Verify in parallel
        tokens_to_fill = max_new_tokens
        tokens_to_verify = max_new_tokens
        ref_offset = 0          # ORACLE DEBUG — remove after testing
        initial_prompt_len = prompt_ids.shape[1]  # ORACLE DEBUG — remove after testing
        z_rnd_all = torch.rand([prompt_ids.shape[0], max_new_tokens + tpsc + (num_proposed_tokens if fixed_tokens else 0)], device=dev, dtype=torch.float32)
        if fixed_tokens:
            # 1st call is ignored anyway
            n_props = ((num_proposed_tokens // tpsc) * [tpsc] + [(num_proposed_tokens % tpsc)] + tpsc * [0])[:tpsc]
        else:
            n_props = self.proposals(self.H_fn(metrics), n_verify=0, num_tokens=num_proposed_tokens)
        
        if eos is None:
            eos = getattr(self.model.tokenizer, 'eos_token_id', None)
            if eos is None:
                raise ValueError("eos token id must be provided either via model.tokenizer.eos_token_id or the eos argument")

        # Reuse cached KV state for a matching prompt prefix
        if past_kv_cache is not None:
            past_prompt_ids, cache = past_kv_cache
            end = min(prompt_ids.shape[1], past_prompt_ids.shape[1])
            match = prompt_ids[:, :end] == past_prompt_ids[:, :end]
            keep = end if match.all() else match.float().argmin().item()
            cache.crop(keep)
            kv_cache = cache
        else:
            kv_cache = DynamicCache()
        if not fixed_tokens:
            self.model.set_gate_window(0)
        # Prefill: process prompt tokens (without proposing new ones)
        outputs = self.model.inference_forward(
            input_ids=prompt_ids[:, kv_cache.get_seq_length():-1],
            auxiliaries=z_rnd_all[:, :num_proposed_tokens if fixed_tokens else 0],
            past_key_values=kv_cache,
            use_cache=True,
        )
        kv_cache = outputs.past_key_values
        if fixed_tokens:
            # Pad to length
            kv_cache.crop(prompt_ids.shape[1] - 1)
            prompt_ids = torch.cat([
                prompt_ids,
                pad_token * ones_tpsc[:, :self.tokens_per_student_call - 1]
            ], dim=1)
            tokens_to_fill -= self.tokens_per_student_call - 1
        # torch.cuda.synchronize()
        # timing['call'] += [time.time() - scall]

        while tokens_to_verify > 0:
            if not fixed_tokens:
                n_props = [min(n, max(0, tokens_to_verify - d)) for d, n in enumerate(n_props)]
            n_verify = tokens_to_verify - tokens_to_fill
            # assert n_verify == len(n_props) - 1
            seq_len = prompt_ids.shape[1] + sum(n_props)
            pos = prompt_ids.shape[1] - kv_cache.get_seq_length()
            metrics['N'] += [pos + sum(n_props)]

            # Prepare inputs
            z_idx = max_new_tokens - tokens_to_verify
            z_rnd = torch.cat([z_rnd_all[:, z_idx + d:z_idx + d + n_prop] for d, n_prop in enumerate(n_props)], dim=1)
            input_ids = prompt_ids[:, kv_cache.get_seq_length():]
            K = kv_cache.get_seq_length()
            P = prompt_ids.shape[1]
            Q_LEN = seq_len - K
            # Position IDs: proposals for hypothesis d are shifted to position P-n_verify+d-1
            input_position_ids = torch.arange(K, seq_len, device=dev)[None, :]
            midx = pos
            for d, n_prop in enumerate(n_props):
                input_position_ids[:, midx: midx + n_prop] -= midx - pos + n_verify - d + 1
                midx += n_prop
            midx = pos
            input_mask = torch.tril(torch.ones(Q_LEN, seq_len, device=dev), diagonal=K)
            for d, n_prop in enumerate(n_props):
                input_mask[midx: midx + n_prop, P - n_verify + d : K + midx] = 0
                midx += n_prop
            input_mask = (1 - input_mask[None, None].to(next(self.parameters()).dtype)) * -1e15

            # Student proposals
            if not fixed_tokens:
                self.model.set_gate_window(sum(n_props))
            outputs = self.model.inference_forward(
                input_ids=input_ids,
                attention_mask=input_mask,
                position_ids=input_position_ids,
                auxiliaries=z_rnd,
                past_key_values=kv_cache,
                use_cache=True
            )
            if self.model.output_hidden_states:
                # Context right before this call's proposals begin — same position
                # convention as training's completion_starts-1 (see PHeadLightningModule).
                self._last_context_hidden = outputs.hidden_states[-1][:, pos - 1]

            kv_cache = outputs.past_key_values
            full_logits = outputs.logits
            student_logits = full_logits[:, pos:]
            if self.temperature is not None and self.temperature != 1.0:
                full_logits[:, :pos] = full_logits[:, :pos] / self.temperature
            full_p = torch.softmax(full_logits, dim=-1)
            student_p = full_p[:, pos:]
            tgt_p, tgt_indices = self.adapt_p(full_p[:, :pos])
            student_predicted = student_logits.argmax(dim=-1)
            student_p_max = student_p.gather(-1, student_predicted[..., None])[..., 0]

            # Verify last speculated tokens
            if n_verify > 0:
                # assert n_verify == tgt_logits.shape[1] - 1
                # with record_function("loop: verify + accept"):
                right_bin_edges = tgt_p.cumsum(dim=-1)  # [1, n_verify+1, top_k]
                right_bin_edges[..., -1] = 1
                z_idx = max_new_tokens - tokens_to_verify
                z_rnd = z_rnd_all[:, z_idx:z_idx + n_verify + 1]
                bin_idx = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
                correct_tokens = tgt_indices.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
                check_tokens = correct_tokens[:, :-1]
                predict_tokens = prompt_ids[..., - n_verify:]
                matches = (predict_tokens == check_tokens)
                num_correct = matches.float().argmin(dim=1)
                num_correct[matches.all(dim=1)] = n_verify
                num_correct = int(num_correct[0])
                num_new = num_correct + 1
                # ORACLE DEBUG — remove after testing
                if oracle_ref_ids is not None:
                    metrics['correct'] += [num_new]
                    ref_offset += num_new
                    # Discard verify window + proposals; keep oracle prefix in KV
                    kv_cache.crop(initial_prompt_len + ref_offset - num_new)
                    # prompt_ids = original prompt + full accepted oracle prefix
                    prompt_ids = torch.cat([
                        prompt_ids[:, :initial_prompt_len],
                        oracle_ref_ids[:, :ref_offset].to(dev),
                    ], dim=1)
                    # All proposals at depth 0: assume none of to-be-verified are correct
                    n_props = self.proposals(self.H_fn(metrics), n_verify=0, num_tokens=num_proposed_tokens, metrics=metrics)
                    n_props = [tpsc] + [0] * (len(n_props) - 1)
                    tokens_to_fill = tokens_to_verify  # n_verify = 0 → else branch fires
                    if eos in oracle_ref_ids[:, ref_offset - num_new : ref_offset]:
                        break
                    continue
                # END ORACLE DEBUG
                tokens_to_verify -= num_correct
                kv_cache.crop(prompt_ids.shape[-1] - (n_verify - num_correct))
                prev_prop = sum(n_props[:num_correct])
                ths_student_predicted = student_predicted[:, prev_prop: prev_prop + n_props[num_correct]]
                ths_student_p = student_p_max[:, prev_prop: prev_prop + n_props[num_correct]]
                metrics['Nrel'] += [num_correct / n_verify]
                # Verify first speculated and add other speculated tokens
                if fixed_tokens and ths_student_predicted.shape[1] < self.tokens_per_student_call:
                    # Pad with linbebreaks
                    ths_student_predicted = torch.cat([ths_student_predicted, pad_token * ones_tpsc[:, :self.tokens_per_student_call - ths_student_predicted.shape[1]]], dim=1)
                    ths_student_p = torch.cat([ths_student_p, 0 * ones_tpsc[:, :self.tokens_per_student_call - ths_student_p.shape[1]]], dim=1)
                if ths_student_predicted.shape[1] == 0:
                    match = False
                else:
                    ths_student_predicted[0, 0] = correct_tokens[0, num_correct]
                    match = ths_student_predicted[0, 0] == correct_tokens[0, num_correct]
                if not match:
                    # Discard speculated tokens
                    prompt_ids = torch.cat([
                        prompt_ids[:, :prompt_ids.shape[1] - n_verify],
                        correct_tokens[:, :num_correct + 1],
                    ], dim=1)
                    tokens_to_verify -= 1
                    tokens_to_fill = tokens_to_verify
                    n_props = self.proposals(self.H_fn(metrics), n_verify=0, num_tokens=num_proposed_tokens, metrics=metrics)
                    if callback is not None:
                        callback(prompt_ids[0], prompt_ids.shape[1])
                else:
                    # Add new speculated tokens
                    prompt_ids = torch.cat([
                        prompt_ids[:, :prompt_ids.shape[1] - (n_verify - num_correct)],
                        ths_student_predicted,
                    ], dim=1)
                    tokens_to_fill = tokens_to_verify - ths_student_predicted.shape[1]
                    tokens_to_verify -= 1
                    n_props = self.proposals(self.H_fn(metrics), student_p=ths_student_p[:, 1:], num_tokens=num_proposed_tokens, metrics=metrics)
                    if callback is not None:
                        callback(prompt_ids[0], prompt_ids.shape[1] - ths_student_predicted.shape[1] + 1)
                metrics['correct'] += [num_new]
                if eos in correct_tokens[:, :num_correct + 1]:
                    break
            else:
                # ORACLE DEBUG — remove after testing
                # n_verify == 0: place student proposals into verify window for next step
                if oracle_ref_ids is not None:
                    ths_proposals = student_predicted[:, :n_props[0]]
                    prompt_ids = torch.cat([prompt_ids, ths_proposals], dim=1)
                    # Crop KV to remove proposals so next step re-processes them
                    kv_cache.crop(prompt_ids.shape[1] - n_props[0])
                    tokens_to_fill = tokens_to_verify - n_props[0]  # n_verify_next = n_props[0] - 1
                    tokens_to_verify -= 1
                    n_props = [n_props[0]] + [0] * (len(n_props) - 1)
                else:
                    raise NotImplementedError
                # END ORACLE DEBUG
            # else:
            #     # Accept one token
            #     tgt_logits = self.adapt_logits(tgt_logits[:, -1:])
            #     right_bin_edges = torch.softmax(tgt_logits, dim=-1).cumsum(dim=-1)
            #     right_bin_edges[..., -1] = 1
            #     right_bin_edges[..., 0] = 0
            #     z_idx = max_new_tokens - tokens_to_verify
            #     z_rnd = z_rnd_all[:, z_idx:z_idx + 1]
            #     correct_token = (right_bin_edges > z_rnd[..., None]).max(dim=-1).indices
            #     kv_cache.crop(int(prompt_ids.shape[-1]))
            #
            #     if fixed_tokens and student_predicted.shape[1] < self.tokens_per_student_call:
            #         # Pad with linbebreaks
            #         student_predicted = torch.cat([student_predicted, 13 * torch.ones([1, self.tokens_per_student_call - ths_student_predicted.shape[1]], dtype=int, device=device)], dim=1)
            #         student_p_max = torch.cat([student_p_max, 13 * torch.ones([1, self.tokens_per_student_call - ths_student_p.shape[1]], device=device)], dim=1)
            #     if student_predicted.shape[1] == 0:
            #         match = False
            #     else:
            #         student_predicted[0, 0] = correct_token[0, 0]
            #         match = student_predicted[0, 0] == correct_token[0, 0]
            #     if not match:
            #         # Discard speculated tokens
            #         prompt_ids = torch.cat([
            #             prompt_ids,
            #             correct_token,
            #         ], dim=1)
            #         tokens_to_verify -= 1
            #         tokens_to_fill = tokens_to_verify
            #         n_props = self.proposals(n_verify=0, num_tokens=num_proposed_tokens, metrics=metrics)
            #     else:
            #         # Add new speculated tokens
            #         prompt_ids = torch.cat([
            #             prompt_ids,
            #             student_predicted[:, :n_props[0]],
            #         ], dim=1)
            #         tokens_to_fill = tokens_to_verify - n_props[0]
            #         tokens_to_verify -= 1
            #         n_props = self.proposals(student_p=student_p_max[:, 1:], num_tokens=num_proposed_tokens, metrics=metrics)
            #     metrics['correct'] += [1]
            #     if eos == correct_token[0, 0]:
            #         break

            # torch.cuda.synchronize()
            # timing['step'] += [time.time() - s]

        # Remove last prediction
        try:
            if ths_student_predicted.shape[1] > 1:
                prompt_ids = prompt_ids[:, :-ths_student_predicted.shape[1]+1]
        except NameError:
            pass

        # print(1000 * np.mean(timing['call'][1:]), 1000 * timing['call'][0], 1000 * np.mean(timing['step']))
        # plt.scatter(metrics['off'], metrics['offp'], s=2, alpha=0.5); plt.gca().set(xlabel='If the predicted token is wrong, the k-th one after is', ylabel='student confidence for actual correct token'); plt.show()
        metrics = {
            'completion': prompt_ids,
            'correct_per_call': np.mean(metrics['correct']),
            'correct_all': metrics['correct'],
            'num_calls': len(metrics['correct']),
        }

        if return_past_key_values and return_metrics:
            return prompt_ids, (prompt_ids, kv_cache), metrics
        if return_past_key_values:
            return prompt_ids, (prompt_ids, kv_cache)
        if return_metrics:
            return prompt_ids, metrics
        return prompt_ids

    @torch.inference_mode()
    def generate_tree(self, batch, max_new_tokens, k: int, nucleus_threshold: float,
                           return_metrics=False, eos=None, **kwargs):
        """
        Single-call choice-k PTP: maintains k independent candidate strands, all
        verified AND re-proposed together in one masked forward call per round --
        generalizes generate()'s own fused propose+verify design (a single call's
        real rows verify last round's proposal while its aux/z rows simultaneously
        propose the next one) from k=1 to k candidates, block-diagonal (no
        cross-candidate attention).

        Every round: verify the k current candidates as real embedded rows (each
        candidate's own accepted-correct length determined via exact-match-OR-top-p-
        nucleus against this call's own resampled target distribution, reusing the
        exact z each candidate's tokens were originally proposed with -- see cand_z
        below); simultaneously, at each candidate's own tip, propose a fresh k-wide
        fan of new children (so whichever candidate wins already has its own k
        next-round candidates ready, computed in this same call). The winner is the
        candidate with the longest verified-accepted length (ties -> higher
        confidence at the correction token, then lowest index).

        Losing candidates' (and the winner's own speculative) cache entries are never
        kept: kv_cache is cropped back to this round's starting frontier
        unconditionally, and the winner's newly-confirmed tokens are re-embedded via
        one small ordinary causal forward (same call shape as this method's own
        prefill) to actually get cached -- avoids needing any new, non-contiguous
        cache-repack machinery, at the cost of re-processing (not re-deciding) a
        handful of already-confirmed tokens each round.

        Total new fan tokens per round <= k * tokens_per_student_call: the per-tip
        depth budget is jointly allocated across the k tips via proposals() (using
        confidence A_i = candidate i's own cumulative acceptance probability so far)
        under a shared depth budget of tokens_per_student_call, then each tip's
        chosen depth is reused for k sibling children -- so total width is
        (sum of per-tip depths <= tokens_per_student_call) * k siblings <= k * cap.

        Self-verification (every row, real and fan alike, using merged-LoRA weights)
        is the caller's responsibility via _full_lora_mode wrapping this whole call,
        exactly as FullLoRAPTPInference wraps plain generate() for "ptp_self".
        """
        prompt_ids = batch['prompt_ids']
        assert prompt_ids.shape[0] == 1, "Batch size must be 1"
        assert self.model.inference_mode, "Call enter_inference_mode() before generate_tree()"
        dev = prompt_ids.device
        tpsc = self.tokens_per_student_call
        metrics = {'correct': []}

        if eos is None:
            eos = getattr(self.model.tokenizer, 'eos_token_id', None)
            if eos is None:
                raise ValueError("eos token id must be provided either via model.tokenizer.eos_token_id or the eos argument")

        kv_cache = DynamicCache()
        outputs = self.model.inference_forward(
            input_ids=prompt_ids[:, :-1], auxiliaries=None, past_key_values=kv_cache, use_cache=True,
        )
        kv_cache = outputs.past_key_values

        # committed_ids always holds exactly 1 more token than kv_cache.get_seq_length()
        # -- that extra trailing token is the uncached "bridge", mirroring generate()'s
        # own convention (see its `kv_cache.crop(prompt_ids.shape[1] - 1)` prefill step).
        committed_ids = prompt_ids
        cand_ids: list[torch.Tensor] | None = None    # k tensors [1, len_i], not yet cached
        cand_z: list[torch.Tensor] | None = None       # k tensors [len_i], z used to propose them
        cand_p: list[torch.Tensor] | None = None       # k tensors [len_i], own top-1 student prob

        tokens_generated = 0
        while tokens_generated < max_new_tokens:
            K = kv_cache.get_seq_length()
            bridge = committed_ids[:, K:]
            assert bridge.shape[1] == 1

            if cand_ids is None:
                lens = [0] * k
                A = torch.ones(k, dtype=torch.float64)
                real_ids = [torch.zeros(1, 0, dtype=torch.long, device=dev) for _ in range(k)]
            else:
                lens = [c.shape[1] for c in cand_ids]
                # proposals()/H_fn operate on CPU tensors (matches proposals()'s own
                # student_p[0].cpu() convention) -- move candidate confidences off GPU.
                A = torch.stack([
                    torch.cumprod(cand_p[i].cpu(), dim=-1)[-1].double() if lens[i] > 0
                    else torch.tensor(1.0, dtype=torch.float64)
                    for i in range(k)
                ])
                real_ids = cand_ids

            n_real = sum(lens)
            # Depth budget shared across the k tips (not multiplied by k here -- the k
            # siblings-per-tip multiplication below is what brings total width up to
            # k * tpsc, see docstring).
            B = self.proposals(self.H_fn(metrics), num_tokens=tpsc, A=A)  # length k, each in [0, tpsc]
            n_fan = k * sum(B)

            input_ids = torch.cat([bridge] + real_ids, dim=1)
            z_fan = torch.rand(1, n_fan, device=dev, dtype=torch.float32)

            Q_real = 1 + n_real
            Q_LEN = Q_real + n_fan
            S = K + Q_LEN
            mask = torch.full((1, 1, Q_LEN, S), float('-inf'), device=dev, dtype=torch.float32)
            pos_ids = torch.zeros(1, Q_LEN, dtype=torch.long, device=dev)

            # Bridge row (row 0): plain causal over the cache + itself.
            mask[0, 0, 0, :K + 1] = 0.0
            pos_ids[0, 0] = K

            cand_start = []  # row index of candidate i's first real token
            row = 1
            for i in range(k):
                cand_start.append(row)
                for j in range(lens[i]):
                    mask[0, 0, row, :K + 1] = 0.0                          # cache + bridge
                    mask[0, 0, row, K + cand_start[i]:K + row + 1] = 0.0   # own earlier real + self
                    pos_ids[0, row] = K + 1 + j
                    row += 1
            assert row == Q_real

            fan_start = []  # row index of candidate i's k-wide fan block start
            for i in range(k):
                fan_start.append(row)
                cand_real_lo = K + cand_start[i]
                cand_real_hi = K + cand_start[i] + lens[i]
                for c in range(k):
                    chain_start = row
                    for o in range(B[i]):
                        mask[0, 0, row, :K + 1] = 0.0                        # cache + bridge
                        mask[0, 0, row, cand_real_lo:cand_real_hi] = 0.0     # this tip's full real block
                        mask[0, 0, row, K + chain_start:K + row + 1] = 0.0   # own earlier fan + self
                        pos_ids[0, row] = K + 1 + lens[i] + o
                        row += 1
            assert row == Q_LEN

            outputs = self.model.inference_forward(
                input_ids=input_ids, auxiliaries=z_fan,
                attention_mask=mask, position_ids=pos_ids,
                past_key_values=kv_cache, use_cache=True,
            )
            kv_cache = outputs.past_key_values
            kv_cache.crop(K)  # discard this round's additions unconditionally (see docstring)

            full_logits = outputs.logits
            fan_logits = full_logits[:, Q_real:]
            full_p = torch.softmax(full_logits, dim=-1)
            real_p = full_p[:, :Q_real]
            fan_p = full_p[:, Q_real:]
            tgt_p, tgt_indices = self.adapt_p(real_p)

            # --- Verify: resample the target at each real row, reusing the SAME z
            # each candidate's own tokens were originally proposed with (bridge gets
            # a fresh z -- its "correct" continuation isn't checked against anything). ---
            right_bin_edges = tgt_p.cumsum(dim=-1)
            right_bin_edges[..., -1] = 1
            z_bridge = torch.rand(1, 1, device=dev, dtype=torch.float32)
            z_real = torch.cat([z_bridge] + [
                (cand_z[i][None] if cand_z is not None and lens[i] > 0 else torch.zeros(1, 0, device=dev))
                for i in range(k)
            ], dim=1)
            bin_idx = (right_bin_edges > z_real[..., None]).max(dim=-1).indices
            correct_tokens = tgt_indices.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)  # [1, Q_real]

            best_i, best_key, best_num_correct = -1, None, 0
            for i in range(k):
                if lens[i] == 0:
                    num_correct = 0
                else:
                    predict = real_ids[i]
                    check_target = torch.cat([
                        correct_tokens[:, 0:1],
                        correct_tokens[:, cand_start[i]:cand_start[i] + lens[i] - 1],
                    ], dim=1)
                    probs_i = torch.cat([
                        real_p[:, 0:1],
                        real_p[:, cand_start[i]:cand_start[i] + lens[i] - 1],
                    ], dim=1)
                    exact = (predict == check_target)[0]
                    nucleus = self._in_nucleus(probs_i, predict, nucleus_threshold)
                    match = exact | nucleus
                    num_correct = lens[i] if bool(match.all()) else int(match.float().argmin().item())
                total_len = num_correct + 1
                correction_idx = 0 if num_correct == 0 else cand_start[i] + num_correct - 1
                confidence = float(tgt_p[0, correction_idx].max())
                key = (total_len, confidence)
                if best_key is None or key > best_key:
                    best_i, best_key, best_num_correct = i, key, num_correct

            correction_idx = 0 if best_num_correct == 0 else cand_start[best_i] + best_num_correct - 1
            correction_token = correct_tokens[:, correction_idx:correction_idx + 1]
            winner_tokens = torch.cat([real_ids[best_i][:, :best_num_correct], correction_token], dim=1)
            n_new = winner_tokens.shape[1]

            committed_ids = torch.cat([committed_ids[:, :K + 1], winner_tokens], dim=1)
            metrics['correct'].append(n_new)
            tokens_generated += n_new

            # Commit the bridge + winner's confirmed tokens into the cache via one
            # small ordinary causal forward (same shape as this method's own prefill
            # above) -- holds back the last token uncached, restoring the "committed_ids
            # is exactly 1 longer than kv_cache" invariant for the next round.
            commit_input = torch.cat([bridge, winner_tokens], dim=1)[:, :-1]
            commit_out = self.model.inference_forward(
                input_ids=commit_input, auxiliaries=None, past_key_values=kv_cache, use_cache=True,
            )
            kv_cache = commit_out.past_key_values

            if eos in winner_tokens[0].tolist():
                break

            # Next round's k candidates: the winner's own precomputed tip-fan is only
            # valid if the winner was FULLY accepted (its fan's mask assumed the whole
            # real block was context -- a mismatch invalidates that conditioning, so
            # fall back to a fresh bootstrap-style round, same as round 1).
            if best_num_correct == lens[best_i]:
                lo = fan_start[best_i]
                depth = B[best_i]
                cand_ids, cand_z, cand_p = [], [], []
                for c in range(k):
                    chain_lo = lo + c * depth
                    chain_hi = chain_lo + depth
                    if depth == 0:
                        cand_ids.append(torch.zeros(1, 0, dtype=torch.long, device=dev))
                        cand_z.append(torch.zeros(0, device=dev))
                        cand_p.append(torch.zeros(0, device=dev))
                        continue
                    logits_c = fan_logits[:, chain_lo:chain_hi]
                    probs_c = fan_p[:, chain_lo:chain_hi]
                    tok_c = logits_c.argmax(dim=-1)
                    p_c = probs_c.gather(-1, tok_c[..., None])[..., 0]
                    cand_ids.append(tok_c)
                    cand_z.append(z_fan[0, chain_lo:chain_hi])
                    cand_p.append(p_c[0])
            else:
                cand_ids = None
                cand_z = None
                cand_p = None

        if committed_ids.shape[1] > prompt_ids.shape[1] + max_new_tokens:
            committed_ids = committed_ids[:, :prompt_ids.shape[1] + max_new_tokens]

        metrics = {
            'completion': committed_ids,
            'correct_per_call': np.mean(metrics['correct']),
            'correct_all': metrics['correct'],
            'num_calls': len(metrics['correct']),
        }
        if return_metrics:
            return committed_ids, metrics
        return committed_ids