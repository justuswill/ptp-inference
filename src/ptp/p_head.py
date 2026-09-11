import torch
import torch.nn as nn
from torch import Tensor

from ptp.lit import ParallelSamplingLightningModule


def gather_call_context(ar_outputs, completion_starts: Tensor) -> Tensor:
    """
    AR hidden state right before each call's proposals begin, at completion_starts - 1
    (same position convention as bin-edge gathering; see PHeadLightningModule).
    Returns (B, N, H).
    """
    hidden = ar_outputs.hidden_states[-1]  # (B, S, H)
    call_pos = (completion_starts - 1).clamp(min=0, max=hidden.shape[1] - 1)  # (B, N)
    return torch.gather(hidden, 1, call_pos[:, :, None].expand(-1, -1, hidden.shape[-1]))


class PHead(nn.Module):
    """Predicts p in (0,1) for the G = 1 + Geometric(p) per-call acceptance model."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, context: Tensor) -> Tensor:
        return torch.sigmoid(self.linear(context.float()).squeeze(-1))


class PHeadLightningModule(ParallelSamplingLightningModule):
    """
    ParallelSamplingLightningModule with an additional frozen-base P-head:
    predicts p per call from the AR context hidden state, trained via a
    censored Geometric NLL against the observed #correct-per-call.
    Everything except p_head is expected to be frozen via freeze_base().
    """

    def __init__(self, *args, p_loss_weight: float = 1.0, completion_loss_weight: float = 0.0,
                 self_mode: bool = False, nucleus_threshold: float | None = None, **kwargs):
        super().__init__(*args, completion_loss_weight=completion_loss_weight, **kwargs)
        self.p_loss_weight = p_loss_weight
        self.p_head: PHead | None = None
        self.self_mode = self_mode
        self.nucleus_threshold = nucleus_threshold

    def add_p_head(self) -> None:
        self.p_head = PHead(self.model.model.config.hidden_size)

    def freeze_base(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.output_hidden_states = True
        self.freeze_backbone_forward = True

    @staticmethod
    def geometric_nll(p_pred: Tensor, correct_counts: Tensor, completion_length: int) -> tuple[Tensor, Tensor]:
        """
        Censored NLL of G = 1 + Geometric(p) given observed correct_counts (0..completion_length).
        k=0 (floor violation, outside G>=1's support) is excluded from the loss.
        k==completion_length is right-censored: -log P(G >= completion_length), not a point mass.
        """
        eps = 1e-6
        p = p_pred.clamp(eps, 1 - eps)
        log_p, log1m_p = torch.log(p), torch.log1p(-p)
        k = correct_counts.float()
        is_censored = correct_counts >= completion_length
        valid = correct_counts >= 1
        exact_nll = -(log_p + (k - 1).clamp(min=0) * log1m_p)
        censored_nll = -(completion_length - 1) * log1m_p
        nll = torch.where(is_censored, censored_nll, exact_nll)
        loss = (nll * valid).sum() / valid.float().sum().clamp(min=1.0)
        return loss, valid

    def _compute_extra_losses(self, ar_outputs, completion_starts, completion_length,
                               metrics, batch_size, num_completions) -> dict:
        if self.p_head is None:
            return {}
        context = gather_call_context(ar_outputs, completion_starts)  # (B, N, H)
        p_pred = self.p_head(context)  # (B, N)
        correct_counts = metrics['correct_counts'].reshape(batch_size, num_completions)
        loss, valid = self.geometric_nll(p_pred, correct_counts, completion_length)
        return {
            'loss': self.p_loss_weight * loss,
            'metrics': {
                'p_loss': loss.detach(),
                'p_mean': p_pred.mean().detach(),
                'p_valid_frac': valid.float().mean().detach(),
            },
        }


class CHead(nn.Module):
    """
    Predicts a full categorical (non-parametric) distribution over #correct in
    {0, ..., NUM_CLASSES-1} directly, instead of assuming a parametric shifted-Geometric
    shape (contrast with PHead). Returns raw logits; NUM_CLASSES=21 matches the same
    0..20 support used everywhere else in this codebase (hist_base, PTPInference.geom_H).
    """
    NUM_CLASSES = 21

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, self.NUM_CLASSES)

    def forward(self, context: Tensor) -> Tensor:
        return self.linear(context.float())  # (..., NUM_CLASSES) logits


class CHeadLightningModule(ParallelSamplingLightningModule):
    """
    ParallelSamplingLightningModule with an additional frozen-base C-head: predicts a
    full categorical distribution over #correct per call from the AR context hidden
    state, trained via a censored categorical NLL against the observed #correct-per-call.
    Everything except c_head is expected to be frozen via freeze_base().
    """

    def __init__(self, *args, c_loss_weight: float = 1.0, completion_loss_weight: float = 0.0,
                 self_mode: bool = False, nucleus_threshold: float | None = None, **kwargs):
        super().__init__(*args, completion_loss_weight=completion_loss_weight, **kwargs)
        self.c_loss_weight = c_loss_weight
        self.self_mode = self_mode
        self.nucleus_threshold = nucleus_threshold
        self.c_head: CHead | None = None

    def add_c_head(self) -> None:
        self.c_head = CHead(self.model.model.config.hidden_size)

    def freeze_base(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.output_hidden_states = True
        self.freeze_backbone_forward = True

    @staticmethod
    def categorical_nll(logits: Tensor, correct_counts: Tensor, completion_length: int) -> Tensor:
        """
        Censored categorical NLL: exact cross-entropy for k < completion_length,
        -log P(correct >= completion_length) (summed tail mass) for the right-censored
        k == completion_length case. No floor exclusion needed (k=0 is a normal class here).
        """
        assert completion_length <= CHead.NUM_CLASSES - 1, \
            f"completion_length ({completion_length}) must be <= {CHead.NUM_CLASSES - 1} for CHead"
        log_probs = torch.log_softmax(logits, dim=-1)
        k = correct_counts.long().clamp(max=CHead.NUM_CLASSES - 1)
        is_censored = correct_counts >= completion_length
        exact_nll = -log_probs.gather(-1, k.unsqueeze(-1)).squeeze(-1)
        censored_nll = -torch.logsumexp(log_probs[..., completion_length:], dim=-1)
        nll = torch.where(is_censored, censored_nll, exact_nll)
        return nll.mean()

    def _compute_extra_losses(self, ar_outputs, completion_starts, completion_length,
                               metrics, batch_size, num_completions) -> dict:
        if self.c_head is None:
            return {}
        context = gather_call_context(ar_outputs, completion_starts)  # (B, N, H)
        logits = self.c_head(context)  # (B, N, NUM_CLASSES)
        correct_counts = metrics['correct_counts'].reshape(batch_size, num_completions)
        loss = self.categorical_nll(logits, correct_counts, completion_length)
        return {
            'loss': self.c_loss_weight * loss,
            'metrics': {'c_loss': loss.detach()},
        }
