"""CPU-only tests for src/ptp/p_head.py — no GPU/checkpoint required."""
import torch
from types import SimpleNamespace

from ptp.p_head import PHead, PHeadLightningModule, CHead, CHeadLightningModule


def test_geometric_nll_excludes_floor_violations():
    p_pred = torch.tensor([0.5, 0.5, 0.5])
    correct_counts = torch.tensor([0, 3, 5])
    loss, valid = PHeadLightningModule.geometric_nll(p_pred, correct_counts, completion_length=16)
    assert valid.tolist() == [False, True, True]
    # k=0 row must not affect the loss: recompute with only the valid rows and compare.
    loss_valid_only, _ = PHeadLightningModule.geometric_nll(
        p_pred[1:], correct_counts[1:], completion_length=16)
    assert torch.allclose(loss, loss_valid_only)


def test_geometric_nll_censored_matches_survival_formula():
    p = torch.tensor([0.3])
    L = 16
    k_censored = torch.tensor([L])
    loss, valid = PHeadLightningModule.geometric_nll(p, k_censored, completion_length=L)
    assert valid.item() is True
    expected = -(L - 1) * torch.log1p(-p)
    assert torch.allclose(loss, expected.squeeze())


def test_geometric_nll_exact_matches_pmf():
    p = torch.tensor([0.4])
    k = torch.tensor([3])
    L = 16
    loss, _ = PHeadLightningModule.geometric_nll(p, k, completion_length=L)
    expected_pmf = p * (1 - p) ** (k.float() - 1)
    expected_nll = -torch.log(expected_pmf)
    assert torch.allclose(loss, expected_nll.squeeze())


def test_geometric_nll_recovers_p_via_sgd():
    torch.manual_seed(0)
    p_true = 0.35
    n = 2000
    # P(G=g) = p*(1-p)^(g-1) for g=1,2,... matches torch.distributions.Geometric(probs=p),
    # whose pmf is (1-probs)^k * probs for k=0,1,2,... via G = 1 + k.
    L = 20
    geom = torch.distributions.Geometric(probs=p_true)
    samples = (1 + geom.sample((n,))).clamp(max=L).long()

    logit = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([logit], lr=0.05)
    for _ in range(300):
        p_pred = torch.sigmoid(logit).expand(n)
        loss, _ = PHeadLightningModule.geometric_nll(p_pred, samples, completion_length=L)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    p_fit = torch.sigmoid(logit).item()
    assert abs(p_fit - p_true) < 0.03, f"fit p={p_fit}, true p={p_true}"


def test_compute_extra_losses_wiring_and_gradient_isolation():
    hidden_size = 8
    batch_size, num_completions, seq_len = 2, 3, 10

    module = PHeadLightningModule.__new__(PHeadLightningModule)
    torch.nn.Module.__init__(module)
    module.p_head = PHead(hidden_size)
    module.p_loss_weight = 1.0

    frozen_param = torch.nn.Parameter(torch.randn(4))
    ar_outputs = SimpleNamespace(hidden_states=(torch.randn(batch_size, seq_len, hidden_size, requires_grad=True),))
    completion_starts = torch.randint(1, seq_len, (batch_size, num_completions))
    correct_counts = torch.randint(0, 17, (batch_size * num_completions,))
    metrics = {"correct_counts": correct_counts}

    out = module._compute_extra_losses(ar_outputs, completion_starts, 16, metrics, batch_size, num_completions)
    assert "loss" in out and "metrics" in out
    assert out["loss"].requires_grad

    out["loss"].backward()
    for name, p in module.p_head.named_parameters():
        assert p.grad is not None, f"p_head.{name} should have a gradient"
    assert frozen_param.grad is None


def test_categorical_nll_exact_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(5, CHead.NUM_CLASSES)
    k = torch.tensor([0, 3, 5, 10, 15])
    L = 16  # all k < L, so no censoring
    loss = CHeadLightningModule.categorical_nll(logits, k, completion_length=L)
    expected = torch.nn.functional.cross_entropy(logits, k)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_categorical_nll_censored_matches_tail_logsumexp():
    torch.manual_seed(1)
    logits = torch.randn(1, CHead.NUM_CLASSES)
    L = 16
    k = torch.tensor([L])  # censored: >= L
    loss = CHeadLightningModule.categorical_nll(logits, k, completion_length=L)
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = -torch.logsumexp(log_probs[..., L:], dim=-1)
    assert torch.allclose(loss, expected.mean(), atol=1e-6)


def test_categorical_nll_recovers_distribution_via_sgd():
    torch.manual_seed(2)
    true_probs = torch.tensor([0.5, 0.3, 0.15, 0.05] + [0.0] * (CHead.NUM_CLASSES - 4))
    n = 4000
    samples = torch.multinomial(true_probs, n, replacement=True)
    L = 20  # no censoring in this support

    logits = torch.zeros(CHead.NUM_CLASSES, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.05)
    for _ in range(400):
        loss = CHeadLightningModule.categorical_nll(logits.unsqueeze(0).expand(n, -1), samples, completion_length=L)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fit_probs = torch.softmax(logits, dim=-1).detach()
    assert torch.allclose(fit_probs[:4], true_probs[:4], atol=0.03)


def test_chead_compute_extra_losses_wiring_and_gradient_isolation():
    hidden_size = 8
    batch_size, num_completions, seq_len = 2, 3, 10

    module = CHeadLightningModule.__new__(CHeadLightningModule)
    torch.nn.Module.__init__(module)
    module.c_head = CHead(hidden_size)
    module.c_loss_weight = 1.0

    frozen_param = torch.nn.Parameter(torch.randn(4))
    ar_outputs = SimpleNamespace(hidden_states=(torch.randn(batch_size, seq_len, hidden_size, requires_grad=True),))
    completion_starts = torch.randint(1, seq_len, (batch_size, num_completions))
    correct_counts = torch.randint(0, 17, (batch_size * num_completions,))
    metrics = {"correct_counts": correct_counts}

    out = module._compute_extra_losses(ar_outputs, completion_starts, 16, metrics, batch_size, num_completions)
    assert "loss" in out and "metrics" in out
    assert out["loss"].requires_grad

    out["loss"].backward()
    for name, p in module.c_head.named_parameters():
        assert p.grad is not None, f"c_head.{name} should have a gradient"
    assert frozen_param.grad is None
