"""Zero-inflated Beta-Binomial branching model with shared per-generation frailty.

Mirrors ``SeqPTPChoiceKInference._branch_transition_matrix`` /
``_branch_survival_table`` / ``_exact_optimal_lengths`` / ``_solve_width_profile``
(``scripts/inference.py:2193-2310``) as free functions so the paper's figures can
be regenerated without importing torch or touching a GPU. Keep in sync with that
source; do not refactor the source to match this file.

Model (artifact "Strand Survival Knapsack", eqs. 1.1-1.4):

    Theta_i ~ pi0 * delta_0 + (1 - pi0) * Beta(alpha, beta)
    S_i | S_{i-1} = m, Theta_i ~ Binomial(m, Theta_i)
    nu = (1 - rho) / rho,  alpha = p * nu,  beta = (1 - p) * nu
    T = min{i >= 1 : S_i = 0},   G = 1 + T

Generation 1 is exempt from zero-inflation; the terminal generation is a forced
death reflecting the hard proposal-length cap.
"""

from __future__ import annotations

import numpy as np
from scipy import special


def transition_matrix(pi0: float, alpha: float, beta: float, w_max: int) -> np.ndarray:
    """Zero-inflated Beta-Binomial transition matrix, rows/cols 0..w_max."""
    m = np.arange(w_max + 1)[:, None]
    j = np.arange(w_max + 1)[None, :]
    with np.errstate(invalid="ignore"):
        log_c = special.gammaln(m + 1) - special.gammaln(j + 1) - special.gammaln(m - j + 1)
        log_b1 = (
            special.gammaln(alpha + j)
            + special.gammaln(beta + m - j)
            - special.gammaln(alpha + beta + m)
        )
        log_b0 = special.gammaln(alpha) + special.gammaln(beta) - special.gammaln(alpha + beta)
    mat = np.exp(log_c + log_b1 - log_b0)
    mat[j > m] = 0.0
    mat[0, :] = 0.0
    mat = (1 - pi0) * mat
    mat[:, 0] += pi0
    mat[0, 0] = 1.0
    return mat


def survival_table(w_max: int, cap_t: int, pi0: float, p: float, rho: float) -> np.ndarray:
    """S[w, d] = P(T_homogeneous(w) >= d), d = 1..cap_t."""
    nu = (1 - rho) / rho
    alpha, beta = p * nu, (1 - p) * nu
    p0 = transition_matrix(0.0, alpha, beta, w_max)  # generation 1: pi0 exempt
    pc = transition_matrix(pi0, alpha, beta, w_max)  # generations 2..
    surv = np.zeros((w_max + 1, cap_t + 1))
    surv[:, 1] = 1.0
    surv[0, :] = 0.0
    v = np.eye(w_max + 1)
    for d in range(2, cap_t + 1):
        gen = d - 1
        v = v @ (p0 if gen == 1 else pc)
        surv[:, d] = 1 - v[:, 0]
    return surv


def t_pmf(k: int, cap_t: int, pi0: float, p: float, rho: float) -> np.ndarray:
    """P(T = d) for d = 1..cap_t, with all surviving mass forced dead at cap_t.

    Derived from the survival table: P(T = d) = S[k, d] - S[k, d+1], and
    P(T = cap_t) = S[k, cap_t] because the structural cap kills whatever is
    still alive. Index 0 of the returned array is unused (T >= 1).
    """
    surv = survival_table(k, cap_t, pi0, p, rho)[k]
    pmf = np.zeros(cap_t + 1)
    for d in range(1, cap_t):
        pmf[d] = surv[d] - surv[d + 1]
    pmf[cap_t] = surv[cap_t]
    return pmf


def expected_T(k: int, cap_t: int, pi0: float, p: float, rho: float) -> float:
    """E[T] via the tail sum (artifact eq. 2.1), truncated at the cap."""
    surv = survival_table(k, cap_t, pi0, p, rho)[k]
    return float(surv[1 : cap_t + 1].sum())


def expected_T_profile(surv: np.ndarray, widths: list[int] | np.ndarray) -> float:
    """E[T] = sum_d S[W(d), d] for a width profile (artifact eq. 3.2)."""
    return float(sum(surv[w, d + 1] for d, w in enumerate(widths)))


def solve_width_profile(surv: np.ndarray, n_budget: int, cap_t: int, w_max: int) -> list[int]:
    """DP over depth for the width-profile knapsack (artifact eq. 5.1)."""
    W, B = w_max + 1, n_budget + 1
    V = {cap_t + 1: np.zeros((W, B))}
    for d in range(cap_t, 0, -1):
        v_next = V[d + 1]
        gain = np.full((W, B), -np.inf)
        for w in range(min(W, B)):
            gain[w, w:] = surv[w, d] + v_next[w, : B - w]
        running_best = np.maximum.accumulate(gain, axis=0)
        v_d = np.empty((W, B))
        b_idx = np.arange(B)
        for w_cap in range(W):
            v_d[w_cap, :] = running_best[np.minimum(w_cap, b_idx), b_idx]
        V[d] = v_d

    widths, w_cap, b = [], w_max, n_budget
    for d in range(1, cap_t + 1):
        v_next = V[d + 1]
        ws = np.arange(min(w_cap, b) + 1)
        best_w = int(np.argmax(surv[ws, d] + v_next[ws, b - ws]))
        widths.append(best_w)
        w_cap, b = best_w, b - best_w
    return widths


def optimal_width_profile(
    n_budget: int, cap_t: int, pi0: float, p: float, rho: float
) -> tuple[list[int], np.ndarray]:
    """Exact optimum; doubles the width bound until it stops binding."""
    if n_budget <= 0:
        return [], survival_table(1, cap_t, pi0, p, rho)
    w_max = max(4 * (n_budget // cap_t) + 4 * cap_t, 1)
    while True:
        surv = survival_table(w_max, cap_t, pi0, p, rho)
        widths = solve_width_profile(surv, n_budget, cap_t, w_max)
        if widths[0] < w_max:
            return widths, surv
        w_max *= 2


def lengths_from_widths(widths: list[int], cap_t: int) -> list[int]:
    """Nonincreasing width profile -> strand-length multiset."""
    lengths: list[int] = []
    for d in range(1, cap_t + 1):
        nxt = widths[d] if d < cap_t else 0
        lengths.extend([d] * (widths[d - 1] - nxt))
    return lengths


def widths_from_lengths(lengths: list[int], cap_t: int) -> list[int]:
    """W(d) = #{s : L_s >= d} (artifact eq. 3.1)."""
    return [int(sum(1 for L in lengths if L >= d)) for d in range(1, cap_t + 1)]
