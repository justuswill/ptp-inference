"""
Generalization of the per-question hierarchical analysis from the plain
Geometric model (K=1) to the full v3 branching model (pi0, p, rho) at K=2 -
but with only p left free per question; pi0 and rho are fixed at their
pooled (population) values, per the plan: identifiability of pi0/rho per
question with only 2-85 calls each is hopeless, and there's no evidence yet
they're context-dependent rather than intrinsic to the K-ensemble mechanism.

Reuses the v3 model exactly as specified in correct.md section 5 (generation
1 exempt from pi0, constant pi0 for generations 2..capT-1, certain death at
capT) - the model version currently adopted in the artifact.

Four things get compared, mirroring the K=1 Geometric analysis:
  (a) pooled single-p branching fit (baseline - pi0,p,rho all pooled)
  (b) marginal Beta-averaged branching fit: p_i ~ Beta(a,b) per question,
      (a,b) fit via numerically-integrated marginal MLE (no closed form here,
      unlike the Geometric case - branching pmf has no conjugate prior)
  (c) mixture of 100 raw per-question free-p MLEs
  (d) mixture of 100 shrinkage-adjusted (posterior mean) p_i's under the
      fitted Beta(a,b) prior
"""
import json
import re
import numpy as np
from scipy import stats, special, optimize

K = 2
CAPT = 19  # correct.md sec 5: 19 for choice-2/5/50/seq-n-choice-1000
MAX_I = CAPT
FLOOR = 2  # T = G - 1, dropping G < 2 floor violations (correct.md sec 3.4)

G_PATH = "scratch/choice2_percall_full.json"
LOG_PATH = "scratch/collect_choice2.log"
CALLS_KEY = "n_calls"


def ab_from_prho(p, rho):
    nu = (1 - rho) / rho
    return p * nu, (1 - p) * nu


def transition_matrix_fast(pi0, p, rho, K):
    alpha, beta = ab_from_prho(p, rho)
    m = np.arange(K + 1)[:, None]
    j = np.arange(K + 1)[None, :]
    with np.errstate(invalid="ignore"):
        logC = special.gammaln(m + 1) - special.gammaln(j + 1) - special.gammaln(m - j + 1)
        logB1 = special.gammaln(alpha + j) + special.gammaln(beta + m - j) - special.gammaln(alpha + beta + m)
        logB0 = special.gammaln(alpha) + special.gammaln(beta) - special.gammaln(alpha + beta)
    P = np.exp(logC + logB1 - logB0)
    P[j > m] = 0.0
    P[0, :] = 0.0
    P = (1 - pi0) * P
    P[:, 0] += pi0
    P[0, 0] = 1.0
    return P


def T_pmf_v3(pi0, p, rho, K, capT, max_i):
    """pmf[i-1] = P(T=i), i=1..max_i. Generation 1 exempt from pi0; certain
    death forced at generation capT."""
    P0 = transition_matrix_fast(0.0, p, rho, K)
    Pc = transition_matrix_fast(pi0, p, rho, K)
    v = np.zeros(K + 1); v[K] = 1.0
    pmf, prev0 = [], v[0]
    for i in range(1, max_i + 1):
        if i >= capT:
            pmf.append(1.0 - prev0); prev0 = 1.0
            continue
        P = P0 if i == 1 else Pc
        v = v @ P
        pmf.append(v[0] - prev0); prev0 = v[0]
    return np.array(pmf)


def parse_log(path, calls_key):
    pat = re.compile(r"example=(\d+) \(\d+/\d+\) " + calls_key + r"=(\d+) sum=(\d+)")
    rows = []
    for line in open(path):
        m = pat.search(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return rows


def r2(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ss_res = ((obs - pred) ** 2).sum()
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


# --- load data, reconstruct per-question T-sequences via log-boundary slicing ---
G_all = np.array(json.load(open(G_PATH)))
rows = parse_log(LOG_PATH, CALLS_KEY)
assert sum(r[1] for r in rows) == len(G_all)

per_q_T = []          # list of arrays: this question's own T values (floor violations dropped)
per_q_counts = []     # list of arrays: histogram of T counts, length MAX_I, index t-1 = count of T=t
idx = 0
n_floor_violations = 0
for _, n_calls, _ in rows:
    Gq = G_all[idx: idx + n_calls]
    idx += n_calls
    Gq_valid = Gq[Gq >= FLOOR]
    n_floor_violations += len(Gq) - len(Gq_valid)
    Tq = Gq_valid - (FLOOR - 1)  # T = G - 1, support {1,2,...}
    per_q_T.append(Tq)
    counts = np.bincount(np.clip(Tq, 1, MAX_I), minlength=MAX_I + 1)[1:]  # index t-1
    per_q_counts.append(counts)
assert idx == len(G_all)
per_q_counts = np.array(per_q_counts, dtype=float)  # (100, MAX_I)
n_i = per_q_counts.sum(axis=1)
print(f"n_floor_violations dropped: {n_floor_violations}")

T_pooled = np.concatenate(per_q_T)


# --- (0) pi0, rho fixed at the trusted 5-way JOINT MLE fit (correct.md sec 5,
# v3 "joint MLE" row: pi0=0.062, p=0.713, rho=0.858) - NOT a standalone
# per-dataset refit. A standalone K=2-only fit was tried first and landed on a
# degenerate boundary (pi0=0.000, rho=0.9999), the known small-K identifiability
# issue correct.md already flags; pi0/rho only pin down cleanly with a joint
# multi-K fit, so that's what gets held fixed here (and should be, per-question,
# for any other K too - only p is ever free per question, everywhere).
pi0_pop, rho_pop = 0.062, 0.858
print(f"pi0, rho fixed at trusted 5-way joint MLE values: pi0={pi0_pop:.4f} rho={rho_pop:.4f}")


# --- precompute pmf on a grid of p, holding pi0_pop/rho_pop fixed - reused for
# everything below (free-per-question fits, marginal integral, shrinkage means) ---
p_grid = np.linspace(1e-3, 1 - 1e-3, 400)
pmf_grid = np.array([T_pmf_v3(pi0_pop, pg, rho_pop, K, CAPT, MAX_I) for pg in p_grid])  # (400, MAX_I)
log_pmf_grid = np.log(np.clip(pmf_grid, 1e-300, None))

# per-question, per-grid-point log-likelihood: (100, 400)
loglik_grid = per_q_counts @ log_pmf_grid.T


# --- (a) pooled single-p (fixing pi0,rho at pop values, p at its own pooled MLE
#     given those - refit p alone for a clean apples-to-apples "(a) pooled" baseline) ---
def nll_p_only(p):
    pmf = T_pmf_v3(pi0_pop, p[0], rho_pop, K, CAPT, MAX_I)
    probs = np.clip(pmf[T_pooled - 1], 1e-300, None)
    return -np.sum(np.log(probs))
res_p = optimize.minimize(nll_p_only, x0=[0.7], method="Nelder-Mead")
p_pooled_only = res_p.x[0]
print(f"pooled p (pi0,rho fixed at pop values): p={p_pooled_only:.4f}")


# --- (c) free per-question p_i via 1D MLE (using the actual continuous
#     function around the best grid point, for precision beyond grid resolution) ---
p_hat_i = np.zeros(len(rows))
for i in range(len(rows)):
    j0 = np.argmax(loglik_grid[i])
    p0 = p_grid[j0]
    def nll_i(p, i=i):
        pmf = T_pmf_v3(pi0_pop, p[0], rho_pop, K, CAPT, MAX_I)
        counts = per_q_counts[i]
        return -np.sum(counts * np.log(np.clip(pmf, 1e-300, None)))
    r = optimize.minimize(nll_i, x0=[p0], method="Nelder-Mead",
                           options=dict(xatol=1e-6, fatol=1e-6))
    p_hat_i[i] = np.clip(r.x[0], 1e-3, 1 - 1e-3)

print(f"\n[free per-question p_i, pi0/rho fixed]")
print(f"  mean={p_hat_i.mean():.4f}  std={p_hat_i.std():.4f}  min={p_hat_i.min():.4f}  max={p_hat_i.max():.4f}")

# LRT: free per-question p_i vs single pooled p (pi0,rho fixed either way - cancels)
ll_free = sum(-optimize.minimize(lambda p, i=i: -(per_q_counts[i] @ np.log(np.clip(
    T_pmf_v3(pi0_pop, p[0], rho_pop, K, CAPT, MAX_I), 1e-300, None))),
    x0=[p_hat_i[i]], method="Nelder-Mead").fun for i in range(len(rows)))
ll_pooled = -nll_p_only([p_pooled_only])
lrt_stat = 2 * (ll_free - ll_pooled)
df = len(rows) - 1
p_value = stats.chi2.sf(lrt_stat, df)
print(f"  LRT free-p_i vs pooled-p (pi0,rho fixed): stat={lrt_stat:.1f} df={df} p={p_value:.2e}")


# --- (b) marginal Beta(a,b)-averaged fit: numerically-integrated MLE over the
#     precomputed grid (no closed form for this kernel, unlike Geometric) ---
def neg_marginal_ll(params):
    log_a, log_b = params
    a, b = np.exp(log_a), np.exp(log_b)
    log_beta_w = stats.beta.logpdf(p_grid, a, b)
    # log-sum-exp over grid, trapezoidal weights via np.trapezoid on the log scale:
    # marginal_i = trapz_p [ exp(loglik_grid[i,:] + log_beta_w) ] dp
    integrand = np.exp(loglik_grid + log_beta_w[None, :] - loglik_grid.max(axis=1, keepdims=True))
    marg = np.trapezoid(integrand, p_grid, axis=1)
    ll = np.log(marg) + loglik_grid.max(axis=1)
    return -ll.sum()

res_bg = optimize.minimize(neg_marginal_ll, x0=[np.log(5), np.log(5)], method="Nelder-Mead",
                            options=dict(xatol=1e-6, fatol=1e-6, maxfev=2000))
a_hat, b_hat = np.exp(res_bg.x)
beta_mean = a_hat / (a_hat + b_hat)
beta_std = np.sqrt(a_hat * b_hat / ((a_hat + b_hat) ** 2 * (a_hat + b_hat + 1)))
ll_bg = -res_bg.fun
print(f"\n[marginal Beta(a,b) fit over p_i, pi0/rho fixed]")
print(f"  a={a_hat:.3f} b={b_hat:.3f} -> Beta mean={beta_mean:.4f} std={beta_std:.4f}")
aic_pooled = 2 * 1 - 2 * ll_pooled
aic_bg = 2 * 2 - 2 * ll_bg
lrt_bg = 2 * (ll_bg - ll_pooled)
p_value_bg = stats.chi2.sf(lrt_bg, df=1)
print(f"  AIC pooled={aic_pooled:.1f}  AIC beta-marginal={aic_bg:.1f}  delta={aic_pooled-aic_bg:.1f}")
print(f"  LRT pooled vs beta-marginal: stat={lrt_bg:.1f} df=1 p={p_value_bg:.2e}")

# (d) shrinkage-adjusted posterior mean p_i under the fitted Beta(a,b) prior
log_beta_w = stats.beta.logpdf(p_grid, a_hat, b_hat)
post_unnorm = np.exp(loglik_grid + log_beta_w[None, :] - loglik_grid.max(axis=1, keepdims=True))
post_norm = post_unnorm / np.trapezoid(post_unnorm, p_grid, axis=1, )[:, None]
p_i_bayes = np.trapezoid(post_norm * p_grid[None, :], p_grid, axis=1)
print(f"\n[shrinkage-adjusted p_i_bayes]")
print(f"  mean={p_i_bayes.mean():.4f}  std={p_i_bayes.std():.4f}  min={p_i_bayes.min():.4f}  max={p_i_bayes.max():.4f}")


# --- aggregate T-histogram fit comparison ---
obs_hist = per_q_counts.sum(axis=0) / per_q_counts.sum()

pred_pooled = T_pmf_v3(pi0_pop, p_pooled_only, rho_pop, K, CAPT, MAX_I)

log_beta_w_final = stats.beta.logpdf(p_grid, a_hat, b_hat)
w_final = np.exp(log_beta_w_final - log_beta_w_final.max())
w_final /= np.trapezoid(w_final, p_grid)
pred_bg = np.trapezoid(pmf_grid * w_final[:, None], p_grid, axis=0)
pred_bg /= pred_bg.sum()

w_n = n_i / n_i.sum()
pred_mix_raw = np.zeros(MAX_I)
pred_mix_bayes = np.zeros(MAX_I)
for i in range(len(rows)):
    pred_mix_raw += w_n[i] * T_pmf_v3(pi0_pop, p_hat_i[i], rho_pop, K, CAPT, MAX_I)
    pred_mix_bayes += w_n[i] * T_pmf_v3(pi0_pop, p_i_bayes[i], rho_pop, K, CAPT, MAX_I)

panel = json.load(open("scratch/panel_data_5way.json"))
r2_existing = panel["datasets"]["choice2"]["jointM_r2"]

print(f"\n=== choice-2 (K=2) aggregate T-histogram fit, v3 branching model, only p_i free ===")
print(f"R^2  pooled single-p (pi0,rho fixed):            {r2(obs_hist, pred_pooled):.5f}")
print(f"R^2  marginal Beta(a,b)-averaged over hierarchy:  {r2(obs_hist, pred_bg):.5f}")
print(f"R^2  mixture of 100 raw per-question p_i:         {r2(obs_hist, pred_mix_raw):.5f}")
print(f"R^2  mixture of 100 shrinkage-adjusted p_i_bayes: {r2(obs_hist, pred_mix_bayes):.5f}")
print(f"R^2  existing 5-way joint MLE fit (pi0,p,rho all pooled across 5 datasets): {r2_existing:.5f}")
