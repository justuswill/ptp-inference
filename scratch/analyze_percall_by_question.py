"""
Does p (the per-call geometric success rate underlying G = 1 + Geometric(p),
tokens-accepted-per-call) vary across questions/contexts, or is a single
pooled p adequate?

Uses the three datasets where the collect_*.log files record each question's
(n_calls, sum) before the per-call samples were flattened into
scratch/*_percall_full.json — confirmed byte-for-byte consistent with those
logs (same total n_calls, same total sum), so log order == percall_full order
and per-question boundaries can be recovered exactly:
  - choice-2   (K=2),    log=collect_choice2.log,        100 questions
  - choice-50  (K=50),   log=collect_choice50.log,        100 questions
  - seq-n-choice-1000 (K=1000), log=collect_seqn1000_full.log, 100 questions

G is supported on {1,2,...} (shift=1 floor), so T = G - 1 ~ Geometric(p) on
{0,1,2,...} with mean (1-p)/p. Per-question sufficient stats: n_i calls,
S_i = sum of T over that question's calls. Geometric MLE: p_hat = n_i / (n_i + S_i).

Three questions:
  1. Per-question free-p MLE vs one pooled p: likelihood-ratio test.
  2. Is the excess variance in p_hat_i bigger than sampling noise would give
     under a true single p (a "funnel plot" / dispersion check)?
  3. A shrinkage hierarchical alternative: p_i ~ Beta(a,b) per question
     (Beta-Geometric == the same frailty construction already used in
     correct.md for K-dependence, just one level up: per-question instead of
     per-generation-within-a-call). Fit (a,b) by marginal MLE (Beta-NegBinom)
     and compare AIC to the pooled single-p model.
"""
import re
import numpy as np
from scipy import stats, optimize
from scipy.special import betaln

SCRATCH = "/home/jcwill/Projects/ptp/scratch"

DATASETS = [
    dict(name="choice-2 (K=2)", log=f"{SCRATCH}/collect_choice2.log", calls_key="n_calls"),
    dict(name="choice-50 (K=50)", log=f"{SCRATCH}/collect_choice50.log", calls_key="n_calls"),
    dict(name="seq-n-choice-1000 (K=1000)", log=f"{SCRATCH}/collect_seqn1000_full.log", calls_key="n_rounds"),
]


def parse_log(path, calls_key):
    pat = re.compile(r"example=(\d+) \(\d+/\d+\) " + calls_key + r"=(\d+) sum=(\d+)")
    rows = []
    for line in open(path):
        m = pat.search(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return rows


def geom_loglik(n, S, p):
    # T ~ Geometric(p) on {0,1,...}, pmf (1-p)^t * p. sum of n iid: n*log(p) + S*log(1-p).
    # Handle p in (0,1); guard S==0 (1-p)^0 fine even if p==1.
    ll = n * np.log(p)
    if S > 0:
        ll += S * np.log(1 - p)
    return ll


def analyze(name, log, calls_key):
    rows = parse_log(log, calls_key)
    assert len(rows) == 100, f"{name}: expected 100 questions, got {len(rows)}"
    n = np.array([r[1] for r in rows], dtype=float)   # calls per question
    S_total = np.array([r[2] for r in rows], dtype=float)  # sum of G per question
    T = S_total - n  # sum of (G-1) = T per question, i.e. "successes-minus-n" for geometric

    p_hat_i = n / (n + T)
    p_pooled = n.sum() / (n.sum() + T.sum())

    # 1. LRT: free p_i per question vs single pooled p
    ll_free = sum(geom_loglik(ni, Ti, pi) for ni, Ti, pi in zip(n, T, p_hat_i))
    ll_pooled = sum(geom_loglik(ni, Ti, p_pooled) for ni, Ti in zip(n, T))
    lrt_stat = 2 * (ll_free - ll_pooled)
    df = len(rows) - 1
    p_value = stats.chi2.sf(lrt_stat, df)

    # 2. Dispersion check: standardize p_hat_i against the Fisher-information
    # SE a single true p_pooled would predict for a geometric MLE with n_i obs:
    # Var(p_hat) ~ p^2(1-p)/n  (asymptotic Fisher-info approx).
    se_i = np.sqrt(p_pooled**2 * (1 - p_pooled) / n)
    z_i = (p_hat_i - p_pooled) / se_i
    dispersion_ratio = np.var(z_i, ddof=1)  # should be ~1 under homogeneity

    # 3. Beta-Geometric hierarchical fit: p_i ~ Beta(a,b) per question, and the
    # n_i individual calls in that question are an iid Geometric(p_i) *sequence*
    # (not just their sum) - same basis as geom_loglik above, so no NegBinom
    # combinatorial (order-counting) term here; that term would double as a
    # constant offset relative to ll_pooled/ll_free and invalidate the AIC
    # comparison below (it does not depend on p so it cancels inside the [1] LRT,
    # but it does not cancel across the pooled-vs-Beta-Geometric comparison).
    # Marginal likelihood of the sequence: integral of p^n (1-p)^T Beta(p;a,b) dp.
    def beta_negbinom_loglik(a, b):
        ll = 0.0
        for ni, Ti in zip(n, T):
            ll += betaln(a + ni, b + Ti) - betaln(a, b)
        return ll

    def neg_ll(params):
        log_a, log_b = params
        a, b = np.exp(log_a), np.exp(log_b)
        return -beta_negbinom_loglik(a, b)

    x0 = [np.log(p_pooled * 10), np.log((1 - p_pooled) * 10)]
    res = optimize.minimize(neg_ll, x0, method="Nelder-Mead",
                             options=dict(xatol=1e-8, fatol=1e-8, maxfev=2000))
    a_hat, b_hat = np.exp(res.x)
    ll_betageom = -res.fun
    beta_mean = a_hat / (a_hat + b_hat)
    beta_std = np.sqrt(a_hat * b_hat / ((a_hat + b_hat) ** 2 * (a_hat + b_hat + 1)))

    # AIC comparison: pooled single-p (1 param) vs Beta-Geometric hierarchical (2 params)
    aic_pooled = 2 * 1 - 2 * ll_pooled
    aic_betageom = 2 * 2 - 2 * ll_betageom
    lrt_bg = 2 * (ll_betageom - ll_pooled)
    p_value_bg = stats.chi2.sf(lrt_bg, df=1)

    print(f"=== {name} ===")
    print(f"  n_questions=100  total_calls={int(n.sum())}  pooled p_hat={p_pooled:.4f}")
    print(f"  per-question p_hat: mean={p_hat_i.mean():.4f} std={p_hat_i.std():.4f} "
          f"min={p_hat_i.min():.4f} max={p_hat_i.max():.4f}")
    print(f"  [1] LRT free-per-question vs pooled: stat={lrt_stat:.1f} df={df} p={p_value:.2e}")
    print(f"  [2] dispersion ratio Var(z_i) (expect ~1 if homogeneous): {dispersion_ratio:.2f}")
    print(f"  [3] Beta-Geometric hierarchical fit: a={a_hat:.3f} b={b_hat:.3f} "
          f"-> Beta mean={beta_mean:.4f} std={beta_std:.4f}")
    print(f"      AIC pooled={aic_pooled:.1f}  AIC beta-geom={aic_betageom:.1f}  "
          f"(lower better; delta={aic_pooled - aic_betageom:.1f})")
    print(f"      LRT pooled vs beta-geom: stat={lrt_bg:.1f} df=1 p={p_value_bg:.2e}")
    print()

    return dict(name=name, rows=rows, n=n, T=T, p_hat_i=p_hat_i, p_pooled=p_pooled,
                z_i=z_i, a_hat=a_hat, b_hat=b_hat, beta_mean=beta_mean, beta_std=beta_std)


results = [analyze(**d) for d in DATASETS]
