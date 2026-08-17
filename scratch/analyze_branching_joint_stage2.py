"""
Stage 2: using the jointly-fit (pi0*, rho*) from
scratch/joint_per_question_pi0_rho.json (shared across all 5 K's, found by
profiling out a free per-question p), do the detailed per-dataset analysis:
free per-question p_i, marginal Beta(a_K,b_K) population fit (separate per K
since baseline predictability differs by K), shrinkage-adjusted p_i_bayes,
and aggregate T-histogram R^2 comparison (pooled / Beta-marginal / raw-mix /
shrinkage-mix) against each dataset's own observed histogram.
"""
import json
import re
import numpy as np
from scipy import stats, special, optimize

DATASETS = [
    dict(name="seqptp", K=1, capT=20, G_path="scratch/choice1_percall_full.json",
         log_path="scratch/collect_choice1.log", calls_key="n_calls"),
    dict(name="choice2", K=2, capT=19, G_path="scratch/choice2_percall_full.json",
         log_path="scratch/collect_choice2.log", calls_key="n_calls"),
    dict(name="choice5", K=5, capT=19, G_path="scratch/choice5_percall_full.json",
         log_path="scratch/collect_choice5.log", calls_key="n_calls"),
    dict(name="choice50", K=50, capT=19, G_path="scratch/choice50_percall_full.json",
         log_path="scratch/collect_choice50.log", calls_key="n_calls"),
    dict(name="seqn1000", K=1000, capT=19, G_path="scratch/seqn_choice1000_percall_full.json",
         log_path="scratch/collect_seqn1000_full.log", calls_key="n_rounds"),
]
FLOOR = 2
MAX_I = 20


def parse_log(path, calls_key):
    pat = re.compile(r"example=(\d+) \(\d+/\d+\) " + calls_key + r"=(\d+) sum=(\d+)")
    rows = []
    for line in open(path):
        m = pat.search(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return rows


def load_per_question_counts(ds):
    G_all = np.array(json.load(open(ds["G_path"])))
    rows = parse_log(ds["log_path"], ds["calls_key"])
    assert sum(r[1] for r in rows) == len(G_all)
    per_q_counts, idx = [], 0
    for _, n_calls, _ in rows:
        Gq = G_all[idx: idx + n_calls]
        idx += n_calls
        Gq_valid = Gq[Gq >= FLOOR]
        Tq = Gq_valid - (FLOOR - 1)
        counts = np.bincount(np.clip(Tq, 1, MAX_I), minlength=MAX_I + 1)[1:]
        per_q_counts.append(counts)
    return np.array(per_q_counts, dtype=float)


def transition_matrix_batch(pi0, p_arr, rho, K):
    nu = (1 - rho) / rho
    alpha, beta = p_arr * nu, (1 - p_arr) * nu
    m = np.arange(K + 1)[None, :, None]
    j = np.arange(K + 1)[None, None, :]
    a, b = alpha[:, None, None], beta[:, None, None]
    with np.errstate(invalid="ignore"):
        logC = special.gammaln(m + 1) - special.gammaln(j + 1) - special.gammaln(m - j + 1)
        logB1 = special.gammaln(a + j) + special.gammaln(b + m - j) - special.gammaln(a + b + m)
        logB0 = special.gammaln(a) + special.gammaln(b) - special.gammaln(a + b)
    P = np.exp(logC + logB1 - logB0)
    P[:, j[0] > m[0]] = 0.0
    P[:, 0, :] = 0.0
    P = (1 - pi0) * P
    P[:, :, 0] += pi0
    P[:, 0, 0] = 1.0
    return P


def T_pmf_v3_batch(pi0, p_arr, rho, K, capT, max_i):
    G = len(p_arr)
    P0 = transition_matrix_batch(0.0, p_arr, rho, K)
    Pc = transition_matrix_batch(pi0, p_arr, rho, K)
    v = np.zeros((G, K + 1)); v[:, K] = 1.0
    pmf, prev0 = [], v[:, 0].copy()
    for i in range(1, max_i + 1):
        if i >= capT:
            pmf.append(1.0 - prev0); prev0 = np.ones(G)
            continue
        P = P0 if i == 1 else Pc
        v = np.einsum('gk,gkl->gl', v, P)
        pmf.append(v[:, 0] - prev0)
        prev0 = v[:, 0]
    return np.array(pmf).T


def T_pmf_v3_single(pi0, p, rho, K, capT, max_i):
    return T_pmf_v3_batch(pi0, np.array([p]), rho, K, capT, max_i)[0]


def r2(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ss_res = ((obs - pred) ** 2).sum()
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


fit = json.load(open("scratch/joint_per_question_pi0_rho.json"))
pi0_star, rho_star = fit["pi0"], fit["rho"]
print(f"Using joint fit: pi0*={pi0_star:.4f} rho*={rho_star:.4f}\n")

panel = json.load(open("scratch/panel_data_5way.json"))
panel_key = {"seqptp": "seqptp", "choice2": "choice2", "choice5": "choice5",
             "choice50": "choice50", "seqn1000": "seqn1000"}

results = {}
for ds in DATASETS:
    name, K, capT = ds["name"], ds["K"], ds["capT"]
    counts = load_per_question_counts(ds)
    n_i = counts.sum(axis=1)
    T_pooled_counts = counts.sum(axis=0)
    obs_hist = T_pooled_counts / T_pooled_counts.sum()

    p_grid = np.linspace(1e-3, 1 - 1e-3, 400)
    pmf_grid = T_pmf_v3_batch(pi0_star, p_grid, rho_star, K, capT, MAX_I)
    log_pmf_grid = np.log(np.clip(pmf_grid, 1e-300, None))
    loglik_grid = counts @ log_pmf_grid.T  # (100, 400)

    # pooled single-p (pi0*,rho* fixed)
    def nll_pooled_p(p):
        pmf = T_pmf_v3_single(pi0_star, p[0], rho_star, K, capT, MAX_I)
        return -np.sum(T_pooled_counts * np.log(np.clip(pmf, 1e-300, None)))
    j0 = np.argmax(loglik_grid.sum(axis=0))
    res_p = optimize.minimize(nll_pooled_p, x0=[p_grid[j0]], method="Nelder-Mead")
    p_pooled = res_p.x[0]
    ll_pooled = -res_p.fun

    # free per-question p_i (continuous refinement around grid argmax)
    p_hat_i = np.zeros(len(counts))
    ll_free = 0.0
    for i in range(len(counts)):
        j = np.argmax(loglik_grid[i])
        def nll_i(p, i=i):
            pmf = T_pmf_v3_single(pi0_star, p[0], rho_star, K, capT, MAX_I)
            return -np.sum(counts[i] * np.log(np.clip(pmf, 1e-300, None)))
        r = optimize.minimize(nll_i, x0=[p_grid[j]], method="Nelder-Mead",
                               options=dict(xatol=1e-6, fatol=1e-6))
        p_hat_i[i] = np.clip(r.x[0], 1e-3, 1 - 1e-3)
        ll_free += -r.fun

    lrt_stat = 2 * (ll_free - ll_pooled)
    df = len(counts) - 1
    p_value = stats.chi2.sf(lrt_stat, df)

    # marginal Beta(a,b) fit
    def neg_marginal_ll(params):
        log_a, log_b = params
        a, b = np.exp(log_a), np.exp(log_b)
        log_beta_w = stats.beta.logpdf(p_grid, a, b)
        integrand = np.exp(loglik_grid + log_beta_w[None, :] - loglik_grid.max(axis=1, keepdims=True))
        marg = np.trapezoid(integrand, p_grid, axis=1)
        return -(np.log(marg) + loglik_grid.max(axis=1)).sum()
    res_bg = optimize.minimize(neg_marginal_ll, x0=[np.log(5), np.log(5)], method="Nelder-Mead",
                                options=dict(xatol=1e-6, fatol=1e-6, maxfev=2000))
    a_hat, b_hat = np.exp(res_bg.x)
    ll_bg = -res_bg.fun
    beta_mean = a_hat / (a_hat + b_hat)
    beta_std = np.sqrt(a_hat * b_hat / ((a_hat + b_hat) ** 2 * (a_hat + b_hat + 1)))
    aic_pooled, aic_bg = 2 - 2 * ll_pooled, 4 - 2 * ll_bg
    lrt_bg = 2 * (ll_bg - ll_pooled)
    p_value_bg = stats.chi2.sf(lrt_bg, df=1)

    # shrinkage-adjusted posterior mean
    log_beta_w = stats.beta.logpdf(p_grid, a_hat, b_hat)
    post_unnorm = np.exp(loglik_grid + log_beta_w[None, :] - loglik_grid.max(axis=1, keepdims=True))
    post_norm = post_unnorm / np.trapezoid(post_unnorm, p_grid, axis=1)[:, None]
    p_i_bayes = np.trapezoid(post_norm * p_grid[None, :], p_grid, axis=1)

    # aggregate histogram predictions
    pred_pooled = T_pmf_v3_single(pi0_star, p_pooled, rho_star, K, capT, MAX_I)
    w_final = np.exp(log_beta_w - log_beta_w.max())
    w_final /= np.trapezoid(w_final, p_grid)
    pred_bg = np.trapezoid(pmf_grid * w_final[:, None], p_grid, axis=0)
    pred_bg /= pred_bg.sum()
    w_n = n_i / n_i.sum()
    pred_mix_raw = (T_pmf_v3_batch(pi0_star, p_hat_i, rho_star, K, capT, MAX_I) * w_n[:, None]).sum(axis=0)
    pred_mix_bayes = (T_pmf_v3_batch(pi0_star, p_i_bayes, rho_star, K, capT, MAX_I) * w_n[:, None]).sum(axis=0)

    r2_pooled, r2_bg = r2(obs_hist, pred_pooled), r2(obs_hist, pred_bg)
    r2_raw, r2_bayes = r2(obs_hist, pred_mix_raw), r2(obs_hist, pred_mix_bayes)
    r2_existing = panel["datasets"][panel_key[name]]["jointM_r2"]

    print(f"=== {name} (K={K}) ===")
    print(f"  pooled p={p_pooled:.4f}   free p_i: mean={p_hat_i.mean():.4f} std={p_hat_i.std():.4f}")
    print(f"  LRT free-p_i vs pooled: stat={lrt_stat:.1f} df={df} p={p_value:.2e}")
    print(f"  Beta(a={a_hat:.2f},b={b_hat:.2f}) mean={beta_mean:.4f} std={beta_std:.4f}   "
          f"AIC delta(pooled-beta)={aic_pooled-aic_bg:.1f}  LRT p={p_value_bg:.2e}")
    print(f"  shrinkage p_i_bayes: mean={p_i_bayes.mean():.4f} std={p_i_bayes.std():.4f}")
    print(f"  R^2  pooled={r2_pooled:.5f}  Beta-marginal={r2_bg:.5f}  raw-mix={r2_raw:.5f}  "
          f"shrink-mix={r2_bayes:.5f}   (old pooled-joint-fit R^2={r2_existing:.5f})")
    print()

    results[name] = dict(p_pooled=p_pooled, p_hat_i=p_hat_i.tolist(), p_i_bayes=p_i_bayes.tolist(),
                          a=a_hat, b=b_hat, beta_mean=beta_mean, beta_std=beta_std,
                          r2_pooled=r2_pooled, r2_bg=r2_bg, r2_raw=r2_raw, r2_bayes=r2_bayes,
                          r2_existing=r2_existing, lrt_stat=lrt_stat, p_value=p_value)

with open("scratch/joint_per_question_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved detailed results to scratch/joint_per_question_results.json")
