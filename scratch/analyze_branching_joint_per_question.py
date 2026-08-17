"""
Generalizes the per-question hierarchical analysis to the full v3 branching
model (pi0, p, rho), replicating exactly how the "joint MLE" fit in
correct.md sec 5 was made - pi0 and rho shared across ALL FIVE datasets
(K=1,2,5,50,1000) - but now with p free per QUESTION (within each dataset)
instead of a single global p per dataset.

Two-level fit:
  (1) Outer: profile-likelihood MLE over (pi0, rho), shared across all 5
      datasets. For each (pi0,rho) candidate, p is profiled out per question
      per dataset via a precomputed p-grid (batched across the grid for
      speed - the branching pmf is expensive at K=1000, so grid points are
      evaluated together rather than one at a time).
  (2) At the converged (pi0*, rho*): per dataset, fit the marginal
      Beta(a_K,b_K) population distribution over that dataset's 100
      per-question p_i's (separate Beta per K, since baseline predictability
      clearly differs by K - see the p(k) power-law fit elsewhere), get
      shrinkage-adjusted p_i_bayes, and compare aggregate-histogram R^2:
      pooled vs Beta-marginal vs raw-mixture vs shrinkage-mixture.
"""
import json
import re
import sys
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
MAX_I = 20  # covers capT=20 (seqptp) and capT=19 (rest)


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
    assert sum(r[1] for r in rows) == len(G_all), f"{ds['name']}: log/data length mismatch"
    per_q_counts, idx, n_dropped = [], 0, 0
    for _, n_calls, _ in rows:
        Gq = G_all[idx: idx + n_calls]
        idx += n_calls
        Gq_valid = Gq[Gq >= FLOOR]
        n_dropped += len(Gq) - len(Gq_valid)
        Tq = Gq_valid - (FLOOR - 1)
        counts = np.bincount(np.clip(Tq, 1, MAX_I), minlength=MAX_I + 1)[1:]
        per_q_counts.append(counts)
    assert idx == len(G_all)
    return np.array(per_q_counts, dtype=float), n_dropped


def transition_matrix_batch(pi0, p_arr, rho, K):
    """p_arr: (G,). Returns (G, K+1, K+1)."""
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
    """Returns (G, max_i) array; pmf[:,i-1] = P(T=i)."""
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
    return np.array(pmf).T  # (G, max_i)


def T_pmf_v3_single(pi0, p, rho, K, capT, max_i):
    return T_pmf_v3_batch(pi0, np.array([p]), rho, K, capT, max_i)[0]


def r2(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ss_res = ((obs - pred) ** 2).sum()
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


print("Loading per-question data for all 5 datasets...", flush=True)
for ds in DATASETS:
    ds["counts"], ds["n_dropped"] = load_per_question_counts(ds)
    ds["n_i"] = ds["counts"].sum(axis=1)
    print(f"  {ds['name']} (K={ds['K']}): {len(ds['counts'])} questions, "
          f"{int(ds['n_i'].sum())} calls, {ds['n_dropped']} floor violations dropped", flush=True)

OUTER_GRID = np.linspace(0.01, 0.99, 40)


def outer_neg_profile_ll(params):
    log_odds_pi0, log_odds_rho = params
    pi0 = 1 / (1 + np.exp(-log_odds_pi0))
    rho = 1 / (1 + np.exp(-log_odds_rho))
    total = 0.0
    for ds in DATASETS:
        pmf_grid = T_pmf_v3_batch(pi0, OUTER_GRID, rho, ds["K"], ds["capT"], MAX_I)  # (G, MAX_I)
        log_pmf_grid = np.log(np.clip(pmf_grid, 1e-300, None))
        loglik_grid = ds["counts"] @ log_pmf_grid.T  # (n_questions, G)
        total += loglik_grid.max(axis=1).sum()
    return -total


print("\nRunning outer joint profile-likelihood optimization over (pi0, rho)...", flush=True)
warm_start_pi0 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.062
warm_start_rho = float(sys.argv[2]) if len(sys.argv) > 2 else 0.858
x0 = [np.log(warm_start_pi0 / (1 - warm_start_pi0)), np.log(warm_start_rho / (1 - warm_start_rho))]
print(f"warm start: pi0={warm_start_pi0}, rho={warm_start_rho}", flush=True)
res = optimize.minimize(outer_neg_profile_ll, x0=x0, method="Nelder-Mead",
                         options=dict(xatol=1e-4, fatol=1e-2, maxfev=250, disp=True))
pi0_star = 1 / (1 + np.exp(-res.x[0]))
rho_star = 1 / (1 + np.exp(-res.x[1]))
print(f"\nConverged: pi0*={pi0_star:.4f} rho*={rho_star:.4f}  (prior pooled joint fit was pi0=0.062, rho=0.858)")
print(f"profile NLL={res.fun:.1f}  nfev={res.nfev}", flush=True)

with open("scratch/joint_per_question_pi0_rho.json", "w") as f:
    json.dump(dict(pi0=pi0_star, rho=rho_star, nll=res.fun, nfev=res.nfev), f)
print("saved pi0*/rho* to scratch/joint_per_question_pi0_rho.json")
