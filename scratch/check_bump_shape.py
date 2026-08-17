import json
import re
import numpy as np
from scipy import special

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
    return [(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for line in open(path) for m in [pat.search(line)] if m]


def load_per_question_counts(ds):
    G_all = np.array(json.load(open(ds["G_path"])))
    rows = parse_log(ds["log_path"], ds["calls_key"])
    per_q_counts, idx = [], 0
    for _, n_calls, _ in rows:
        Gq = G_all[idx: idx + n_calls]; idx += n_calls
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
        pmf.append(v[:, 0] - prev0); prev0 = v[:, 0]
    return np.array(pmf).T


fit = json.load(open("scratch/joint_per_question_pi0_rho.json"))
pi0_star, rho_star = fit["pi0"], fit["rho"]
results = json.load(open("scratch/joint_per_question_results.json"))

for ds in DATASETS:
    name, K, capT = ds["name"], ds["K"], ds["capT"]
    counts = load_per_question_counts(ds)
    n_i = counts.sum(axis=1)
    obs_hist = counts.sum(axis=0) / counts.sum()

    r = results[name]
    p_pooled = r["p_pooled"]
    p_hat_i = np.array(r["p_hat_i"])
    p_i_bayes = np.array(r["p_i_bayes"])

    pred_pooled = T_pmf_v3_batch(pi0_star, np.array([p_pooled]), rho_star, K, capT, MAX_I)[0]
    w_n = n_i / n_i.sum()
    pred_mix_raw = (T_pmf_v3_batch(pi0_star, p_hat_i, rho_star, K, capT, MAX_I) * w_n[:, None]).sum(axis=0)
    pred_mix_bayes = (T_pmf_v3_batch(pi0_star, p_i_bayes, rho_star, K, capT, MAX_I) * w_n[:, None]).sum(axis=0)

    n_show = 6
    print(f"=== {name} (K={K}) ===  T=1..{n_show}")
    print(f"  obs:         {np.round(obs_hist[:n_show], 4)}")
    print(f"  pooled:      {np.round(pred_pooled[:n_show], 4)}")
    print(f"  raw-mix:     {np.round(pred_mix_raw[:n_show], 4)}")
    print(f"  shrink-mix:  {np.round(pred_mix_bayes[:n_show], 4)}")
    obs_bump = obs_hist[1] > obs_hist[0]
    pooled_bump = pred_pooled[1] > pred_pooled[0]
    raw_bump = pred_mix_raw[1] > pred_mix_raw[0]
    shrink_bump = pred_mix_bayes[1] > pred_mix_bayes[0]
    print(f"  bump (T=2 > T=1)?  obs={obs_bump}  pooled={pooled_bump}  raw-mix={raw_bump}  shrink-mix={shrink_bump}")
    print()
