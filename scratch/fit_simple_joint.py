"""
Fits the simple single-(pi0, p, rho) v3 branching model shared across ALL FIVE
datasets (K=1,2,5,50,1000) AND across all questions (no per-question/per-K free p) --
replicating exactly how vicuna's _BRANCH_PI0/_BRANCH_P/_BRANCH_RHO (scratch/v3_jointM_fit.json:
pi0=0.0616, p=0.713, rho=0.858) were fit, for a different checkpoint/prefix.

Usage: fit_simple_joint.py [PREFIX]
"""
import json
import re
import sys
import numpy as np
from scipy import special, optimize

_PREFIX = sys.argv[1] if len(sys.argv) > 1 else ""

DATASETS = [
    dict(name="seqptp", K=1, capT=20, G_path=f"scratch/{_PREFIX}choice1_percall_full.json",
         log_path=f"scratch/collect_{_PREFIX}choice1.log", calls_key="n_calls"),
    dict(name="choice2", K=2, capT=19, G_path=f"scratch/{_PREFIX}choice2_percall_full.json",
         log_path=f"scratch/collect_{_PREFIX}choice2.log", calls_key="n_calls"),
    dict(name="choice5", K=5, capT=19, G_path=f"scratch/{_PREFIX}choice5_percall_full.json",
         log_path=f"scratch/collect_{_PREFIX}choice5.log", calls_key="n_calls"),
    dict(name="choice50", K=50, capT=19, G_path=f"scratch/{_PREFIX}choice50_percall_full.json",
         log_path=f"scratch/collect_{_PREFIX}choice50.log", calls_key="n_calls"),
    dict(name="seqn1000", K=1000, capT=19, G_path=f"scratch/{_PREFIX}seqn_choice1000_percall_full.json",
         log_path=f"scratch/collect_{_PREFIX}seqn1000_full.log", calls_key="n_rounds"),
]
FLOOR = 2
MAX_I = 20


def parse_log(path, calls_key):
    pat = re.compile(r"example=(\d+) \(\d+/\d+\) " + calls_key + r"=(\d+) sum=(\d+)")
    return [(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in (pat.search(line) for line in open(path)) if m]


def load_pooled_counts(ds):
    G_all = np.array(json.load(open(ds["G_path"])))
    rows = parse_log(ds["log_path"], ds["calls_key"])
    assert sum(r[1] for r in rows) == len(G_all)
    Gv = G_all[G_all >= FLOOR]
    Tv = Gv - (FLOOR - 1)
    return np.bincount(np.clip(Tv, 1, MAX_I), minlength=MAX_I + 1)[1:]


def transition_matrix(pi0, alpha, beta, w_max):
    m = np.arange(w_max + 1)[:, None]
    j = np.arange(w_max + 1)[None, :]
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


def T_pmf(pi0, p, rho, K, capT, max_i):
    nu = (1 - rho) / rho
    alpha, beta = p * nu, (1 - p) * nu
    P0 = transition_matrix(0.0, alpha, beta, K)
    Pc = transition_matrix(pi0, alpha, beta, K)
    v = np.zeros(K + 1); v[K] = 1.0
    pmf, prev0 = [], v[0]
    for i in range(1, max_i + 1):
        if i >= capT:
            pmf.append(1.0 - prev0); prev0 = 1.0
            continue
        v = v @ (P0 if i == 1 else Pc)
        pmf.append(v[0] - prev0)
        prev0 = v[0]
    return np.array(pmf)


print(f"Loading pooled per-call counts (prefix={_PREFIX!r})...", flush=True)
counts = {}
for ds in DATASETS:
    counts[ds["name"]] = load_pooled_counts(ds)
    print(f"  {ds['name']} (K={ds['K']}): {int(counts[ds['name']].sum())} calls", flush=True)


def neg_ll(params):
    pi0_logit, p_logit, rho_logit = params
    pi0 = 1 / (1 + np.exp(-pi0_logit))
    p = 1 / (1 + np.exp(-p_logit))
    rho = 1 / (1 + np.exp(-rho_logit))
    total = 0.0
    for ds in DATASETS:
        pmf = T_pmf(pi0, p, rho, ds["K"], ds["capT"], MAX_I)
        total += np.sum(counts[ds["name"]] * np.log(np.clip(pmf, 1e-300, None)))
    return -total


x0 = [np.log(0.062 / (1 - 0.062)), np.log(0.713 / (1 - 0.713)), np.log(0.858 / (1 - 0.858))]
print("\nFitting single (pi0, p, rho) shared across all 5 datasets...", flush=True)
res = optimize.minimize(neg_ll, x0=x0, method="Nelder-Mead",
                         options=dict(xatol=1e-5, fatol=1e-2, maxfev=500, disp=True))
pi0_hat = 1 / (1 + np.exp(-res.x[0]))
p_hat = 1 / (1 + np.exp(-res.x[1]))
rho_hat = 1 / (1 + np.exp(-res.x[2]))
print(f"\nFit: pi0={pi0_hat:.6f} p={p_hat:.6f} rho={rho_hat:.6f}  nll={res.fun:.2f}")

out_path = f"scratch/{_PREFIX}v3_jointM_fit.json"
with open(out_path, "w") as f:
    json.dump(dict(p0=pi0_hat, p=p_hat, rho=rho_hat, nll=res.fun), f)
print(f"saved to {out_path}")
