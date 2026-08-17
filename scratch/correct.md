# Branching-process model for "tokens accepted per call"

Self-contained description of the model used to fit `choice-k` / `seq-ptp` /
`seq-n-ptp-choice-k` per-call acceptance-count data, plus runnable code for
computing `E[T]`, `P(T=1)` from parameters, and for fitting parameters back
out of observed data (aggregate `E[T]`, aggregate `P(T=1)`, or raw samples of
`T`).

## 1. The model

### 1.1 Generative process

Start with `K` candidate "strands" (e.g. `K` = choice-k ensemble size, or
`K=1` for a single-strand algorithm like plain `seq-ptp`). At each generation
`i = 1, 2, ...`:

```
S_0 = K
S_i | S_{i-1} = m  ~  BetaBinomial(m, alpha, beta)
```

i.e. a shared "frailty" `Theta_i ~ Beta(alpha, beta)` is drawn fresh each
generation and applied to *all* `m` surviving strands that round:

```
Theta_i ~ Beta(alpha, beta)
S_i | S_{i-1}=m, Theta_i  ~  Binomial(m, Theta_i)
```

Marginalizing out `Theta_i` gives the Beta-Binomial transition above. The
shared `Theta_i` is what correlates the strands within a generation — it is
*not* redrawn per-strand.

`T = min{ i >= 1 : S_i = 0 }` is the **extinction time** — the number of
generations until every strand has failed. In the token-acceptance
application, `G` (tokens accepted per call) equals `T` plus a fixed
structural floor (usually 1, from a guaranteed first-token match): `G = 1 + T`.

### 1.2 Reparametrization: (alpha, beta) <-> (p, rho)

Work in `(p, rho)` instead of `(alpha, beta)` — `p` is the marginal
per-strand success probability and `rho` is the pairwise correlation between
any two strands in the same generation:

```
nu = (1 - rho) / rho              # "concentration", alpha+beta
alpha = p * nu
beta  = (1 - p) * nu

# inverse:
p   = alpha / (alpha + beta)
rho = 1 / (alpha + beta + 1)
```

Properties: `E[Theta] = p`, `Var(Theta) = p(1-p)*rho`, and `rho` is exactly
the Beta-Binomial intraclass correlation, `Corr(X_i, X_j) = rho` for `i != j`
within one generation. `rho in (0,1)` — the Beta-Binomial can only represent
**positive** correlation.

### 1.3 Zero-inflation (a nonzero floor as K -> infinity)

The plain model above has `P(T=1; K) -> 0` as `K -> infinity` (see §2.5) —
i.e. it's always possible to rescue an arbitrarily large ensemble from total
first-round failure. Empirically this is false: `P(X=2)` (equivalently
`P(T=1)`) plateaus at a nonzero floor as `k` grows. Fix: give `Theta` a point
mass at 0 —

```
Theta ~ pi0 * delta_0  +  (1 - pi0) * Beta(alpha, beta)
```

With probability `pi0` the round is a guaranteed total wipeout regardless of
`K` (e.g. a genuinely unpredictable/high-entropy position). Transition law
becomes:

```
P(S_i=0 | S_{i-1}=m) = pi0 + (1-pi0) * BetaBinom(0; m, alpha, beta)
P(S_i=j | S_{i-1}=m) = (1-pi0) * BetaBinom(j; m, alpha, beta)     for j >= 1
```

Setting `pi0 = 0` recovers the plain model.

### 1.4 Structural properties

- **Monotonicity of the S_i sequence**: `S_i <= S_{i-1}` always, with
  probability 1, for every `K` and every realization — a Beta-Binomial with
  `m` trials only has support `{0,...,m}`, so you can't draw a value larger
  than `m`. This is what makes `T` a well-defined absorption time.
- **`P(T=x)` is *not* necessarily monotonically decreasing in `x`.** For
  small `K` it usually is (mode at `T=1`), but for a "just right" `K` (large
  enough that instant wipeout is somewhat suppressed, small enough that
  extinction by generation 2 is still very likely) the pmf can have a small
  bump at `T=2` instead. Don't assume unimodality-at-1 without checking.

## 2. Computing distributional quantities from parameters

### 2.1 Transition kernel (code)

Two implementations: a simple one using `scipy.stats.betabinom` (fine for
small `K`, e.g. K<=50), and a vectorized log-gamma version (needed for large
`K`, e.g. K=1000 — building the matrix via a Python loop of per-row
`scipy.stats.betabinom.pmf` calls is the dominant cost and is ~10x slower).

```python
import numpy as np
from scipy import stats, special

def ab_from_prho(p, rho):
    nu = (1 - rho) / rho
    return p * nu, (1 - p) * nu

# --- simple version (fine for K <~ 50) ---
def transition_matrix(pi0, p, rho, K):
    alpha, beta = ab_from_prho(p, rho)
    P = np.zeros((K + 1, K + 1))
    P[0, 0] = 1.0                      # 0 is absorbing
    for m in range(1, K + 1):
        bb = stats.betabinom.pmf(np.arange(m + 1), m, alpha, beta)
        row = (1 - pi0) * bb
        row[0] += pi0
        P[m, :m + 1] = row
    return P

# --- vectorized version (fast for large K, e.g. K=1000) ---
def transition_matrix_fast(pi0, p, rho, K):
    alpha, beta = ab_from_prho(p, rho)
    m = np.arange(K + 1)[:, None]        # rows
    j = np.arange(K + 1)[None, :]        # cols
    with np.errstate(invalid='ignore'):
        logC  = special.gammaln(m + 1) - special.gammaln(j + 1) - special.gammaln(m - j + 1)
        logB1 = special.gammaln(alpha + j) + special.gammaln(beta + m - j) - special.gammaln(alpha + beta + m)
        logB0 = special.gammaln(alpha) + special.gammaln(beta) - special.gammaln(alpha + beta)
    P = np.exp(logC + logB1 - logB0)
    P[j > m] = 0.0
    P[0, :] = 0.0
    P = (1 - pi0) * P
    P[:, 0] += pi0
    P[0, 0] = 1.0
    return P
```

### 2.2 P(T=1; K) — closed form + code

`T=1` means `S_1 = 0`, i.e. total wipeout in the very first round:

```
P(T=1; K) = pi0 + (1-pi0) * BetaBinom(0; K, alpha, beta)
          = pi0 + (1-pi0) * B(alpha, beta+K) / B(alpha, beta)
```

`B(alpha, beta+K)/B(alpha,beta)` can be computed two ways — a product form
(exact, integer `K` only, no Gamma functions needed) or via `lnGamma` (works
for non-integer `K`, useful for smooth-curve plotting in JS where a
`lnGamma`/Lanczos implementation is easy to write but no Beta-function
library is available):

```python
def beta_binom_zero_pmf_product(m, alpha, beta):
    """P(S=0; m,alpha,beta), integer m only — no Gamma functions."""
    prod = 1.0
    for i in range(m):
        prod *= (beta + i) / (alpha + beta + i)
    return prod

def beta_binom_zero_pmf_lngamma(m, alpha, beta):
    """Same, via lnGamma — works for non-integer m too."""
    from scipy.special import gammaln
    log_ratio = (gammaln(beta + m) - gammaln(beta)) + (gammaln(alpha + beta) - gammaln(alpha + beta + m))
    return np.exp(log_ratio)

def P_T1(K, pi0, p, rho):
    alpha, beta = ab_from_prho(p, rho)
    return pi0 + (1 - pi0) * beta_binom_zero_pmf_lngamma(K, alpha, beta)
```

JS port (Lanczos approximation for `lnGamma`, used in the artifact so the
fitted curve renders smoothly for non-integer `k` on a log-scale x-axis):

```js
const LANCZOS_G = 7;
const LANCZOS_C = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028,
  771.32342877765313, -176.61502916214059, 12.507343278686905,
  -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];
function lnGamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
  x -= 1;
  let a = LANCZOS_C[0];
  const t = x + LANCZOS_G + 0.5;
  for (let i = 1; i < LANCZOS_G + 2; i++) a += LANCZOS_C[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function betaBinomZeroPmf(m, alpha, beta) {
  const lnRatio = (lnGamma(beta + m) - lnGamma(beta)) + (lnGamma(alpha + beta) - lnGamma(alpha + beta + m));
  return Math.exp(lnRatio);
}
function P_T1(K, pi0, p, rho) {
  const nu = (1 - rho) / rho;
  const alpha = p * nu, beta = (1 - p) * nu;
  return pi0 + (1 - pi0) * betaBinomZeroPmf(K, alpha, beta);
}
```

### 2.3 Asymptotics as K -> infinity

Using `Gamma(K+beta)/Gamma(K+alpha+beta) ~ K^(-alpha)`:

```
P(T=1; K)  ~  pi0 + (1-pi0) * [Gamma(alpha+beta)/Gamma(beta)] * K^(-alpha)     as K -> infinity
           ->  pi0
```

So `P(T=1;K)` decays as a **power law in K with exponent `-alpha`**, plateauing
at `pi0` — not an exponential decay like independent trials would give
(`(1-p)^K`). This is the mechanistic explanation for why an empirical
`p(k) = c + a*k^(-b)` curve fit works: `c ~ pi0`, `b ~ alpha`.

### 2.4 Full pmf of T via DP + code

Propagate the state-distribution vector through the transition matrix; `T`'s
pmf is the increment in absorbed mass at each step:

```python
def T_pmf(pi0, p, rho, K, max_i=200):
    """Returns array pmf where pmf[i-1] = P(T=i), i=1..max_i."""
    P = transition_matrix_fast(pi0, p, rho, K)   # or transition_matrix for small K
    v = np.zeros(K + 1); v[K] = 1.0
    pmf = []
    prev_mass0 = v[0]
    for _ in range(max_i):
        v = v @ P
        pmf.append(v[0] - prev_mass0)
        prev_mass0 = v[0]
    return np.array(pmf)
```

### 2.5 E[T] via the tail-sum identity + code

For a non-negative-integer-valued `T >= 1`:

```
E[T] = sum_{i=0}^infinity P(S_i > 0) = sum_{i=0}^infinity [1 - v_i(0)]
```

(the `i=0` term is 1 since `S_0=K>0`). No closed form in general — compute
numerically via the same propagation as §2.4, summed as survival
probabilities instead of first-passage probabilities:

```python
def E_T(pi0, p, rho, K, max_i=300):
    P = transition_matrix_fast(pi0, p, rho, K)
    v = np.zeros(K + 1); v[K] = 1.0
    total = 1.0                       # i=0 term
    for _ in range(max_i):
        v = v @ P
        surv = 1 - v[0]
        total += surv
        if surv < 1e-13:
            break
    return total
```

`E[G] = 1 + E[T]` in the application (adjust the `+1` if your structural
floor differs).

## 3. Fitting parameters from data

Three regimes, from least to most information-rich: (a) you only have
aggregate `E[T]` (or `E[G]`) at a few `K` values, (b) you have `P(T=1)` (a
Binomial count `x` out of `n` trials) at a few `K` values, (c) you have raw
per-call samples of `T` at one or more `K` values (the richest signal — full
histogram, not just a summary statistic).

**Performance note**: for `K` in the hundreds/thousands, use
`transition_matrix_fast` (§2.1) — building the matrix with a Python loop over
`scipy.stats.betabinom.pmf` calls per row is the dominant cost in a fit and
is slow enough to make optimizer runs take minutes instead of seconds.

### 3.1 Fitting from aggregate E[T] values (nonlinear least squares)

Use when you only have mean tokens/call per `K` (e.g. summarized results
directories), not raw per-call data.

```python
from scipy import optimize

def fit_from_ET(Ks, ET_obs, x0=(0.1, 0.6, 0.6)):
    """Ks: list of K values. ET_obs: matching list of observed E[T] = E[G]-1."""
    Ks = list(Ks)
    ET_obs = np.asarray(ET_obs)

    def sse(params):
        pi0, p, rho = params
        if not (0 <= pi0 < 1 and 1e-4 < p < 1 - 1e-4 and 1e-4 < rho < 1 - 1e-4):
            return 1e10
        preds = np.array([E_T(pi0, p, rho, k) for k in Ks])
        return np.sum((preds - ET_obs) ** 2)

    res = optimize.minimize(sse, x0=x0, method='Nelder-Mead',
                             options={'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 400})
    pi0, p, rho = res.x
    preds = np.array([E_T(pi0, p, rho, k) for k in Ks])
    ss_tot = ((ET_obs - ET_obs.mean()) ** 2).sum()
    r2 = 1 - res.fun / ss_tot
    return dict(pi0=pi0, p=p, rho=rho, r2=r2, preds=preds, success=res.success)
```

### 3.2 Fitting from P(T=1) counts (Binomial likelihood)

Use when you have, per `K`, an exact count `x` (calls landing at `T=1`) out
of `n` trials — richer than a plain mean since it's an exact likelihood, not
a least-squares proxy.

```python
def fit_from_PT1_counts(rows, x0=(0.1, 0.5, 0.5)):
    """rows: list of (K, x, n) tuples."""
    Ks = np.array([r[0] for r in rows], dtype=float)
    xs = np.array([r[1] for r in rows], dtype=float)
    ns = np.array([r[2] for r in rows], dtype=float)

    def neg_log_lik(params):
        pi0, p, rho = params
        if not (0 <= pi0 < 1 and 1e-4 < p < 1 - 1e-4 and 1e-4 < rho < 1 - 1e-4):
            return 1e10
        probs = np.array([P_T1(k, pi0, p, rho) for k in Ks])
        probs = np.clip(probs, 1e-12, 1 - 1e-12)
        ll = xs * np.log(probs) + (ns - xs) * np.log(1 - probs)
        return -ll.sum()

    res = optimize.minimize(neg_log_lik, x0=x0, method='Nelder-Mead',
                             options={'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 400})
    pi0, p, rho = res.x
    return dict(pi0=pi0, p=p, rho=rho, nll=res.fun, success=res.success)

# example usage:
rows = [(1, 45, 100), (2, 31, 85), (5, 20, 79), (50, 11, 70), (250, 8, 64), (1000, 8, 60)]
fit = fit_from_PT1_counts(rows)
```

### 3.3 Fitting from raw T samples at a single K (full-histogram MLE)

The richest single-K fit — uses the *entire* observed distribution, not just
its mean or `P(T=1)`.

```python
def fit_from_samples(T_samples, K, x0=(0.05, 0.6, 0.6), fix_pi0=None):
    """
    T_samples: array of observed T values (already shifted: T = G - floor).
    fix_pi0: pass a float to hold pi0 fixed (e.g. calibrated from a separate
             multi-K fit) and only fit (p, rho) — recommended when a single K
             can't identify pi0 on its own (see 3.5).
    """
    T_samples = np.asarray(T_samples)
    max_i = int(T_samples.max()) + 10

    def neg_log_lik(params):
        if fix_pi0 is not None:
            pi0 = fix_pi0
            p, rho = params
        else:
            pi0, p, rho = params
        if not (0 <= pi0 < 1 and 1e-4 < p < 1 - 1e-4 and 1e-4 < rho < 1 - 1e-4):
            return 1e10
        pmf = T_pmf(pi0, p, rho, K, max_i)
        probs = np.clip(pmf[T_samples - 1], 1e-300, None)
        return -np.sum(np.log(probs))

    x0_use = x0[1:] if fix_pi0 is not None else x0
    res = optimize.minimize(neg_log_lik, x0=x0_use, method='Nelder-Mead',
                             options={'xatol': 1e-7, 'fatol': 1e-8, 'maxfev': 2000})
    if fix_pi0 is not None:
        p, rho = res.x
        pi0 = fix_pi0
    else:
        pi0, p, rho = res.x
    return dict(pi0=pi0, p=p, rho=rho, nll=res.fun, success=res.success)
```

### 3.4 Joint fit across multiple K's using raw samples

**First, exclude floor violations from every dataset before fitting** — `T`
is always `>= 1` by construction (it can never be 0), so any observed sample
with `G < 1 + shift` (i.e. below the dataset's expected structural floor)
falls outside the model's support and must be dropped from the likelihood,
not passed through. This matters most for datasets whose floor isn't
perfectly hard (e.g. `seq-ptp` has a ~0.6% rate of `G=1` samples via
early-EOS truncation, against an expected floor of `G=2`).

```python
def prepare_T(G_samples, floor):
    """G_samples: raw observed token-counts. floor: expected structural floor
    (2 for choice-k algorithms, 2 for seq-ptp too — same guaranteed-token
    mechanism, just with rare violations). Returns T = G - (floor - 1),
    dropping any sample below the floor."""
    G_samples = np.asarray(G_samples)
    kept = G_samples[G_samples >= floor]
    n_dropped = len(G_samples) - len(kept)
    return kept - (floor - 1), n_dropped
```

Then sum the per-dataset negative log-likelihoods, sharing whichever
parameters you want shared:

```python
def fit_joint(datasets, x0):
    """
    datasets: list of dicts, each {'T': array, 'K': int} (already floor-cleaned
    via prepare_T above).
    Fits ONE (pi0, p, rho) shared across every dataset.
    """
    max_is = [int(d['T'].max()) + 10 for d in datasets]

    def neg_log_lik(params):
        pi0, p, rho = params
        if not (0 <= pi0 < 1 and 1e-4 < p < 1 - 1e-4 and 1e-4 < rho < 1 - 1e-4):
            return 1e10
        total = 0.0
        for d, max_i in zip(datasets, max_is):
            pmf = T_pmf(pi0, p, rho, d['K'], max_i)
            probs = np.clip(pmf[d['T'] - 1], 1e-300, None)
            total -= np.sum(np.log(probs))
        return total

    res = optimize.minimize(neg_log_lik, x0=x0, method='Nelder-Mead',
                             options={'xatol': 1e-7, 'fatol': 1e-8, 'maxfev': 6000})
    pi0, p, rho = res.x
    return dict(pi0=pi0, p=p, rho=rho, nll=res.fun, success=res.success)
```

**Worked example** — `choice-5` (K=5, n=3472, 0 floor violations) jointly
with `seq-ptp` (K=1, n=4159, 26 floor violations dropped), fully sharing
`(pi0, p, rho)`:

```python
T5, _  = prepare_T(choice5_samples, floor=2)   # K=5
T1, n_dropped = prepare_T(seqptp_samples, floor=2)   # K=1, drops 26/4159

fit = fit_joint([{'T': T5, 'K': 5}, {'T': T1, 'K': 1}], x0=[0.1, 0.7, 0.7])
# -> pi0=0.000, p=0.678, rho=0.822
# R^2: choice-5=0.975, seq-ptp=0.998  (standalone fits were 0.981 and 0.997)
```

Sharing all three parameters here costs almost nothing versus fitting each
dataset separately — see §3.5 for why.

### 3.5 Weighted least-squares — prioritizing a specific G range

Standard full-histogram MLE (§3.3) weights each *sample* equally, which in
practice weights each *bin* roughly by its natural frequency — but the
likelihood's multiplicative/relative-error character can still trade a bit
of absolute accuracy on the big low-`G` bins for reduced relative error on
the thin tail. If what you actually care about is nailing a specific range
(e.g. `G=2..5`, the range that dominates real throughput), fit a weighted
least-squares objective directly on the histogram instead:

```python
def fit_weighted_histogram(G_samples, K, target_range, weight_in=1.0, weight_out=0.1,
                            x0=(0.1, 0.6, 0.6), maxv=20):
    """
    G_samples: raw observed token counts (not yet shifted to T).
    target_range: (lo, hi) inclusive G values to prioritize.
    """
    G_samples = np.asarray(G_samples)
    counts = np.bincount(G_samples.clip(max=maxv), minlength=maxv + 1).astype(float)
    obs = counts / counts.sum()

    lo, hi = target_range
    weights = np.full(maxv + 1, weight_out)
    weights[lo:hi + 1] = weight_in

    def pred_for(pi0, p, rho, max_i=30):
        pmf_full = T_pmf(pi0, p, rho, K, max_i)
        pred = np.zeros(maxv + 1)
        for t in range(1, max_i):
            g = t + 1                      # G = 1 + T convention
            if g <= maxv:
                pred[g] = pmf_full[t - 1]
        pred[maxv] += max(0, 1 - pred.sum())
        return pred

    def weighted_sse(params):
        pi0, p, rho = params
        if not (0 <= pi0 < 1 and 1e-4 < p < 1 - 1e-4 and 1e-4 < rho < 1 - 1e-4):
            return 1e10
        pred = pred_for(pi0, p, rho)
        return np.sum(weights * (obs - pred) ** 2)

    res = optimize.minimize(weighted_sse, x0=x0, method='Nelder-Mead',
                             options={'xatol': 1e-7, 'fatol': 1e-10, 'maxfev': 4000})
    pi0, p, rho = res.x
    return dict(pi0=pi0, p=p, rho=rho, pred=pred_for(pi0, p, rho), obs=obs)
```

**Worked example** on the real choice-5 data, prioritizing `G=2..5` at 10x
the weight of everything else:

```
standard MLE (unweighted):  p=0.657, rho=0.762  -> G=2..5 R^2=0.744, full R^2=0.981
weighted (G=2..5 x10):      p=0.582, rho=0.583  -> G=2..5 R^2=0.939, full R^2=0.991
```

The weighted fit won on *both* the targeted range and the full histogram
here — not guaranteed in general, but a sign that unweighted MLE was
trading away some low-`G` accuracy for tail accuracy that mattered less.
Always check the full-range R² after weighting, in case the trade goes the
other way for a different dataset.

### 3.6 Identifiability — read this before fitting

- **`pi0` needs a spread of `K` that includes large values.** Its signature
  is "does `P(T=1;K)` keep decaying or does it plateau" — with only small
  `K` (e.g. K=1 and K=5), the continuous Beta-Binomial part alone can fit the
  data just as well, and the optimizer will converge to `pi0 -> 0` every
  time. This isn't a bug; there just isn't enough information in the data to
  tell the two mechanisms apart at small `K`.
- **`rho` has zero effect at `K=1`.** `BetaBinom(1, alpha, beta)` is exactly
  `Bernoulli(alpha/(alpha+beta)) = Bernoulli(p)` regardless of the
  concentration `alpha+beta`. This is actually good news for joint fits: a
  `K=1` dataset's likelihood term simply doesn't depend on `rho` at all, so
  sharing `rho` with a `K>1` dataset costs the `K=1` fit nothing.
- **Fully sharing `(pi0, p, rho)` across structurally different
  algorithms/datasets works fine in practice** as long as (a) the
  floor/shift convention is correctly matched per-dataset first (§3.4's
  `prepare_T`), and (b) each dataset's individually-optimal `p` isn't too
  far from the others' — check this by fitting standalone first (§3.3) and
  comparing. `choice-5` alone wants `p=0.657`; `seq-ptp` alone wants
  `p=0.682` — close enough that a shared compromise (`p=0.678`) barely hurts
  either. If the individual `p`'s were far apart, forcing them to share would
  degrade the worse-fitting dataset badly — check the per-dataset R² after
  any shared fit to make sure this hasn't happened.
- **An earlier version of this analysis wrongly concluded fully-sharing
  `(p,rho)` was a bad idea** (seq-ptp's R² appeared to collapse to ~0.40).
  That was an indexing bug — `T` was computed inconsistently between the two
  datasets (mixing the "no shift" and "shift by floor-1" conventions), not a
  real incompatibility. Once both datasets use the *same*, correctly-derived
  `T = G - (floor - 1)` with floor violations dropped (not force-fit and not
  silently mis-indexed), the shared fit works well. Moral: get the
  shift/exclusion convention exactly right and identically applied across
  every dataset *before* concluding parameters can't be shared.

## 4. Model variant: pi0 exempted on the first generation

**Hypothesis**: zero-inflation (`pi0`, the point mass at "total wipeout this
generation") shouldn't apply to the very first generation `S_0 -> S_1` — only
from the second generation onward. Concretely: `Theta_1 ~ Beta(alpha,beta)`
(no mixture), while `Theta_i ~ pi0*delta_0 + (1-pi0)*Beta(alpha,beta)` for
`i >= 2`, same as before.

### 4.1 Consequence: P(T=1;K) has no pi0 dependence at all

Since the first transition is always pure `BetaBinomial(K,alpha,beta)`:

```
P(T=1; K) = BetaBinom(0; K, alpha, beta)      -- no "pi0 + (1-pi0)*..." wrapper
```

This means **pi0 is unidentifiable from a P(T=1)-only fit** under this
convention — a fit against just the `(K, successes, n)` sweep can only
recover `(p, rho)`. pi0 only becomes identifiable once the fit target
involves generation >= 2 (full per-call histograms, or E[T]/E[G] means,
which integrate over all generations).

### 4.2 Code change

Only the transition-application logic changes — `transition_matrix_fast`
itself is unchanged and reused for both roles:

```python
def T_pmf_v2(pi0, p, rho, K, max_i=200):
    P_first = transition_matrix_fast(0.0, p, rho, K)   # pure BB, first step only
    P_rest  = transition_matrix_fast(pi0, p, rho, K)    # zero-inflated, i>=2
    v = np.zeros(K + 1); v[K] = 1.0
    pmf, prev_mass0 = [], v[0]
    for i in range(max_i):
        P = P_first if i == 0 else P_rest
        v = v @ P
        pmf.append(v[0] - prev_mass0)
        prev_mass0 = v[0]
    return np.array(pmf)
```

`E_T_v2` is the same substitution inside the tail-sum loop (§2.5). Sanity
check: at `pi0=0` this must exactly match the original `T_pmf`/`E_T`, since
both kernels collapse to the same pure-BB matrix — verified numerically.

### 4.3 Refit results (same 5 real 100-question-run datasets: seq-ptp K=1,
choice-2 K=2, choice-5 K=5, choice-50 K=50, seq-n-choice-1000 K=1000)

| fit | v1 (pi0 always applies) | v2 (pi0 exempt on step 1) |
|---|---|---|
| P(T=1) fit | pi0=0.095, p=0.726, rho=0.630, R&sup2;=0.990 | pi0=n/a, p=0.675, rho=0.793, R&sup2;=0.961 |
| E[T] fit | pi0=0.188, p=0.838, rho=0.674, R&sup2;=0.999 | pi0=0.232, p=0.807, rho=0.677, R&sup2;=0.999 |
| joint MLE | pi0=0.000, p=0.675, rho=0.868, NLL=38183.2 | pi0=0.063, p=0.712, rho=0.856, NLL=38098.9 |
| joint weighted | pi0=0.074, p=0.709, rho=0.754, wSSE=0.0119 | pi0=0.061, p=0.679, rho=0.801, wSSE=0.0051 |

Per-dataset full-histogram R&sup2; under each joint fit:

| dataset | jointM v1 | jointM v2 | jointW v1 | jointW v2 |
|---|---|---|---|---|
| seq-ptp (K=1) | 0.998 | 0.979 | 0.998 | 0.990 |
| choice-2 (K=2) | 0.999 | 0.986 | 0.999 | 0.990 |
| choice-5 (K=5) | 0.960 | 0.995 | 0.975 | 0.993 |
| choice-50 (K=50) | 0.902 | 0.970 | 0.939 | 0.986 |
| seq-n-choice-1000 (K=1000) | 0.921 | 0.981 | 0.899 | 0.973 |

**Takeaway**: v2's joint fits (the richest calibration — full per-call
histograms across all 5 K's) are a clear improvement: lower NLL, less than
half the weighted SSE, and a much more even R&sup2; spread across K (v1 fit
the small-K datasets very well at the expense of choice-50/seq-n-1000; v2
trades a little accuracy at K=1,2 for a lot more at K=50,1000). The
standalone E[T] fit is a wash (same R&sup2; on its own target, but *worse*
at explaining full histograms directly — 0.74-0.81 vs 0.83-1.00 under v1 —
since it was only ever calibrated against 5 summary means either way). The
standalone P(T=1) fit is weaker under v2 (0.961 vs 0.990) because it loses a
free parameter (pi0) for that specific target — not a fair comparison, since
pi0 structurally cannot help there anymore.

**Net**: v2 adopted as the current model in the artifact. A backup of the v1
version is kept at `shift1_geometric_fit.BACKUP_pi0_always_applies.html` in
both the scratchpad and `scratch/` — republish that file to revert.

## 5. Model variant: pi0 exempt on generation 1 + forced at the known cap (v3)

Combines two findings from the per-step exploration (seq-ptp and choice-1000,
both fully free per-generation pi0, then logistic/power-law/linear/step
schedules — see prior session notes): (a) generation 1 behaves differently
from the rest and is best left pure BetaBinomial (no pi0), and (b) the only
real, AIC-justified structure beyond a single constant pi0 is a deterministic
cap at each dataset's own last observed generation — reflecting the actual
`tokens_per_student_call=20` proposal-length limit, not a fitted pattern.

```
generation 1:          pure BetaBinomial(K,alpha,beta)      -- pi0 exempt
generations 2..capT-1: zero-inflated with constant pi0
generation capT:       certain death (pi0=1)                -- capT = 20 for
                       seq-ptp, 19 for choice-2/5/50/seq-n-choice-1000
```

```python
def T_pmf_v3(pi0, p, rho, K, capT, max_i):
    P0 = transition_matrix_fast(0.0, p, rho, K)     # generation 1, and the "no pi0" baseline
    Pc = transition_matrix_fast(pi0, p, rho, K)      # generations 2..capT-1
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
```

### Refit results (same 5 datasets)

| fit | pi0 | p | rho | note |
|---|---|---|---|---|
| P(T=1) fit | n/a | 0.675 | 0.793 | unchanged from v2 — gen-1 exemption means pi0 never enters this fit |
| E[T] fit | 0.230 | 0.806 | 0.678 | ~unchanged from v2 (cap barely affects the mean) |
| joint MLE | 0.062 | 0.713 | 0.858 | **NLL 37993.2** vs v2's 38098.9 — ~106 nat improvement, same (p,rho) region |
| joint weighted | 0.061 | 0.679 | 0.801 | unchanged from v2 (weighted objective barely touches the capped tail bin) |

Per-dataset R&sup2; barely moves between v2 and v3 (e.g. choice-50 jointM:
0.970&rarr;0.970, seq-n-1000 jointM: 0.981&rarr;0.980) — **R&sup2; is the
wrong metric to see this improvement**, since it's dominated by the large
early bins where v2/v3 predict almost identically. The improvement is
entirely in the tail-probability calibration (one or two bins near each
dataset's cap), which is invisible to an aggregate-proportion R&sup2; but
costs real log-likelihood when real samples land there.

**Standalone confirmation on choice-1000 alone** (isolating just this
dataset, not the shared joint fit) found the same structure independently:
free MLE over (pi0_first=0 fixed, pi0_mid free, pi0_last=1 fixed, p, rho)
gave pi0_mid=0.088, p=0.697, rho=0.833, NLL=7041.2 (vs 7064.8 for the
best single-constant-pi0 + free-t0 alternative) — the biggest single-dataset
improvement found in this whole exploration, confirming exempting generation
1 matters independently of the terminal cap.

**Net**: v3 adopted as the current model. Backups of both prior versions
kept for easy revert:
- `shift1_geometric_fit.BACKUP_pi0_always_applies.html` (pi0 constant, every generation)
- `shift1_geometric_fit.BACKUP_v2_pi0_exempt_first_step_only.html` (pi0 constant, gen&ge;2, no cap)
