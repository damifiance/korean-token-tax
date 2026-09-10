"""Pooled-ratio estimands and paired stratified cluster bootstrap.

Primary estimands are corpus-level ratio-of-sums (never means of per-message
ratios):

    R_T   = sum(T_K) / sum(T_E)
    R_B   = sum(B_K) / sum(B_E)
    R_eta = (sum(B_K)/sum(T_K)) / (sum(B_E)/sum(T_E)) = R_B / R_T
    C_K   = sum(T_K,korean-fit) / sum(T_K,baseline)

The tokenizer-gap contraction statistic

    G = R_T,korean-fit / R_T,baseline

is retained as exploratory only. G can fall because the comparison tokenizer
uses more English tokens even when Korean token count does not fall; C_K is
therefore the confirmatory statistic for the stated absolute-reduction control.

Bootstrap resamples complete clusters (all pairs of a cluster together, both
languages together) with replacement, independently within each genre stratum,
preserving per-stratum cluster counts.

Note: R_T, R_B, R_eta are functionally dependent (R_eta = R_B/R_T) and must
not be treated as independent evidence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Logical per-pair columns expected by pooled_stats.
COL_TK = "tokens_ko_baseline"
COL_TE = "tokens_en_baseline"
COL_BK = "bits_ko"
COL_BE = "bits_en"
COL_TK_KF = "tokens_ko_koreanfit"
COL_TE_KF = "tokens_en_koreanfit"


def pooled_stats(df: pd.DataFrame, include_control: bool = True) -> dict:
    s = df.sum(numeric_only=True)
    R_T = s[COL_TK] / s[COL_TE]
    R_B = s[COL_BK] / s[COL_BE]
    R_eta = R_B / R_T
    out = {
        "R_T": float(R_T),
        "R_B": float(R_B),
        "R_eta": float(R_eta),
        "one_minus_R_eta": float(1.0 - R_eta),
    }
    if include_control and COL_TK_KF in df.columns and df[COL_TK_KF].notna().all():
        ratio_kf = s[COL_TK_KF] / s[COL_TE_KF]
        out["C_K"] = float(s[COL_TK_KF] / s[COL_TK])
        out["R_T_koreanfit"] = float(ratio_kf)
        out["G"] = float(ratio_kf / R_T)
    return out


def cluster_bootstrap(df: pd.DataFrame, n_replicates: int, seed: int,
                      cluster_col: str = "cluster_id",
                      stratum_col: str = "genre",
                      include_control: bool = True) -> pd.DataFrame:
    """Paired stratified cluster bootstrap of pooled_stats.

    Within each stratum, resample its clusters with replacement (same count),
    take all rows of each sampled cluster (with multiplicity), and recompute
    the pooled statistics on the union across strata.
    """
    rng = np.random.default_rng(seed)

    # Precompute per-cluster sums; pooled stats depend only on column sums,
    # so summing per cluster first is exact and fast.
    value_cols = [c for c in (COL_TK, COL_TE, COL_BK, COL_BE, COL_TK_KF, COL_TE_KF)
                  if c in df.columns]
    per_cluster = (
        df.groupby([stratum_col, cluster_col], observed=True)[value_cols]
        .sum()
        .reset_index()
    )
    strata = []
    for stratum, sub in per_cluster.groupby(stratum_col, observed=True):
        strata.append(sub[value_cols].to_numpy(dtype=np.float64))

    rows = []
    for _ in range(n_replicates):
        sums = np.zeros(len(value_cols), dtype=np.float64)
        for arr in strata:
            n = arr.shape[0]
            idx = rng.integers(0, n, size=n)
            sums += arr[idx].sum(axis=0)
        rep = pd.DataFrame([sums], columns=value_cols)
        rows.append(pooled_stats(rep, include_control=include_control))
    return pd.DataFrame(rows)


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def one_sided_lower(samples: np.ndarray, alpha: float = 0.05) -> float:
    return float(np.quantile(samples, alpha))


def one_sided_upper(samples: np.ndarray, alpha: float = 0.05) -> float:
    return float(np.quantile(samples, 1 - alpha))


def summarize(point: dict, boot: pd.DataFrame, alpha_two_sided: float = 0.05) -> dict:
    out = {}
    for k, v in point.items():
        if k not in boot.columns:
            continue
        samples = boot[k].to_numpy()
        lo, hi = percentile_ci(samples, alpha_two_sided)
        out[k] = {
            "point": v,
            "ci_low": lo,
            "ci_high": hi,
            "alpha_two_sided": alpha_two_sided,
            "n_replicates": int(len(samples)),
        }
    return out


def confirmatory_decisions(boot: pd.DataFrame, thresholds: dict,
                           alpha: float = 0.05) -> dict:
    """Prespecified decision rules. `thresholds` must contain c_T, L_B, U_B,
    c_eta (and optionally c_C). Raises if any needed threshold is None —
    thresholds may never be inferred from the data.

    R_T, R_B, R_eta decisions are NOT independent (R_eta = R_B / R_T).
    """
    for key in ("c_T", "L_B", "U_B", "c_eta"):
        if thresholds.get(key) is None:
            raise ValueError(f"threshold {key} is not set; confirmatory decisions blocked")

    rt = boot["R_T"].to_numpy()
    rb = boot["R_B"].to_numpy()
    reta = boot["R_eta"].to_numpy()

    rt_lower = one_sided_lower(rt, alpha)
    # TOST-style equivalence at level alpha: (1 - 2*alpha) two-sided CI in [L_B, U_B]
    rb_eq_lo, rb_eq_hi = percentile_ci(rb, 2 * alpha)
    reta_upper = one_sided_upper(reta, alpha)

    out = {
        "R_T": {
            "rule": f"one-sided {1-alpha:.0%} lower bound > c_T={thresholds['c_T']}",
            "lower_bound": rt_lower,
            "pass": bool(rt_lower > thresholds["c_T"]),
        },
        "R_B_equivalence": {
            "rule": (f"{1-2*alpha:.0%} CI inside [{thresholds['L_B']}, {thresholds['U_B']}]"),
            "ci": [rb_eq_lo, rb_eq_hi],
            "pass": bool(thresholds["L_B"] < rb_eq_lo and rb_eq_hi < thresholds["U_B"]),
        },
        "R_eta": {
            "rule": f"one-sided {1-alpha:.0%} upper bound < c_eta={thresholds['c_eta']}",
            "upper_bound": reta_upper,
            "pass": bool(reta_upper < thresholds["c_eta"]),
        },
        "note": "R_eta = R_B / R_T; these three endpoints are functionally dependent.",
    }
    if "C_K" in boot.columns and thresholds.get("c_C") is not None:
        ck_upper = one_sided_upper(boot["C_K"].to_numpy(), alpha)
        out["C_K"] = {
            "rule": f"one-sided {1-alpha:.0%} upper bound < c_C={thresholds['c_C']}",
            "upper_bound": ck_upper,
            "pass": bool(ck_upper < thresholds["c_C"]),
            "note": ("C_K < 1 is an absolute reduction in Korean token count under "
                     "the comparison tokenizer. Attribution to training-language balance "
                     "still requires controlling algorithm, vocab size, normalization, "
                     "pre-tokenization, and training-corpus size."),
        }
    return out


def simulate_power(pilot_df: pd.DataFrame, thresholds: dict,
                   n_clusters_grid: list[int], n_sim: int = 200,
                   n_boot: int = 1000, alpha: float = 0.05,
                   seed: int = 0, cluster_col: str = "cluster_id",
                   stratum_col: str = "genre") -> pd.DataFrame:
    """Simulation-based power using pilot per-cluster sums as the population.

    Independent sample size = number of clusters. For each target size m,
    draw m clusters with replacement from the pilot (ignoring strata if the
    pilot is too small to stratify), run the bootstrap decision rules, and
    report the pass rate per endpoint.
    """
    rng = np.random.default_rng(seed)
    value_cols = [c for c in (COL_TK, COL_TE, COL_BK, COL_BE, COL_TK_KF, COL_TE_KF)
                  if c in pilot_df.columns]
    per_cluster = (
        pilot_df.groupby(cluster_col, observed=True)[value_cols].sum().reset_index()
    )
    arr = per_cluster[value_cols].to_numpy(dtype=np.float64)

    records = []
    for m in n_clusters_grid:
        passes = {"R_T": 0, "R_B_equivalence": 0, "R_eta": 0}
        for _ in range(n_sim):
            idx = rng.integers(0, arr.shape[0], size=m)
            sim = pd.DataFrame(arr[idx], columns=value_cols)
            sim[cluster_col] = np.arange(m)
            sim[stratum_col] = "all"
            boot = cluster_bootstrap(sim, n_boot, int(rng.integers(0, 2**31)),
                                     cluster_col=cluster_col, stratum_col=stratum_col,
                                     include_control=False)
            dec = confirmatory_decisions(boot, thresholds, alpha)
            for k in passes:
                passes[k] += int(dec[k]["pass"])
        rec = {"n_clusters": m}
        rec.update({f"power_{k}": v / n_sim for k, v in passes.items()})
        records.append(rec)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Exploratory / secondary analyses (never feed confirmatory decisions)
# ---------------------------------------------------------------------------

def per_pair_ratio_summary(df: pd.DataFrame) -> dict:
    """Distribution of per-message ratios. Exploratory description only: the
    estimand remains the pooled ratio-of-sums, and the mean of per-pair ratios
    is a different (and biased-upward) quantity."""
    rt = df[COL_TK] / df[COL_TE]
    rb = df[COL_BK] / df[COL_BE]
    reta = rb / rt
    out = {}
    for name, s in (("R_T", rt), ("R_B", rb), ("R_eta", reta)):
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        out[name] = {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "median": float(q[0.5]),
            "geometric_mean": float(np.exp(np.log(s[s > 0]).mean())),
            "p05": float(q[0.05]), "p25": float(q[0.25]),
            "p75": float(q[0.75]), "p95": float(q[0.95]),
        }
    out["note"] = ("Per-pair ratios are descriptive only; the estimand is the pooled "
                   "ratio-of-sums and is not the mean of these ratios.")
    return out


def assign_length_strata(df: pd.DataFrame, length_col: str, k: int = 3) -> pd.Series:
    """Label rows by quantile bins of `length_col` (k bins, 'Q1' shortest).
    Bin edges are computed from the analysed data and reported by the caller."""
    ranks = df[length_col].rank(method="first")
    labels = [f"Q{i + 1}" for i in range(k)]
    return pd.qcut(ranks, q=k, labels=labels).astype(str)


def stratified_estimates(df: pd.DataFrame, by: pd.Series, n_replicates: int,
                         seed: int, alpha: float = 0.05,
                         cluster_col: str = "cluster_id",
                         stratum_col: str = "genre") -> dict:
    """Pooled estimands with cluster-bootstrap CIs inside each level of `by`.

    Within a level the bootstrap still resamples whole clusters within genre
    strata. Levels are analysed independently; a cluster whose rows fall into
    several levels contributes to each. Results are descriptive."""
    out = {}
    rng = np.random.default_rng(seed)
    for level in sorted(pd.unique(by)):
        sub = df.loc[by == level]
        if len(sub) == 0:
            continue
        point = pooled_stats(sub)
        boot = cluster_bootstrap(sub, n_replicates, int(rng.integers(0, 2**31)),
                                 cluster_col=cluster_col, stratum_col=stratum_col)
        summ = summarize(point, boot, alpha)
        out[str(level)] = {
            "n_pairs": int(len(sub)),
            "n_clusters": int(sub[cluster_col].nunique()),
            "estimands": summ,
        }
    return out


def cluster_sign_flip_test(df: pd.DataFrame, n_permutations: int, seed: int,
                           cluster_col: str = "cluster_id") -> dict:
    """Paired cluster-level permutation (sign-flip) test.

    H0 for R_T, R_B, R_eta: the Korean/English labels are exchangeable within
    every pair, so each pooled ratio equals 1. A permutation swaps the Korean
    and English columns (tokens and bits together) for every pair of a
    randomly chosen set of clusters — clusters are the exchangeable units, so
    all pairs of a cluster flip together. Statistic: log of the pooled ratio.
    Two-sided p = (1 + #{|stat_perm| >= |stat_obs|}) / (1 + n_permutations).

    H0 for C_K: the two tokenizers' Korean counts are exchangeable within
    every pair, so C_K = 1; the flip swaps the baseline and Korean-fit Korean
    counts cluster-wise.

    Uses per-cluster sums, which is exact because every pooled ratio is a
    function of column sums only. Exploratory: the confirmatory rules are the
    prespecified bootstrap bounds, not these p-values.
    """
    rng = np.random.default_rng(seed)
    has_control = COL_TK_KF in df.columns and df[COL_TK_KF].notna().all()
    cols = [COL_TK, COL_TE, COL_BK, COL_BE] + ([COL_TK_KF] if has_control else [])
    per_cluster = df.groupby(cluster_col, observed=True)[cols].sum().to_numpy(np.float64)
    K = per_cluster.shape[0]
    TK, TE, BK, BE = (per_cluster[:, i] for i in range(4))

    def _log_ratios(tk, te, bk, be):
        rt = np.log(tk.sum(axis=-1) / te.sum(axis=-1))
        rb = np.log(bk.sum(axis=-1) / be.sum(axis=-1))
        return rt, rb, rb - rt

    obs_rt, obs_rb, obs_reta = _log_ratios(TK, TE, BK, BE)

    flips = rng.integers(0, 2, size=(n_permutations, K)).astype(np.float64)  # 1 = swap
    keep = 1.0 - flips
    p_tk = keep * TK + flips * TE
    p_te = keep * TE + flips * TK
    p_bk = keep * BK + flips * BE
    p_be = keep * BE + flips * BK
    perm_rt, perm_rb, perm_reta = _log_ratios(p_tk, p_te, p_bk, p_be)

    def _p(obs, perm):
        return float((1 + np.sum(np.abs(perm) >= abs(obs))) / (1 + len(perm)))

    out = {
        "method": "cluster-level sign-flip permutation of the Korean/English labels; statistic = log pooled ratio; two-sided",
        "n_permutations": int(n_permutations),
        "n_clusters": int(K),
        "seed": int(seed),
        "R_T": {"log_ratio_obs": float(obs_rt), "p_value": _p(obs_rt, perm_rt)},
        "R_B": {"log_ratio_obs": float(obs_rb), "p_value": _p(obs_rb, perm_rb)},
        "R_eta": {"log_ratio_obs": float(obs_reta), "p_value": _p(obs_reta, perm_reta)},
        "note": ("R_T, R_B and R_eta are functionally dependent; p-values are not "
                 "independent evidence. Minimum attainable p is 1/(1+n_permutations)."),
    }
    if has_control:
        TKF = per_cluster[:, 4]
        obs_ck = np.log(TKF.sum() / TK.sum())
        flips_c = rng.integers(0, 2, size=(n_permutations, K)).astype(np.float64)
        keep_c = 1.0 - flips_c
        num = (keep_c * TKF + flips_c * TK).sum(axis=1)
        den = (keep_c * TK + flips_c * TKF).sum(axis=1)
        perm_ck = np.log(num / den)
        out["C_K"] = {"log_ratio_obs": float(obs_ck), "p_value": _p(obs_ck, perm_ck),
                      "h0": "baseline and Korean-fit Korean token counts exchangeable within pairs"}
    return out


def overall_decision(decisions: dict, confirmatory: list[str], rule: str) -> dict:
    """Combine per-comparison decisions under the prespecified rule.

    Only comparisons listed as confirmatory count; every other computed
    decision is reported separately as exploratory. The only implemented rule
    is `all_confirmatory_pass`.
    """
    if rule != "all_confirmatory_pass":
        raise ValueError(f"unknown decision rule {rule!r}")
    missing = [c for c in confirmatory if c not in decisions]
    if missing:
        raise ValueError(f"confirmatory comparison(s) {missing} were not computed")
    per = {c: bool(decisions[c]["pass"]) for c in confirmatory}
    return {
        "rule": rule,
        "confirmatory_comparisons": list(confirmatory),
        "per_comparison_pass": per,
        "overall_pass": all(per.values()),
        "exploratory_decisions_reported_separately": sorted(
            k for k, v in decisions.items() if isinstance(v, dict) and "pass" in v and k not in confirmatory
        ),
    }
