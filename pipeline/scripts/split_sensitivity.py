#!/usr/bin/env python3
"""Post hoc split-sensitivity analysis (NOT confirmatory).

Question: was the held-out R_eta outcome a property of the particular
pilot/held-out split, or a stable property of the corpus? Per-pair bits and
token counts do not depend on the split, so the whole corpus (pilot + held-out,
9,017 pairs, 347 articles) is pooled and many random article-level splits are
drawn with the same 278/69 sizes. For each draw the frozen decision rule for
R_eta (one-sided 95% upper bound < 0.95) is evaluated on the 278-article part.
The frozen thresholds are held fixed; nothing here may be used to revise them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from klen import config, data, stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = config.load_spec(ROOT / "spec/measurement_spec.yaml")
th = spec["thresholds"]
C_ETA, C_T, C_C = th["c_eta"], th["c_T"], th["c_C"]

pilot = pd.read_csv(ROOT / "out/pilot/per_pair_pilot.csv")
full = pd.read_csv(ROOT / "out/confirmatory/per_pair_full.csv")
pilot["split_original"] = "pilot"; full["split_original"] = "held_out"
df = pd.concat([pilot, full], ignore_index=True)
df, excl = data.apply_exclusions(df, spec["exclusions"])
cols = [stats.COL_TK, stats.COL_TE, stats.COL_BK, stats.COL_BE, stats.COL_TK_KF, stats.COL_TE_KF]
per_art = df.groupby("cluster_id")[cols].sum()
A = per_art.to_numpy(np.float64)
K = A.shape[0]
print(f"corpus after exclusions: {len(df)} pairs, {K} articles")

def ratios(s):
    return dict(R_T=s[..., 0] / s[..., 1], R_B=s[..., 2] / s[..., 3],
                R_eta=(s[..., 2] / s[..., 3]) / (s[..., 0] / s[..., 1]), C_K=s[..., 4] / s[..., 0])

def boot_bounds(sub, n_boot, rng):
    m = sub.shape[0]
    b = ratios(sub[rng.integers(0, m, size=(n_boot, m))].sum(axis=1))
    return dict(R_eta_upper95=float(np.quantile(b["R_eta"], 0.95)),
                R_T_lower95=float(np.quantile(b["R_T"], 0.05)),
                C_K_upper95=float(np.quantile(b["C_K"], 0.95)),
                R_eta_ci=[float(np.quantile(b["R_eta"], 0.025)), float(np.quantile(b["R_eta"], 0.975))])

rng = np.random.default_rng(20260911)

# 1. Whole corpus (all 347 articles) — descriptive, post hoc.
whole_point = {k: float(v) for k, v in ratios(A.sum(axis=0)).items()}
whole_b = boot_bounds(A, 10000, rng)
print("\nWHOLE CORPUS (347 articles):", {k: round(v, 4) for k, v in whole_point.items()})
print("  R_eta 95% CI", [round(x, 4) for x in whole_b["R_eta_ci"]], " upper95", round(whole_b["R_eta_upper95"], 4),
      "-> rule", "PASS" if whole_b["R_eta_upper95"] < C_ETA else "FAIL")

# 2. Random re-splits: 278-article 'held-out' parts drawn WITHOUT replacement.
n_splits, n_boot = 1000, 2000
held_n = 278
recs = []
for i in range(n_splits):
    idx = rng.permutation(K)
    held, pil = A[idx[:held_n]], A[idx[held_n:]]
    hp = ratios(held.sum(axis=0)); pp = ratios(pil.sum(axis=0))
    bb = boot_bounds(held, n_boot, rng)
    recs.append(dict(split=i, held_R_eta=float(hp["R_eta"]), held_R_T=float(hp["R_T"]), held_C_K=float(hp["C_K"]),
                     pilot_R_eta=float(pp["R_eta"]), **{k: v for k, v in bb.items() if k != "R_eta_ci"},
                     pass_R_eta=bool(bb["R_eta_upper95"] < C_ETA), pass_R_T=bool(bb["R_T_lower95"] > C_T),
                     pass_C_K=bool(bb["C_K_upper95"] < C_C)))
res = pd.DataFrame(recs)
res["pass_all"] = res.pass_R_eta & res.pass_R_T & res.pass_C_K
q = res[["held_R_eta", "R_eta_upper95", "pilot_R_eta", "held_R_T", "held_C_K"]].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T
print(f"\nRANDOM RE-SPLITS ({n_splits} draws, 278 held-out articles each):")
print(q.round(4).to_string())
print("\nfraction of re-splits passing R_eta rule (upper95 < 0.95):", res.pass_R_eta.mean())
print("fraction passing R_T rule:", res.pass_R_T.mean(), " C_K rule:", res.pass_C_K.mean(), " all three:", res.pass_all.mean())
print("fraction of re-splits with held-out point R_eta < 0.95:", (res.held_R_eta < C_ETA).mean())
print("smallest c_eta that >=95% of re-splits would pass:", round(float(res.R_eta_upper95.quantile(0.95)), 4))

# 3. Per-article distribution: is the effect widespread or driven by a few articles?
art = ratios(A)
art_df = pd.DataFrame({"R_eta": art["R_eta"], "n_pairs": df.groupby("cluster_id").size().reindex(per_art.index).values,
                       "tokens_en": A[:, 1]}, index=per_art.index)
print("\nPER-ARTICLE R_eta: median", round(float(art_df.R_eta.median()), 4),
      " share of articles with R_eta<1:", round(float((art_df.R_eta < 1).mean()), 3),
      " share with R_eta<0.95:", round(float((art_df.R_eta < 0.95).mean()), 3))
# leave-one-article-out on the whole corpus
loo = []
tot = A.sum(axis=0)
for k in range(K):
    loo.append(float(ratios(tot - A[k])["R_eta"]))
loo = np.array(loo)
print("leave-one-article-out R_eta range:", round(loo.min(), 4), "-", round(loo.max(), 4))

out = ROOT / "out/post_hoc"; out.mkdir(exist_ok=True)
res.to_csv(out / "resplit_results.csv", index=False)
art_df.to_csv(out / "per_article_ratios.csv")
summary = dict(
    note="POST HOC split-sensitivity; not a confirmatory analysis. Frozen thresholds held fixed.",
    corpus=dict(pairs=int(len(df)), articles=int(K), exclusions=excl["rules"]),
    whole_corpus=dict(point=whole_point, R_eta_ci95=whole_b["R_eta_ci"], R_eta_upper95=whole_b["R_eta_upper95"]),
    resplits=dict(n_splits=n_splits, n_boot=n_boot, held_out_articles=held_n, seed=20260911,
                  quantiles=q.round(6).to_dict(orient="index"),
                  pass_rate=dict(R_eta=float(res.pass_R_eta.mean()), R_T=float(res.pass_R_T.mean()),
                                 C_K=float(res.pass_C_K.mean()), all=float(res.pass_all.mean())),
                  point_below_c_eta_rate=float((res.held_R_eta < C_ETA).mean()),
                  c_eta_for_95pct_pass=float(res.R_eta_upper95.quantile(0.95))),
    per_article=dict(median_R_eta=float(art_df.R_eta.median()), share_below_1=float((art_df.R_eta < 1).mean()),
                     share_below_c_eta=float((art_df.R_eta < 0.95).mean()),
                     leave_one_out_R_eta_range=[float(loo.min()), float(loo.max())]),
)
(out / "split_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {out}")
