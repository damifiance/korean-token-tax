"""Statistical analysis of a per-pair results file.

Usage:
  python -m klen.analyze SPEC.yaml PER_PAIR.csv OUTDIR [--exploratory]

Confirmatory mode (default) requires a frozen spec with all thresholds set,
and refuses pilot-tagged inputs. --exploratory computes point estimates and
CIs without confirmatory decisions (for pilot variance / power work).

In both modes the spec's `exclusions` block is applied first (default: no
exclusions) and an `exploratory` section is written: per-pair ratio
distribution, per-genre and per-length-stratum pooled estimates, and paired
cluster permutation tests. That section never feeds the confirmatory rules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config, data, provenance, stats

INTERPRETATION_GUARDS = [
    "B_M is model-relative surprisal, not semantic information or intrinsic entropy.",
    "Parallel meaning does not guarantee equal textual entropy or model NLL.",
    "Korean information differences and tokenizer inefficiency may coexist.",
    "R_T, R_B, R_eta are functionally dependent (R_eta = R_B/R_T).",
    "1 - R_eta is a relative shortfall under an English-efficiency counterfactual, "
    "not a literal fraction of zero-information tokens.",
    "C_K < 1 measures absolute Korean-token reduction under the comparison "
    "tokenizer; attribution to training-language balance still requires controlling "
    "algorithm, vocab size, normalization, pre-tokenization, and training-corpus size.",
    "G is exploratory gap contraction only: it can fall because English token count "
    "rises even when Korean token count does not fall. Attribution likewise "
    "requires controlling algorithm, vocab size, normalization, pre-tokenization, "
    "and training-corpus size.",
    "Per-genre, per-length-stratum, per-pair-distribution and permutation results are "
    "exploratory; the confirmatory decisions are the prespecified bootstrap bounds only.",
]


def exploratory_section(df: pd.DataFrame, spec: dict, n_rep: int, seed: int,
                        alpha: float) -> dict:
    ex_cfg = spec.get("exploratory") or {}
    length_col = ex_cfg.get("length_variable") or "n_chars_en"
    k = int(ex_cfg.get("n_length_strata") or 3)
    n_perm = int(ex_cfg.get("n_permutations") or 10000)
    n_rep_strata = int(ex_cfg.get("n_bootstrap_replicates_strata") or n_rep)

    section = {"per_pair_ratio_distribution": stats.per_pair_ratio_summary(df)}

    genres = df["genre"].astype(str)
    section["per_genre"] = {
        "note": "pooled estimands within each genre; cluster bootstrap within genre",
        "levels": stats.stratified_estimates(df, genres, n_rep_strata, seed + 1, alpha),
    }

    if length_col in df.columns and len(df) >= k:
        strata = stats.assign_length_strata(df, length_col, k)
        edges = {lvl: [float(df.loc[strata == lvl, length_col].min()),
                       float(df.loc[strata == lvl, length_col].max())]
                 for lvl in sorted(strata.unique())}
        section["length_strata"] = {
            "length_variable": length_col,
            "n_strata": k,
            "ranges": edges,
            "note": ("quantile bins of the length variable over the analysed rows; "
                     "clusters may span bins; Q1 = shortest"),
            "levels": stats.stratified_estimates(df, strata, n_rep_strata, seed + 2, alpha),
        }
    else:
        section["length_strata"] = {"skipped": f"column {length_col!r} not available"}

    section["permutation_tests"] = stats.cluster_sign_flip_test(df, n_perm, seed + 3)
    return section


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("per_pair_csv")
    ap.add_argument("outdir")
    ap.add_argument("--exploratory", action="store_true")
    ap.add_argument("--skip-exploratory-section", action="store_true",
                    help="omit the exploratory section (faster; confirmatory decisions unchanged)")
    args = ap.parse_args(argv)

    spec = config.load_spec(args.spec)
    if not args.exploratory:
        config.assert_ready_for_confirmatory(spec)
        if "pilot" in Path(args.per_pair_csv).name:
            raise SystemExit("refusing confirmatory analysis on pilot-tagged data")

    raw = pd.read_csv(args.per_pair_csv)
    df, exclusion_report = data.apply_exclusions(raw, spec.get("exclusions"))
    if len(df) == 0:
        raise SystemExit("exclusion rules removed every row")

    boot_cfg = spec.get("bootstrap", {})
    n_rep = int(boot_cfg.get("n_replicates", 10000))
    seed = int(boot_cfg.get("seed", 0))
    alpha = float(boot_cfg.get("alpha_two_sided", 0.05))

    point = stats.pooled_stats(df)
    boot = stats.cluster_bootstrap(df, n_rep, seed)
    summary = stats.summarize(point, boot, alpha)

    result = {
        "mode": "exploratory" if args.exploratory else "confirmatory",
        "input_file": str(args.per_pair_csv),
        "input_sha256": provenance.file_sha256(args.per_pair_csv),
        "spec_content_sha256": config.spec_content_hash(spec),
        "exclusions": exclusion_report,
        "estimands": summary,
        "identity_check_R_eta_equals_RB_over_RT": abs(
            point["R_eta"] - point["R_B"] / point["R_T"]
        ),
        "n_pairs": int(len(df)),
        "n_clusters": int(df["cluster_id"].nunique()),
        "genres": {str(k): int(v) for k, v in df["genre"].value_counts().items()},
        "bootstrap": {"n_replicates": n_rep, "seed": seed},
        "interpretation_guards": INTERPRETATION_GUARDS,
    }
    if not args.exploratory:
        alpha_eq = float(boot_cfg.get("alpha_equivalence", 0.05))
        all_dec = stats.confirmatory_decisions(boot, spec["thresholds"], alpha_eq)
        ep = spec.get("endpoints") or {}
        conf = list(ep.get("confirmatory") or [])
        result["endpoints"] = ep
        result["confirmatory_decisions"] = {k: v for k, v in all_dec.items()
                                            if k in conf or not isinstance(v, dict)}
        result["exploratory_decisions"] = {
            "note": "computed with the frozen thresholds but NOT part of the confirmatory rule",
            **{k: v for k, v in all_dec.items() if isinstance(v, dict) and "pass" in v and k not in conf},
        }
        result["overall_decision"] = stats.overall_decision(all_dec, conf, ep.get("decision_rule"))
    if not args.skip_exploratory_section:
        result["exploratory"] = exploratory_section(df, spec, n_rep, seed, alpha)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    boot.to_csv(outdir / "bootstrap_replicates.csv", index=False)
    provenance.write_json(result, outdir / "analysis_summary.json")
    print(f"wrote {outdir / 'analysis_summary.json'}")
    print(f"rows analysed: {len(df)} of {len(raw)} "
          f"(dropped {exclusion_report['n_rows_dropped']}); clusters: {result['n_clusters']}")
    for k, v in summary.items():
        print(f"{k}: {v['point']:.6f}  95% CI [{v['ci_low']:.6f}, {v['ci_high']:.6f}]")
    if "confirmatory_decisions" in result:
        for k, v in result["confirmatory_decisions"].items():
            if isinstance(v, dict) and "pass" in v:
                print(f"decision {k}: {'PASS' if v['pass'] else 'FAIL'}  ({v['rule']})")
        od = result["overall_decision"]
        print(f"OVERALL ({od['rule']}): {'PASS' if od['overall_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
