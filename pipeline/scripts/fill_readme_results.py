#!/usr/bin/env python3
"""Render the Results section of the top-level README from analysis JSON.

Numbers only come from results/*/analysis_summary.json; the surrounding
prose is fixed here so a re-run regenerates the section identically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT.parent / "README.md"


def load(run: str) -> dict:
    return json.loads((ROOT / "results" / run / "analysis_summary.json").read_text())


def est_row(name: str, label: str, e: dict) -> str:
    v = e[name]
    return f"| {label} | {v['point']:.4f} | [{v['ci_low']:.4f}, {v['ci_high']:.4f}] |"


def main() -> int:
    conf, pilot = load("confirmatory"), load("pilot")
    e, p = conf["estimands"], pilot["estimands"]
    dec, od = conf["confirmatory_decisions"], conf["overall_decision"]
    exd = conf["exploratory_decisions"]
    ex = conf["exploratory"]
    excl = conf["exclusions"]

    lines = []
    lines.append(f"Held-out set after exclusions: **{conf['n_pairs']:,} pairs in {conf['n_clusters']} articles** "
                 f"({excl['n_rows_dropped']} rows dropped: "
                 + ", ".join(f"{k} {v}" for k, v in excl['dropped_by_rule_sequential'].items()) + "). "
                 f"Pilot after the same rules: {pilot['n_pairs']:,} pairs in {pilot['n_clusters']} articles.\n")
    lines.append("### Pooled estimands (held-out, 95 % cluster-bootstrap percentile CI)\n")
    lines.append("| Quantity | Point | 95 % CI |")
    lines.append("|---|---|---|")
    lines.append(est_row("R_T", "`R_T` token ratio KR/EN (o200k)", e))
    lines.append(est_row("R_B", "`R_B` bit ratio KR/EN (Qwen3-1.7B)", e))
    lines.append(est_row("R_eta", "`R_η` = `R_B` / `R_T`", e))
    lines.append(est_row("one_minus_R_eta", "`1 − R_η` encoding shortfall", e))
    lines.append(est_row("C_K", "`C_K` Korean tokens, polyglot-ko / o200k", e))
    lines.append(est_row("R_T_koreanfit", "`R_T` under polyglot-ko (exploratory)", e))
    lines.append(est_row("G", "`G` gap contraction (exploratory)", e))
    lines.append("")
    lines.append("### Pre-registered decisions\n")
    lines.append("| Comparison | Rule | Bound observed | Result |")
    lines.append("|---|---|---|---|")
    for k in od["confirmatory_comparisons"]:
        d = dec[k]
        bound = d.get("lower_bound", d.get("upper_bound"))
        lines.append(f"| `{k}` | {d['rule']} | {bound:.4f} | {'PASS' if d['pass'] else 'FAIL'} |")
    lines.append(f"\n**Overall ({od['rule']}): {'PASS' if od['overall_pass'] else 'FAIL'}.**\n")
    failed = [k for k in od["confirmatory_comparisons"] if not dec[k]["pass"]]
    if failed:
        parts = []
        for k in failed:
            d = dec[k]; bound = d.get("lower_bound", d.get("upper_bound"))
            parts.append(f"`{k}` (bound {bound:.4f} vs threshold in rule \"{d['rule']}\"; point estimate {e[k]['point']:.4f})")
        lines.append("The pre-registered rule was not met because " + "; ".join(parts) + ". "
                     "The thresholds were frozen before this data was scored and are not revised here; "
                     "the point estimates and intervals above are reported as observed.\n")
    rb = exd["R_B_equivalence"]
    lines.append(f"Exploratory bit-ratio equivalence: 90 % CI for `R_B` = "
                 f"[{rb['ci'][0]:.4f}, {rb['ci'][1]:.4f}] against [0.8333, 1.20] → "
                 f"{'inside' if rb['pass'] else 'not inside'} the interval. "
                 f"Korean text costs more bits under this model as well as more tokens; "
                 f"the pre-registered primary question is whether bits rise *as fast as* tokens, "
                 f"which is what `R_η` measures.\n")

    lines.append("### Pilot vs held-out\n")
    lines.append("| Quantity | Pilot (69 articles) | Held-out (278 articles) |")
    lines.append("|---|---|---|")
    for k, lab in (("R_T", "`R_T`"), ("R_B", "`R_B`"), ("R_eta", "`R_η`"), ("C_K", "`C_K`")):
        lines.append(f"| {lab} | {p[k]['point']:.4f} | {e[k]['point']:.4f} |")
    lines.append("")

    ls = ex["length_strata"]
    lines.append(f"### By message length (exploratory; terciles of `{ls['length_variable']}`)\n")
    lines.append("| Stratum | Range (chars) | n | `R_T` | `R_B` | `R_η` [95 % CI] | `C_K` |")
    lines.append("|---|---|---|---|---|---|---|")
    for lvl, v in ls["levels"].items():
        s = v["estimands"]; r = ls["ranges"][lvl]
        lines.append(f"| {lvl} | {r[0]:.0f}–{r[1]:.0f} | {v['n_pairs']} | {s['R_T']['point']:.3f} | {s['R_B']['point']:.3f} | "
                     f"{s['R_eta']['point']:.3f} [{s['R_eta']['ci_low']:.3f}, {s['R_eta']['ci_high']:.3f}] | {s['C_K']['point']:.3f} |")
    lines.append("")

    pt = ex["permutation_tests"]
    lines.append(f"### Paired cluster permutation tests (exploratory; {pt['n_permutations']:,} sign-flips of whole articles)\n")
    lines.append("| Quantity | log pooled ratio | two-sided p |")
    lines.append("|---|---|---|")
    for k in ("R_T", "R_B", "R_eta", "C_K"):
        lines.append(f"| `{k}` | {pt[k]['log_ratio_obs']:+.4f} | {pt[k]['p_value']:.5f} |")
    lines.append(f"\nMinimum attainable p is 1/(1+{pt['n_permutations']:,}); the three language ratios are one dependent finding.\n")

    pp = ex["per_pair_ratio_distribution"]
    lines.append("### Per-message ratio distribution (descriptive only; not the estimand)\n")
    lines.append("| Ratio | mean | median | geometric mean | 5th pct | 95th pct |")
    lines.append("|---|---|---|---|---|---|")
    for k in ("R_T", "R_B", "R_eta"):
        d = pp[k]
        lines.append(f"| `{k}` | {d['mean']:.3f} | {d['median']:.3f} | {d['geometric_mean']:.3f} | {d['p05']:.3f} | {d['p95']:.3f} |")
    lines.append("")
    lines.append("Full outputs, including 10,000 bootstrap replicates, provenance (package versions, "
                 "tokenizer file hashes, dataset hash, device) and the pilot precision simulation, are in `pipeline/results/`.")

    block = "\n".join(lines)
    text = README.read_text()
    new = re.sub(r"<!-- RESULTS:BEGIN -->.*?<!-- RESULTS:END -->",
                 "<!-- RESULTS:BEGIN -->\n" + block + "\n<!-- RESULTS:END -->", text, flags=re.S)
    README.write_text(new)
    print("README results section updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
