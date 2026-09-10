import json

import numpy as np
import pandas as pd
import pytest
import yaml

from klen import analyze, data, stats


def per_pair_df(n_clusters=8, rows_per_cluster=4, seed=0, effect=True):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        for r in range(rows_per_cluster):
            te = int(rng.integers(20, 60))
            be = float(rng.uniform(100, 300))
            if effect:
                tk, bk = int(round(te * 1.5)), be * 1.2
            else:
                tk, bk = te, be
            rows.append({
                "korean_text": "한국어 문장" if r % 2 == 0 else "same text",
                "english_text": "English sentence" if r % 2 == 0 else "same text",
                "pair_id": f"p{c}-{r}", "cluster_id": f"c{c}", "genre": "g1" if c % 2 == 0 else "g2",
                "direction": "unknown",
                "n_chars_ko": 5 + r, "n_chars_en": 10 + 3 * r,
                "flag_ko_no_hangul": r % 2 == 1,
                "flag_ko_control_chars": False, "flag_en_control_chars": r == 3,
                "tokens_ko_baseline": tk, "tokens_en_baseline": te,
                "bits_ko": bk, "bits_en": be,
                "tokens_ko_koreanfit": int(round(tk * 0.8)), "tokens_en_koreanfit": te * 2,
            })
    return pd.DataFrame(rows)


# ---- exclusions -------------------------------------------------------------

def test_no_rules_keeps_everything():
    df = per_pair_df()
    out, rep = data.apply_exclusions(df, None)
    assert len(out) == len(df)
    assert rep["n_rows_dropped"] == 0
    out2, _ = data.apply_exclusions(df, {"drop_ko_equals_en": None, "min_chars_ko": None})
    assert len(out2) == len(df)


def test_each_rule_drops_expected_rows():
    df = per_pair_df()
    n = len(df)
    out, rep = data.apply_exclusions(df, {"drop_ko_equals_en": True})
    assert len(out) == n / 2 and rep["dropped_by_rule_sequential"]["drop_ko_equals_en"] == n / 2
    out, _ = data.apply_exclusions(df, {"drop_ko_no_hangul": True})
    assert len(out) == n / 2
    out, _ = data.apply_exclusions(df, {"drop_control_chars": True})
    assert len(out) == n * 3 / 4
    out, _ = data.apply_exclusions(df, {"min_chars_ko": 7})       # r in {0,1} dropped
    assert len(out) == n / 2
    out, _ = data.apply_exclusions(df, {"min_chars_en": 16})      # n_chars_en=10+3r: r in {0,1} dropped
    assert len(out) == n / 2
    # ratios: r=0 -> .5, r=1 -> .4615, r=2 -> .4375, r=3 -> .421
    out, _ = data.apply_exclusions(df, {"char_ratio_ko_over_en": [0.43, 0.49]})
    assert len(out) == n / 2
    # sequential accounting: second rule only counts rows still present
    out, rep = data.apply_exclusions(df, {"drop_ko_equals_en": True, "drop_ko_no_hangul": True})
    assert len(out) == n / 2
    assert rep["dropped_by_rule_sequential"]["drop_ko_no_hangul"] == 0


def test_unknown_rule_is_an_error():
    with pytest.raises(ValueError, match="unknown exclusion rule"):
        data.apply_exclusions(per_pair_df(), {"drop_short": True})
    with pytest.raises(ValueError):
        data.apply_exclusions(per_pair_df(), {"char_ratio_ko_over_en": [2, 1]})


def test_identical_text_flag_in_validation():
    df = pd.DataFrame({
        "korean_text": ["CC BY-ND", "안녕"], "english_text": ["CC BY-ND ", "Hi"],
        "pair_id": ["a", "b"], "cluster_id": ["c", "c"], "genre": ["g", "g"],
        "direction": ["u", "u"],
    })
    out, report = data.normalize_and_flag(df)
    assert out["flag_ko_equals_en"].tolist() == [True, False]
    assert report["flag_counts"]["flag_ko_equals_en"] == 1


# ---- per-pair distribution ----------------------------------------------------

def test_per_pair_summary_hand_computed():
    df = pd.DataFrame({
        "tokens_ko_baseline": [2, 4, 6], "tokens_en_baseline": [1, 1, 1],
        "bits_ko": [1.0, 1.0, 1.0], "bits_en": [1.0, 2.0, 4.0],
    })
    s = stats.per_pair_ratio_summary(df)
    assert s["R_T"]["mean"] == pytest.approx(4.0)
    assert s["R_T"]["median"] == pytest.approx(4.0)
    assert s["R_B"]["median"] == pytest.approx(0.5)
    assert s["R_B"]["geometric_mean"] == pytest.approx((1 * 0.5 * 0.25) ** (1 / 3))
    assert s["R_eta"]["median"] == pytest.approx(0.5 / 4.0)


# ---- strata ------------------------------------------------------------------

def test_length_strata_assignment_and_estimates():
    df = per_pair_df(n_clusters=6)
    strata = stats.assign_length_strata(df, "n_chars_en", 3)
    assert set(strata) == {"Q1", "Q2", "Q3"}
    assert (df.loc[strata == "Q1", "n_chars_en"].max()
            <= df.loc[strata == "Q3", "n_chars_en"].min())
    est = stats.stratified_estimates(df, strata, 50, seed=1)
    assert set(est) == {"Q1", "Q2", "Q3"}
    for lvl in est.values():
        assert lvl["estimands"]["R_T"]["point"] == pytest.approx(1.5, rel=0.02)


def test_per_genre_estimates_match_subset_pooled():
    df = per_pair_df()
    est = stats.stratified_estimates(df, df["genre"], 20, seed=3)
    for g in ("g1", "g2"):
        sub = df[df["genre"] == g]
        assert est[g]["estimands"]["R_B"]["point"] == pytest.approx(stats.pooled_stats(sub)["R_B"])
        assert est[g]["n_clusters"] == sub["cluster_id"].nunique()


# ---- permutation test ----------------------------------------------------------

def test_sign_flip_p_is_one_under_exact_symmetry():
    df = per_pair_df(effect=False)  # ko == en exactly -> observed log ratio 0
    res = stats.cluster_sign_flip_test(df, 500, seed=0)
    for k in ("R_T", "R_B", "R_eta"):
        assert res[k]["log_ratio_obs"] == pytest.approx(0.0)
        assert res[k]["p_value"] == pytest.approx(1.0)


def test_sign_flip_detects_strong_effect_and_respects_floor():
    df = per_pair_df(n_clusters=12, effect=True)  # R_T = 1.5 in every cluster
    n_perm = 999
    res = stats.cluster_sign_flip_test(df, n_perm, seed=0)
    assert res["R_T"]["log_ratio_obs"] == pytest.approx(np.log(1.5), rel=1e-2)  # tk rounded
    # only the all-keep and all-flip assignments reach |log 1.5|: p near 1/(1+n)
    assert res["R_T"]["p_value"] <= 5 / (1 + n_perm)
    assert res["R_T"]["p_value"] >= 1 / (1 + n_perm)
    assert res["C_K"]["p_value"] <= 5 / (1 + n_perm)


def test_sign_flip_uses_whole_clusters():
    # Two clusters whose pairs individually point in opposite directions but
    # cancel within each cluster. Flipping whole clusters therefore never
    # changes the pooled ratio, so every permutation statistic equals the
    # observed one and p must be exactly 1. Pair-level flipping would not.
    df = pd.DataFrame({
        "cluster_id": ["a", "a", "b", "b"],
        "tokens_ko_baseline": [10, 1, 10, 1], "tokens_en_baseline": [1, 10, 1, 10],
        "bits_ko": [10.0, 1.0, 10.0, 1.0], "bits_en": [1.0, 10.0, 1.0, 10.0],
    })
    res = stats.cluster_sign_flip_test(df, 300, seed=1)
    assert res["R_T"]["p_value"] == 1.0


# ---- analyze CLI end-to-end ----------------------------------------------------

def test_analyze_applies_exclusions_and_writes_exploratory(tmp_path):
    df = per_pair_df()
    csv = tmp_path / "per_pair_pilot.csv"
    df.to_csv(csv, index=False)
    spec = {
        "thresholds": {}, "endpoints": {},
        "exclusions": {"drop_ko_equals_en": True},
        "exploratory": {"n_permutations": 200, "n_bootstrap_replicates_strata": 30},
        "bootstrap": {"n_replicates": 40, "seed": 5, "alpha_two_sided": 0.05},
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    analyze.main([str(spec_path), str(csv), str(tmp_path / "out"), "--exploratory"])
    res = json.loads((tmp_path / "out" / "analysis_summary.json").read_text())
    assert res["exclusions"]["n_rows_after"] == len(df) / 2
    assert res["n_pairs"] == len(df) / 2
    assert res["estimands"]["R_T"]["point"] == pytest.approx(1.5, rel=1e-6)
    ex = res["exploratory"]
    assert set(ex["per_genre"]["levels"]) == {"g1", "g2"}
    assert set(ex["length_strata"]["levels"]) == {"Q1", "Q2", "Q3"}
    assert ex["permutation_tests"]["n_permutations"] == 200
    assert "C_K" in ex["permutation_tests"]
    assert "mean" in ex["per_pair_ratio_distribution"]["R_T"]
    assert "confirmatory_decisions" not in res


# ---- overall decision rule -----------------------------------------------------

def test_overall_decision_uses_only_confirmatory_list():
    dec = {"R_T": {"pass": True}, "R_B_equivalence": {"pass": False},
           "R_eta": {"pass": True}, "C_K": {"pass": True}, "note": "x"}
    od = stats.overall_decision(dec, ["R_eta", "R_T", "C_K"], "all_confirmatory_pass")
    assert od["overall_pass"] is True
    assert od["exploratory_decisions_reported_separately"] == ["R_B_equivalence"]
    dec["C_K"]["pass"] = False
    assert stats.overall_decision(dec, ["R_eta", "R_T", "C_K"], "all_confirmatory_pass")["overall_pass"] is False
    with pytest.raises(ValueError):
        stats.overall_decision(dec, ["R_eta"], "any_pass")
    with pytest.raises(ValueError, match="not computed"):
        stats.overall_decision({"R_T": {"pass": True}}, ["C_K"], "all_confirmatory_pass")


def test_config_validates_endpoint_names():
    from klen import config
    good = {"endpoints": {"primary": "R_eta", "confirmatory": ["R_eta", "R_T", "C_K"],
                          "decision_rule": "all_confirmatory_pass"}}
    assert config.invalid_fields(good) == []
    bad = {"endpoints": {"primary": "eta", "confirmatory": ["R_B"], "decision_rule": "majority"}}
    msgs = " ".join(config.invalid_fields(bad))
    assert "primary" in msgs and "unknown comparisons" in msgs and "decision_rule" in msgs
    assert any("decision_rule must be set" in m for m in
               config.invalid_fields({"endpoints": {"confirmatory": ["R_T"]}}))


def test_analyze_confirmatory_end_to_end_with_frozen_spec(tmp_path):
    from klen import config
    df = per_pair_df(n_clusters=10)
    csv = tmp_path / "per_pair_full.csv"
    df.to_csv(csv, index=False)
    spec = {
        "corpus": {"path": "x", "columns": {k: k for k in ("korean_text", "english_text", "pair_id", "cluster_id", "genre", "direction")}},
        "estimator": {"model_id": "m", "revision": "r"},
        "tokenizers": {"baseline": {"id": "a", "revision": "1"}, "korean_fit": {"id": "b", "revision": "2"}},
        "token_accounting": "content_only", "scoring_policy": {"bos": "prepend_bos", "eos": "none"},
        "thresholds": {"c_T": 1.2, "L_B": 0.8333, "U_B": 1.2, "c_eta": 0.95, "c_C": 0.9},
        "endpoints": {"primary": "R_eta", "confirmatory": ["R_eta", "R_T", "C_K"],
                      "decision_rule": "all_confirmatory_pass", "exploratory": ["R_B_equivalence"]},
        "exclusions": {"drop_ko_equals_en": True},
        "exploratory": {"n_permutations": 100, "n_bootstrap_replicates_strata": 20},
        "bootstrap": {"n_replicates": 60, "seed": 1, "alpha_two_sided": 0.05, "alpha_equivalence": 0.05},
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    config.freeze_spec(spec_path)
    analyze.main([str(spec_path), str(csv), str(tmp_path / "out")])
    res = json.loads((tmp_path / "out" / "analysis_summary.json").read_text())
    # synthetic data: R_T = 1.5 (> 1.2 passes), R_eta = 1.2/1.5 = 0.8 (< 0.95 passes),
    # C_K = 0.8 (< 0.9 passes); R_B = 1.2 sits on the interval edge -> exploratory only
    assert set(res["confirmatory_decisions"]) >= {"R_eta", "R_T", "C_K", "note"}
    assert "R_B_equivalence" not in res["confirmatory_decisions"]
    assert "R_B_equivalence" in res["exploratory_decisions"]
    assert res["overall_decision"]["overall_pass"] is True
    assert res["exclusions"]["n_rows_dropped"] == len(df) / 2
