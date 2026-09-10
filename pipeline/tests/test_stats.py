import numpy as np
import pandas as pd
import pytest

from klen import stats


def hand_df():
    # Hand-computed pooled ratios:
    #   sum T_K = 10+20+30 = 60, sum T_E = 5+10+15 = 30      -> R_T = 2.0
    #   sum B_K = 40+50+60 = 150, sum B_E = 50+40+10 = 100   -> R_B = 1.5
    #   R_eta = (150/60)/(100/30) = 2.5/3.3333 = 0.75 = R_B/R_T
    return pd.DataFrame({
        "pair_id": ["a", "b", "c"],
        "cluster_id": ["c1", "c1", "c2"],
        "genre": ["g1", "g1", "g1"],
        "tokens_ko_baseline": [10, 20, 30],
        "tokens_en_baseline": [5, 10, 15],
        "bits_ko": [40.0, 50.0, 60.0],
        "bits_en": [50.0, 40.0, 10.0],
        "tokens_ko_koreanfit": [8, 16, 24],
        "tokens_en_koreanfit": [5, 10, 15],
    })


def test_pooled_ratios_hand_computed():
    p = stats.pooled_stats(hand_df())
    assert p["R_T"] == pytest.approx(2.0)
    assert p["R_B"] == pytest.approx(1.5)
    assert p["R_eta"] == pytest.approx(0.75)
    assert p["one_minus_R_eta"] == pytest.approx(0.25)
    # absolute Korean reduction: 48/60 = 0.8
    assert p["C_K"] == pytest.approx(0.8)
    # exploratory gap contraction: (48/30) / (60/30) = 1.6 / 2.0 = 0.8
    assert p["R_T_koreanfit"] == pytest.approx(1.6)
    assert p["G"] == pytest.approx(0.8)


def test_pooled_identity_R_eta_equals_RB_over_RT():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "cluster_id": [f"c{i%5}" for i in range(50)],
        "genre": ["g1"] * 25 + ["g2"] * 25,
        "tokens_ko_baseline": rng.integers(5, 500, 50),
        "tokens_en_baseline": rng.integers(5, 500, 50),
        "bits_ko": rng.uniform(10, 2000, 50),
        "bits_en": rng.uniform(10, 2000, 50),
    })
    p = stats.pooled_stats(df, include_control=False)
    assert p["R_eta"] == pytest.approx(p["R_B"] / p["R_T"], rel=1e-12)


def test_ratio_of_sums_not_mean_of_ratios():
    df = hand_df()
    per_msg = (df["tokens_ko_baseline"] / df["tokens_en_baseline"]).mean()
    p = stats.pooled_stats(df)
    # In this construction they happen to be equal (all ratios 2.0) for T,
    # but for bits they differ; assert the estimator follows the pooled form.
    mean_bit_ratio = (df["bits_ko"] / df["bits_en"]).mean()
    assert p["R_B"] == pytest.approx(1.5)
    assert mean_bit_ratio != pytest.approx(1.5)
    assert per_msg == pytest.approx(p["R_T"])  # degenerate here by design


def test_bootstrap_seed_determinism():
    df = hand_df()
    b1 = stats.cluster_bootstrap(df, 50, seed=123)
    b2 = stats.cluster_bootstrap(df, 50, seed=123)
    b3 = stats.cluster_bootstrap(df, 50, seed=124)
    pd.testing.assert_frame_equal(b1, b2)
    assert not b1.equals(b3)


def test_bootstrap_identity_holds_in_every_replicate():
    df = hand_df()
    boot = stats.cluster_bootstrap(df, 200, seed=1)
    np.testing.assert_allclose(boot["R_eta"], boot["R_B"] / boot["R_T"], rtol=1e-12)
    np.testing.assert_allclose(boot["one_minus_R_eta"], 1 - boot["R_eta"], rtol=1e-12)
    assert "C_K" in boot.columns


def test_gap_contraction_does_not_imply_korean_token_reduction():
    # The Korean-fit tokenizer worsens Korean token count by 20%, but worsens
    # English even more. G contracts to 0.5 although the promised Korean-token
    # control fails: C_K = 1.2.
    df = pd.DataFrame({
        "cluster_id": ["c1"],
        "genre": ["g"],
        "tokens_ko_baseline": [10],
        "tokens_en_baseline": [5],
        "bits_ko": [10.0],
        "bits_en": [10.0],
        "tokens_ko_koreanfit": [12],
        "tokens_en_koreanfit": [12],
    })
    p = stats.pooled_stats(df)
    assert p["C_K"] == pytest.approx(1.2)
    assert p["G"] == pytest.approx(0.5)


def test_bootstrap_resamples_whole_clusters():
    # Two clusters with distinctive sums; every replicate's totals must be an
    # integer combination of complete cluster sums (pairs never split).
    df = pd.DataFrame({
        "cluster_id": ["c1", "c1", "c2"],
        "genre": ["g", "g", "g"],
        "tokens_ko_baseline": [1, 2, 100],
        "tokens_en_baseline": [1, 1, 1],
        "bits_ko": [1.0, 1.0, 1.0],
        "bits_en": [1.0, 1.0, 1.0],
    })
    boot = stats.cluster_bootstrap(df, 300, seed=5, include_control=False)
    # cluster sums: c1 = (ko 3, en 2), c2 = (ko 100, en 1); 2 draws w/ replacement:
    # {c1,c1} -> 6/4, {c1,c2} -> 103/3, {c2,c2} -> 200/2. If a pair were ever
    # split from its cluster, other values would appear.
    valid = {round(6.0 / 4.0, 9), round(103.0 / 3.0, 9), round(200.0 / 2.0, 9)}
    assert set(np.round(boot["R_T"], 9)) <= valid
    assert len(set(np.round(boot["R_T"], 9))) == 3  # all combos observed


def test_bootstrap_preserves_strata():
    # One huge-token cluster in stratum g2. Stratified resampling draws
    # exactly one cluster from g2 every time, so its contribution is constant.
    df = pd.DataFrame({
        "cluster_id": ["a", "b", "z"],
        "genre": ["g1", "g1", "g2"],
        "tokens_ko_baseline": [1, 2, 1000],
        "tokens_en_baseline": [1, 2, 1000],
        "bits_ko": [1.0, 2.0, 1000.0],
        "bits_en": [1.0, 2.0, 1000.0],
    })
    boot = stats.cluster_bootstrap(df, 200, seed=9, include_control=False)
    # g2's single cluster appears exactly once per replicate -> numerator and
    # denominator both include exactly 1000 from it; R_T bounded accordingly.
    assert boot["R_T"].min() >= (1000 + 1 * 2) / (1000 + 2 * 2) - 1e-12
    assert boot["R_T"].max() <= (1000 + 2 * 2) / (1000 + 1 * 2) + 1e-12


def test_confirmatory_requires_thresholds():
    boot = stats.cluster_bootstrap(hand_df(), 50, seed=2)
    with pytest.raises(ValueError, match="threshold c_T is not set"):
        stats.confirmatory_decisions(boot, {"c_T": None, "L_B": 0.9, "U_B": 1.1,
                                            "c_eta": 0.9})


def test_confirmatory_decision_logic():
    # Synthetic bootstrap distributions with known quantiles.
    n = 1000
    boot = pd.DataFrame({
        "R_T": np.linspace(1.5, 2.5, n),      # 5% quantile ~ 1.55
        "R_B": np.linspace(0.95, 1.05, n),    # 90% CI ~ [0.955, 1.045]
        "R_eta": np.linspace(0.5, 0.7, n),    # 95% quantile ~ 0.69
        "one_minus_R_eta": 1 - np.linspace(0.5, 0.7, n),
    })
    th = {"c_T": 1.3, "L_B": 0.9, "U_B": 1.1, "c_eta": 0.8, "c_C": None}
    dec = stats.confirmatory_decisions(boot, th, alpha=0.05)
    assert dec["R_T"]["pass"] is True
    assert dec["R_B_equivalence"]["pass"] is True
    assert dec["R_eta"]["pass"] is True
    # Tighten thresholds until each fails.
    th_fail = {"c_T": 2.6, "L_B": 0.99, "U_B": 1.01, "c_eta": 0.5}
    dec2 = stats.confirmatory_decisions(boot, th_fail, alpha=0.05)
    assert dec2["R_T"]["pass"] is False
    assert dec2["R_B_equivalence"]["pass"] is False
    assert dec2["R_eta"]["pass"] is False


def test_equivalence_uses_90pct_ci_at_alpha_05():
    n = 1000
    rb = np.linspace(0.90, 1.10, n)  # 90% CI ~ [0.91, 1.09]; 95% CI wider
    boot = pd.DataFrame({
        "R_T": np.full(n, 2.0), "R_B": rb, "R_eta": rb / 2.0,
        "one_minus_R_eta": 1 - rb / 2.0,
    })
    th = {"c_T": 1.0, "L_B": 0.905, "U_B": 1.095, "c_eta": 0.99}
    dec = stats.confirmatory_decisions(boot, th, alpha=0.05)
    lo, hi = dec["R_B_equivalence"]["ci"]
    assert lo == pytest.approx(np.quantile(rb, 0.05))
    assert hi == pytest.approx(np.quantile(rb, 0.95))
    assert dec["R_B_equivalence"]["pass"] is True


def test_confirmatory_control_uses_absolute_korean_reduction_not_G():
    n = 1000
    boot = pd.DataFrame({
        "R_T": np.full(n, 2.0),
        "R_B": np.full(n, 1.0),
        "R_eta": np.full(n, 0.5),
        "one_minus_R_eta": np.full(n, 0.5),
        "C_K": np.linspace(1.10, 1.20, n),
        "G": np.linspace(0.40, 0.50, n),
    })
    th = {"c_T": 1.3, "L_B": 0.9, "U_B": 1.1, "c_eta": 0.8, "c_C": 0.9}
    dec = stats.confirmatory_decisions(boot, th, alpha=0.05)
    assert "C_K" in dec
    assert "G" not in dec
    assert dec["C_K"]["pass"] is False
