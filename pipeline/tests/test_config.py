import copy

import pytest
import yaml

from klen import config

SPEC_PATH = "spec/measurement_spec_template.yaml"


def complete_spec():
    spec = config.load_spec(SPEC_PATH)
    spec = copy.deepcopy(spec)
    spec["corpus"]["path"] = "corpus.csv"
    for k in spec["corpus"]["columns"]:
        spec["corpus"]["columns"][k] = k
    spec["estimator"]["model_id"] = "m"
    spec["estimator"]["revision"] = "r"
    spec["tokenizers"]["baseline"] = {"id": "b", "revision": "r"}
    spec["tokenizers"]["korean_fit"] = {"id": "k", "revision": "r"}
    spec["token_accounting"] = "content_only"
    spec["scoring_policy"] = {"bos": "prepend_bos", "eos": "none"}
    spec["thresholds"] = {"c_T": 1.3, "L_B": 0.9, "U_B": 1.1, "c_eta": 0.85, "c_C": 0.9}
    spec["endpoints"]["primary"] = "R_eta"
    spec["endpoints"]["confirmatory"] = ["R_T", "R_B_equivalence", "R_eta"]
    spec["endpoints"]["decision_rule"] = "all_confirmatory_pass"
    return spec


def test_template_spec_reports_all_blocking_fields():
    spec = config.load_spec(SPEC_PATH)
    missing = config.missing_blocking_fields(spec)
    assert set(missing) == set(config.BLOCKING_FIELDS)


def test_complete_spec_has_no_missing_or_invalid():
    spec = complete_spec()
    assert config.missing_blocking_fields(spec) == []
    assert config.invalid_fields(spec) == []


def test_threshold_sanity_checks():
    spec = complete_spec()
    spec["thresholds"]["c_T"] = 0.9
    spec["thresholds"]["c_eta"] = 1.2
    spec["thresholds"]["L_B"] = 1.05
    bad = config.invalid_fields(spec)
    assert any("c_T" in b for b in bad)
    assert any("c_eta" in b for b in bad)
    assert any("equivalence interval" in b for b in bad)


def test_confirmatory_blocked_when_unfrozen():
    spec = complete_spec()
    with pytest.raises(config.SpecNotReady, match="not frozen"):
        config.assert_ready_for_confirmatory(spec)


def test_freeze_and_tamper_detection(tmp_path):
    p = tmp_path / "spec.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(complete_spec(), f, allow_unicode=True)
    frozen = config.freeze_spec(p)
    config.assert_ready_for_confirmatory(frozen)  # no raise
    # Tampering after freeze must be detected.
    frozen["thresholds"]["c_T"] = 1.01
    with pytest.raises(config.SpecNotReady, match="hash mismatch"):
        config.assert_ready_for_confirmatory(frozen)


def test_freeze_refuses_incomplete_spec(tmp_path):
    spec = complete_spec()
    spec["thresholds"]["c_eta"] = None
    p = tmp_path / "spec.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, allow_unicode=True)
    with pytest.raises(config.SpecNotReady, match="thresholds.c_eta"):
        config.freeze_spec(p)
