"""Measurement-spec loading, validation, and freeze checking.

The spec file is the single source of truth for every researcher-supplied
decision. Fields listed in BLOCKING_FIELDS must be non-null before the
confirmatory pipeline will run. This module never fills in defaults for
researcher decisions.
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Dotted paths of researcher decisions that block the confirmatory run.
BLOCKING_FIELDS = [
    "corpus.path",
    "corpus.columns.korean_text",
    "corpus.columns.english_text",
    "corpus.columns.pair_id",
    "corpus.columns.cluster_id",
    "corpus.columns.genre",
    "corpus.columns.direction",
    "estimator.model_id",
    "estimator.revision",
    "tokenizers.baseline.id",
    "tokenizers.baseline.revision",
    "tokenizers.korean_fit.id",
    "tokenizers.korean_fit.revision",
    "token_accounting",
    "scoring_policy.bos",
    "scoring_policy.eos",
    "thresholds.c_T",
    "thresholds.L_B",
    "thresholds.U_B",
    "thresholds.c_eta",
    "thresholds.c_C",
    "endpoints.primary",
    "endpoints.confirmatory",
]

ALLOWED_VALUES = {
    "token_accounting": {"content_only", "with_special_tokens", "chat_template"},
    "scoring_policy.bos": {"prepend_bos", "none_skip_first"},
    "scoring_policy.eos": {"none", "append_scored"},
}


PRIMARY_ENDPOINTS = {"R_T", "R_B", "R_eta", "one_minus_R_eta", "C_K"}
CONFIRMATORY_COMPARISONS = {"R_T", "R_B_equivalence", "R_eta", "C_K"}
DECISION_RULES = {"all_confirmatory_pass"}


def _get(d: dict, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_spec(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict):
        raise ValueError(f"spec file {path} did not parse to a mapping")
    return spec


def missing_blocking_fields(spec: dict) -> list[str]:
    missing = []
    for field in BLOCKING_FIELDS:
        val = _get(spec, field)
        if val is None or (isinstance(val, (list, str)) and len(val) == 0):
            missing.append(field)
    return missing


def invalid_fields(spec: dict) -> list[str]:
    """Fields that are set but fail sanity constraints."""
    bad = []
    for dotted, allowed in ALLOWED_VALUES.items():
        val = _get(spec, dotted)
        if val is not None and val not in allowed:
            bad.append(f"{dotted}={val!r} not in {sorted(allowed)}")
    th = spec.get("thresholds") or {}
    c_T, L_B, U_B = th.get("c_T"), th.get("L_B"), th.get("U_B")
    c_eta, c_C = th.get("c_eta"), th.get("c_C")
    if c_T is not None and not c_T > 1:
        bad.append(f"thresholds.c_T={c_T} must be > 1")
    if c_eta is not None and not 0 < c_eta < 1:
        bad.append(f"thresholds.c_eta={c_eta} must be in (0, 1)")
    if c_C is not None and not 0 < c_C < 1:
        bad.append(f"thresholds.c_C={c_C} must be in (0, 1)")
    if L_B is not None and U_B is not None and not (0 < L_B < 1 < U_B):
        bad.append(f"equivalence interval [{L_B}, {U_B}] must straddle 1 with L_B in (0,1)")
    if (spec.get("token_accounting") == "chat_template"
            and spec.get("chat_template_config") is None):
        bad.append("token_accounting=chat_template requires chat_template_config")
    ep = spec.get("endpoints") or {}
    primary = ep.get("primary")
    if primary is not None and primary not in PRIMARY_ENDPOINTS:
        bad.append(f"endpoints.primary={primary!r} not in {sorted(PRIMARY_ENDPOINTS)}")
    conf = ep.get("confirmatory") or []
    unknown = [c for c in conf if c not in CONFIRMATORY_COMPARISONS]
    if unknown:
        bad.append(f"endpoints.confirmatory has unknown comparisons {unknown}; "
                   f"allowed: {sorted(CONFIRMATORY_COMPARISONS)}")
    rule = ep.get("decision_rule")
    if conf and rule is None:
        bad.append("endpoints.decision_rule must be set when confirmatory comparisons are listed")
    if rule is not None and rule not in DECISION_RULES:
        bad.append(f"endpoints.decision_rule={rule!r} not in {sorted(DECISION_RULES)}")
    return bad


def spec_content_hash(spec: dict) -> str:
    """Hash of the spec content excluding the freeze bookkeeping fields."""
    clean = copy.deepcopy(spec)
    for k in ("frozen", "frozen_at", "frozen_sha256"):
        clean.pop(k, None)
    canonical = yaml.safe_dump(clean, sort_keys=True, allow_unicode=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assert_ready_for_confirmatory(spec: dict) -> None:
    missing = missing_blocking_fields(spec)
    bad = invalid_fields(spec)
    problems = [f"missing: {m}" for m in missing] + [f"invalid: {b}" for b in bad]
    if not spec.get("frozen"):
        problems.append("spec is not frozen (run klen.freeze_spec first)")
    elif spec.get("frozen_sha256") != spec_content_hash(spec):
        problems.append("spec content changed after freezing (hash mismatch)")
    if problems:
        raise SpecNotReady(problems)


class SpecNotReady(RuntimeError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(
            "confirmatory run blocked:\n" + "\n".join(f"  - {p}" for p in problems)
        )


def freeze_spec(path: str | Path) -> dict:
    """Mark the spec frozen, recording content hash and timestamp. Refuses if
    blocking fields are missing or invalid."""
    spec = load_spec(path)
    missing = missing_blocking_fields(spec)
    bad = invalid_fields(spec)
    if missing or bad:
        raise SpecNotReady([f"missing: {m}" for m in missing] + [f"invalid: {b}" for b in bad])
    spec["frozen"] = True
    spec["frozen_at"] = datetime.now(timezone.utc).isoformat()
    spec["frozen_sha256"] = spec_content_hash(spec)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
    return spec
