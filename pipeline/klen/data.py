"""Corpus loading, NFC normalization, and validation.

One row per aligned Korean-English pair. Validation produces per-row flags and
a corpus-level report; nothing is silently dropped — exclusion decisions belong
to the researcher and are recorded via the flags.
"""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import pandas as pd

REQUIRED_LOGICAL_COLUMNS = [
    "korean_text", "english_text", "pair_id", "cluster_id", "genre", "direction",
]

_CONTROL_CATEGORIES = {"Cc", "Cf"}
_ALLOWED_CONTROLS = {"\n", "\t", "\r"}


def load_corpus(path: str | Path, fmt: str | None = None) -> pd.DataFrame:
    path = Path(path)
    fmt = fmt or path.suffix.lstrip(".").lower()
    if fmt == "csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    if fmt == "jsonl":
        return pd.read_json(path, lines=True, dtype=str)
    if fmt == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported corpus format: {fmt}")


def rename_to_logical(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Map researcher-supplied column names to the pipeline's logical names."""
    missing_spec = [k for k in REQUIRED_LOGICAL_COLUMNS if not column_map.get(k)]
    if missing_spec:
        raise ValueError(f"column mapping incomplete for: {missing_spec}")
    missing_cols = [v for v in column_map.values() if v not in df.columns]
    if missing_cols:
        raise ValueError(f"corpus lacks columns: {missing_cols}; has {list(df.columns)}")
    inverse = {v: k for k, v in column_map.items()}
    return df.rename(columns=inverse)[REQUIRED_LOGICAL_COLUMNS].copy()


def _has_disallowed_control(s: str) -> bool:
    return any(
        unicodedata.category(ch) in _CONTROL_CATEGORIES and ch not in _ALLOWED_CONTROLS
        for ch in s
    )


def _has_hangul(s: str) -> bool:
    return any(
        "가" <= ch <= "힣" or "ᄀ" <= ch <= "ᇿ" or "㄰" <= ch <= "㆏"
        for ch in s
    )


def normalize_and_flag(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply NFC to both text columns; add validation flags and size columns.

    Returns (dataframe, report). Rows are never dropped here.
    """
    out = df.copy()
    for side, col in (("ko", "korean_text"), ("en", "english_text")):
        raw = out[col].astype(str)
        norm = raw.map(lambda s: unicodedata.normalize("NFC", s))
        out[col] = norm
        out[f"flag_{side}_nfc_changed"] = raw.values != norm.values
        out[f"flag_{side}_empty"] = norm.str.strip().str.len() == 0
        out[f"flag_{side}_control_chars"] = norm.map(_has_disallowed_control)
        out[f"n_chars_{side}"] = norm.str.len()
        out[f"n_bytes_utf8_{side}"] = norm.map(lambda s: len(s.encode("utf-8")))
    out["flag_ko_no_hangul"] = ~out["korean_text"].map(_has_hangul)
    out["flag_ko_equals_en"] = (
        out["korean_text"].str.strip() == out["english_text"].str.strip()
    )
    out["flag_en_has_hangul"] = out["english_text"].map(_has_hangul)
    out["flag_duplicate_pair_id"] = out["pair_id"].duplicated(keep=False)
    out["flag_missing_meta"] = (
        out[["pair_id", "cluster_id", "genre", "direction"]].isna().any(axis=1)
    )

    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    report = {
        "n_rows": int(len(out)),
        "n_clusters": int(out["cluster_id"].nunique()),
        "genres": {str(k): int(v) for k, v in out["genre"].value_counts().items()},
        "directions": {str(k): int(v) for k, v in out["direction"].value_counts().items()},
        "flag_counts": {c: int(out[c].sum()) for c in flag_cols},
    }
    return out, report


def hard_errors(df: pd.DataFrame) -> list[str]:
    """Conditions under which scoring must not proceed."""
    errs = []
    if df["flag_duplicate_pair_id"].any():
        dups = df.loc[df["flag_duplicate_pair_id"], "pair_id"].unique().tolist()
        errs.append(f"duplicate pair_id values: {dups[:10]}")
    for side in ("ko", "en"):
        n = int(df[f"flag_{side}_empty"].sum())
        if n:
            errs.append(f"{n} rows have empty {side} text")
    if df["flag_missing_meta"].any():
        errs.append(f"{int(df['flag_missing_meta'].sum())} rows missing metadata")
    return errs


def dataset_sha256(df: pd.DataFrame) -> str:
    """Order-sensitive hash over normalized text + identifiers."""
    h = hashlib.sha256()
    cols = REQUIRED_LOGICAL_COLUMNS
    for row in df[cols].itertuples(index=False):
        for val in row:
            h.update(str(val).encode("utf-8"))
            h.update(b"\x1f")
        h.update(b"\x1e")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Researcher-declared exclusion rules
# ---------------------------------------------------------------------------
# The spec's optional `exclusions` mapping declares which flagged rows are
# removed before analysis. Every key defaults to "no exclusion" (null/false),
# so an absent block means "analyse every scored row". The block is part of
# the frozen spec content, so the rules cannot change after freezing. Rules
# are applied at analysis time to the per-pair results file; measurement
# always scores every row so that exclusions remain auditable.

EXCLUSION_RULES = {
    "drop_ko_equals_en": "drop rows whose Korean and English text are identical after NFC and strip",
    "drop_ko_no_hangul": "drop rows whose Korean side contains no Hangul",
    "drop_control_chars": "drop rows with disallowed control/format characters on either side",
    "min_chars_ko": "drop rows with fewer Korean characters than this integer",
    "min_chars_en": "drop rows with fewer English characters than this integer",
    "char_ratio_ko_over_en": "drop rows whose n_chars_ko / n_chars_en lies outside [lo, hi]",
}


def ko_equals_en(df: pd.DataFrame) -> pd.Series:
    return (df["korean_text"].astype(str).str.strip()
            == df["english_text"].astype(str).str.strip())


def apply_exclusions(df: pd.DataFrame, rules: dict | None) -> tuple[pd.DataFrame, dict]:
    """Apply the spec's exclusion rules to a per-pair results frame.

    Returns (filtered frame, report). Unknown rule names are an error so a
    typo can never silently keep rows. Row order is preserved.
    """
    rules = dict(rules or {})
    unknown = sorted(set(rules) - set(EXCLUSION_RULES))
    if unknown:
        raise ValueError(f"unknown exclusion rule(s): {unknown}; allowed: {sorted(EXCLUSION_RULES)}")

    keep = pd.Series(True, index=df.index)
    dropped_by_rule: dict[str, int] = {}

    def _drop(name: str, mask: pd.Series) -> None:
        nonlocal keep
        mask = mask.fillna(False).astype(bool)
        dropped_by_rule[name] = int((mask & keep).sum())
        keep &= ~mask

    if rules.get("drop_ko_equals_en"):
        _drop("drop_ko_equals_en", ko_equals_en(df))
    if rules.get("drop_ko_no_hangul"):
        col = df["flag_ko_no_hangul"] if "flag_ko_no_hangul" in df.columns \
            else ~df["korean_text"].map(_has_hangul)
        _drop("drop_ko_no_hangul", col.astype(bool))
    if rules.get("drop_control_chars"):
        if {"flag_ko_control_chars", "flag_en_control_chars"} <= set(df.columns):
            mask = df["flag_ko_control_chars"].astype(bool) | df["flag_en_control_chars"].astype(bool)
        else:
            mask = df["korean_text"].map(_has_disallowed_control) | df["english_text"].map(_has_disallowed_control)
        _drop("drop_control_chars", mask)
    if rules.get("min_chars_ko") is not None:
        _drop("min_chars_ko", df["n_chars_ko"] < int(rules["min_chars_ko"]))
    if rules.get("min_chars_en") is not None:
        _drop("min_chars_en", df["n_chars_en"] < int(rules["min_chars_en"]))
    if rules.get("char_ratio_ko_over_en") is not None:
        lo, hi = rules["char_ratio_ko_over_en"]
        if not (0 < float(lo) < float(hi)):
            raise ValueError("char_ratio_ko_over_en must be [lo, hi] with 0 < lo < hi")
        ratio = df["n_chars_ko"] / df["n_chars_en"]
        _drop("char_ratio_ko_over_en", (ratio < float(lo)) | (ratio > float(hi)))

    out = df.loc[keep].copy()
    report = {
        "rules": {k: rules.get(k) for k in EXCLUSION_RULES},
        "n_rows_before": int(len(df)),
        "n_rows_after": int(len(out)),
        "n_rows_dropped": int(len(df) - len(out)),
        "n_clusters_before": int(df["cluster_id"].nunique()) if "cluster_id" in df.columns else None,
        "n_clusters_after": int(out["cluster_id"].nunique()) if "cluster_id" in out.columns else None,
        "dropped_by_rule_sequential": dropped_by_rule,
    }
    return out, report
