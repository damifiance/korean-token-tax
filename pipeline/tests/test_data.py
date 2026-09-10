import unicodedata

import pandas as pd
import pytest

from klen import data


def make_df(**overrides):
    base = {
        "korean_text": ["안녕하세요", "감사합니다"],
        "english_text": ["Hello", "Thank you"],
        "pair_id": ["p1", "p2"],
        "cluster_id": ["c1", "c1"],
        "genre": ["greeting", "greeting"],
        "direction": ["ko->en", "ko->en"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_nfc_normalization_applied():
    # NFD-decomposed Hangul must be converted to NFC and flagged.
    decomposed = unicodedata.normalize("NFD", "안녕")
    assert decomposed != "안녕"
    df = make_df(korean_text=[decomposed, "감사합니다"])
    out, report = data.normalize_and_flag(df)
    assert out.loc[0, "korean_text"] == "안녕"
    assert bool(out.loc[0, "flag_ko_nfc_changed"]) is True
    assert bool(out.loc[1, "flag_ko_nfc_changed"]) is False
    assert report["flag_counts"]["flag_ko_nfc_changed"] == 1


def test_char_and_byte_counts():
    df = make_df()
    out, _ = data.normalize_and_flag(df)
    assert out.loc[0, "n_chars_ko"] == 5
    assert out.loc[0, "n_bytes_utf8_ko"] == 15  # 5 Hangul syllables x 3 bytes
    assert out.loc[0, "n_chars_en"] == 5
    assert out.loc[0, "n_bytes_utf8_en"] == 5


def test_empty_text_is_hard_error():
    df = make_df(english_text=["Hello", "   "])
    out, _ = data.normalize_and_flag(df)
    errs = data.hard_errors(out)
    assert any("empty en text" in e for e in errs)


def test_duplicate_pair_id_is_hard_error():
    df = make_df(pair_id=["p1", "p1"])
    out, _ = data.normalize_and_flag(df)
    errs = data.hard_errors(out)
    assert any("duplicate pair_id" in e for e in errs)


def test_language_sanity_flags():
    df = make_df(korean_text=["no hangul here", "감사합니다"])
    out, _ = data.normalize_and_flag(df)
    assert bool(out.loc[0, "flag_ko_no_hangul"]) is True
    assert bool(out.loc[1, "flag_ko_no_hangul"]) is False


def test_rename_to_logical_missing_column_raises():
    df = pd.DataFrame({"ko": ["a"], "en": ["b"]})
    with pytest.raises(ValueError, match="column mapping incomplete"):
        data.rename_to_logical(df, {"korean_text": "ko", "english_text": "en"})


def test_dataset_hash_stable_and_content_sensitive():
    df1, _ = data.normalize_and_flag(make_df())
    df2, _ = data.normalize_and_flag(make_df())
    assert data.dataset_sha256(df1) == data.dataset_sha256(df2)
    df3, _ = data.normalize_and_flag(make_df(english_text=["Hello!", "Thank you"]))
    assert data.dataset_sha256(df1) != data.dataset_sha256(df3)
