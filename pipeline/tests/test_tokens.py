import pytest

from klen import tokens


class FakeTok:
    """Counts characters as tokens; add_special_tokens=True adds 2 (CLS/SEP-like)."""

    def __call__(self, text, add_special_tokens=False):
        ids = list(range(len(text)))
        if add_special_tokens:
            ids = [101] + ids + [102]
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        n = sum(len(m["content"]) for m in messages)
        return list(range(n + 7 + (3 if add_generation_prompt else 0)))


def test_content_only_counting():
    assert tokens.count_tokens(FakeTok(), "hello", "content_only") == 5


def test_with_special_tokens_counting():
    assert tokens.count_tokens(FakeTok(), "hello", "with_special_tokens") == 7


def test_chat_template_counting():
    n = tokens.count_tokens(FakeTok(), "hello", "chat_template",
                            {"role": "user", "add_generation_prompt": False})
    assert n == 12
    n2 = tokens.count_tokens(FakeTok(), "hello", "chat_template",
                             {"role": "user", "add_generation_prompt": True})
    assert n2 == 15


def test_chat_template_requires_config():
    with pytest.raises(ValueError, match="requires chat_template_config"):
        tokens.count_tokens(FakeTok(), "hello", "chat_template", None)


def test_unknown_accounting_rejected():
    with pytest.raises(ValueError, match="unknown token accounting"):
        tokens.count_tokens(FakeTok(), "hello", "api_billed")


def test_tiktoken_o200k_content_only_and_no_automatic_specials():
    tok = tokens.load_counting_tokenizer("tiktoken:o200k_base", "0.14.0")
    plain = tokens.count_tokens(tok, "한국어 and English", "content_only")
    wrapped = tokens.count_tokens(tok, "한국어 and English", "with_special_tokens")
    assert plain > 0
    assert wrapped == plain
    assert tok.n_vocab == 200019


def test_tiktoken_revision_must_match_installed_package():
    with pytest.raises(RuntimeError, match="revision mismatch"):
        tokens.load_counting_tokenizer("tiktoken:o200k_base", "not-installed")


def test_tiktoken_rejects_chat_template_accounting():
    tok = tokens.load_counting_tokenizer("tiktoken:o200k_base", "0.14.0")
    with pytest.raises(ValueError, match="no chat template"):
        tokens.count_tokens(
            tok, "hello", "chat_template",
            {"role": "user", "add_generation_prompt": False},
        )
