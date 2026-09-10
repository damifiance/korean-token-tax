"""Billing-token counting under arbitrary tokenizers.

These tokenizers are used ONLY for counting |tau_b(s)|. Their token IDs are
never fed to the estimator model (enforced by keeping counting purely
integer-valued here — no ID sequences are returned).

Hugging Face tokenizers are addressed by their repository ID plus immutable
commit revision. OpenAI ``tiktoken`` encodings use IDs such as
``tiktoken:o200k_base`` and pin the installed package version as the revision.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
from contextlib import contextmanager
from pathlib import Path


_TIKTOKEN_ASSETS = {
    "r50k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
        "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    ),
    "cl100k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    ),
    "o200k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    ),
}


@contextmanager
def _tiktoken_cache(cache_dir: Path):
    """Temporarily direct tiktoken's verified downloads to project storage."""
    key = "TIKTOKEN_CACHE_DIR"
    old = os.environ.get(key)
    os.environ[key] = str(cache_dir)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


class _TiktokenCountingTokenizer:
    """Small Transformers-like wrapper used by :func:`count_tokens`.

    Raw tiktoken encodings do not prepend or append default special tokens, so
    ``add_special_tokens=True`` and ``False`` intentionally have equal counts.
    ``encode_ordinary`` treats literal strings such as ``<|endoftext|>`` as
    message content instead of control tokens.
    """

    def __init__(self, encoding, encoding_name: str, asset_path: Path,
                 source_id: str, revision: str):
        self._encoding = encoding
        self.encoding_name = encoding_name
        self.tokenizer_file = str(asset_path)
        self.init_kwargs = {
            "name_or_path": str(asset_path.parent),
            "tokenizer_file": str(asset_path),
        }
        self._klen_source_id = source_id
        self._klen_revision = revision

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        del add_special_tokens  # raw encodings have no automatic wrapper
        return {"input_ids": self._encoding.encode_ordinary(text)}

    def apply_chat_template(self, *args, **kwargs):
        del args, kwargs
        raise ValueError(
            "raw tiktoken encodings have no chat template; use content_only "
            "or supply a Hugging Face chat tokenizer"
        )

    @property
    def n_vocab(self) -> int:
        return self._encoding.n_vocab


def count_tokens(tokenizer, text: str, accounting: str,
                 chat_template_config: dict | None = None) -> int:
    if accounting == "content_only":
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])
    if accounting == "with_special_tokens":
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])
    if accounting == "chat_template":
        if not chat_template_config:
            raise ValueError("chat_template accounting requires chat_template_config")
        role = chat_template_config.get("role", "user")
        ids = tokenizer.apply_chat_template(
            [{"role": role, "content": text}],
            tokenize=True,
            add_generation_prompt=bool(chat_template_config.get("add_generation_prompt", False)),
        )
        return len(ids)
    raise ValueError(f"unknown token accounting: {accounting}")


def load_counting_tokenizer(tokenizer_id: str, revision: str):
    if tokenizer_id.startswith("tiktoken:"):
        encoding_name = tokenizer_id.split(":", 1)[1]
        if encoding_name not in _TIKTOKEN_ASSETS:
            raise ValueError(
                f"unsupported tiktoken encoding {encoding_name!r}; "
                f"choose one of {sorted(_TIKTOKEN_ASSETS)}"
            )
        try:
            installed = importlib.metadata.version("tiktoken")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("tiktoken is required for tiktoken:* tokenizers") from exc
        if revision != installed:
            raise RuntimeError(
                f"tiktoken revision mismatch: spec pins {revision!r}, "
                f"but installed package is {installed!r}"
            )

        import tiktoken
        from tiktoken.load import read_file_cached

        url, expected_hash = _TIKTOKEN_ASSETS[encoding_name]
        cache_dir = (
            Path(__file__).resolve().parents[1]
            / "artifacts" / "tokenizers" / "tiktoken-cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha1(url.encode()).hexdigest()
        asset_path = cache_dir / cache_key
        with _tiktoken_cache(cache_dir):
            # Ensures a local copy exists and verifies its SHA-256 before use.
            read_file_cached(url, expected_hash)
            encoding = tiktoken.get_encoding(encoding_name)
        return _TiktokenCountingTokenizer(
            encoding, encoding_name, asset_path, tokenizer_id, revision
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, revision=revision, use_fast=True
    )
    tokenizer._klen_source_id = tokenizer_id
    tokenizer._klen_revision = revision
    try:
        from huggingface_hub import snapshot_download
        tokenizer._klen_snapshot_dir = snapshot_download(
            repo_id=tokenizer_id,
            revision=revision,
            local_files_only=True,
            allow_patterns=[
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "vocab.json", "vocab.txt",
                "merges.txt", "*.model", "*.tiktoken",
            ],
        )
    except Exception:
        # Tokenization itself already succeeded; provenance can still record
        # the immutable source ID/revision even if no snapshot path is exposed.
        tokenizer._klen_snapshot_dir = None
    return tokenizer
