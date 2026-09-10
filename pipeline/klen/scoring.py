"""Summed autoregressive NLL in bits under the fixed estimator model.

Guarantees:
  - tokenization uses the estimator model's native tokenizer only,
    with add_special_tokens=False; BOS/EOS are added explicitly per policy;
  - raw next-token logits, model.eval(), torch.inference_mode();
  - log-softmax computed in float32 regardless of model dtype;
  - every target token is scored exactly once (asserted), including under
    chunked scoring with non-overlapping target accounting;
  - no padding and no chat-template tokens are ever scored.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

LN2 = math.log(2.0)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_local_estimator_snapshot(
    local_path: str | Path, model_id: str, revision: str,
) -> Path:
    """Verify that an offline snapshot is the declared canonical estimator.

    ``MODEL_MANIFEST.json`` binds the local files to the canonical Hub model
    ID and immutable revision.  File hashes prevent a silently incomplete or
    altered checkpoint from being scored as the frozen estimator, while the
    identity checks catch a mismatched model/tokenizer configuration even if a
    manifest was assembled incorrectly.
    """
    root = Path(local_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"estimator.local_path is not a directory: {root}")

    manifest_path = root / "MODEL_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"local estimator snapshot lacks {manifest_path.name}: {root}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("model_id") != model_id:
        raise ValueError(
            "local estimator model_id mismatch: "
            f"manifest={manifest.get('model_id')!r}, spec={model_id!r}"
        )
    if manifest.get("revision") != revision:
        raise ValueError(
            "local estimator revision mismatch: "
            f"manifest={manifest.get('revision')!r}, spec={revision!r}"
        )

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("local estimator manifest has no file inventory")
    for name, expected in files.items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe path in local estimator manifest: {name!r}")
        path = root / rel
        if not path.is_file():
            raise ValueError(f"local estimator file missing: {path}")
        expected_size = expected.get("size")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(
                f"local estimator size mismatch for {name}: "
                f"got {path.stat().st_size}, expected {expected_size}"
            )
        expected_hash = expected.get("sha256")
        if not expected_hash or _file_sha256(path) != expected_hash:
            raise ValueError(f"local estimator SHA-256 mismatch for {name}")

    config_path = root / "config.json"
    tokenizer_path = root / "tokenizer.json"
    tokenizer_config_path = root / "tokenizer_config.json"
    for required in (config_path, tokenizer_path, tokenizer_config_path):
        if not required.is_file():
            raise ValueError(f"local estimator identity file missing: {required}")
    with config_path.open("r", encoding="utf-8") as f:
        model_config = json.load(f)
    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    with tokenizer_config_path.open("r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    identity = manifest.get("identity") or {}
    expected_model = identity.get("model") or {}
    for key, expected in expected_model.items():
        if model_config.get(key) != expected:
            raise ValueError(
                f"local estimator config identity mismatch for {key}: "
                f"got {model_config.get(key)!r}, expected {expected!r}"
            )

    expected_tok = (identity.get("tokenizer") or {})
    actual_tok = {
        "model_type": (tokenizer_json.get("model") or {}).get("type"),
        "normalizer_type": (tokenizer_json.get("normalizer") or {}).get("type"),
        "regular_vocab_size": len((tokenizer_json.get("model") or {}).get("vocab") or {}),
        "add_bos_token": tokenizer_config.get("add_bos_token"),
    }
    eot_id = expected_tok.get("endoftext_token_id")
    if eot_id is not None:
        added = {
            int(item["id"]): item.get("content")
            for item in tokenizer_json.get("added_tokens", [])
            if "id" in item
        }
        actual_tok["endoftext_token_id"] = eot_id if eot_id in added else None
        actual_tok["endoftext_token"] = added.get(eot_id)
    for key, expected in expected_tok.items():
        if actual_tok.get(key) != expected:
            raise ValueError(
                f"local estimator tokenizer identity mismatch for {key}: "
                f"got {actual_tok.get(key)!r}, expected {expected!r}"
            )
    return root


@dataclass(frozen=True)
class ScoringPolicy:
    bos: str                       # "prepend_bos" | "none_skip_first"
    eos: str                       # "none" | "append_scored"
    context_length: int | None = None   # None -> single pass, no chunking
    context_overlap: int | None = None  # carried context when chunking; default W//2

    def __post_init__(self):
        if self.bos not in ("prepend_bos", "none_skip_first"):
            raise ValueError(f"unknown bos policy: {self.bos}")
        if self.eos not in ("none", "append_scored"):
            raise ValueError(f"unknown eos policy: {self.eos}")


@dataclass
class ScoreResult:
    bits_total: float
    n_targets: int                 # tokens actually scored
    n_content_tokens: int          # estimator-tokenizer content tokens (no BOS/EOS)
    per_token_bits: list[float] = field(default_factory=list)
    full_ids: list[int] = field(default_factory=list)
    n_chunks: int = 1


def build_sequence(tokenizer, text: str, policy: ScoringPolicy) -> tuple[list[int], int, int]:
    """Return (full_ids, first_target_index, n_content_tokens).

    Content tokens come from add_special_tokens=False; BOS/EOS handled
    explicitly so no unintended special tokens can enter.
    """
    content_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(content_ids) == 0:
        raise ValueError("text tokenized to zero content tokens")

    ids = list(content_ids)
    if policy.eos == "append_scored":
        if tokenizer.eos_token_id is None:
            raise ValueError("eos policy requires tokenizer.eos_token_id")
        ids.append(tokenizer.eos_token_id)

    if policy.bos == "prepend_bos":
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError("bos policy 'prepend_bos' requires tokenizer.bos_token_id")
        full = [bos_id] + ids
    else:  # none_skip_first: first content token has no context and is not scored
        full = ids
    first_target_index = 1
    if len(full) < 2:
        raise ValueError("sequence too short to score any target under this policy")
    return full, first_target_index, len(content_ids)


def _chunk_windows(n: int, first_target: int, policy: ScoringPolicy):
    """Yield (start, target_start, end): model input full[start:end], scored
    targets full[target_start:end]. Target ranges partition
    [first_target, n) — non-overlapping accounting; context may overlap."""
    W = policy.context_length
    if W is None or n <= W:
        yield (0, first_target, n)
        return
    if W < 2:
        raise ValueError("context_length must be >= 2")
    overlap = policy.context_overlap if policy.context_overlap is not None else W // 2
    if not 1 <= overlap <= W - 1:
        raise ValueError("context_overlap must be in [1, context_length-1]")

    target_start = first_target
    start = 0
    while target_start < n:
        end = min(start + W, n)
        yield (start, target_start, end)
        target_start = end
        start = end - overlap


def score_text(model, tokenizer, text: str, policy: ScoringPolicy,
               device: str | torch.device = "cpu",
               keep_per_token: bool = False) -> ScoreResult:
    full, first_target, n_content = build_sequence(tokenizer, text, policy)
    model.eval()

    per_token_bits: list[float] = []
    total_bits = 0.0
    n_scored = 0
    n_chunks = 0

    with torch.inference_mode():
        for start, t0, end in _chunk_windows(len(full), first_target, policy):
            n_chunks += 1
            input_ids = torch.tensor([full[start:end]], dtype=torch.long, device=device)
            logits = model(input_ids).logits  # [1, L, V], raw logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            # target at absolute position p is predicted by logits at p-1
            rows = torch.arange(t0 - 1 - start, end - 1 - start, device=device)
            cols = input_ids[0, t0 - start: end - start]
            sel = logprobs[0, rows, cols]
            # MPS does not implement float64 tensors.  Move the selected
            # float32 log-probabilities to CPU before converting to float64
            # for stable accumulation and Python-list output.
            bits = (-sel / LN2).cpu().double()
            total_bits += float(bits.sum().item())
            n_scored += int(bits.numel())
            if keep_per_token:
                per_token_bits.extend(bits.tolist())

    expected = len(full) - first_target
    if n_scored != expected:
        raise AssertionError(
            f"scored {n_scored} targets but expected {expected}: accounting bug"
        )
    return ScoreResult(
        bits_total=total_bits,
        n_targets=n_scored,
        n_content_tokens=n_content,
        per_token_bits=per_token_bits,
        full_ids=full,
        n_chunks=n_chunks,
    )


def load_estimator(model_id: str, revision: str, dtype: str = "float32",
                   device: str = "cpu", local_path: str | Path | None = None):
    """Load the fixed estimator model and its native tokenizer at a pinned
    revision. The estimator is only ever paired with its own tokenizer.

    ``local_path`` is a loading optimization, not the estimator identity: its
    manifest must match ``model_id`` and ``revision``, which remain the
    canonical provenance fields.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    if local_path is not None:
        source = validate_local_estimator_snapshot(local_path, model_id, revision)
        load_kwargs = {"local_files_only": True}
    else:
        source = model_id
        load_kwargs = {"revision": revision}

    tok = AutoTokenizer.from_pretrained(source, use_fast=True, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch_dtype, **load_kwargs
    ).to(device)
    # Some causal-LM repositories (including Qwen3 Base) declare a BOS id in
    # the model configuration while intentionally disabling automatic BOS
    # insertion in the tokenizer.  Synchronize only the missing tokenizer
    # field so an explicit ``prepend_bos`` scoring policy can use the model's
    # declared boundary token; never override a tokenizer-defined BOS.
    if tok.bos_token_id is None and model.config.bos_token_id is not None:
        tok.bos_token_id = model.config.bos_token_id
    # Preserve canonical identity in tokenizer provenance even when files were
    # loaded from an offline project directory.
    tok._klen_source_id = model_id
    tok._klen_revision = revision
    if local_path is not None:
        tok._klen_snapshot_dir = str(source)
    model.eval()
    return model, tok
