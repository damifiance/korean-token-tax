"""Provenance recording: versions, hashes, seeds, environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

TRACKED_PACKAGES = [
    "torch", "transformers", "tokenizers", "numpy", "pandas", "scipy",
    "pyyaml", "huggingface_hub", "safetensors", "tiktoken",
]


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_file_hashes(tokenizer) -> dict:
    """Hash files backing a loaded HF tokenizer or tiktoken encoding."""
    hashes = {}
    try:
        vocab_files = tokenizer.init_kwargs.get("name_or_path")
    except AttributeError:
        vocab_files = None
    # Most reliable: hash all files in the resolved snapshot directory.
    candidates = set()
    for attr in ("vocab_file", "merges_file", "tokenizer_file"):
        p = getattr(tokenizer, attr, None) or tokenizer.init_kwargs.get(attr)
        if p and Path(str(p)).is_file():
            candidates.add(Path(str(p)))
    snapshot_dir = getattr(tokenizer, "_klen_snapshot_dir", None)
    if snapshot_dir and Path(str(snapshot_dir)).is_dir():
        for p in Path(str(snapshot_dir)).iterdir():
            if p.suffix in (".json", ".txt", ".model", ".tiktoken"):
                candidates.add(p)
    if vocab_files and Path(str(vocab_files)).is_dir():
        for p in Path(str(vocab_files)).iterdir():
            if p.suffix in (".json", ".txt", ".model"):
                candidates.add(p)
    for p in sorted(candidates):
        hashes[p.name] = file_sha256(p)
    source_id = getattr(tokenizer, "_klen_source_id", None)
    revision = getattr(tokenizer, "_klen_revision", None)
    if source_id is not None:
        hashes["_source_id"] = source_id
    if revision is not None:
        hashes["_revision"] = revision
    return hashes


def package_versions() -> dict:
    versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def environment_record(spec: dict, dataset_hash: str,
                       tokenizer_hashes: dict | None = None,
                       extra: dict | None = None) -> dict:
    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "spec": spec,
        "dataset_sha256": dataset_hash,
        "tokenizer_file_hashes": tokenizer_hashes or {},
    }
    if extra:
        rec.update(extra)
    return rec


def write_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
