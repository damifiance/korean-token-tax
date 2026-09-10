#!/usr/bin/env python3
"""Download, validate, and align the gated FLORES+ Korean/English files."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPOSITORY = "openlanguagedata/flores_plus"
DATASET_VERSION = "4.6"
REVISION = "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06"
LICENSE = "CC-BY-SA-4.0"
EXPECTED_ROWS = {"dev": 997, "devtest": 1012}
LANGUAGES = {"en": "eng_Latn", "ko": "kor_Hang"}
SPLITS = ("dev", "devtest")
REQUIRED_SOURCE_FIELDS = {
    "id",
    "text",
    "url",
    "domain",
    "topic",
    "last_updated",
    "split",
}
OUTPUT_FIELDS = [
    "pair_id",
    "cluster_id",
    "genre",
    "translation_direction",
    "ko_text",
    "en_text",
    "source_url",
    "split",
    "flores_id",
    "domain",
    "topic",
    "en_last_updated",
    "ko_last_updated",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_url(split: str, language_code: str) -> str:
    return (
        f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
        f"{REVISION}/{split}/{language_code}.jsonl"
    )


def download_file(url: str, destination: Path, token: str) -> None:
    if destination.exists():
        print(f"Reusing existing source file: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "flores-plus-ko-en-research-preparation/1.0",
        },
    )

    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
        temporary_path.replace(destination)
    except urllib.error.HTTPError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        if exc.code in (401, 403):
            raise RuntimeError(
                "FLORES+ access was rejected. Log in to Hugging Face, accept "
                "the dataset conditions personally, and provide a read token "
                "through the HF_TOKEN environment variable."
            ) from exc
        raise
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_source(path: Path, expected_split: str, expected_language: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank line in {path} at line {line_number}")
            row = json.loads(line)
            missing = REQUIRED_SOURCE_FIELDS.difference(row)
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is missing fields: {sorted(missing)}"
                )
            if str(row["split"]) != expected_split:
                raise ValueError(
                    f"{path}:{line_number} has split={row['split']!r}; "
                    f"expected {expected_split!r}"
                )
            expected_iso = "eng" if expected_language == "eng_Latn" else "kor"
            if str(row.get("iso_639_3", "")) != expected_iso:
                raise ValueError(
                    f"{path}:{line_number} has iso_639_3={row.get('iso_639_3')!r}; "
                    f"expected {expected_iso!r}"
                )
            row_id = str(row["id"])
            if row_id in rows:
                raise ValueError(f"Duplicate id {row_id!r} in {path}")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"Empty text in {path} for id {row_id!r}")
            rows[row_id] = row
    return rows


def stable_id_order(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def cluster_id(source: str, pair_id: str) -> str:
    if not source:
        return pair_id
    digest = hashlib.sha256(source.strip().encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"


def align_split(
    split: str,
    english: dict[str, dict[str, Any]],
    korean: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    english_ids = set(english)
    korean_ids = set(korean)
    if english_ids != korean_ids:
        missing_ko = sorted(english_ids - korean_ids, key=stable_id_order)
        missing_en = sorted(korean_ids - english_ids, key=stable_id_order)
        raise ValueError(
            f"Unaligned {split} IDs: missing Korean={missing_ko[:10]}, "
            f"missing English={missing_en[:10]}"
        )

    expected = EXPECTED_ROWS[split]
    if len(english_ids) != expected:
        raise ValueError(
            f"Unexpected {split} row count: got {len(english_ids)}, expected {expected}"
        )

    aligned: list[dict[str, str]] = []
    for row_id in sorted(english_ids, key=stable_id_order):
        en = english[row_id]
        ko = korean[row_id]
        for field in ("url", "domain", "topic"):
            if str(en[field]) != str(ko[field]):
                raise ValueError(
                    f"Metadata mismatch for {split}/{row_id}: {field} is "
                    f"{en[field]!r} in English and {ko[field]!r} in Korean"
                )
        pair = f"floresplus-{split}-{row_id}"
        url = str(en["url"]).strip()
        domain = str(en["domain"]).strip()
        aligned.append(
            {
                "pair_id": pair,
                "cluster_id": cluster_id(url, pair),
                "genre": domain,
                "translation_direction": "en->ko",
                "ko_text": str(ko["text"]),
                "en_text": str(en["text"]),
                "source_url": url,
                "split": split,
                "flores_id": row_id,
                "domain": domain,
                "topic": str(en["topic"]).strip(),
                "en_last_updated": str(en["last_updated"]),
                "ko_last_updated": str(ko["last_updated"]),
            }
        )
    return aligned


def refuse_overwrite(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite generated files. Move or remove them first: "
            + ", ".join(existing)
        )


def write_outputs(rows: list[dict[str, str]], jsonl_path: Path, csv_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_record(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def main() -> int:
    base = Path(__file__).resolve().parent
    raw_dir = base / "raw"
    processed_dir = base / "processed"
    jsonl_output = processed_dir / "flores_plus_ko_en.jsonl"
    csv_output = processed_dir / "flores_plus_ko_en.csv"
    manifest_path = base / "manifest.json"
    refuse_overwrite([jsonl_output, csv_output, manifest_path])

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Accept the FLORES+ gate personally and export "
            "a Hugging Face read token before running this script."
        )

    source_paths: dict[tuple[str, str], Path] = {}
    for split in SPLITS:
        for language, language_code in LANGUAGES.items():
            path = raw_dir / split / f"{language_code}.jsonl"
            download_file(source_url(split, language_code), path, token)
            source_paths[(split, language)] = path

    all_rows: list[dict[str, str]] = []
    source_records: list[dict[str, Any]] = []
    for split in SPLITS:
        en_path = source_paths[(split, "en")]
        ko_path = source_paths[(split, "ko")]
        english = read_source(en_path, split, LANGUAGES["en"])
        korean = read_source(ko_path, split, LANGUAGES["ko"])
        all_rows.extend(align_split(split, english, korean))
        source_records.extend(
            [
                {
                    "language": "eng_Latn",
                    "split": split,
                    "download_url": source_url(split, LANGUAGES["en"]),
                    **file_record(en_path, len(english)),
                },
                {
                    "language": "kor_Hang",
                    "split": split,
                    "download_url": source_url(split, LANGUAGES["ko"]),
                    **file_record(ko_path, len(korean)),
                },
            ]
        )

    expected_total = sum(EXPECTED_ROWS.values())
    if len(all_rows) != expected_total:
        raise AssertionError(f"Got {len(all_rows)} aligned rows; expected {expected_total}")

    write_outputs(all_rows, jsonl_output, csv_output)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": REPOSITORY,
            "dataset_version": DATASET_VERSION,
            "revision": REVISION,
            "license": LICENSE,
            "dataset_page": f"https://huggingface.co/datasets/{REPOSITORY}",
            "gated": True,
            "languages": LANGUAGES,
            "splits": list(SPLITS),
            "files": source_records,
        },
        "alignment": {
            "join_key": ["split", "id"],
            "translation_direction": "en->ko",
            "rows": len(all_rows),
            "rows_by_split": dict(Counter(row["split"] for row in all_rows)),
            "clusters": len({row["cluster_id"] for row in all_rows}),
            "rows_by_genre": dict(Counter(row["genre"] for row in all_rows)),
        },
        "outputs": [
            file_record(jsonl_output, len(all_rows)),
            file_record(csv_output, len(all_rows)),
        ],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Prepared {len(all_rows)} aligned pairs in {processed_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

