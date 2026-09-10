#!/usr/bin/env python3
"""Validate and convert OPUS GlobalVoices v2018q4 en-ko to an aligned CSV."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ARCHIVE_NAME = "OPUS-GlobalVoices-v2018q4-en-ko.txt.zip"
ARCHIVE_SHA256 = "b8d91a43ac305bd54f98edcf2598686cc7b1e6ab3bdae05fb0983a3644a68144"
DOWNLOAD_URL = "https://object.pouta.csc.fi/OPUS-GlobalVoices/v2018q4/moses/en-ko.txt.zip"
RELEASE = "v2018q4"
EXPECTED_ARCHIVE_BYTES = 891_364
EXPECTED_DOCUMENTS = 365
EXPECTED_NONEMPTY_DOCUMENTS = 347
EXPECTED_ALIGNMENT_UNITS = 9_017
SPLIT_SEED = "20260910"
PILOT_FRACTION = 0.20
EXPECTED_COMPONENTS = 325
EXPECTED_MULTI_CLUSTER_COMPONENTS = 8
EXPECTED_PILOT_CLUSTERS = 69
EXPECTED_CONFIRMATORY_CLUSTERS = 278
EXPECTED_PILOT_ROWS = 1_732
EXPECTED_CONFIRMATORY_ROWS = 7_285
EN_MEMBER = "GlobalVoices.en-ko.en"
KO_MEMBER = "GlobalVoices.en-ko.ko"
XML_MEMBER = "GlobalVoices.en-ko.xml"
README_MEMBER = "README"
LICENSE_MEMBER = "LICENSE"
DOCUMENT_PATH_PATTERN = re.compile(
    r"^en/(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})_"
    r"(?P<slug>.+)_\.xml\.gz$"
)
OUTPUT_FIELDS = [
    "pair_id",
    "cluster_id",
    "genre",
    "translation_direction",
    "ko_text",
    "en_text",
    "source_url",
    "split",
    "corpus_release",
    "document_index",
    "alignment_index",
    "en_document_path",
    "ko_document_path",
    "xtargets",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return unicodedata.normalize("NFC", value.strip())


def document_chunks(text: str) -> list[list[str]]:
    """Split a Moses file on article separators, retaining empty article groups."""
    chunks: list[list[str]] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = normalized_text(raw_line)
        if line:
            current.append(line)
        else:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def source_url_from_document(path: str) -> str:
    match = DOCUMENT_PATH_PATTERN.fullmatch(path)
    if match is None:
        raise ValueError(f"Cannot derive a Global Voices URL from {path!r}")
    values = match.groupdict()
    return (
        "https://globalvoices.org/"
        f"{values['year']}/{values['month']}/{values['day']}/{values['slug']}/"
    )


def stable_cluster_id(en_document_path: str) -> str:
    digest = hashlib.sha256(en_document_path.encode("utf-8")).hexdigest()[:16]
    return f"globalvoices-{RELEASE}-doc-{digest}"


def checked_member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except KeyError as exc:
        raise ValueError(f"Archive is missing required member {name!r}") from exc


def count_blank_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if not line.strip())


def build_rows(archive_path: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError(
            f"Archive has {archive_path.stat().st_size} bytes; "
            f"expected {EXPECTED_ARCHIVE_BYTES}"
        )
    archive_hash = sha256_file(archive_path)
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError(
            f"Archive SHA-256 is {archive_hash}; expected {ARCHIVE_SHA256}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP integrity check failed for {bad_member!r}")
        en_bytes = checked_member(archive, EN_MEMBER)
        ko_bytes = checked_member(archive, KO_MEMBER)
        xml_bytes = checked_member(archive, XML_MEMBER)
        readme_bytes = checked_member(archive, README_MEMBER)
        license_bytes = checked_member(archive, LICENSE_MEMBER)

    en_text = en_bytes.decode("utf-8-sig")
    ko_text = ko_bytes.decode("utf-8-sig")
    en_documents = document_chunks(en_text)
    ko_documents = document_chunks(ko_text)
    root = ET.fromstring(xml_bytes)
    groups = root.findall("linkGrp")

    counts = {
        "english_physical_lines": len(en_text.splitlines()),
        "korean_physical_lines": len(ko_text.splitlines()),
        "english_blank_separators": count_blank_lines(en_text),
        "korean_blank_separators": count_blank_lines(ko_text),
        "english_document_chunks": len(en_documents),
        "korean_document_chunks": len(ko_documents),
        "xml_link_groups": len(groups),
        "xml_nonempty_link_groups": sum(bool(group.findall("link")) for group in groups),
        "xml_empty_link_groups": sum(not group.findall("link") for group in groups),
        "xml_links": sum(len(group.findall("link")) for group in groups),
    }
    if not (
        len(groups)
        == len(en_documents)
        == len(ko_documents)
        == EXPECTED_DOCUMENTS
    ):
        raise ValueError(f"Document-count mismatch: {counts}")
    if counts["xml_links"] != EXPECTED_ALIGNMENT_UNITS:
        raise ValueError(f"Alignment-count mismatch: {counts}")

    rows: list[dict[str, str]] = []
    for document_offset, (group, en_lines, ko_lines) in enumerate(
        zip(groups, en_documents, ko_documents, strict=True), start=1
    ):
        links = group.findall("link")
        if not (len(links) == len(en_lines) == len(ko_lines)):
            raise ValueError(
                f"Document {document_offset} has {len(links)} XML links, "
                f"{len(en_lines)} English rows, and {len(ko_lines)} Korean rows"
            )
        en_document = group.attrib.get("fromDoc", "")
        ko_document = group.attrib.get("toDoc", "")
        if not en_document or not ko_document:
            raise ValueError(f"Document {document_offset} lacks source document metadata")
        source_url = source_url_from_document(en_document)
        cluster_id = stable_cluster_id(en_document)

        for alignment_offset, (link, en_line, ko_line) in enumerate(
            zip(links, en_lines, ko_lines, strict=True), start=1
        ):
            xtargets = link.attrib.get("xtargets", "")
            if not xtargets or ";" not in xtargets:
                raise ValueError(
                    f"Document {document_offset}, alignment {alignment_offset} "
                    "has invalid xtargets metadata"
                )
            rows.append(
                {
                    "pair_id": (
                        f"globalvoices-{RELEASE}-d{document_offset:04d}-"
                        f"a{alignment_offset:04d}"
                    ),
                    "cluster_id": cluster_id,
                    "genre": "news",
                    "translation_direction": "unknown",
                    "ko_text": ko_line,
                    "en_text": en_line,
                    "source_url": source_url,
                    "split": "unassigned",
                    "corpus_release": RELEASE,
                    "document_index": str(document_offset),
                    "alignment_index": str(alignment_offset),
                    "en_document_path": en_document,
                    "ko_document_path": ko_document,
                    "xtargets": xtargets,
                }
            )

    if len(rows) != EXPECTED_ALIGNMENT_UNITS:
        raise AssertionError(f"Prepared {len(rows)} rows; expected {EXPECTED_ALIGNMENT_UNITS}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise AssertionError("pair_id is not unique")
    if len({row["cluster_id"] for row in rows}) != EXPECTED_NONEMPTY_DOCUMENTS:
        raise AssertionError("cluster_id does not map one-to-one to XML documents")
    if any(not row["en_text"] or not row["ko_text"] for row in rows):
        raise AssertionError("Prepared data contains an empty aligned text")

    member_metadata = {
        EN_MEMBER: {"bytes": len(en_bytes), "sha256": sha256_bytes(en_bytes)},
        KO_MEMBER: {"bytes": len(ko_bytes), "sha256": sha256_bytes(ko_bytes)},
        XML_MEMBER: {"bytes": len(xml_bytes), "sha256": sha256_bytes(xml_bytes)},
        README_MEMBER: {
            "bytes": len(readme_bytes),
            "sha256": sha256_bytes(readme_bytes),
        },
        LICENSE_MEMBER: {
            "bytes": len(license_bytes),
            "sha256": sha256_bytes(license_bytes),
        },
    }
    return rows, {
        "archive_sha256": archive_hash,
        "observed_counts": counts,
        "archive_members": member_metadata,
    }


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def assign_cluster_splits(
    rows: list[dict[str, str]],
) -> tuple[list[list[str]], set[str], dict[str, object]]:
    """Split connected article components so exact duplicates cannot leak."""
    clusters = sorted({row["cluster_id"] for row in rows})
    parent = {cluster: cluster for cluster in clusters}

    def find(cluster: str) -> str:
        while parent[cluster] != cluster:
            parent[cluster] = parent[parent[cluster]]
            cluster = parent[cluster]
        return cluster

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            parent[larger] = smaller

    pair_clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        pair_clusters[(row["en_text"], row["ko_text"])].add(row["cluster_id"])
    for linked_clusters in pair_clusters.values():
        ordered = sorted(linked_clusters)
        for cluster in ordered[1:]:
            union(ordered[0], cluster)

    component_map: dict[str, list[str]] = defaultdict(list)
    for cluster in clusters:
        component_map[find(cluster)].append(cluster)
    components = [sorted(component) for component in component_map.values()]
    multi_cluster_components = sum(len(component) > 1 for component in components)
    if len(components) != EXPECTED_COMPONENTS:
        raise AssertionError(
            f"Found {len(components)} duplicate-linked components; "
            f"expected {EXPECTED_COMPONENTS}"
        )
    if multi_cluster_components != EXPECTED_MULTI_CLUSTER_COMPONENTS:
        raise AssertionError(
            f"Found {multi_cluster_components} multi-cluster components; "
            f"expected {EXPECTED_MULTI_CLUSTER_COMPONENTS}"
        )

    def component_serialization(component: list[str]) -> str:
        return "\n".join(component)

    ranked = sorted(
        components,
        key=lambda component: (
            hashlib.sha256(
                f"{SPLIT_SEED}\0{component_serialization(component)}".encode("utf-8")
            ).hexdigest(),
            component_serialization(component),
        ),
    )
    target_clusters = int(len(clusters) * PILOT_FRACTION)
    cumulative_clusters = 0
    pilot_component_count = 0
    while pilot_component_count < len(ranked):
        next_size = len(ranked[pilot_component_count])
        if cumulative_clusters + next_size > target_clusters:
            break
        cumulative_clusters += next_size
        pilot_component_count += 1
    if pilot_component_count < len(ranked):
        next_size = len(ranked[pilot_component_count])
        current_distance = abs(cumulative_clusters - target_clusters)
        next_distance = abs(cumulative_clusters + next_size - target_clusters)
        if next_distance < current_distance:
            cumulative_clusters += next_size
            pilot_component_count += 1

    pilot_clusters = {
        cluster
        for component in ranked[:pilot_component_count]
        for cluster in component
    }
    if len(pilot_clusters) != EXPECTED_PILOT_CLUSTERS:
        raise AssertionError(
            f"Split rule selected {len(pilot_clusters)} pilot clusters; "
            f"expected {EXPECTED_PILOT_CLUSTERS}"
        )
    for row in rows:
        row["split"] = (
            "pilot" if row["cluster_id"] in pilot_clusters else "confirmatory"
        )

    split_memberships: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        split_memberships[(row["en_text"], row["ko_text"])].add(row["split"])
    cross_split_duplicate_groups = sum(
        len(splits) > 1 for splits in split_memberships.values()
    )
    if cross_split_duplicate_groups:
        raise AssertionError(
            f"{cross_split_duplicate_groups} exact bilingual duplicates cross the split"
        )

    component_details: dict[str, object] = {
        "components": len(components),
        "multi_cluster_components": multi_cluster_components,
        "component_size_distribution": dict(
            sorted(Counter(len(component) for component in components).items())
        ),
        "pilot_components": pilot_component_count,
        "confirmatory_components": len(ranked) - pilot_component_count,
        "target_pilot_clusters": target_clusters,
        "cross_split_exact_duplicate_groups": cross_split_duplicate_groups,
    }
    return ranked, pilot_clusters, component_details


def main() -> int:
    pipeline_dir = Path(__file__).resolve().parents[1]
    archive_path = pipeline_dir / "data" / "raw" / ARCHIVE_NAME
    processed_dir = pipeline_dir / "data" / "processed"
    pilot_path = processed_dir / "globalvoices_v2018q4_en_ko_pilot.csv"
    confirmatory_path = (
        processed_dir / "globalvoices_v2018q4_en_ko_confirmatory.csv"
    )
    manifest_path = processed_dir / "globalvoices_v2018q4_en_ko_split_manifest.json"
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Missing {archive_path}. Download it from {DOWNLOAD_URL} without renaming."
        )
    existing_outputs = [
        str(path)
        for path in (pilot_path, confirmatory_path, manifest_path)
        if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing output: " + ", ".join(existing_outputs)
        )

    rows, validation = build_rows(archive_path)
    ranked_components, pilot_clusters, component_details = assign_cluster_splits(rows)
    pilot_rows = [row for row in rows if row["split"] == "pilot"]
    confirmatory_rows = [row for row in rows if row["split"] == "confirmatory"]
    if len(pilot_rows) != EXPECTED_PILOT_ROWS:
        raise AssertionError(
            f"Pilot has {len(pilot_rows)} rows; expected {EXPECTED_PILOT_ROWS}"
        )
    if len(confirmatory_rows) != EXPECTED_CONFIRMATORY_ROWS:
        raise AssertionError(
            "Confirmatory file has "
            f"{len(confirmatory_rows)} rows; expected {EXPECTED_CONFIRMATORY_ROWS}"
        )
    if {row["cluster_id"] for row in pilot_rows}.intersection(
        row["cluster_id"] for row in confirmatory_rows
    ):
        raise AssertionError("Pilot and confirmatory cluster sets overlap")
    write_csv(pilot_path, pilot_rows)
    write_csv(confirmatory_path, confirmatory_rows)

    cluster_sizes = Counter(row["cluster_id"] for row in rows)
    exact_pair_counts = Counter((row["en_text"], row["ko_text"]) for row in rows)
    pair_clusters: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        pair_clusters[(row["en_text"], row["ko_text"])].add(row["cluster_id"])
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "corpus": "GlobalVoices",
            "release": RELEASE,
            "format": "OPUS Moses",
            "languages": ["en", "ko"],
            "download_url": DOWNLOAD_URL,
            "archive_path": str(archive_path),
            "expected_archive_bytes": EXPECTED_ARCHIVE_BYTES,
            **validation,
        },
        "processing": {
            "unicode_normalization": "NFC",
            "surrounding_whitespace": "removed",
            "alignment_mapping": "blank-separated Moses chunks matched in order to XML linkGrp/link elements",
            "genre": "news",
            "translation_direction": "unknown",
            "split": {
                "unit": "cluster_id (one OPUS XML linkGrp / article)",
                "indivisible_unit": "connected component of article clusters linked by identical (en_text, ko_text) pairs",
                "seed": SPLIT_SEED,
                "pilot_fraction": PILOT_FRACTION,
                "rule": "union article clusters sharing an exact bilingual pair; rank components by SHA256(seed + NUL + newline-joined sorted cluster IDs); choose the ranked prefix with article count closest to floor(fraction * N), breaking an equal-distance cutoff toward the smaller pilot",
                "pilot_cluster_ids": sorted(pilot_clusters),
                "ranked_component_list_sha256": sha256_bytes(
                    ("\n\n".join("\n".join(component) for component in ranked_components) + "\n").encode("utf-8")
                ),
                **component_details,
            },
        },
        "output": {
            "total_rows": len(rows),
            "total_clusters": len(cluster_sizes),
            "format": "csv",
            "encoding": "utf-8",
            "columns": OUTPUT_FIELDS,
            "pilot": {
                "path": str(pilot_path),
                "rows": len(pilot_rows),
                "clusters": len({row["cluster_id"] for row in pilot_rows}),
                "sha256": sha256_file(pilot_path),
                "bytes": pilot_path.stat().st_size,
            },
            "confirmatory": {
                "path": str(confirmatory_path),
                "rows": len(confirmatory_rows),
                "clusters": len({row["cluster_id"] for row in confirmatory_rows}),
                "sha256": sha256_file(confirmatory_path),
                "bytes": confirmatory_path.stat().st_size,
            },
            "minimum_rows_per_cluster": min(cluster_sizes.values()),
            "maximum_rows_per_cluster": max(cluster_sizes.values()),
        },
        "quality_audit": {
            "filtering": "none; all nonempty upstream alignment units retained",
            "empty_english_rows": sum(not row["en_text"] for row in rows),
            "empty_korean_rows": sum(not row["ko_text"] for row in rows),
            "non_nfc_english_rows": sum(
                unicodedata.normalize("NFC", row["en_text"]) != row["en_text"]
                for row in rows
            ),
            "non_nfc_korean_rows": sum(
                unicodedata.normalize("NFC", row["ko_text"]) != row["ko_text"]
                for row in rows
            ),
            "korean_rows_without_hangul_syllables": sum(
                re.search(r"[\uac00-\ud7a3]", row["ko_text"]) is None for row in rows
            ),
            "english_rows_without_ascii_letters": sum(
                re.search(r"[A-Za-z]", row["en_text"]) is None for row in rows
            ),
            "exact_duplicate_pair_groups": sum(
                count > 1 for count in exact_pair_counts.values()
            ),
            "exact_duplicate_rows_beyond_first": sum(
                count - 1 for count in exact_pair_counts.values() if count > 1
            ),
            "exact_duplicate_groups_spanning_clusters": sum(
                len(clusters) > 1 for clusters in pair_clusters.values()
            ),
            "maximum_exact_pair_occurrences": max(exact_pair_counts.values()),
            "cross_split_exact_duplicate_groups": component_details[
                "cross_split_exact_duplicate_groups"
            ],
            "note": "Flags are descriptive only. Rows without Hangul or ASCII letters and exact duplicates were not automatically treated as invalid.",
        },
        "limitations": [
            "The archive supplies no per-document translation-direction field.",
            "The release supplies no native pilot/confirmatory split; this preparation adds a deterministic article-level split.",
            "Pilot observations must not be reused in the confirmatory analysis.",
            "Because the material predates modern models, training-data contamination cannot generally be ruled out.",
            "The archive license file points to original-source licenses without an SPDX identifier; preserve attribution and inspect historical source pages before republication.",
        ],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Prepared {len(rows)} aligned rows across {len(cluster_sizes)} articles")
    print(
        f"Pilot: {pilot_path} ({len(pilot_rows)} rows, "
        f"SHA-256 {manifest['output']['pilot']['sha256']})"
    )
    print(
        f"Confirmatory: {confirmatory_path} ({len(confirmatory_rows)} rows, "
        f"SHA-256 {manifest['output']['confirmatory']['sha256']})"
    )
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
