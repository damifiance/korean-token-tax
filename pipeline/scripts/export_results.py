#!/usr/bin/env python3
"""Copy analysis outputs from out/ into results/ for publication.

Per-pair measurement files are copied WITHOUT the text columns (corpus text
is not redistributed); every numeric column, identifier and flag is kept, so
all statistics can be recomputed from results/ alone. Summaries, bootstrap
replicates, provenance and precision-simulation files are copied verbatim.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, RES = ROOT / "out", ROOT / "results"
TEXT_COLS = {"korean_text", "english_text"}
RUNS = {"pilot": "per_pair_pilot.csv", "confirmatory": "per_pair_full.csv"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    manifest = {}
    for run, per_pair in RUNS.items():
        src, dst = OUT / run, RES / run
        if not src.exists():
            print(f"skip {run}: {src} missing", file=sys.stderr)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        files = {}
        for f in sorted(src.iterdir()):
            if f.name == per_pair:
                df = pd.read_csv(f)
                kept = [c for c in df.columns if c not in TEXT_COLS]
                target = dst / f.name.replace(".csv", "_no_text.csv")
                df[kept].to_csv(target, index=False)
                files[target.name] = {"sha256": sha(target), "rows": int(len(df)),
                                      "source_sha256": sha(f), "dropped_columns": sorted(TEXT_COLS)}
            elif f.suffix in {".json", ".csv", ".log"}:
                shutil.copy2(f, dst / f.name)
                files[f.name] = {"sha256": sha(dst / f.name)}
        manifest[run] = files
    for extra in (ROOT / "data/processed/globalvoices_v2018q4_en_ko_split_manifest.json",):
        if extra.exists():
            shutil.copy2(extra, RES / extra.name)
            manifest[extra.name] = {"sha256": sha(RES / extra.name)}
    (RES / "RESULTS_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"exported to {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
