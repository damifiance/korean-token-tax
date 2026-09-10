"""End-to-end measurement run: validate corpus -> score NLL -> count tokens
-> write one row per aligned pair + provenance JSON.

Usage:
  python -m klen.run_pipeline SPEC.yaml OUTDIR [--device mps|cpu] [--pilot N]
                                               [--allow-unfrozen]

--pilot N scores only the first N pairs and tags the output as pilot (excluded
from confirmatory analysis). --allow-unfrozen permits running before the spec
is frozen (pilot/debug only; output is tagged accordingly).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config, data, provenance, scoring, tokens
from .stats import (COL_BE, COL_BK, COL_TE, COL_TE_KF, COL_TK, COL_TK_KF)


def resolve_estimator_local_path(spec_path: str | Path, local_path: str | None):
    """Resolve estimator.local_path relative to the measurement-spec file."""
    if local_path is None:
        return None
    path = Path(local_path).expanduser()
    if not path.is_absolute():
        path = Path(spec_path).expanduser().resolve().parent / path
    return str(path.resolve())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("outdir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pilot", type=int, default=None)
    ap.add_argument("--allow-unfrozen", action="store_true")
    args = ap.parse_args(argv)

    spec = config.load_spec(args.spec)
    missing = config.missing_blocking_fields(spec)
    bad = config.invalid_fields(spec)

    # Measurement can run on an unfrozen spec only in explicit pilot/debug mode;
    # thresholds/endpoints are not needed for measurement, but corpus, model,
    # tokenizer, accounting, and BOS/EOS policy are.
    measurement_fields = [m for m in missing if not m.startswith(("thresholds", "endpoints"))]
    if measurement_fields or bad:
        raise config.SpecNotReady(
            [f"missing: {m}" for m in measurement_fields] + [f"invalid: {b}" for b in bad]
        )
    if not spec.get("frozen") and not args.allow_unfrozen:
        raise config.SpecNotReady(["spec not frozen; use --allow-unfrozen for pilot/debug runs"])

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- load + validate corpus -------------------------------------------
    cols = spec["corpus"]["columns"]
    raw = data.load_corpus(spec["corpus"]["path"], spec["corpus"].get("format"))
    df = data.rename_to_logical(raw, cols)
    df, report = data.normalize_and_flag(df)
    errs = data.hard_errors(df)
    if errs:
        for e in errs:
            print(f"HARD ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    ds_hash = data.dataset_sha256(df)

    if args.pilot:
        df = df.head(args.pilot).copy()

    # ---- estimator model (native tokenizer ONLY) --------------------------
    est = spec["estimator"]
    estimator_local_path = resolve_estimator_local_path(
        args.spec, est.get("local_path")
    )
    model, est_tok = scoring.load_estimator(
        est["model_id"], est["revision"], est.get("dtype", "float32"),
        args.device, local_path=estimator_local_path,
    )
    policy = scoring.ScoringPolicy(
        bos=spec["scoring_policy"]["bos"],
        eos=spec["scoring_policy"]["eos"],
        context_length=est.get("context_length"),
        context_overlap=est.get("context_overlap"),
    )

    # ---- counting tokenizers (never fed to the estimator) -----------------
    accounting = spec["token_accounting"]
    ct_cfg = spec.get("chat_template_config")
    tok_base = tokens.load_counting_tokenizer(
        spec["tokenizers"]["baseline"]["id"], spec["tokenizers"]["baseline"]["revision"]
    )
    tok_kf = tokens.load_counting_tokenizer(
        spec["tokenizers"]["korean_fit"]["id"], spec["tokenizers"]["korean_fit"]["revision"]
    )

    # ---- per-pair measurement ---------------------------------------------
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        rec = {c: getattr(r, c) for c in data.REQUIRED_LOGICAL_COLUMNS}
        for c in df.columns:
            if c.startswith(("flag_", "n_chars_", "n_bytes_")):
                rec[c] = getattr(r, c)
        sk = scoring.score_text(model, est_tok, r.korean_text, policy, args.device)
        se = scoring.score_text(model, est_tok, r.english_text, policy, args.device)
        rec[COL_BK] = sk.bits_total
        rec[COL_BE] = se.bits_total
        rec["n_targets_ko"] = sk.n_targets
        rec["n_targets_en"] = se.n_targets
        rec["est_tokens_ko"] = sk.n_content_tokens
        rec["est_tokens_en"] = se.n_content_tokens
        rec["n_chunks_ko"] = sk.n_chunks
        rec["n_chunks_en"] = se.n_chunks
        rec[COL_TK] = tokens.count_tokens(tok_base, r.korean_text, accounting, ct_cfg)
        rec[COL_TE] = tokens.count_tokens(tok_base, r.english_text, accounting, ct_cfg)
        rec[COL_TK_KF] = tokens.count_tokens(tok_kf, r.korean_text, accounting, ct_cfg)
        rec[COL_TE_KF] = tokens.count_tokens(tok_kf, r.english_text, accounting, ct_cfg)
        rows.append(rec)
        if (i + 1) % 25 == 0:
            print(f"scored {i + 1}/{len(df)} pairs", file=sys.stderr)

    out = pd.DataFrame(rows)
    tag = "pilot" if args.pilot else "full"
    out_path = outdir / f"per_pair_{tag}.csv"
    out.to_csv(out_path, index=False)

    prov = provenance.environment_record(
        spec, ds_hash,
        tokenizer_hashes={
            "estimator": provenance.tokenizer_file_hashes(est_tok),
            "baseline": provenance.tokenizer_file_hashes(tok_base),
            "korean_fit": provenance.tokenizer_file_hashes(tok_kf),
        },
        extra={
            "run_tag": tag,
            "n_pairs_scored": len(out),
            "device": args.device,
            "estimator_load_source": estimator_local_path or est["model_id"],
            "estimator_local_path_used": estimator_local_path is not None,
            "validation_report": report,
            "spec_frozen": bool(spec.get("frozen")),
        },
    )
    provenance.write_json(prov, outdir / f"provenance_{tag}.json")
    print(f"wrote {out_path} ({len(out)} rows) and provenance_{tag}.json")


if __name__ == "__main__":
    main()
