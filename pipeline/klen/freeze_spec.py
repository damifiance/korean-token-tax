"""Freeze the measurement spec: python -m klen.freeze_spec SPEC.yaml

Refuses to freeze while any blocking researcher decision is missing or
invalid. After freezing, any content change invalidates the recorded hash and
blocks confirmatory analysis.
"""

from __future__ import annotations

import argparse

from . import config


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    args = ap.parse_args(argv)
    try:
        spec = config.freeze_spec(args.spec)
    except config.SpecNotReady as e:
        raise SystemExit(str(e))
    print(f"spec frozen at {spec['frozen_at']}")
    print(f"content sha256: {spec['frozen_sha256']}")


if __name__ == "__main__":
    main()
