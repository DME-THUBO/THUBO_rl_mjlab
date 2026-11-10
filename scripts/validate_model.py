#!/usr/bin/env python3
"""Validate the checked-in MJCF and actuator metadata before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dme_thubo.model_validation import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--model", type=Path, default=root / "mjcf" / "THUBO.xml")
    parser.add_argument("--actuator-config", type=Path, default=root / "sts3250_official.json")
    args = parser.parse_args()
    import json
    try:
        print(json.dumps(validate(args.model, args.actuator_config), indent=2))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
