#!/usr/bin/env python3
"""
XNV advanced market maker bot.

Usage:
    python bots/run_xnv_mm.py examples/xnv_market_maker.yml
    python bots/run_xnv_mm.py examples/xnv_market_maker.yml --dry-run
    python bots/run_xnv_mm.py examples/xnv_market_maker.yml --monitor-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    from engine.xnv_mm_runner import run_xnv_mm
    from utils.logging_config import setup_logging

    parser = argparse.ArgumentParser(description="XNV advanced market maker bot")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Log-only mode, no order placement",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulated execution (logs orders but does not place them)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.monitor_only:
        config["mode"] = "monitor"
    elif args.dry_run:
        config["mode"] = "dry-run"
    else:
        config["mode"] = config.get("mode", "live")

    state_path = Path(config.get("state_path", "state/xnv_mm_state.json"))
    state_path.parent.mkdir(parents=True, exist_ok=True)

    run_xnv_mm(config, state_path)


if __name__ == "__main__":
    main()
