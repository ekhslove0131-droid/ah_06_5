"""Minimal launch/status CLI; deliberately no stop or kill operation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runner import configured_worker, launch_detached, status


def build_parser():
    parser = argparse.ArgumentParser(prog="team-ensemble")
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch = subparsers.add_parser("launch", help="launch the sealed bundle")
    launch.add_argument("--run-root", required=True)
    launch.add_argument("--config")
    launch.add_argument("--start", action="store_true")
    inspect_status = subparsers.add_parser("status", help="read current durable status")
    inspect_status.add_argument("--run-root", required=True)
    preflight = subparsers.add_parser("preflight", help="validate sealed inputs without fitting")
    preflight.add_argument("--config", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(args.run_root), sort_keys=True))
        return 0
    if args.command == "preflight":
        from workflow import preflight_workflow
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        print(json.dumps(preflight_workflow(config), sort_keys=True))
        return 0
    if not args.start:
        print(json.dumps({"status": "START_FLAG_REQUIRED"}, sort_keys=True))
        return 2
    if not args.config or not Path(args.config).is_file():
        print(json.dumps({"status": "CONFIG_REQUIRED"}, sort_keys=True))
        return 2
    result = launch_detached(
        args.run_root,
        lambda: configured_worker(args.config, args.run_root),
        start=True,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "STARTED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
