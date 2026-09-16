from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training_bank import preflight, prepare_bank


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cbj-stack7-bank")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "prepare-bank"):
        current = sub.add_parser(name); current.add_argument("--config", required=True)
        if name == "prepare-bank": current.add_argument("--start", action="store_true")
    args = parser.parse_args(argv); path = Path(args.config); config = json.loads(path.read_text())
    config["_runtime_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if args.command == "preflight":
        result = preflight(config); print(json.dumps(result, sort_keys=True)); return 0 if result["ready"] else 3
    if not args.start:
        print(json.dumps({"status": "START_FLAG_REQUIRED"})); return 2
    result = prepare_bank(config, config["bank_output_root"])
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
