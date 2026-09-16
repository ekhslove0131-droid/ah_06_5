from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from deployment import deploy
from evidence import prepare_evidence, prerequisite_status
from search import run_search
from source_identity import source_identity


def read_json(path): return json.loads(Path(path).read_text())


def _runner(config, run_root, callback):
    sys.path.insert(0, config["v2_root"])
    from runner import launch_detached
    return launch_detached(run_root, callback, start=True)


def _verify_source(config):
    source_sha, rows = source_identity(Path(__file__).parent)
    expected = config.get("meta_source_sha256")
    if expected and expected != source_sha: raise ValueError("combo meta source hash mismatch")
    return source_sha, rows


def status(config):
    result = {"prerequisite": prerequisite_status(config)}
    for name, root in (("search", config["combo_search_run_root"]), ("deployment", config["combo_deployment_run_root"])):
        root = Path(root); complete = root / ("SEARCH_COMPLETE.json" if name == "search" else "COMPLETE.json")
        sys.path.insert(0, config["v2_root"])
        from runner import _runner_live
        result[name] = {
            "runner_live": _runner_live(root),
            "complete": read_json(complete) if complete.is_file() else None,
            "failure": read_json(root / "FAILURE.json") if (root / "FAILURE.json").is_file() else None,
        }
    return result


def build_parser():
    parser = argparse.ArgumentParser(prog="cbj-combo-search")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "prepare-evidence", "search", "deploy", "status"):
        current = sub.add_parser(name); current.add_argument("--config", required=True)
        if name in {"prepare-evidence", "search", "deploy"}: current.add_argument("--start", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv); config = read_json(args.config)
    config["_runtime_config_sha256"] = hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    if args.command == "preflight": print(json.dumps(prerequisite_status(config), sort_keys=True)); return 0
    if args.command == "status": print(json.dumps(status(config), sort_keys=True)); return 0
    if not args.start: print(json.dumps({"status": "START_FLAG_REQUIRED"})); return 2
    source_sha, rows = _verify_source(config)
    if args.command == "prepare-evidence":
        result = prepare_evidence(config, config["combo_evidence_root"])
        if result.get("status") == "EVIDENCE_PREPARED":
            Path(config["combo_evidence_root"], "META_SOURCE.json").write_text(json.dumps({
                "schema_version": "COMBO_META_SOURCE_V1", "source_sha256": source_sha, "files": rows,
            }, indent=2) + "\n")
            frozen_config = {key: value for key, value in config.items() if not key.startswith("_")}
            frozen_config["evidence_sha256"] = result["evidence_sha256"]
            frozen_config["grid_manifest_sha256"] = hashlib.sha256(Path(config["grid_manifest_json"]).read_bytes()).hexdigest()
            Path(config["combo_evidence_root"], "runtime_config.search.json").write_text(
                json.dumps(frozen_config, indent=2) + "\n"
            )
        print(json.dumps(result, sort_keys=True)); return 0 if result.get("ready") else 3
    prereq = prerequisite_status(config)
    if not prereq.get("ready"): print(json.dumps(prereq, sort_keys=True)); return 3
    if args.command == "search":
        sys.path.insert(0, config["v2_root"])
        from runner import _runner_live
        if _runner_live(config["combo_deployment_run_root"]):
            print(json.dumps({"status": "DEPLOYMENT_ALREADY_RUNNING"})); return 3
        result = _runner(config, config["combo_search_run_root"], lambda: run_search(config, config["combo_search_run_root"]))
    else:
        sys.path.insert(0, config["v2_root"])
        from runner import _runner_live
        if _runner_live(config["combo_search_run_root"]):
            print(json.dumps({"status": "SEARCH_STILL_RUNNING"})); return 3
        if not Path(config["combo_search_run_root"], "SEARCH_COMPLETE.json").is_file():
            print(json.dumps({"status": "SEARCH_NOT_COMPLETE"})); return 3
        result = _runner(config, config["combo_deployment_run_root"], lambda: deploy(config, config["combo_deployment_run_root"]))
    print(json.dumps(result, sort_keys=True)); return 0 if result.get("status") == "STARTED" else 3


if __name__ == "__main__": raise SystemExit(main())
