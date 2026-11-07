"""Run or verify the RHPC Stage-A constructed-mechanism experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

from rhpc.admission import admit_official_spark_run, source_checkout_state
from rhpc.artifacts import publish_stage_a_artifact, verify_stage_a_artifact
from rhpc.errors import RHPCError
from rhpc.stage_a import StageAConfig, run_stage_a


SOURCE_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = SOURCE_ROOT / "docs" / "research" / "rhpc-stage-a-method-dev-protocol-2026-08-25.md"
OFFICIAL_ENV = "CHELATED_RHPC_STAGE_A"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run and atomically publish one observation")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-id", type=int, required=True)
    run.add_argument("--trials", type=int, default=32, help="development only; official mode always uses 256")
    run.add_argument("--spark-official", action="store_true")
    run.add_argument("--allow-official", action="store_true")
    run.add_argument("--predecessor-projection", type=Path)
    run.add_argument("--predecessor-public-key", type=Path)
    run.add_argument("--predecessor-protocol-digest")
    run.add_argument("--predecessor-evaluator-digest")
    verify = subparsers.add_parser("verify", help="verify a closed result and regenerate it exactly")
    verify.add_argument("artifact", type=Path)
    return parser


def _require_official_arguments(args: argparse.Namespace) -> None:
    if not args.allow_official or os.environ.get(OFFICIAL_ENV) != "1":
        raise RHPCError(f"official Spark mode requires --allow-official and {OFFICIAL_ENV}=1")
    required = {
        "--predecessor-projection": args.predecessor_projection,
        "--predecessor-public-key": args.predecessor_public_key,
        "--predecessor-protocol-digest": args.predecessor_protocol_digest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RHPCError("official Spark mode is missing " + ", ".join(missing))


def _run(args: argparse.Namespace) -> int:
    if os.path.lexists(str(args.output)):
        raise RHPCError("RHPC artifact destination must be absent before observation")
    protocol_sha256 = _sha256_file(PROTOCOL_PATH)
    if args.spark_official:
        _require_official_arguments(args)
        config = StageAConfig(run_id=args.run_id)
        if not config.official:
            raise RHPCError("official Spark mode permits only frozen run IDs 7 and 11")
        admission = admit_official_spark_run(
            SOURCE_ROOT,
            args.predecessor_projection,
            args.predecessor_public_key,
            protocol_digest=args.predecessor_protocol_digest,
            evaluator_digest=args.predecessor_evaluator_digest,
        )
        execution_mode = "SPARK_OFFICIAL"
        source = admission["source"]
        predecessor = admission["predecessor"]
    else:
        config = StageAConfig(run_id=args.run_id, trials=args.trials)
        execution_mode = "DEVELOPMENT"
        source = source_checkout_state(SOURCE_ROOT)
        predecessor = None
    result = run_stage_a(config)
    verified = publish_stage_a_artifact(
        args.output,
        result,
        execution_mode=execution_mode,
        protocol_sha256=protocol_sha256,
        source_git_sha=source["source_git_sha"],
        source_tree_clean=source["source_tree_clean"],
        predecessor_completion=predecessor,
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output.resolve()),
                "execution_mode": execution_mode,
                "run_id": result["run_id"],
                "status": result["status"],
                "source_git_sha": verified["manifest"]["source_git_sha"],
                "protocol_sha256": protocol_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            verified = verify_stage_a_artifact(args.artifact, exact_regeneration=True)
            print(
                json.dumps(
                    {
                        "artifact": str(args.artifact.resolve()),
                        "execution_mode": verified["manifest"]["execution_mode"],
                        "run_id": verified["result"]["run_id"],
                        "status": verified["result"]["status"],
                        "verified": True,
                    },
                    sort_keys=True,
                )
            )
            return 0
        return _run(args)
    except (OSError, RHPCError) as exc:
        print(f"RHPC_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
