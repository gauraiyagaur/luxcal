"""Run one brand brief through the full pipeline.

    python -m scripts.run_single --case data/cases/case_001.json --variant full --replicate 1

The case file is JSON with `case_id` and `brief`. Paths are relative to the
repository root, which is also where this must be run from: the config's
`rubric_path` and the run logger's git calls both resolve against the process
working directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from luxcal.core.config import load_config
from luxcal.core.graph import build_graph
from luxcal.core.state import LuxcalState
from luxcal.logging.run_logger import RunLogger

DEFAULT_CONFIG = Path("luxcal/configs/full.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Run one brand brief through the LUXCAL pipeline."
    )
    parser.add_argument(
        "--case", required=True, type=Path, help="Path to the case JSON file."
    )
    parser.add_argument(
        "--variant", default="full", help="Ablation condition; names the run."
    )
    parser.add_argument(
        "--replicate", type=int, default=1, help="Which repeat of this configuration."
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Variant config file."
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("runs"), help="Where run directories go."
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Run against an uncommitted working tree. The recorded commit SHA is "
            "suffixed '-dirty' and the run is not reproducible; do not use for runs "
            "that produce results."
        ),
    )
    return parser.parse_args(argv)


def initial_state(run_id: str, case_id: str, brief: str) -> LuxcalState:
    """Build the state the graph starts from, with every field present."""
    return {
        "run_id": run_id,
        "case_id": case_id,
        "brief": brief,
        "profile": None,
        "calibration": None,
        "concept": None,
        "verdict": None,
        "critic_history": [],
        "iteration": 0,
        "terminal_state": None,
        "messages": [],
    }


def resolve_terminal_state(state: dict[str, Any], variant: str) -> str | None:
    """Fill in the terminal state where nothing along the path set one.

    Agent 2 writes GATE_NO on a refusal and a failed agent writes ERROR, but
    nothing writes PASS or ESCALATE: the Critic produces the verdict those are
    read from and does not set the field itself. Derived here so that
    `manifest.terminal_state` is populated for every run rather than being
    null on exactly the successful ones. This properly belongs in the Critic.

    The absent-verdict clause is scoped to `minus_critic` deliberately. That
    variant has no Critic, so no verdict is the designed outcome. In every
    other variant an absent verdict means the Critic did not return — retry
    exhaustion, or an error — which is a failure and must not be relabelled
    PASS.
    """
    if state.get("terminal_state") is not None:
        return state["terminal_state"]

    verdict = state.get("verdict")

    if (
        verdict is None
        and variant == "minus_critic"
        and state.get("concept") is not None
    ):
        return "PASS"

    if verdict is None:
        return None
    if verdict.verdict in {"PASS", "ESCALATE"}:
        return verdict.verdict
    # A REVISE that reached the end of the graph means the loop router cut the
    # loop on the iteration cap without the Critic having upgraded the verdict.
    return "ESCALATE"


async def run(args: argparse.Namespace) -> Path:
    """Execute one run and return its directory."""
    config = load_config(args.config)
    case = json.loads(args.case.read_text(encoding="utf-8-sig"))

    logger = RunLogger(
        case_id=case["case_id"],
        variant=args.variant,
        replicate_id=args.replicate,
        config_path=args.config,
        rubric_path=Path(config["rubric_path"]),
        runs_dir=args.runs_dir,
        allow_dirty=args.allow_dirty,
    )
    logger.save_brief(case["brief"])

    state = initial_state(logger.run_dir.name, case["case_id"], case["brief"])
    app = build_graph(config, logger)

    try:
        final_state = await app.ainvoke(state)
    except Exception:
        # A run that dies mid-graph must still leave a manifest behind, or the
        # run directory is an unreadable fragment (SPEC §7).
        traceback.print_exc()
        final_state = {**state, "terminal_state": "ERROR"}

    final_state = {
        **final_state,
        "terminal_state": resolve_terminal_state(
            dict(final_state), config["variant"]
        ),
    }
    logger.finalise(final_state)
    return logger.run_dir


def print_summary(run_dir: Path) -> None:
    """Print the run's outcome, read back from the manifest it wrote."""
    manifest: dict[str, Any] = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )

    def show(label: str, key: str) -> None:
        print(f"  {label:<20} {manifest.get(key)}")

    print("\n" + "-" * 60)
    show("terminal state", "terminal_state")
    show("gate", "gate_decision")
    show("visibility ceiling", "visibility_band")
    show("intensity ceiling", "intensity_band")
    show("selected locus", "selected_locus")
    show("critic iterations", "critic_iterations")
    show("input tokens", "total_input_tokens")
    show("output tokens", "total_output_tokens")
    print(f"  {'est. cost (USD)':<20} {manifest.get('est_cost_usd')}")
    show("wall clock (s)", "wall_clock_seconds")

    unpriced = manifest.get("unpriced_models")
    if unpriced:
        print(f"  {'UNPRICED MODELS':<20} {unpriced} (cost above excludes these)")

    print("-" * 60)
    print(f"\nRun directory: {run_dir}\n")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a non-zero exit code if the run did not pass."""
    load_dotenv()
    args = parse_args(argv)

    run_dir = asyncio.run(run(args))
    print_summary(run_dir)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return 0 if manifest.get("terminal_state") in {"PASS", "GATE_NO"} else 1


if __name__ == "__main__":
    sys.exit(main())
