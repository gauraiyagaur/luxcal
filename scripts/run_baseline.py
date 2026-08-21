"""Run one brief through a bare model — the `baseline` ablation.

    python -m scripts.run_baseline --case data/cases/case_001.json --replicate 1

None of the architecture: no profile, no gate, no ceilings, no locus filter, no
Critic. One prompt, one call, one `Concept`. It exists to answer what the rest
of the system buys over asking a capable model directly, so it deliberately
does not import `build_graph`, `LuxcalState` or any agent.

It does use `RunLogger`, so a baseline run is logged, costed and snapshotted
exactly like any other and its row joins the others on `case_id`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

from luxcal.agents._llm import RETRYABLE, call_with_retries
from luxcal.core.config import load_config
from luxcal.core.schemas import Concept
from luxcal.logging.run_logger import RunLogger

DEFAULT_CONFIG = Path("luxcal/configs/baseline.yaml")

# Matches Agent 3's budget so the two are comparable on output length.
_MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are a luxury innovation strategist. You will be given a
brand brief describing a business problem for which AI integration is being
considered. Propose one concept for how AI could be used.

Describe where the concept sits and how far it personalises, using the fields
below. Write in plain professional English.

Return a single JSON object and nothing else. No preamble, no code fences.

{
  "name": "<short name for the concept>",
  "locus": "<PRE_BACKSTAGE | PRE_ADVISOR | PRE_CLIENT | AT_BACKSTAGE | AT_ADVISOR | AT_CLIENT | POST_BACKSTAGE | POST_ADVISOR | POST_CLIENT | NON_BACKSTAGE | NON_CLIENT | UNMAPPED>",
  "touchpoint": "<who interacts, at what moment, through what surface>",
  "mechanism": "<what the AI actually does>",
  "ai_position": "<BACKSTAGE | ADVISOR_MEDIATED | CLIENT_FACING>",
  "differentiation_unit": "<SEGMENT | COHORT | INDIVIDUAL | ONE_OF_ONE>",
  "claimed_visibility": "<LOW | MEDIUM | HIGH>",
  "claimed_intensity": "<LOW | MEDIUM | HIGH>",
  "evidence_ids": []
}

The locus values name where in the business the AI acts: the prefix is the
phase relative to the client encounter (PRE, AT, POST, NON) and the suffix is
who it is perceptible to (BACKSTAGE, ADVISOR, CLIENT). Use UNMAPPED if none
fits."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Run one brief through a bare model (baseline ablation)."
    )
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def baseline_state(run_id: str, case_id: str, brief: str) -> dict[str, Any]:
    """The state shape `RunLogger.finalise` reads, with the unused fields empty.

    Every calibration and critic field stays None or empty: this variant has no
    gate, no ceilings, no locus filter and no verdict, so those manifest fields
    are genuinely not applicable rather than merely unset.
    """
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


async def run(args: argparse.Namespace) -> Path:
    """Execute one baseline run and return its directory."""
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
    state = baseline_state(logger.run_dir.name, case["case_id"], case["brief"])

    try:
        async with anthropic.AsyncAnthropic() as client:
            concept = await call_with_retries(
                client=client,
                model=config["model_generation"],
                temperature=0.7,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=(
                    f"Here is the brand brief.\n\n<brief>\n{case['brief']}\n</brief>"
                ),
                max_tokens=_MAX_TOKENS,
                logger=logger,
                node="baseline",
                parse=Concept.model_validate,
            )
        state = {**state, "concept": concept, "terminal_state": "PASS"}
    except RETRYABLE:
        state = {**state, "terminal_state": "ERROR"}
    except Exception:
        traceback.print_exc()
        state = {**state, "terminal_state": "ERROR"}

    logger.finalise(state)
    return logger.run_dir


def main(argv: list[str] | None = None) -> int:
    """Entry point. Non-zero exit if no concept was produced."""
    load_dotenv()
    args = parse_args(argv)
    run_dir = asyncio.run(run(args))

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    print("\n" + "-" * 60)
    for key in (
        "terminal_state",
        "selected_locus",
        "total_input_tokens",
        "total_output_tokens",
        "est_cost_usd",
        "wall_clock_seconds",
    ):
        print(f"  {key:<20} {manifest.get(key)}")
    print("-" * 60)
    print(f"\nRun directory: {run_dir}\n")

    return 0 if manifest.get("terminal_state") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
