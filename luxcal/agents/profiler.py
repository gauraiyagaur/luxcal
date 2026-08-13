"""Agent 1 — Brand Profiler (SPEC §5.1).

Converts an unstructured brief into a typed five-dimension profile in a single
schema-constrained extraction call. The dimension definitions and diagnostic
questions are injected verbatim from the versioned rubric rather than restated
here, so that revising the rubric revises the prompt.

Abstention is the point of the provenance fields: where the brief does not
evidence a dimension, the agent must say so rather than inferring silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import anthropic

from luxcal.agents._llm import RETRYABLE, call_with_retries
from luxcal.core.config import load_rubric
from luxcal.core.schemas import BrandProfile, Category
from luxcal.core.state import LuxcalState
from luxcal.logging.run_logger import RunLogger

# The profile is a bounded object — five position records plus two short
# strings — so this is roughly four times the expected output. It exists to
# stop a truncated response being mistaken for a malformed one.
_MAX_TOKENS = 8192


async def run_profiler(
    state: LuxcalState,
    config: dict,
    logger: RunLogger,
) -> dict:
    """Extract a `BrandProfile` from the brief. LangGraph node for Agent 1.

    Reads `state["brief"]`, makes up to three extraction calls, and returns
    `{"profile": BrandProfile}` on success. Every attempt is logged in full,
    including the ones that fail validation. If all three attempts fail the
    run is terminated with `{"terminal_state": "ERROR"}` rather than returning
    a partial or fabricated profile.
    """
    rubric = load_rubric(Path(config["rubric_path"]))

    try:
        async with anthropic.AsyncAnthropic() as client:
            profile = await call_with_retries(
                client=client,
                model=config["model_generation"],
                temperature=0.0,
                system_prompt=_build_system_prompt(rubric),
                user_prompt=_build_user_prompt(state["brief"]),
                max_tokens=_MAX_TOKENS,
                logger=logger,
                node="profiler",
                parse=BrandProfile.model_validate,
            )
    except RETRYABLE:
        return {"terminal_state": "ERROR"}

    logger.save_state("profiler", {**state, "profile": profile})
    return {"profile": profile}


def _build_system_prompt(rubric: dict) -> str:
    """Assemble the extraction prompt from the rubric.

    Definitions and diagnostic questions are inserted exactly as the rubric
    states them. The `retrieval_confidence` and `grounding` blocks are
    deliberately not injected: they describe the strength of the literature
    behind each dimension, which is not information the extraction should be
    conditioned on.
    """
    sections = [_dimension_section(dimension) for dimension in rubric["dimensions"]]
    categories = " | ".join(get_args(Category))

    return f"""You are a luxury brand analyst performing structured extraction.

You will be given a brand brief: free text describing a luxury brand, its
category, and a business problem for which AI integration is being considered.
Your task is to read the brief and return a five-dimension brand profile as a
single JSON object.

This is an analytical instrument, not a piece of client-facing writing. Use
plain professional English throughout. Do not use marketing register.

# The five dimensions

Each dimension has a closed vocabulary of positions. You must select exactly
one position per dimension. The definitions and diagnostic questions below are
the calibration standard; assess the brief against them rather than against
your own intuitions about the category.

{chr(10).join(sections)}
# Category

Assign exactly one category from this closed vocabulary:

{categories}

# Provenance and abstention

For each dimension, decide whether the brief actually evidences the position
you have chosen.

- If it does, set `stated_in_brief` to true and set `evidence_span` to a
  verbatim quotation from the brief. The quotation must appear in the brief
  character for character. Do not paraphrase, tidy or join separated phrases.
- If it does not, set `stated_in_brief` to false and set `evidence_span` to
  null. Choose the position most typical of the brief's category, and say in
  the rationale that this is a category fallback rather than something the
  brief states. Set `confidence` low to reflect the absence of evidence.

Inferring a position from silence and presenting it as evidenced is the single
most damaging error you can make here. A dimension the brief does not address
is a normal and expected outcome, not a failure.

`confidence` is your confidence in the position, from 0.0 to 1.0.

`rationale` is one line explaining why that position was chosen. It must be a
single sentence.

`problem_statement` restates the brief's business problem in one sentence.

# Response format

Return a single JSON object and nothing else. No preamble, no explanation, no
code fences. The object must have exactly these keys:

{_response_shape(rubric)}"""


def _dimension_section(dimension: dict) -> str:
    """Render one dimension's rubric entry for injection into the prompt."""
    questions = "\n".join(
        f"- {question}" for question in dimension["diagnostic_questions"]
    )
    positions = " | ".join(dimension["positions"])

    return f"""## {dimension["id"].upper()} — {dimension["name"]}

JSON field: `{dimension["profile_field"]}`
Positions: {positions}

Definition:
{dimension["definition"]}

Diagnostic questions:
{questions}
"""


def _response_shape(rubric: dict) -> str:
    """Render the expected JSON object, keyed and constrained from the rubric.

    Written out explicitly rather than generated from `BrandProfile`'s JSON
    schema: the generic `DimensionPosition` produces `$defs` names such as
    `DimensionPosition_Literal__STRICT____MODERATE____RELAXED___`, which would
    go into the prompt verbatim and read as noise to the model.
    """
    shape: dict[str, Any] = {
        "category": "<one of the category vocabulary above>",
        "problem_statement": "<the business problem, in one sentence>",
    }
    for dimension in rubric["dimensions"]:
        shape[dimension["profile_field"]] = {
            "position": "<one of: " + " | ".join(dimension["positions"]) + ">",
            "evidence_span": "<verbatim quotation from the brief, or null>",
            "stated_in_brief": "<true or false>",
            "confidence": "<number from 0.0 to 1.0>",
            "rationale": "<one sentence>",
        }
    return json.dumps(shape, indent=2)


def _build_user_prompt(brief: str) -> str:
    """Assemble the user turn.

    Appending a prior validation error on a retry is `call_with_retries`'s job,
    not this function's.
    """
    return f"Here is the brand brief.\n\n<brief>\n{brief}\n</brief>"
