"""Agent 3 — Ideation (SPEC §5.3).

Generates a baseline concept for the locus Agent 2 selected, within the
ceilings Agent 2 derived, and makes the concept declare its own position on
both axes so that the Critic has something to check against.

v1 injects theoretical grounding only: the dimension definitions and
diagnostic questions, verbatim from the versioned rubric. Market evidence is
v2, and the `minus_market_rag` ablation measures the difference.

Generation is the one place stochasticity is wanted, so this is the only agent
that does not run at temperature 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic
from pydantic import ValidationError

from luxcal.agents._llm import ResponseError, call_with_retries
from luxcal.core.config import load_rubric
from luxcal.core.locus import LOCUS_DESCRIPTIONS
from luxcal.core.schemas import (
    BrandProfile,
    CalibrationOutput,
    Concept,
    CriticVerdict,
    GridLocus,
)
from luxcal.core.state import LuxcalState
from luxcal.logging.run_logger import RunLogger

# A concept is a short object; this is several times the expected output.
_MAX_TOKENS = 4096


async def run_ideation(
    state: LuxcalState,
    config: dict,
    logger: RunLogger,
) -> dict:
    """Generate a `Concept` for the top-ranked locus. LangGraph node for Agent 3.

    Reads the calibration and profile, and on a revision iteration the current
    verdict and the full critic history. Returns
    `{"concept": Concept, "iteration": int}`; the iteration counter is passed
    through unchanged, since the Critic is what advances it.
    """
    calibration: CalibrationOutput = state["calibration"]
    profile: BrandProfile = state["profile"]
    iteration = state.get("iteration", 0)
    rubric = load_rubric(Path(config["rubric_path"]))

    locus = calibration.viable_loci[0].locus
    verdict = state.get("verdict")
    history = state.get("critic_history") or []

    system_prompt = _system_prompt(profile, calibration, locus, rubric)
    user_prompt = _user_prompt(locus, verdict, history)

    try:
        async with anthropic.AsyncAnthropic() as client:
            concept = await call_with_retries(
                client=client,
                model=config["model_generation"],
                temperature=0.7,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=_MAX_TOKENS,
                logger=logger,
                node="ideate",
                iteration=iteration,
                parse=lambda payload: _validate_concept(payload, locus),
            )
    except (ValidationError, ResponseError, ValueError):
        return {"terminal_state": "ERROR"}

    logger.save_state(
        "ideate",
        {**state, "concept": concept},
        iteration=iteration if iteration > 0 else None,
    )
    return {"concept": concept, "iteration": iteration}


def _validate_concept(payload: Any, locus: GridLocus) -> Concept:
    """Validate a concept and check it was placed at the locus that was asked for.

    A concept at a different locus is schema-valid but defeats the calibration:
    the ceilings were derived for this cell, and the Critic would check it
    against the wrong constraint. Retryable, so the model is told.
    """
    concept = Concept.model_validate(payload)
    if concept.locus != locus:
        raise ResponseError(
            f"the concept must be placed at the selected locus {locus}, "
            f"but was placed at {concept.locus}"
        )
    return concept


def _system_prompt(
    profile: BrandProfile,
    calibration: CalibrationOutput,
    locus: GridLocus,
    rubric: dict,
) -> str:
    """Assemble the generation prompt from the profile, calibration and rubric."""
    dimensions = "\n".join(
        _dimension_section(dimension) for dimension in rubric["dimensions"]
    )

    return f"""You are a luxury innovation strategist. Your task is to design one
concept for a specific, already-chosen place in the business.

Where the concept sits has been decided before you were asked, by a
calibration step you cannot revisit. Your task is what happens there.

# The brand

Category: {profile.category}
Problem: {profile.problem_statement}

{_profile_block(profile, rubric)}

# The calibrated constraint

Visibility ceiling: {calibration.visibility_band}
Intensity ceiling: {calibration.intensity_band}

Visibility is how perceptible the AI is to the client. Intensity is how far
the offering is differentiated per individual. These are different failure
modes and are constrained separately: a conspicuous algorithmic greeting is
low intensity but high visibility, and it is the high visibility that damages
the brand. A one-of-one commission is maximum intensity but low visibility,
and it reinforces rarity rather than eroding it.

Your concept must not exceed either ceiling.

# The selected locus

{locus}: {LOCUS_DESCRIPTIONS[locus]}

The concept must operate here. Do not relocate it, do not propose a second
deployment elsewhere, and do not hedge by describing something that could sit
in several places.

# The calibration standard

These are the dimensions the concept will be judged against. Read them as
constraints on what is appropriate for this house, not as topics to mention.

{dimensions}
# What to produce

One concept that addresses the problem stated above, operates at the selected
locus, and stays within both ceilings.

The concept must declare its own position honestly. `claimed_visibility` and
`claimed_intensity` are your assessment of what you have actually designed,
not a restatement of the ceilings you were given. If the thing you have
described is more visible than the ceiling permits, the correct response is to
design something else — not to declare a lower band than the design warrants.

Write in plain professional English. Do not use marketing register, and do not
name the house or invent a brand voice.

# Response format

Return a single JSON object and nothing else. No preamble, no code fences.

{{
  "name": "<short name for the concept>",
  "locus": "{locus}",
  "touchpoint": "<who interacts, at what moment, through what surface>",
  "mechanism": "<what the AI actually does>",
  "ai_position": "<BACKSTAGE | ADVISOR_MEDIATED | CLIENT_FACING>",
  "differentiation_unit": "<SEGMENT | COHORT | INDIVIDUAL | ONE_OF_ONE>",
  "claimed_visibility": "<LOW | MEDIUM | HIGH>",
  "claimed_intensity": "<LOW | MEDIUM | HIGH>",
  "evidence_ids": ["<the rubric dimension ids that most shaped this concept, e.g. D2>"]
}}"""


def _user_prompt(
    locus: GridLocus,
    verdict: CriticVerdict | None,
    history: list[CriticVerdict],
) -> str:
    """Assemble the user turn, carrying the full critique history on a revision.

    The whole history is carried rather than only the latest verdict so that
    the agent does not oscillate between two previously rejected concepts
    (SPEC §5.3).
    """
    if verdict is None or verdict.verdict != "REVISE":
        return f"Design the concept for {locus} and return the JSON object."

    directive = verdict.revision_directive
    directive_block = (
        f"""Dimension: {directive.dimension}
Axis: {directive.axis}
Required direction of change: {directive.direction}
Instruction: {directive.instruction}"""
        if directive is not None
        else "No directive was issued; address the violations recorded above."
    )

    return f"""Your previous concept was rejected. Revise it.

# The directive you must address

{directive_block}

# Every critique so far

{_history_block(history)}

Address the directive specifically. Do not resubmit a previously rejected
concept with different wording, and do not move the concept to a different
locus to sidestep the problem — it must remain at {locus}.

Return the revised concept as a single JSON object."""


def _history_block(history: list[CriticVerdict]) -> str:
    """Render every verdict so far, oldest first."""
    if not history:
        return "No previous critiques were recorded."

    blocks = []
    for verdict in history:
        lines = [
            f"## Iteration {verdict.iteration} — {verdict.verdict}",
            f"Within the visibility ceiling: {verdict.visibility_within_ceiling}",
            f"Within the intensity ceiling: {verdict.intensity_within_ceiling}",
            f"Misdeclared: {verdict.misdeclared}",
        ]
        if verdict.misdeclaration_rationale:
            lines.append(f"  {verdict.misdeclaration_rationale}")
        if verdict.violations:
            lines.append("Violations:")
            lines.extend(
                f"  {violation.dimension} ({violation.severity}): {violation.detail}"
                for violation in verdict.violations
            )
        if verdict.revision_directive is not None:
            directive = verdict.revision_directive
            lines.append(
                f"Directive: {directive.dimension}, {directive.axis}, "
                f"{directive.direction} — {directive.instruction}"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _dimension_section(dimension: dict) -> str:
    """Render one dimension's rubric entry for injection into the prompt."""
    questions = "\n".join(
        f"- {question}" for question in dimension["diagnostic_questions"]
    )
    return f"""## {dimension["id"].upper()} — {dimension["name"]}

Definition:
{dimension["definition"]}

Diagnostic questions:
{questions}
"""


def _profile_block(profile: BrandProfile, rubric: dict) -> str:
    """Render the five dimension positions, in rubric order."""
    lines = []
    for dimension in rubric["dimensions"]:
        position = getattr(profile, dimension["profile_field"])
        lines.append(
            f"{dimension['id'].upper()} {dimension['name']}: {position.position} "
            f"— {position.rationale}"
        )
    return "\n".join(lines)
