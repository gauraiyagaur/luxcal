"""Agent 2 — Calibration and Gate (SPEC §5.2).

Four steps of deliberately mixed kind, in strict order:

    (i)   gate      — an LLM judgement over the problem statement
    (ii)  ceilings  — deterministic ordinal arithmetic, no model involved
    (iii) filter    — a set operation against the fixed locus grid
    (iv)  ranking   — an LLM call over the surviving loci only

The separation is the auditability claim: a brand manager can be shown exactly
why a locus was excluded, and that exclusion does not depend on model
temperament. Steps (ii) and (iii) are `core.ceilings` and `core.locus`
respectively; nothing here reimplements them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic
from pydantic import Field, TypeAdapter

from luxcal.agents._llm import RETRYABLE, ResponseError, call_with_retries
from luxcal.core.ceilings import compute_ceilings
from luxcal.core.config import load_rubric, ordinal_maps
from luxcal.core.locus import LOCUS_DESCRIPTIONS, LOCUS_GRID, filter_loci
from luxcal.core.schemas import (
    Band,
    BrandProfile,
    CalibrationOutput,
    GateDecision,
    GridLocus,
    LuxcalModel,
    RankedLocus,
)
from luxcal.core.state import LuxcalState
from luxcal.logging.run_logger import RunLogger

# Both responses are small — a paragraph, or eleven short ranking entries.
_MAX_TOKENS = 4096

_RANKED_LOCI = TypeAdapter(list[RankedLocus])


class _GateResponse(LuxcalModel):
    """The gate call's expected response shape."""

    gate_decision: GateDecision
    gate_rationale: str = Field(min_length=1)


async def run_calibration(
    state: LuxcalState,
    config: dict,
    logger: RunLogger,
) -> dict:
    """Gate the problem, derive both ceilings, filter the grid and rank survivors.

    Reads `state["profile"]`. Returns `{"calibration": CalibrationOutput}` on a
    proceed, or that plus `{"terminal_state": "GATE_NO"}` on a refusal. Both
    bands are computed either way, so a refused case remains analysable. If
    either LLM call exhausts its retries the run terminates with
    `{"terminal_state": "ERROR"}`.
    """
    profile: BrandProfile = state["profile"]
    rubric = load_rubric(Path(config["rubric_path"]))
    model = config["model_judge"]

    try:
        async with anthropic.AsyncAnthropic() as client:
            # (i) Gate — semantic judgement.
            gate = await call_with_retries(
                client=client,
                model=model,
                temperature=0.0,
                system_prompt=_gate_system_prompt(profile, rubric),
                user_prompt="Assess this brand profile and return your gate decision.",
                max_tokens=_MAX_TOKENS,
                logger=logger,
                node="calibrate",
                parse=_GateResponse.model_validate,
            )

            # (ii) Ceilings — deterministic, computed on both paths.
            visibility_band, intensity_band = compute_ceilings(
                profile, ordinal_maps(rubric)
            )

            if gate.gate_decision == "REFUSE":
                return _refusal(
                    state, logger, gate, visibility_band, intensity_band
                )

            # (iii) Locus filter — deterministic set operation.
            surviving, excluded = filter_loci(visibility_band, intensity_band)

            # (iv) Ranking — semantic judgement over the survivors only.
            viable_loci = await call_with_retries(
                client=client,
                model=model,
                temperature=0.0,
                system_prompt=_ranking_system_prompt(
                    profile, surviving, visibility_band, intensity_band
                ),
                user_prompt="Rank these loci and return the JSON array.",
                max_tokens=_MAX_TOKENS,
                logger=logger,
                node="calibrate",
                parse=lambda payload: _validate_ranking(payload, surviving),
            )
    except RETRYABLE:
        return {"terminal_state": "ERROR"}

    calibration = CalibrationOutput(
        gate_decision="PROCEED",
        gate_rationale=gate.gate_rationale,
        visibility_band=visibility_band,
        intensity_band=intensity_band,
        viable_loci=viable_loci,
        excluded_loci=excluded,
    )
    logger.save_state("calibrate", {**state, "calibration": calibration})
    return {"calibration": calibration}


def _refusal(
    state: LuxcalState,
    logger: RunLogger,
    gate: _GateResponse,
    visibility_band: Band,
    intensity_band: Band,
) -> dict:
    """Assemble the terminal output for a gated-out case.

    No ranking is performed, so `viable_loci` is empty; no filtering is
    performed either, so `excluded_loci` is empty rather than listing the whole
    grid. The bands are still recorded: a refusal with its calibration attached
    is a reportable result, not an absence of one.
    """
    calibration = CalibrationOutput(
        gate_decision="REFUSE",
        gate_rationale=gate.gate_rationale,
        visibility_band=visibility_band,
        intensity_band=intensity_band,
        viable_loci=[],
        excluded_loci=[],
    )
    logger.save_state("calibrate", {**state, "calibration": calibration})
    return {"calibration": calibration, "terminal_state": "GATE_NO"}


def _validate_ranking(payload: Any, surviving: list[GridLocus]) -> list[RankedLocus]:
    """Validate a ranking and check it actually ranks the loci that survived.

    A ranking that omits a survivor, invents an excluded locus, or repeats a
    rank is schema-valid but wrong for the request: it would put a locus the
    ceilings rejected into the recommendation, or leave the ordering ambiguous.
    Treated as retryable so the model is told what it got wrong.
    """
    ranking = _RANKED_LOCI.validate_python(payload)

    ranked = [entry.locus for entry in ranking]
    if sorted(ranked) != sorted(surviving):
        raise ResponseError(
            f"the ranking must cover exactly the surviving loci {sorted(surviving)}, "
            f"but it ranked {sorted(ranked)}"
        )

    ranks = sorted(entry.rank for entry in ranking)
    if ranks != list(range(1, len(ranking) + 1)):
        raise ResponseError(
            f"ranks must be the consecutive integers 1 to {len(ranking)} with no "
            f"repeats, but were {ranks}"
        )

    return sorted(ranking, key=lambda entry: entry.rank)


def _gate_system_prompt(profile: BrandProfile, rubric: dict) -> str:
    """Build the gate prompt: is this a personalisation problem at all?"""
    return f"""You are a luxury market research strategist. Your task is to decide
whether the business problem described below is one that personalisation could
appropriately address.

This is a screening judgement made before any concept is generated. Answering
it honestly matters more than finding a use case.

Not every luxury business problem requires personalisation. Authentication,
supply chain integrity, sizing accuracy, inventory optimisation and similar
operational problems should be refused. These are real problems, and AI may
well help with them, but they are not personalisation problems and the system
must be able to say so. Refuse also where personalisation would be a solution
in search of a problem, or where the brief's stated difficulty would be
untouched by differentiating the offering per client.

Proceed where the problem concerns how the brand relates to individual
clients: relationship depth, relevance of what is offered to whom, the quality
or consistency of client experience, or client retention and attrition.

# The brand profile

{_profile_block(profile, rubric)}

# Response format

Return a single JSON object and nothing else. No preamble, no code fences.

{{
  "gate_decision": "PROCEED or REFUSE",
  "gate_rationale": "<one paragraph explaining the decision, in plain professional English>"
}}"""


def _ranking_system_prompt(
    profile: BrandProfile,
    surviving: list[GridLocus],
    visibility_band: Band,
    intensity_band: Band,
) -> str:
    """Build the ranking prompt over the surviving loci only."""
    entries = "\n".join(_locus_line(locus) for locus in surviving)

    return f"""You are a luxury market research strategist. Your task is to rank
candidate deployment locations by how well each fits a specific business
problem.

# The problem

Category: {profile.category}
Problem: {profile.problem_statement}

# The constraint already applied

This brand's calibrated ceilings are visibility {visibility_band} and intensity
{intensity_band}. Visibility is how perceptible the AI is to the client;
intensity is how far the offering is differentiated per individual. The loci
below are the ones whose native demands fall within both ceilings — the rest of
the grid has already been excluded and is not available to you.

Do not re-litigate that exclusion, and do not propose a locus that is not on
this list. Your judgement is about fit to the problem, not about whether the
constraint is right.

# The surviving loci

{entries}

# What to return

Rank every locus listed above, and only those. Rank 1 is the best fit for this
particular business problem; ranks must be consecutive integers with no
repeats. Give each a one-line rationale tying that locus to the problem as
stated — not a general description of the locus, which you already have.

Write in plain professional English. Do not use marketing register.

Return a single JSON array and nothing else. No preamble, no code fences.

[
  {{"locus": "<one of the loci above>", "rank": 1, "rationale": "<one sentence>"}}
]"""


def _locus_line(locus: GridLocus) -> str:
    """Render one locus with its description and native demands."""
    native_visibility, native_intensity = LOCUS_GRID[locus]
    return (
        f"- {locus}: {LOCUS_DESCRIPTIONS[locus]}; "
        f"visibility {native_visibility}, intensity {native_intensity}"
    )


def _profile_block(profile: BrandProfile, rubric: dict) -> str:
    """Render the full profile for the gate prompt, in rubric dimension order."""
    lines = [
        f"Category: {profile.category}",
        f"Problem statement: {profile.problem_statement}",
        "",
    ]
    for dimension in rubric["dimensions"]:
        position = getattr(profile, dimension["profile_field"])
        evidence = (
            f'"{position.evidence_span}"'
            if position.evidence_span is not None
            else "none — not stated in the brief, position taken as a category fallback"
        )
        lines.append(f"{dimension['id'].upper()} {dimension['name']}: {position.position}")
        lines.append(f"  Rationale: {position.rationale}")
        lines.append(f"  Stated in brief: {position.stated_in_brief}")
        lines.append(f"  Evidence: {evidence}")
        lines.append("")
    return "\n".join(lines).rstrip()
