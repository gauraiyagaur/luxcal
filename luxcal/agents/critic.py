"""Critic — conformance to the calibration Agent 2 set (SPEC §5.4).

Two checks of different kind, run in sequence and recorded separately.

Check 1 is deterministic: are the concept's claimed bands within the ceilings?
No language model is involved, and it cannot be wrong.

Check 2 is a language-model judgement addressing what check 1 structurally
cannot catch — whether the concept as described actually matches the bands it
claims. A generator will readily describe a client-facing assistant and label
it BACKSTAGE, and detecting that misdeclaration is the one task here for which
model judgement is irreplaceable. Its failures are recorded in `misdeclared`,
apart from the check 1 booleans, because they are the finding of greater
research interest.

The Critic evaluates and directs. It never proposes a concept of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Self

import anthropic
from pydantic import Field, model_validator

from luxcal.agents._llm import RETRYABLE, call_with_retries
from luxcal.core.ceilings import BAND_ORDER
from luxcal.core.config import load_rubric
from luxcal.core.locus import LOCUS_DESCRIPTIONS
from luxcal.core.schemas import (
    BrandProfile,
    CalibrationOutput,
    Concept,
    CriticVerdict,
    DimensionViolation,
    LuxcalModel,
    RevisionDirective,
    Verdict,
)
from luxcal.core.state import LuxcalState
from luxcal.logging.run_logger import RunLogger

_MAX_TOKENS = 4096

# Which dimension a deterministic ceiling breach is attributed to. Visibility
# is governed by D2 and D4 and intensity by D1 and D3 (SPEC §5.2); the
# definitional dimension of each axis is named, since the arithmetic does not
# say which of the pair bound hardest.
_AXIS_DIMENSION = {"VISIBILITY": "D2", "INTENSITY": "D1"}


class _Assessment(LuxcalModel):
    """The check 2 response shape.

    Validated on the way in so that an internally inconsistent judgement —
    a misdeclaration reported alongside a PASS, a REVISE with no directive —
    is retried rather than being carried into the verdict.
    """

    misdeclared: bool
    misdeclaration_rationale: Optional[str] = None
    violations: list[DimensionViolation] = Field(default_factory=list)
    verdict: Verdict
    revision_directive: Optional[RevisionDirective] = None

    @model_validator(mode="after")
    def _internally_consistent(self) -> Self:
        if self.misdeclared and self.misdeclaration_rationale is None:
            raise ValueError(
                "misdeclaration_rationale is required when misdeclared is true"
            )
        if self.verdict == "REVISE" and self.revision_directive is None:
            raise ValueError("revision_directive is required when the verdict is REVISE")
        if self.verdict != "REVISE" and self.revision_directive is not None:
            raise ValueError(
                f"revision_directive must be null when the verdict is {self.verdict}"
            )
        if self.verdict == "PASS" and (self.misdeclared or _has_critical(self.violations)):
            raise ValueError(
                "the verdict cannot be PASS when the concept is misdeclared or has a "
                "CRITICAL violation"
            )
        return self


async def run_critic(
    state: LuxcalState,
    config: dict,
    logger: RunLogger,
) -> dict:
    """Judge the concept against the calibration. LangGraph node for the Critic.

    Returns `{"verdict", "critic_history", "iteration"}`. The history is read
    and rewritten in full rather than appended to, so the trajectory of
    rejections survives the graph's default overwrite semantics.
    """
    concept: Concept = state["concept"]
    calibration: CalibrationOutput = state["calibration"]
    profile: BrandProfile = state["profile"]
    iteration = state["iteration"]
    cap = config.get("critic_max_iterations", 3)

    # Check 1 — deterministic. No model, no call, cannot be wrong.
    visibility_within = _within(
        concept.claimed_visibility, calibration.visibility_band
    )
    intensity_within = _within(concept.claimed_intensity, calibration.intensity_band)

    if visibility_within and intensity_within:
        # Check 2 — the judgement check 1 structurally cannot make.
        rubric = load_rubric(Path(config["rubric_path"]))
        try:
            async with anthropic.AsyncAnthropic() as client:
                assessment = await call_with_retries(
                    client=client,
                    model=config["model_judge"],
                    temperature=0.0,
                    system_prompt=_system_prompt(
                        concept, profile, calibration, rubric
                    ),
                    user_prompt="Assess this concept and return your judgement.",
                    max_tokens=_MAX_TOKENS,
                    logger=logger,
                    node="critic",
                    iteration=iteration,
                    parse=_Assessment.model_validate,
                )
        except RETRYABLE:
            return {"terminal_state": "ERROR"}

        verdict_type = assessment.verdict
        directive = assessment.revision_directive
        misdeclared = assessment.misdeclared
        rationale = assessment.misdeclaration_rationale
        violations = assessment.violations
    else:
        verdict_type = "REVISE"
        directive = _ceiling_directive(
            concept, calibration, visibility_within, intensity_within
        )
        misdeclared = False
        rationale = None
        violations = []

    new_iteration = iteration + 1

    # The cap makes the graph terminate. A directive cannot ride an ESCALATE:
    # `CriticVerdict` requires it to be present exactly when the verdict is
    # REVISE.
    if verdict_type == "REVISE" and new_iteration >= cap:
        verdict_type = "ESCALATE"
        directive = None

    verdict = CriticVerdict(
        verdict=verdict_type,
        iteration=iteration,
        visibility_within_ceiling=visibility_within,
        intensity_within_ceiling=intensity_within,
        misdeclared=misdeclared,
        misdeclaration_rationale=rationale,
        violations=violations,
        revision_directive=directive,
    )
    history = list(state.get("critic_history") or []) + [verdict]

    logger.save_state(
        "critic",
        {**state, "verdict": verdict, "critic_history": history},
        iteration=iteration,
    )
    return {
        "verdict": verdict,
        "critic_history": history,
        "iteration": new_iteration,
    }


def _within(claimed: str, ceiling: str) -> bool:
    """Whether a claimed band falls within a calibrated ceiling."""
    return BAND_ORDER[claimed] <= BAND_ORDER[ceiling]


def _ceiling_directive(
    concept: Concept,
    calibration: CalibrationOutput,
    visibility_within: bool,
    intensity_within: bool,
) -> RevisionDirective:
    """Build the directive for a check 1 breach.

    Where both axes breach, the directive names visibility: of the two failure
    modes it is the one that damages the brand (SPEC §2.1), so it is the one to
    fix first. The instruction states both so the generator sees the whole
    problem.
    """
    axis = "VISIBILITY" if not visibility_within else "INTENSITY"

    breaches = []
    if not visibility_within:
        breaches.append(
            f"claimed visibility {concept.claimed_visibility} exceeds the "
            f"{calibration.visibility_band} ceiling"
        )
    if not intensity_within:
        breaches.append(
            f"claimed intensity {concept.claimed_intensity} exceeds the "
            f"{calibration.intensity_band} ceiling"
        )

    return RevisionDirective(
        dimension=_AXIS_DIMENSION[axis],
        axis=axis,
        direction="REDUCE",
        instruction=(
            f"The concept breaches its calibrated ceilings: {'; and '.join(breaches)}. "
            f"Redesign so that what is actually built sits within both ceilings, "
            f"rather than restating the same concept with lower claimed bands."
        ),
    )


def _has_critical(violations: list[DimensionViolation]) -> bool:
    """Whether any violation is CRITICAL."""
    return any(violation.severity == "CRITICAL" for violation in violations)


def _system_prompt(
    concept: Concept,
    profile: BrandProfile,
    calibration: CalibrationOutput,
    rubric: dict,
) -> str:
    """Assemble the check 2 prompt."""
    dimensions = "\n".join(
        _dimension_section(dimension) for dimension in rubric["dimensions"]
    )

    return f"""You are a luxury brand calibration critic. Your task is to judge
whether a proposed concept conforms to a calibration that was set before the
concept was generated.

You do not propose concepts, and you do not suggest what the concept should
have been. You assess what is in front of you and, where it fails, say which
dimension is at fault and in which direction it must change.

# The concept

Name: {concept.name}
Locus: {concept.locus} — {LOCUS_DESCRIPTIONS[concept.locus]}
Touchpoint: {concept.touchpoint}
Mechanism: {concept.mechanism}
Declared AI position: {concept.ai_position}
Declared differentiation unit: {concept.differentiation_unit}
Claimed visibility: {concept.claimed_visibility}
Claimed intensity: {concept.claimed_intensity}

# The brand

Category: {profile.category}
Problem: {profile.problem_statement}

{_profile_block(profile, rubric)}

# The calibration

Visibility ceiling: {calibration.visibility_band}
Intensity ceiling: {calibration.intensity_band}

An arithmetic check has already confirmed that the claimed bands above fall
within these ceilings. Do not repeat that check.

# Your two tasks

## 1. Does the concept match the bands it claims?

Read the touchpoint and mechanism, and decide what the concept described would
actually be in practice. Then compare that with what it declares.

Visibility is how perceptible the AI is to the client. Intensity is how far
the offering is differentiated per individual.

A concept describing a client-facing chatbot that declares BACKSTAGE is
misdeclared, whatever its stated rationale. So is one that describes a
one-of-one commission while claiming LOW intensity. The declaration is a claim
about the design, and a generator has an incentive to declare whatever keeps
it within bounds.

If the description and the declaration disagree, set `misdeclared` to true and
say in `misdeclaration_rationale` what the concept actually is and which
declared field is wrong. If they agree, set it to false and leave the
rationale null.

## 2. Does the concept violate the calibration standard?

Assess the concept against each dimension's diagnostic questions below. Record
a violation only where the concept genuinely conflicts with a dimension, not
wherever it merely fails to mention one.

Severity:
- MINOR: a point of friction that does not undermine the dimension.
- MAJOR: the concept works against the dimension in a way that needs redesign.
- CRITICAL: the concept is incompatible with the brand's position on that
  dimension and cannot be repaired by adjustment.

{dimensions}
# The verdict

- PASS: no misdeclaration and no critical violation.
- REVISE: a misdeclaration, a critical violation, or enough major violations
  that the concept needs rework. Give a `revision_directive` naming one
  dimension, the axis it concerns, the direction of change required, and a
  specific instruction.
- ESCALATE: the concept cannot be repaired within this calibration.

A PASS is not available when `misdeclared` is true or when any violation is
CRITICAL. A `revision_directive` must be present when the verdict is REVISE
and absent otherwise.

Write in plain professional English. Do not use marketing register.

# Response format

Return a single JSON object and nothing else. No preamble, no code fences.

{{
  "misdeclared": false,
  "misdeclaration_rationale": null,
  "violations": [
    {{"dimension": "<D1 | D2 | D3 | D4 | D5>",
      "severity": "<MINOR | MAJOR | CRITICAL>",
      "detail": "<what conflicts, in one or two sentences>"}}
  ],
  "verdict": "<PASS | REVISE | ESCALATE>",
  "revision_directive": {{
    "dimension": "<D1 | D2 | D3 | D4 | D5>",
    "axis": "<VISIBILITY | INTENSITY>",
    "direction": "<INCREASE | REDUCE | MAINTAIN>",
    "instruction": "<what must change, in one or two sentences>"
  }}
}}"""


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
