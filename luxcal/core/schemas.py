"""Pydantic contracts exchanged between LUXCAL agents.

Every categorical field is a closed `Literal`; free text is confined to
`*_rationale`, `*_statement`, `detail` and `instruction` fields, none of which
are analysed statistically. Models are frozen because no agent may mutate
another agent's output (SPEC §4), and `extra="forbid"` so that a model
returning an unexpected key fails validation rather than having it silently
dropped.

Position vocabularies are transcribed from `rubric/rubric_v1.yaml`; any change
there requires a matching change here and a rubric version increment.
"""

from __future__ import annotations

from typing import Generic, Literal, Optional, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Shared vocabularies (SPEC §3)
# ---------------------------------------------------------------------------

Band = Literal["LOW", "MEDIUM", "HIGH"]

Category = Literal[
    "FASHION_LEATHER",
    "WATCHES_JEWELLERY",
    "HOSPITALITY",
    "AUTOMOTIVE",
    "SPIRITS",
    "BEAUTY",
    "OTHER",
]

# The locus grid (SPEC §3.3), named structurally as <phase>_<facing>. The
# native visibility demand is a property of the facing axis; the native
# intensity demand is given per cell. Both live in `core/locus.py` — this
# module declares only the vocabulary.
#
#   phase \ facing   BACKSTAGE (V=LOW)   ADVISOR (V=MEDIUM)   CLIENT (V=HIGH)
#   PRE              PRE_BACKSTAGE       PRE_ADVISOR          PRE_CLIENT
#   AT               AT_BACKSTAGE        AT_ADVISOR           AT_CLIENT
#   POST             POST_BACKSTAGE      POST_ADVISOR         POST_CLIENT
#   NON              NON_BACKSTAGE       —                    NON_CLIENT
GridLocus = Literal[
    "PRE_BACKSTAGE",
    "PRE_ADVISOR",
    "PRE_CLIENT",
    "AT_BACKSTAGE",
    "AT_ADVISOR",
    "AT_CLIENT",
    "POST_BACKSTAGE",
    "POST_ADVISOR",
    "POST_CLIENT",
    "NON_BACKSTAGE",
    "NON_CLIENT",
]

# UNMAPPED (SPEC §3.3) is not a cell of the grid: it records a brief that maps
# nowhere, so it has no native demand, cannot be filtered against a ceiling and
# cannot be ranked. It therefore belongs to the concept vocabulary but not to
# the calibration one — `RankedLocus` and `ExcludedLocus` take `GridLocus`,
# `Concept.locus` takes `Locus`. The two must be kept in step; `Locus` is
# spelled out rather than derived so that both render as a flat enum in the
# JSON schema sent to the model.
Locus = Literal[
    "PRE_BACKSTAGE",
    "PRE_ADVISOR",
    "PRE_CLIENT",
    "AT_BACKSTAGE",
    "AT_ADVISOR",
    "AT_CLIENT",
    "POST_BACKSTAGE",
    "POST_ADVISOR",
    "POST_CLIENT",
    "NON_BACKSTAGE",
    "NON_CLIENT",
    "UNMAPPED",
]

DimensionId = Literal["D1", "D2", "D3", "D4", "D5"]

Axis = Literal["VISIBILITY", "INTENSITY"]

# ---------------------------------------------------------------------------
# Dimension positions (SPEC §3.1, rubric_v1.yaml)
# ---------------------------------------------------------------------------

D1Position = Literal["OBJECTIVE", "VIRTUAL", "DIFFUSE"]
D2Position = Literal["STRICT", "MODERATE", "RELAXED"]
D3Position = Literal["EMOTIONAL_LED", "BALANCED", "FUNCTIONAL_LED"]
D4Position = Literal["UNM", "MIXED", "ELT"]
D5Position = Literal["STRONG", "PARTIAL", "WEAK"]

PositionT = TypeVar("PositionT")


class LuxcalModel(BaseModel):
    """Base contract: immutable, no undeclared fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DimensionPosition(LuxcalModel, Generic[PositionT]):
    """One dimension's assigned position with its provenance (SPEC §5.1).

    Parametrised by the dimension's own vocabulary so that a D2 position
    cannot be assigned to a D4 field. `evidence_span` is None exactly when
    the brief does not state the dimension; in that case `stated_in_brief`
    is False and the position comes from the declared category prior.
    """

    position: PositionT
    evidence_span: Optional[str] = Field(
        default=None, description="Verbatim quote from the brief; None if unevidenced."
    )
    stated_in_brief: bool
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(
        min_length=1,
        description="One line explaining why this position was chosen.",
    )

    @model_validator(mode="after")
    def _evidence_matches_provenance(self) -> Self:
        if self.stated_in_brief and self.evidence_span is None:
            raise ValueError(
                "evidence_span must be a verbatim quote when stated_in_brief is True"
            )
        if not self.stated_in_brief and self.evidence_span is not None:
            raise ValueError(
                "evidence_span must be None when stated_in_brief is False; "
                "an unevidenced position falls back to the category prior"
            )
        return self


class BrandProfile(LuxcalModel):
    """Agent 1 output — the typed five-dimension profile of the brief."""

    category: Category
    problem_statement: str = Field(
        min_length=1,
        description="The brief's business problem restated in one sentence.",
    )
    d1_rarity: DimensionPosition[D1Position]
    d2_invisibility: DimensionPosition[D2Position]
    d3_value_orientation: DimensionPosition[D3Position]
    d4_motivation: DimensionPosition[D4Position]
    d5_orchestration: DimensionPosition[D5Position]


# ---------------------------------------------------------------------------
# Calibration (SPEC §5.2)
# ---------------------------------------------------------------------------

GateDecision = Literal["PROCEED", "REFUSE"]

ExclusionReason = Literal[
    "VISIBILITY_CEILING",
    "INTENSITY_CEILING",
    "BOTH_CEILINGS",
]


class RankedLocus(LuxcalModel):
    """A locus that survived the ceiling filter, with its ranking rationale."""

    locus: GridLocus
    rank: int = Field(ge=1, description="1 is the best fit for the stated problem.")
    rationale: str = Field(min_length=1)


class ExcludedLocus(LuxcalModel):
    """A locus the ceilings ruled out.

    The native demands are carried rather than recomputed so that the
    exclusion can be shown to a brand manager exactly as it was decided.
    """

    locus: GridLocus
    reason: ExclusionReason
    native_visibility: Band
    native_intensity: Band


class CalibrationOutput(LuxcalModel):
    """Agent 2 output — gate decision, both ceilings, and the filtered grid.

    The bands are computed deterministically from the profile and are present
    even when the gate refuses, so that refused cases remain analysable.
    """

    gate_decision: GateDecision
    gate_rationale: str = Field(min_length=1)
    visibility_band: Band
    intensity_band: Band
    viable_loci: list[RankedLocus] = Field(default_factory=list)
    excluded_loci: list[ExcludedLocus] = Field(default_factory=list)

    @model_validator(mode="after")
    def _refusal_carries_no_ranking(self) -> Self:
        if self.gate_decision == "REFUSE" and self.viable_loci:
            raise ValueError(
                "viable_loci must be empty when gate_decision is REFUSE; "
                "a refusal means no ranking was performed"
            )
        return self


# ---------------------------------------------------------------------------
# Concept (SPEC §5.3)
# ---------------------------------------------------------------------------

AIPosition = Literal["BACKSTAGE", "ADVISOR_MEDIATED", "CLIENT_FACING"]

DifferentiationUnit = Literal["SEGMENT", "COHORT", "INDIVIDUAL", "ONE_OF_ONE"]


class Concept(LuxcalModel):
    """Agent 3 output — a baseline concept that declares its own axis positions."""

    name: str = Field(min_length=1)
    locus: Locus
    touchpoint: str = Field(
        min_length=1,
        description="Who interacts, at what moment, through what surface.",
    )
    mechanism: str = Field(min_length=1, description="What the AI actually does.")
    ai_position: AIPosition
    differentiation_unit: DifferentiationUnit
    claimed_visibility: Band
    claimed_intensity: Band
    evidence_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Critic (SPEC §5.4)
# ---------------------------------------------------------------------------

Verdict = Literal["PASS", "REVISE", "ESCALATE"]

Severity = Literal["MINOR", "MAJOR", "CRITICAL"]

RevisionDirection = Literal["INCREASE", "REDUCE", "MAINTAIN"]


class DimensionViolation(LuxcalModel):
    """A breach of one dimension's diagnostic questions, found by check 2."""

    dimension: DimensionId
    severity: Severity
    detail: str = Field(min_length=1)


class RevisionDirective(LuxcalModel):
    """The semantic gradient returned to Agent 3 on a REVISE verdict.

    Constrained to the dimension vocabulary and to a direction so that
    feedback cannot drift into generic encouragement (SPEC §5.3).
    """

    dimension: DimensionId
    axis: Axis
    direction: RevisionDirection
    instruction: str = Field(min_length=1)


class CriticVerdict(LuxcalModel):
    """Critic output — the two checks kept separate and logged separately.

    `visibility_within_ceiling` / `intensity_within_ceiling` are check 1 and
    contain no model judgement. `misdeclared` is check 2's finding that the
    concept as described does not match the bands it claims; it is recorded
    apart from `violations` because it is the result of greater research
    interest (SPEC §5.4).
    """

    verdict: Verdict
    iteration: int = Field(ge=0)

    visibility_within_ceiling: bool
    intensity_within_ceiling: bool

    misdeclared: bool
    misdeclaration_rationale: Optional[str] = None

    violations: list[DimensionViolation] = Field(default_factory=list)
    revision_directive: Optional[RevisionDirective] = None

    @model_validator(mode="after")
    def _directive_matches_verdict(self) -> Self:
        if self.verdict == "REVISE" and self.revision_directive is None:
            raise ValueError("revision_directive is required when verdict is REVISE")
        if self.verdict != "REVISE" and self.revision_directive is not None:
            raise ValueError(
                f"revision_directive must be None when verdict is {self.verdict}"
            )
        return self

    @model_validator(mode="after")
    def _misdeclaration_is_explained(self) -> Self:
        if self.misdeclared and self.misdeclaration_rationale is None:
            raise ValueError(
                "misdeclaration_rationale is required when misdeclared is True"
            )
        return self
