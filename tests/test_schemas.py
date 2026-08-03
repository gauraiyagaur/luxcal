"""Contract tests for the inter-agent Pydantic models.

These guard the two properties downstream code relies on: that a profile
survives a serialisation round trip unchanged, and that the schema-level
invariants reject malformed agent output rather than passing it on.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from luxcal.core.schemas import (
    BrandProfile,
    CalibrationOutput,
    CriticVerdict,
    DimensionPosition,
    D1Position,
    ExcludedLocus,
    RankedLocus,
    RevisionDirective,
)


def _position(position: str, *, stated: bool = True) -> dict:
    return {
        "position": position,
        "evidence_span": "a verbatim span from the brief" if stated else None,
        "stated_in_brief": stated,
        "confidence": 0.8,
        "rationale": "The brief describes tightly controlled allocation.",
    }


@pytest.fixture
def profile_payload() -> dict:
    return {
        "category": "WATCHES_JEWELLERY",
        "problem_statement": (
            "The house wants to shorten waitlist attrition without widening "
            "allocation."
        ),
        "d1_rarity": _position("OBJECTIVE"),
        "d2_invisibility": _position("STRICT"),
        "d3_value_orientation": _position("EMOTIONAL_LED"),
        "d4_motivation": _position("UNM", stated=False),
        "d5_orchestration": _position("STRONG"),
    }


def test_brand_profile_round_trips(profile_payload: dict) -> None:
    profile = BrandProfile.model_validate(profile_payload)

    assert profile.category == "WATCHES_JEWELLERY"
    assert profile.d2_invisibility.position == "STRICT"
    assert profile.d4_motivation.stated_in_brief is False
    assert profile.d4_motivation.evidence_span is None

    dumped = profile.model_dump()
    assert BrandProfile.model_validate(dumped) == profile

    # The run logger serialises to JSON, so the dump must survive that too.
    assert BrandProfile.model_validate(json.loads(profile.model_dump_json())) == profile


def test_brand_profile_json_schema_generates() -> None:
    schema = BrandProfile.model_json_schema()

    assert set(schema["required"]) == {
        "category",
        "problem_statement",
        "d1_rarity",
        "d2_invisibility",
        "d3_value_orientation",
        "d4_motivation",
        "d5_orchestration",
    }
    # One $def per parametrisation of the generic — five distinct vocabularies.
    assert len(schema["$defs"]) == 5

    print(json.dumps(schema, indent=2))


def test_stated_position_requires_evidence_span() -> None:
    with pytest.raises(ValidationError, match="evidence_span"):
        DimensionPosition[D1Position].model_validate(
            {
                "position": "OBJECTIVE",
                "evidence_span": None,
                "stated_in_brief": True,
                "confidence": 0.9,
                "rationale": "Production is capped by atelier capacity.",
            }
        )


def test_unevidenced_position_forbids_evidence_span() -> None:
    with pytest.raises(ValidationError, match="evidence_span"):
        DimensionPosition[D1Position].model_validate(
            {
                "position": "OBJECTIVE",
                "evidence_span": "invented span",
                "stated_in_brief": False,
                "confidence": 0.4,
                "rationale": "Fallback to the category prior.",
            }
        )


def _verdict_payload(**overrides: object) -> dict:
    payload: dict = {
        "verdict": "PASS",
        "iteration": 1,
        "visibility_within_ceiling": True,
        "intensity_within_ceiling": True,
        "misdeclared": False,
        "misdeclaration_rationale": None,
        "violations": [],
        "revision_directive": None,
    }
    payload.update(overrides)
    return payload


def test_revise_verdict_requires_directive() -> None:
    with pytest.raises(ValidationError, match="revision_directive"):
        CriticVerdict.model_validate(_verdict_payload(verdict="REVISE"))


def test_non_revise_verdict_forbids_directive() -> None:
    directive = RevisionDirective(
        dimension="D2",
        axis="VISIBILITY",
        direction="REDUCE",
        instruction="Move the assistant behind the advisor.",
    )
    with pytest.raises(ValidationError, match="revision_directive"):
        CriticVerdict.model_validate(
            _verdict_payload(verdict="PASS", revision_directive=directive.model_dump())
        )


def test_revise_verdict_with_directive_is_valid() -> None:
    verdict = CriticVerdict.model_validate(
        _verdict_payload(
            verdict="REVISE",
            visibility_within_ceiling=False,
            revision_directive={
                "dimension": "D2",
                "axis": "VISIBILITY",
                "direction": "REDUCE",
                "instruction": "Move the assistant behind the advisor.",
            },
        )
    )

    assert verdict.revision_directive is not None
    assert verdict.revision_directive.direction == "REDUCE"


def test_misdeclaration_requires_rationale() -> None:
    with pytest.raises(ValidationError, match="misdeclaration_rationale"):
        CriticVerdict.model_validate(
            _verdict_payload(misdeclared=True, misdeclaration_rationale=None)
        )


def test_position_from_wrong_dimension_is_rejected(profile_payload: dict) -> None:
    profile_payload["d1_rarity"] = _position("STRICT")  # a D2 position

    with pytest.raises(ValidationError, match="d1_rarity"):
        BrandProfile.model_validate(profile_payload)


def _ranked_locus(locus: str = "AT_BACKSTAGE") -> dict:
    return {
        "locus": locus,
        "rank": 1,
        "rationale": "Provenance checks sit behind the encounter.",
    }


def test_refusal_may_not_carry_ranked_loci() -> None:
    with pytest.raises(ValidationError, match="viable_loci"):
        CalibrationOutput.model_validate(
            {
                "gate_decision": "REFUSE",
                "gate_rationale": "Authentication is not a personalisation problem.",
                "visibility_band": "LOW",
                "intensity_band": "LOW",
                "viable_loci": [_ranked_locus()],
                "excluded_loci": [],
            }
        )


def test_refusal_with_empty_ranking_is_valid() -> None:
    calibration = CalibrationOutput.model_validate(
        {
            "gate_decision": "REFUSE",
            "gate_rationale": "Authentication is not a personalisation problem.",
            "visibility_band": "LOW",
            "intensity_band": "LOW",
            "viable_loci": [],
            "excluded_loci": [],
        }
    )

    # The ceilings are still recorded, so refused cases remain analysable.
    assert calibration.visibility_band == "LOW"
    assert calibration.viable_loci == []


def test_ranked_locus_rejects_unmapped() -> None:
    with pytest.raises(ValidationError, match="locus"):
        RankedLocus.model_validate(_ranked_locus("UNMAPPED"))


def test_excluded_locus_rejects_unmapped() -> None:
    with pytest.raises(ValidationError, match="locus"):
        ExcludedLocus.model_validate(
            {
                "locus": "UNMAPPED",
                "reason": "VISIBILITY_CEILING",
                "native_visibility": "HIGH",
                "native_intensity": "HIGH",
            }
        )
