"""Hand-worked tests for the deterministic ceiling arithmetic (SPEC §5.2).

Every expected band below is computed by hand in the accompanying comment.
The ordinal maps are written out literally rather than loaded from the rubric,
so that a test failure distinguishes a change in the arithmetic from a change
in the rubric; `test_literal_maps_match_the_rubric` covers the drift between
the two.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from luxcal.core.ceilings import BAND_ORDER, compute_ceilings, to_band
from luxcal.core.schemas import Band, BrandProfile

ORDINAL_MAPS: dict[str, dict[str, int]] = {
    "D1": {"OBJECTIVE": 0, "VIRTUAL": 1, "DIFFUSE": 2},
    "D2": {"STRICT": 0, "MODERATE": 1, "RELAXED": 2},
    "D3": {"EMOTIONAL_LED": 0, "BALANCED": 1, "FUNCTIONAL_LED": 2},
    "D4": {"UNM": 0, "MIXED": 1, "ELT": 2},
    "D5": {"STRONG": 1, "PARTIAL": 0, "WEAK": -1},
}

RUBRIC_PATH = Path(__file__).resolve().parents[1] / "luxcal" / "rubric" / "rubric_v1.yaml"


def _profile(
    *,
    d1: str = "VIRTUAL",
    d2: str = "MODERATE",
    d3: str = "BALANCED",
    d4: str = "MIXED",
    d5: str = "PARTIAL",
) -> BrandProfile:
    """A profile whose unspecified dimensions sit at the neutral middle."""

    def position(value: str) -> dict:
        return {
            "position": value,
            "evidence_span": "a verbatim span from the brief",
            "stated_in_brief": True,
            "confidence": 0.8,
            "rationale": "Hand-worked fixture.",
        }

    return BrandProfile.model_validate(
        {
            "category": "WATCHES_JEWELLERY",
            "problem_statement": "A fixture profile for the ceiling arithmetic.",
            "d1_rarity": position(d1),
            "d2_invisibility": position(d2),
            "d3_value_orientation": position(d3),
            "d4_motivation": position(d4),
            "d5_orchestration": position(d5),
        }
    )


# ---------------------------------------------------------------------------
# Visibility — D2 + D4, adjusted by D5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d2", "d4", "d5", "expected"),
    [
        ("STRICT", "UNM", "STRONG", "LOW"),  # 0 + 0 + 1 =  1          -> LOW
        ("RELAXED", "ELT", "WEAK", "MEDIUM"),  # 2 + 2 - 1 =  3          -> MEDIUM
        ("RELAXED", "ELT", "PARTIAL", "HIGH"),  # 2 + 2 + 0 =  4          -> HIGH
        ("STRICT", "UNM", "WEAK", "LOW"),  # 0 + 0 - 1 = -1 clamped 0 -> LOW
        ("RELAXED", "ELT", "STRONG", "HIGH"),  # 2 + 2 + 1 =  5          -> HIGH
    ],
)
def test_visibility_ceiling(d2: str, d4: str, d5: str, expected: Band) -> None:
    visibility, _ = compute_ceilings(_profile(d2=d2, d4=d4, d5=d5), ORDINAL_MAPS)

    assert visibility == expected


# ---------------------------------------------------------------------------
# Intensity — D1 + D3, adjusted by D5
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d1", "d3", "d5", "expected"),
    [
        ("OBJECTIVE", "EMOTIONAL_LED", "STRONG", "LOW"),  # 0 + 0 + 1 =  1  -> LOW
        ("DIFFUSE", "FUNCTIONAL_LED", "WEAK", "MEDIUM"),  # 2 + 2 - 1 =  3  -> MEDIUM
        ("DIFFUSE", "FUNCTIONAL_LED", "PARTIAL", "HIGH"),  # 2 + 2 + 0 =  4  -> HIGH
        ("OBJECTIVE", "EMOTIONAL_LED", "WEAK", "LOW"),  # 0 + 0 - 1 = -1 -> 0 -> LOW
        ("DIFFUSE", "FUNCTIONAL_LED", "STRONG", "HIGH"),  # 2 + 2 + 1 =  5  -> HIGH
    ],
)
def test_intensity_ceiling(d1: str, d3: str, d5: str, expected: Band) -> None:
    _, intensity = compute_ceilings(_profile(d1=d1, d3=d3, d5=d5), ORDINAL_MAPS)

    assert intensity == expected


def test_the_two_axes_are_computed_independently() -> None:
    """A profile permissive on visibility and restrictive on intensity.

    visibility: RELAXED 2 + ELT   2           + PARTIAL 0 = 4 -> HIGH
    intensity:  OBJECTIVE 0 + EMOTIONAL_LED 0 + PARTIAL 0 = 0 -> LOW
    """
    profile = _profile(
        d1="OBJECTIVE", d2="RELAXED", d3="EMOTIONAL_LED", d4="ELT", d5="PARTIAL"
    )

    assert compute_ceilings(profile, ORDINAL_MAPS) == ("HIGH", "LOW")


def test_d5_adjusts_both_axes_in_the_same_direction() -> None:
    """STRONG curatorial control raises both ceilings by one point.

    with PARTIAL: v = 1 + 1 + 0 = 2 -> MEDIUM;  i = 1 + 1 + 0 = 2 -> MEDIUM
    with STRONG:  v = 1 + 1 + 1 = 3 -> MEDIUM;  i = 1 + 1 + 1 = 3 -> MEDIUM
    with WEAK:    v = 1 + 1 - 1 = 1 -> LOW;     i = 1 + 1 - 1 = 1 -> LOW
    """
    assert compute_ceilings(_profile(d5="PARTIAL"), ORDINAL_MAPS) == ("MEDIUM", "MEDIUM")
    assert compute_ceilings(_profile(d5="STRONG"), ORDINAL_MAPS) == ("MEDIUM", "MEDIUM")
    assert compute_ceilings(_profile(d5="WEAK"), ORDINAL_MAPS) == ("LOW", "LOW")


# ---------------------------------------------------------------------------
# to_band
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0, "LOW"),  # <= 1
        (1, "LOW"),  # <= 1, upper edge of LOW
        (2, "MEDIUM"),  # <= 3, lower edge of MEDIUM
        (3, "MEDIUM"),  # <= 3, upper edge of MEDIUM
        (4, "HIGH"),  # >= 4, lower edge of HIGH
        (5, "HIGH"),  # >= 4
    ],
)
def test_to_band_at_every_boundary(score: int, expected: Band) -> None:
    assert to_band(score) == expected


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-1, "LOW"),  # clamped to 0
        (-3, "LOW"),  # clamped to 0
        (6, "HIGH"),  # clamped to 5
        (9, "HIGH"),  # clamped to 5
    ],
)
def test_to_band_clamps_out_of_range_scores(score: int, expected: Band) -> None:
    assert to_band(score) == expected


def test_to_band_is_deterministic() -> None:
    for score in range(-3, 9):
        assert to_band(score) == to_band(score)


# ---------------------------------------------------------------------------
# Guards against the rubric and the code drifting apart
# ---------------------------------------------------------------------------


def test_literal_maps_match_the_rubric() -> None:
    """The literal maps above must equal the rubric's, normalised as the loader will.

    This is the normalisation `config.py` owes `compute_ceilings`: merge the
    two rubric blocks, then upper-case the dimension keys into `DimensionId`.
    """
    rubric = yaml.safe_load(RUBRIC_PATH.read_text(encoding="utf-8-sig"))
    ceilings = rubric["ceilings"]

    merged = {**ceilings["ordinal_maps"], **ceilings["adjustment_map"]}
    from_rubric = {dimension.upper(): positions for dimension, positions in merged.items()}

    assert from_rubric == ORDINAL_MAPS


def test_band_order_matches_the_rubric() -> None:
    rubric = yaml.safe_load(RUBRIC_PATH.read_text(encoding="utf-8-sig"))

    assert list(BAND_ORDER) == rubric["bands"]["ordinal_order"]
    assert BAND_ORDER == {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def test_unknown_position_fails_loudly() -> None:
    maps = {**ORDINAL_MAPS, "D2": {"STRICT": 0}}  # MODERATE and RELAXED removed

    with pytest.raises(KeyError, match="D2.MODERATE"):
        compute_ceilings(_profile(d2="MODERATE"), maps)
