"""Hand-worked tests for the locus grid and the ceiling filter (SPEC §3.3, §5.2).

Each case lists the surviving cells and the reason every other cell fails, so
the partition can be checked against the SPEC §3.3 table by hand.
"""

from __future__ import annotations

from typing import get_args

import pytest

from luxcal.core.locus import (
    AI_POSITION_VISIBILITY,
    LOCUS_AI_POSITION,
    LOCUS_DESCRIPTIONS,
    LOCUS_GRID,
    filter_loci,
)
from luxcal.core.schemas import Band, ExclusionReason, GridLocus

BACKSTAGE_CELLS = ["PRE_BACKSTAGE", "AT_BACKSTAGE", "POST_BACKSTAGE", "NON_BACKSTAGE"]
LOW_INTENSITY_CELLS = ["AT_BACKSTAGE", "NON_BACKSTAGE"]


def _reasons(visibility: Band, intensity: Band) -> dict[GridLocus, ExclusionReason]:
    _, excluded = filter_loci(visibility, intensity)
    return {entry.locus: entry.reason for entry in excluded}


# ---------------------------------------------------------------------------
# The grid itself
# ---------------------------------------------------------------------------


def test_grid_has_exactly_eleven_cells() -> None:
    """UNMAPPED is not a cell, so the grid covers every GridLocus and no more."""
    assert len(LOCUS_GRID) == 11
    assert set(LOCUS_GRID) == set(get_args(GridLocus))


def test_descriptions_cover_exactly_the_grid() -> None:
    """The two dicts must describe the same eleven cells.

    A cell with no description would reach the Agent 2 ranking prompt as a bare
    name with no context, and a description with no cell would never be
    reachable — both silent, so they are pinned here.
    """
    assert set(LOCUS_DESCRIPTIONS) == set(LOCUS_GRID)
    assert all(description for description in LOCUS_DESCRIPTIONS.values())


def test_ai_position_map_covers_the_grid_and_matches_its_visibility() -> None:
    """`LOCUS_AI_POSITION` must agree with the grid it was extracted from.

    The map states the facing axis a second time, so it can drift from
    `LOCUS_GRID`. Every cell's declared position must imply exactly the native
    visibility the grid already records for that cell.
    """
    assert set(LOCUS_AI_POSITION) == set(LOCUS_GRID)
    for locus, position in LOCUS_AI_POSITION.items():
        native_visibility, _ = LOCUS_GRID[locus]
        assert AI_POSITION_VISIBILITY[position] == native_visibility


def test_native_visibility_follows_the_facing_axis() -> None:
    """Visibility is a property of the column, not an attribute of the cell."""
    for locus, (native_visibility, _) in LOCUS_GRID.items():
        if locus.endswith("_BACKSTAGE"):
            assert native_visibility == "LOW"
        elif locus.endswith("_ADVISOR"):
            assert native_visibility == "MEDIUM"
        else:
            assert native_visibility == "HIGH"


# ---------------------------------------------------------------------------
# Filter cases
# ---------------------------------------------------------------------------


def test_both_ceilings_high_admits_the_whole_grid() -> None:
    """No native demand exceeds HIGH, so all 11 survive and nothing is excluded."""
    surviving, excluded = filter_loci("HIGH", "HIGH")

    assert surviving == list(LOCUS_GRID)
    assert excluded == []


def test_both_ceilings_low_admits_backstage_only() -> None:
    """The SPEC §5.2 backstage-only case.

    survive: AT_BACKSTAGE (LOW, LOW), NON_BACKSTAGE (LOW, LOW)
    excluded 9:
      PRE_BACKSTAGE  (LOW, MED)  -> intensity only
      POST_BACKSTAGE (LOW, MED)  -> intensity only
      PRE_ADVISOR    (MED, HIGH) -> both
      AT_ADVISOR     (MED, HIGH) -> both
      POST_ADVISOR   (MED, MED)  -> both
      PRE_CLIENT     (HIGH, MED) -> both
      AT_CLIENT      (HIGH, HIGH)-> both
      POST_CLIENT    (HIGH, MED) -> both
      NON_CLIENT     (HIGH, MED) -> both
    """
    surviving, excluded = filter_loci("LOW", "LOW")

    assert surviving == LOW_INTENSITY_CELLS
    assert len(excluded) == 9

    reasons = _reasons("LOW", "LOW")
    assert reasons["PRE_BACKSTAGE"] == "INTENSITY_CEILING"
    assert reasons["POST_BACKSTAGE"] == "INTENSITY_CEILING"
    for locus in ("PRE_ADVISOR", "AT_ADVISOR", "POST_ADVISOR"):
        assert reasons[locus] == "BOTH_CEILINGS"
    for locus in ("PRE_CLIENT", "AT_CLIENT", "POST_CLIENT", "NON_CLIENT"):
        assert reasons[locus] == "BOTH_CEILINGS"


def test_low_visibility_high_intensity_admits_backstage_column() -> None:
    """A discreet brand with room to differentiate deeply, out of sight.

    survive: the four backstage cells — intensity never binds at HIGH
    excluded 7: every advisor and client cell, visibility only
    """
    surviving, excluded = filter_loci("LOW", "HIGH")

    assert surviving == BACKSTAGE_CELLS
    assert len(excluded) == 7
    assert all(entry.reason == "VISIBILITY_CEILING" for entry in excluded)


def test_high_visibility_low_intensity_admits_low_intensity_cells_only() -> None:
    """Intensity binds regardless of how visible the AI may be.

    survive: AT_BACKSTAGE (I=LOW), NON_BACKSTAGE (I=LOW)
    excluded 9: every cell with I=MEDIUM or I=HIGH, intensity only —
                visibility never binds at HIGH
    """
    surviving, excluded = filter_loci("HIGH", "LOW")

    assert surviving == LOW_INTENSITY_CELLS
    assert len(excluded) == 9
    assert all(entry.reason == "INTENSITY_CEILING" for entry in excluded)


def test_both_ceilings_medium() -> None:
    """The mixed case, where all three exclusion reasons appear.

    survive 5:
      PRE_BACKSTAGE  (LOW, MED)
      AT_BACKSTAGE   (LOW, LOW)
      POST_BACKSTAGE (LOW, MED)
      POST_ADVISOR   (MED, MED)
      NON_BACKSTAGE  (LOW, LOW)
    excluded 6:
      PRE_ADVISOR (MED, HIGH) -> intensity only
      AT_ADVISOR  (MED, HIGH) -> intensity only
      PRE_CLIENT  (HIGH, MED) -> visibility only
      POST_CLIENT (HIGH, MED) -> visibility only
      NON_CLIENT  (HIGH, MED) -> visibility only
      AT_CLIENT   (HIGH, HIGH)-> both
    """
    surviving, excluded = filter_loci("MEDIUM", "MEDIUM")

    assert surviving == [
        "PRE_BACKSTAGE",
        "AT_BACKSTAGE",
        "POST_BACKSTAGE",
        "POST_ADVISOR",
        "NON_BACKSTAGE",
    ]
    assert len(excluded) == 6


def test_both_ceilings_medium_exclusion_reasons() -> None:
    """Every excluded cell in the mixed case carries the right reason."""
    assert _reasons("MEDIUM", "MEDIUM") == {
        "PRE_ADVISOR": "INTENSITY_CEILING",
        "AT_ADVISOR": "INTENSITY_CEILING",
        "PRE_CLIENT": "VISIBILITY_CEILING",
        "POST_CLIENT": "VISIBILITY_CEILING",
        "NON_CLIENT": "VISIBILITY_CEILING",
        "AT_CLIENT": "BOTH_CEILINGS",
    }


# ---------------------------------------------------------------------------
# Invariants across every ceiling combination
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("visibility", ["LOW", "MEDIUM", "HIGH"])
@pytest.mark.parametrize("intensity", ["LOW", "MEDIUM", "HIGH"])
def test_partition_is_total_and_disjoint(visibility: Band, intensity: Band) -> None:
    """Every cell is either kept or excluded exactly once, for any ceilings."""
    surviving, excluded = filter_loci(visibility, intensity)
    excluded_loci = [entry.locus for entry in excluded]

    assert len(surviving) + len(excluded) == 11
    assert set(surviving).isdisjoint(excluded_loci)
    assert set(surviving) | set(excluded_loci) == set(LOCUS_GRID)


@pytest.mark.parametrize("visibility", ["LOW", "MEDIUM", "HIGH"])
@pytest.mark.parametrize("intensity", ["LOW", "MEDIUM", "HIGH"])
def test_excluded_loci_report_their_native_demands(
    visibility: Band, intensity: Band
) -> None:
    """The carried demands must match the grid, not be recomputed or invented."""
    _, excluded = filter_loci(visibility, intensity)

    for entry in excluded:
        assert (entry.native_visibility, entry.native_intensity) == LOCUS_GRID[
            entry.locus
        ]
