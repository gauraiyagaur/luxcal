"""The fixed locus grid and the ceiling filter (SPEC §3.3, §5.2 step iii).

This module is pure: no LLM call, no I/O. It answers only which loci the
calibrated ceilings permit, and for each excluded locus, why. Ranking the
survivors is a separate LLM call in Agent 2; nothing here orders by merit.

The grid is defined structurally — by who the AI is perceptible to and when it
acts — so the native visibility demand follows from the facing axis rather than
being assigned per cell.
"""

from __future__ import annotations

from luxcal.core.ceilings import BAND_ORDER
from luxcal.core.schemas import (
    AIPosition,
    Band,
    ExcludedLocus,
    ExclusionReason,
    GridLocus,
)

# Native (visibility, intensity) demand per cell, transcribed from the SPEC
# §3.3 table. Insertion order is the grid's reading order — pre, at, post,
# non-encounter, each backstage to client-direct — and is the order survivors
# are returned in.
LOCUS_GRID: dict[GridLocus, tuple[Band, Band]] = {
    "PRE_BACKSTAGE": ("LOW", "MEDIUM"),
    "PRE_ADVISOR": ("MEDIUM", "HIGH"),
    "PRE_CLIENT": ("HIGH", "MEDIUM"),
    "AT_BACKSTAGE": ("LOW", "LOW"),
    "AT_ADVISOR": ("MEDIUM", "HIGH"),
    "AT_CLIENT": ("HIGH", "HIGH"),
    "POST_BACKSTAGE": ("LOW", "MEDIUM"),
    "POST_ADVISOR": ("MEDIUM", "MEDIUM"),
    "POST_CLIENT": ("HIGH", "MEDIUM"),
    "NON_BACKSTAGE": ("LOW", "LOW"),
    "NON_CLIENT": ("HIGH", "MEDIUM"),
}

# Prose label per cell, transcribed from the same SPEC §3.3 table. Kept beside
# `LOCUS_GRID` so the two describe the same eleven cells; consumers that need
# the native demands read them from the grid rather than restating them here.
LOCUS_DESCRIPTIONS: dict[GridLocus, str] = {
    "PRE_BACKSTAGE": "Demand forecasting and allocation; backstage, before the client encounter",
    "PRE_ADVISOR": "Client dossier and relationship preparation; advisor-mediated, before the encounter",
    "PRE_CLIENT": "Targeted outreach and campaign; client-direct, before the encounter",
    "AT_BACKSTAGE": "Inventory, authentication and provenance; backstage, during the encounter",
    "AT_ADVISOR": "Advisor assistant and configurator support; advisor-mediated, during the encounter",
    "AT_CLIENT": "Client-facing assistant or interface; client-direct, during the encounter",
    "POST_BACKSTAGE": "Service and care analytics; backstage, after the encounter",
    "POST_ADVISOR": "Follow-up prompting; advisor-mediated, after the encounter",
    "POST_CLIENT": "Direct aftercare communication; client-direct, after the encounter",
    "NON_BACKSTAGE": "Product and design development input; backstage, outside any client encounter",
    "NON_CLIENT": "Brand content and narrative generation; client-direct, outside any client encounter",
}

# The `ai_position` each grid column corresponds to.
#
# Made explicit for the evaluation. The correspondence was previously only
# implicit — in the grid comment in `schemas.py` and in `test_locus.py`'s
# facing-axis assertion — so anything reading it was inferring rather than
# citing a source. A concept whose declared position disagrees with its
# locus's column is not necessarily wrong: it may be declaring a higher
# visibility than the locus name implies, which is the honest answer. Treat a
# mismatch as a declared inconsistency to be inspected, not an error.
LOCUS_AI_POSITION: dict[GridLocus, AIPosition] = {
    "PRE_BACKSTAGE": "BACKSTAGE",
    "PRE_ADVISOR": "ADVISOR_MEDIATED",
    "PRE_CLIENT": "CLIENT_FACING",
    "AT_BACKSTAGE": "BACKSTAGE",
    "AT_ADVISOR": "ADVISOR_MEDIATED",
    "AT_CLIENT": "CLIENT_FACING",
    "POST_BACKSTAGE": "BACKSTAGE",
    "POST_ADVISOR": "ADVISOR_MEDIATED",
    "POST_CLIENT": "CLIENT_FACING",
    "NON_BACKSTAGE": "BACKSTAGE",
    "NON_CLIENT": "CLIENT_FACING",
}

# The native visibility demand of each position — the same facing axis as
# above, read from the column headers of the SPEC §3.3 grid rather than from
# any single cell.
AI_POSITION_VISIBILITY: dict[AIPosition, Band] = {
    "BACKSTAGE": "LOW",
    "ADVISOR_MEDIATED": "MEDIUM",
    "CLIENT_FACING": "HIGH",
}


def filter_loci(
    visibility_ceiling: Band,
    intensity_ceiling: Band,
) -> tuple[list[GridLocus], list[ExcludedLocus]]:
    """Partition the grid against both ceilings.

    A locus survives when neither of its native demands exceeds the
    corresponding ceiling. Returns the survivors in grid order, unranked, and
    an `ExcludedLocus` for every cell that failed, carrying its native demands
    so the exclusion can be shown exactly as it was decided.
    """
    surviving: list[GridLocus] = []
    excluded: list[ExcludedLocus] = []

    for locus, (native_visibility, native_intensity) in LOCUS_GRID.items():
        visibility_ok = BAND_ORDER[native_visibility] <= BAND_ORDER[visibility_ceiling]
        intensity_ok = BAND_ORDER[native_intensity] <= BAND_ORDER[intensity_ceiling]

        if visibility_ok and intensity_ok:
            surviving.append(locus)
            continue

        reason: ExclusionReason
        if not visibility_ok and not intensity_ok:
            reason = "BOTH_CEILINGS"
        elif not visibility_ok:
            reason = "VISIBILITY_CEILING"
        else:
            reason = "INTENSITY_CEILING"

        excluded.append(
            ExcludedLocus(
                locus=locus,
                reason=reason,
                native_visibility=native_visibility,
                native_intensity=native_intensity,
            )
        )

    return surviving, excluded
