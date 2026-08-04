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
from luxcal.core.schemas import Band, ExcludedLocus, ExclusionReason, GridLocus

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
