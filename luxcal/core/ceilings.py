"""Deterministic ceiling arithmetic (SPEC §5.2).

This module is pure: no LLM call, no I/O, no mutation of its inputs. It is the
piece of the system most likely to be interrogated by an examiner, so the
arithmetic is kept flat and readable rather than generalised.

Visibility is governed by D2 and D4, intensity by D1 and D3, and D5 adjusts
both. The ordinal maps are supplied by the caller from the versioned rubric
rather than being hard-coded here, so that varying the weighting is a
configuration change rather than a code change.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from luxcal.core.schemas import Band, BrandProfile, DimensionId

# Ordinal comparison of bands, imported by `locus.py` so that the ordering is
# stated once. Matches `bands.ordinal_order` in the rubric.
BAND_ORDER: Final[dict[Band, int]] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

_CLAMP_MIN: Final[int] = 0
_CLAMP_MAX: Final[int] = 5
_LOW_MAX_SCORE: Final[int] = 1
_MEDIUM_MAX_SCORE: Final[int] = 3


def to_band(score: int) -> Band:
    """Clamp a raw ordinal score to [0, 5] and resolve it to a band.

    Exposed separately from `compute_ceilings` so that `locus.py` can band a
    score without reimplementing the thresholds.
    """
    clamped = max(_CLAMP_MIN, min(_CLAMP_MAX, score))
    if clamped <= _LOW_MAX_SCORE:
        return "LOW"
    if clamped <= _MEDIUM_MAX_SCORE:
        return "MEDIUM"
    return "HIGH"


def compute_ceilings(
    profile: BrandProfile,
    ordinal_maps: Mapping[DimensionId, Mapping[str, int]],
) -> tuple[Band, Band]:
    """Return the (visibility, intensity) ceilings for a profile.

    `ordinal_maps` is keyed by `DimensionId` — uppercase `D1` through `D5` —
    each mapping that dimension's positions to integers. The D5 entry holds the
    adjustment values (STRONG +1, PARTIAL 0, WEAK -1).

    The rubric spells its dimension keys lowercase and splits these across
    `ceilings.ordinal_maps` (d1-d4) and `ceilings.adjustment_map` (d5).
    Merging those two blocks and upper-casing their keys is the loader's
    responsibility; this module works in the vocabulary schemas.py declares.
    """
    adjustment = _ordinal(ordinal_maps, "D5", profile.d5_orchestration.position)

    visibility = (
        _ordinal(ordinal_maps, "D2", profile.d2_invisibility.position)
        + _ordinal(ordinal_maps, "D4", profile.d4_motivation.position)
        + adjustment
    )
    intensity = (
        _ordinal(ordinal_maps, "D1", profile.d1_rarity.position)
        + _ordinal(ordinal_maps, "D3", profile.d3_value_orientation.position)
        + adjustment
    )

    return to_band(visibility), to_band(intensity)


def _ordinal(
    ordinal_maps: Mapping[DimensionId, Mapping[str, int]],
    dimension: DimensionId,
    position: str,
) -> int:
    """Look up one position's ordinal, failing loudly on a mis-shaped rubric."""
    try:
        return ordinal_maps[dimension][position]
    except KeyError as exc:
        raise KeyError(
            f"ordinal map has no entry for {dimension}.{position}; "
            "the map passed in does not match the rubric vocabulary"
        ) from exc
