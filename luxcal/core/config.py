"""Loading and validation of the run config and the versioned rubric.

This is the single place that reads either file. In particular it owns the one
piece of translation between them: the rubric spells its dimension keys
lowercase and splits the ceiling arithmetic across two blocks, whereas
`core.ceilings` works in the `DimensionId` vocabulary `schemas.py` declares.
Normalising in one place keeps that seam from being re-derived — and
mis-derived — at each call site.

Paths inside the config are interpreted relative to the process working
directory, which for this project is the repository root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from luxcal.core.schemas import DimensionId, LuxcalModel


class RunConfig(LuxcalModel):
    """The keys a run cannot start without.

    `extra="forbid"` is inherited from `LuxcalModel`: an unrecognised key in a
    variant config is far more likely to be a typo than an intention, and a
    silently ignored setting would be invisible in the results.
    """

    model_generation: str = Field(
        min_length=1,
        description="Pinned model for Agent 1 and Agent 3 (SPEC §7.5).",
    )
    model_judge: str = Field(
        min_length=1,
        description="Pinned model for the Agent 2 gate and ranking, and the Critic.",
    )
    rubric_path: Path = Field(description="Path to the versioned rubric file.")
    critic_max_iterations: int = Field(
        default=3,
        ge=1,
        description=(
            "Iteration cap for the ideation-critic loop (SPEC §5.4). Exposed as "
            "configuration so that {1, 3, 5} can be run as an ablation condition."
        ),
    )


def load_config(config_path: Path) -> dict[str, Any]:
    """Read and validate a variant config, returning it as a plain dict.

    A dict rather than the `RunConfig` model because the agent nodes take
    `config: dict`; `RunConfig` is the validation schema behind it. Raises
    `pydantic.ValidationError` if a required key is missing or unusable.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    return RunConfig.model_validate(raw).model_dump()


def load_rubric(rubric_path: Path) -> dict[str, Any]:
    """Read the versioned rubric, tolerating a byte-order mark.

    The BOM tolerance is deliberate: a mark written by an editor changes the
    file's bytes and therefore its `rubric_sha256`, and a loader that refused
    to parse it would fail long after the hash had already drifted.
    """
    return yaml.safe_load(rubric_path.read_text(encoding="utf-8-sig"))


def ordinal_maps(rubric: dict[str, Any]) -> dict[DimensionId, dict[str, int]]:
    """Assemble the ordinal maps `compute_ceilings` expects.

    Merges `ceilings.ordinal_maps` (d1-d4) with `ceilings.adjustment_map` (d5)
    and upper-cases the dimension keys into `DimensionId`. Fails loudly if the
    result is not exactly the five dimensions, since a partial map would
    otherwise surface much later as a `KeyError` on one position.
    """
    ceilings = rubric["ceilings"]
    merged = {**ceilings["ordinal_maps"], **ceilings["adjustment_map"]}
    normalised = {
        dimension.upper(): positions for dimension, positions in merged.items()
    }

    expected = {"D1", "D2", "D3", "D4", "D5"}
    if set(normalised) != expected:
        raise ValueError(
            "the rubric's ceiling blocks must cover exactly D1-D5, but merged to "
            f"{sorted(normalised)}"
        )
    return normalised


def load_ordinal_maps(rubric_path: Path) -> dict[DimensionId, dict[str, int]]:
    """Read a rubric and return its ordinal maps in one step."""
    return ordinal_maps(load_rubric(rubric_path))
