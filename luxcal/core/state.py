"""The single typed state object that flows through every graph node (SPEC §4).

Each agent reads the fields it needs and writes only its own outputs. Node
removal for ablation is therefore a configuration change rather than a code
change.
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

from luxcal.core.schemas import (
    BrandProfile,
    CalibrationOutput,
    Concept,
    CriticVerdict,
)

TerminalState = Literal["PASS", "GATE_NO", "ESCALATE", "ERROR"]


class LuxcalState(TypedDict):
    """Shared LangGraph state.

    `critic_history` accumulates every verdict rather than being overwritten:
    the trajectory of rejections across iterations shows whether ideation
    responds to feedback or merely re-rolls, which is itself a result.
    """

    run_id: str
    case_id: str
    brief: str

    profile: Optional[BrandProfile]  # Agent 1
    calibration: Optional[CalibrationOutput]  # Agent 2
    concept: Optional[Concept]  # Agent 3
    verdict: Optional[CriticVerdict]  # Critic

    critic_history: list[CriticVerdict]
    iteration: int
    terminal_state: Optional[TerminalState]

    messages: list[dict]  # Agent 4 dialogue
