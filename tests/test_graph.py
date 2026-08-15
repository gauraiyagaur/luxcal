"""Branch coverage for the three graph routers (SPEC §6).

The routers are the only decisions the orchestration layer makes, and each is
a pure function of state, so the whole surface is state dicts in and branch
labels out. Every string asserted here must be a key in the corresponding
branch map in `build_graph`; a label that is returned but not mapped fails at
runtime, mid-run, which is the failure these tests exist to prevent.
"""

from __future__ import annotations

import pytest

from luxcal.core.graph import gate_router, loop_router, profiler_router
from luxcal.core.schemas import (
    BrandProfile,
    CalibrationOutput,
    CriticVerdict,
    Verdict,
)

CAP = 3


def _position(value: str) -> dict:
    return {
        "position": value,
        "evidence_span": "a verbatim span from the brief",
        "stated_in_brief": True,
        "confidence": 0.8,
        "rationale": "Stated in the brief.",
    }


def _profile() -> BrandProfile:
    return BrandProfile.model_validate(
        {
            "category": "WATCHES_JEWELLERY",
            "problem_statement": "Waitlist attrition is rising.",
            "d1_rarity": _position("OBJECTIVE"),
            "d2_invisibility": _position("STRICT"),
            "d3_value_orientation": _position("EMOTIONAL_LED"),
            "d4_motivation": _position("UNM"),
            "d5_orchestration": _position("STRONG"),
        }
    )


def _calibration(gate_decision: str) -> CalibrationOutput:
    return CalibrationOutput.model_validate(
        {
            "gate_decision": gate_decision,
            "gate_rationale": "Relationship depth is a personalisation problem.",
            "visibility_band": "LOW",
            "intensity_band": "MEDIUM",
            "viable_loci": (
                [{"locus": "AT_BACKSTAGE", "rank": 1, "rationale": "Fits."}]
                if gate_decision == "PROCEED"
                else []
            ),
            "excluded_loci": [],
        }
    )


def _verdict(kind: Verdict, iteration: int = 0) -> CriticVerdict:
    return CriticVerdict.model_validate(
        {
            "verdict": kind,
            "iteration": iteration,
            "visibility_within_ceiling": True,
            "intensity_within_ceiling": True,
            "misdeclared": False,
            "misdeclaration_rationale": None,
            "violations": [],
            # The schema requires a directive exactly on REVISE.
            "revision_directive": (
                {
                    "dimension": "D2",
                    "axis": "VISIBILITY",
                    "direction": "REDUCE",
                    "instruction": "Move the assistant behind the advisor.",
                }
                if kind == "REVISE"
                else None
            ),
        }
    )


# ---------------------------------------------------------------------------
# profiler_router — proceed | error
# ---------------------------------------------------------------------------


def test_profiler_proceeds_with_a_profile() -> None:
    assert profiler_router({"profile": _profile(), "terminal_state": None}) == "proceed"


def test_profiler_errors_on_terminal_error() -> None:
    """Agent 1 returns terminal_state ERROR when its retries are exhausted."""
    assert profiler_router({"profile": None, "terminal_state": "ERROR"}) == "error"


def test_profiler_errors_on_a_missing_profile() -> None:
    """No profile and no ERROR flag is still not something Agent 2 can consume."""
    assert profiler_router({}) == "error"


def test_profiler_error_wins_over_a_present_profile() -> None:
    """A run already marked ERROR must not continue, whatever else is in state."""
    assert profiler_router({"profile": _profile(), "terminal_state": "ERROR"}) == "error"


# ---------------------------------------------------------------------------
# gate_router — proceed | refuse | error
# ---------------------------------------------------------------------------


def test_gate_proceeds() -> None:
    assert gate_router({"calibration": _calibration("PROCEED")}) == "proceed"


def test_gate_refuses() -> None:
    assert gate_router({"calibration": _calibration("REFUSE")}) == "refuse"


def test_gate_errors_on_terminal_error() -> None:
    assert gate_router({"calibration": None, "terminal_state": "ERROR"}) == "error"


def test_gate_errors_on_a_missing_calibration() -> None:
    assert gate_router({}) == "error"


def test_gate_error_is_not_read_as_a_refusal() -> None:
    """Both end the run, but they are different results and must not be conflated."""
    state = {"calibration": _calibration("PROCEED"), "terminal_state": "ERROR"}

    assert gate_router(state) == "error"


# ---------------------------------------------------------------------------
# loop_router — pass | revise | escalate | error
# ---------------------------------------------------------------------------


def test_loop_passes() -> None:
    assert loop_router({"verdict": _verdict("PASS"), "iteration": 1}, CAP) == "pass"


def test_loop_revises_below_the_cap() -> None:
    assert loop_router({"verdict": _verdict("REVISE"), "iteration": 1}, CAP) == "revise"


def test_loop_escalates_on_an_escalate_verdict() -> None:
    """The Critic's own cap upgrade arrives here as an ESCALATE verdict."""
    state = {"verdict": _verdict("ESCALATE", iteration=2), "iteration": 3}

    assert loop_router(state, CAP) == "escalate"


@pytest.mark.parametrize("iteration", [CAP, CAP + 1])
def test_loop_escalates_a_revise_at_or_past_the_cap(iteration: int) -> None:
    """The graph's own termination guarantee, independent of the Critic's.

    A REVISE that reaches the cap is turned into an escalation here even if the
    Critic did not upgrade it, so the loop cannot run unbounded.
    """
    state = {"verdict": _verdict("REVISE"), "iteration": iteration}

    assert loop_router(state, CAP) == "escalate"


def test_loop_still_revises_on_the_iteration_below_the_cap() -> None:
    """Boundary: the cap fires at `iteration == cap`, not one before it."""
    state = {"verdict": _verdict("REVISE"), "iteration": CAP - 1}

    assert loop_router(state, CAP) == "revise"


def test_loop_errors_on_terminal_error() -> None:
    assert loop_router({"verdict": None, "terminal_state": "ERROR"}, CAP) == "error"


def test_loop_errors_on_a_missing_verdict() -> None:
    assert loop_router({"iteration": 1}, CAP) == "error"


def test_loop_error_wins_over_a_present_verdict() -> None:
    state = {"verdict": _verdict("PASS"), "iteration": 1, "terminal_state": "ERROR"}

    assert loop_router(state, CAP) == "error"


# ---------------------------------------------------------------------------
# The labels the routers emit must be the labels the graph maps
# ---------------------------------------------------------------------------


def test_routers_only_emit_mapped_branch_labels() -> None:
    """Pins each router's output alphabet to its branch map in `build_graph`.

    A router returning a label the map does not contain fails at runtime,
    part-way through a run, rather than at construction.
    """
    profiler_states = [
        {"profile": _profile()},
        {"terminal_state": "ERROR"},
        {},
    ]
    gate_states = [
        {"calibration": _calibration("PROCEED")},
        {"calibration": _calibration("REFUSE")},
        {"terminal_state": "ERROR"},
        {},
    ]
    loop_states = [
        {"verdict": _verdict("PASS"), "iteration": 1},
        {"verdict": _verdict("REVISE"), "iteration": 1},
        {"verdict": _verdict("REVISE"), "iteration": CAP},
        {"verdict": _verdict("ESCALATE", iteration=2), "iteration": 3},
        {"terminal_state": "ERROR"},
    ]

    assert {profiler_router(s) for s in profiler_states} <= {"proceed", "error"}
    assert {gate_router(s) for s in gate_states} <= {"proceed", "refuse", "error"}
    assert {loop_router(s, CAP) for s in loop_states} <= {
        "pass",
        "revise",
        "escalate",
        "error",
    }
