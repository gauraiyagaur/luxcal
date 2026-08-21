"""Pinned behaviour for `resolve_terminal_state`.

This function decides `terminal_state`, the primary outcome variable for every
run in the batch, and its behaviour is variant-dependent: `minus_critic` has no
Critic, so an absent verdict is the designed outcome there and a failure
everywhere else. Getting that backwards would relabel failed runs as successes
in the results table, silently.
"""

from __future__ import annotations

import pytest

from luxcal.core.schemas import Concept, CriticVerdict
from scripts.run_single import resolve_terminal_state

GRAPH_VARIANTS = ["full", "minus_loop", "llm_bands"]


def _concept() -> Concept:
    return Concept.model_validate(
        {
            "name": "Allocation memory",
            "locus": "AT_BACKSTAGE",
            "touchpoint": "The salon director, before allocation rounds",
            "mechanism": "Ranks waitlist entries for human review",
            "ai_position": "BACKSTAGE",
            "differentiation_unit": "INDIVIDUAL",
            "claimed_visibility": "LOW",
            "claimed_intensity": "LOW",
            "evidence_ids": [],
        }
    )


def _verdict(kind: str) -> CriticVerdict:
    return CriticVerdict.model_validate(
        {
            "verdict": kind,
            "iteration": 0,
            "visibility_within_ceiling": True,
            "intensity_within_ceiling": True,
            "misdeclared": False,
            "misdeclaration_rationale": None,
            "violations": [],
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
# The verdict path — unchanged by the minus_critic clause
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", GRAPH_VARIANTS + ["minus_critic"])
def test_pass_verdict_resolves_to_pass(variant: str) -> None:
    state = {"verdict": _verdict("PASS"), "concept": _concept()}

    assert resolve_terminal_state(state, variant) == "PASS"


@pytest.mark.parametrize("variant", GRAPH_VARIANTS)
def test_escalate_verdict_resolves_to_escalate(variant: str) -> None:
    state = {"verdict": _verdict("ESCALATE"), "concept": _concept()}

    assert resolve_terminal_state(state, variant) == "ESCALATE"


@pytest.mark.parametrize("variant", GRAPH_VARIANTS)
def test_trailing_revise_resolves_to_escalate(variant: str) -> None:
    """A REVISE at the end of the graph means the loop router cut the loop."""
    state = {"verdict": _verdict("REVISE"), "concept": _concept()}

    assert resolve_terminal_state(state, variant) == "ESCALATE"


# ---------------------------------------------------------------------------
# The minus_critic clause — and its containment
# ---------------------------------------------------------------------------


def test_minus_critic_with_no_verdict_resolves_to_pass() -> None:
    """The designed outcome for the one variant that has no Critic."""
    state = {"verdict": None, "concept": _concept()}

    assert resolve_terminal_state(state, "minus_critic") == "PASS"


@pytest.mark.parametrize("variant", GRAPH_VARIANTS)
def test_other_variants_with_no_verdict_are_not_pass(variant: str) -> None:
    """The containment that matters.

    In a variant that has a Critic, no verdict means the Critic did not return
    — retry exhaustion, or an error. That is a failure and must never be
    relabelled PASS.
    """
    state = {"verdict": None, "concept": _concept()}

    result = resolve_terminal_state(state, variant)

    assert result != "PASS"
    assert result is None


def test_minus_critic_with_no_verdict_and_no_concept_is_not_pass() -> None:
    """PASS must imply something was produced."""
    state = {"verdict": None, "concept": None}

    assert resolve_terminal_state(state, "minus_critic") != "PASS"


# ---------------------------------------------------------------------------
# An explicit terminal state always wins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", GRAPH_VARIANTS + ["minus_critic"])
@pytest.mark.parametrize("terminal", ["ERROR", "GATE_NO"])
def test_explicit_terminal_state_wins(variant: str, terminal: str) -> None:
    """Set by an agent or the gate; nothing downstream may override it.

    Includes the case that would otherwise trip the new clause: minus_critic,
    no verdict, concept present, but ERROR already recorded.
    """
    state = {"terminal_state": terminal, "verdict": None, "concept": _concept()}

    assert resolve_terminal_state(state, variant) == terminal


def test_explicit_error_wins_over_a_pass_verdict() -> None:
    state = {"terminal_state": "ERROR", "verdict": _verdict("PASS")}

    assert resolve_terminal_state(state, "full") == "ERROR"
