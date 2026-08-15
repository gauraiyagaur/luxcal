"""Orchestration (SPEC §6).

The orchestrator routes; it does not reason. Every decision this module makes
is read directly off a field an agent already wrote — `gate_decision`,
`verdict`, `iteration` — and no field is computed here. The router functions
are pure and side-effect free, which is what makes node removal for an
ablation a configuration change rather than a code change.

Passing config and logger to nodes
----------------------------------
The agent nodes take `(state, config, logger)`, whereas LangGraph calls a node
with `(state)` alone. The two are bridged with a closure per node rather than
by threading the pair through LangGraph's own configurable mechanism: a
single-argument closure leaves the agents as ordinary functions that can be
called and tested directly, and it keeps the live `RunLogger` out of the
config dict, which is validated data rather than a place to hide objects.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from luxcal.agents.calibration import run_calibration
from luxcal.agents.critic import run_critic
from luxcal.agents.ideation import run_ideation
from luxcal.agents.profiler import run_profiler
from luxcal.core.state import LuxcalState
from luxcal.logging.run_logger import RunLogger

AgentNode = Callable[[LuxcalState, dict, RunLogger], Awaitable[dict]]


def profiler_router(state: LuxcalState) -> str:
    """Route out of Agent 1 on whether a profile was produced.

    Agent 1 returns `{"terminal_state": "ERROR"}` when its retries are
    exhausted. Without this branch the graph would carry on to Agent 2, which
    dereferences the profile immediately and fails on `None` — turning a
    handled failure into a traceback.
    """
    if state.get("terminal_state") == "ERROR":
        return "error"
    if state.get("profile") is None:
        return "error"
    return "proceed"


def gate_router(state: LuxcalState) -> str:
    """Route out of Agent 2 on its gate decision.

    Returns `"error"` when an upstream node terminated the run, so that a
    failed extraction ends the graph rather than being read as a refusal.
    """
    if state.get("terminal_state") == "ERROR":
        return "error"

    calibration = state.get("calibration")
    if calibration is None:
        return "error"

    return "proceed" if calibration.gate_decision == "PROCEED" else "refuse"


def loop_router(state: LuxcalState, critic_max_iterations: int) -> str:
    """Route out of the Critic on its verdict.

    The cap is re-checked here as well as in the Critic. That is deliberate
    duplication: the Critic owns the decision and records it in the verdict,
    while this check is the graph's own guarantee of termination, and it holds
    even if a future variant returns a verdict the Critic did not cap.
    """
    if state.get("terminal_state") == "ERROR":
        return "error"

    verdict = state.get("verdict")
    if verdict is None:
        return "error"

    if verdict.verdict == "REVISE" and state.get("iteration", 0) >= critic_max_iterations:
        return "escalate"

    return verdict.verdict.lower()


def build_graph(config: dict, logger: RunLogger) -> CompiledStateGraph:
    """Construct and compile the pipeline.

    Takes the logger alongside the config because the nodes are closures over
    both; SPEC §6 shows `build_graph(config)` only because it predates the
    logger being a per-run object.

    No checkpointer is attached. It is needed for Agent 4's session
    persistence and for nothing before that.
    """
    critic_max_iterations = config.get("critic_max_iterations", 3)

    def node(agent: AgentNode, name: str) -> Callable[[LuxcalState], Awaitable[dict]]:
        """Bind config and logger to an agent, leaving LangGraph a state-only node."""

        async def run(state: LuxcalState) -> dict:
            return await agent(state, config, logger)

        run.__name__ = name
        return run

    def route_loop(state: LuxcalState) -> str:
        return loop_router(state, critic_max_iterations)

    graph = StateGraph(LuxcalState)

    graph.add_node("profiler", node(run_profiler, "profiler"))
    graph.add_node("calibrate", node(run_calibration, "calibrate"))
    graph.add_node("ideate", node(run_ideation, "ideate"))
    graph.add_node("critic", node(run_critic, "critic"))

    graph.set_entry_point("profiler")
    graph.add_conditional_edges(
        "profiler",
        profiler_router,
        {"proceed": "calibrate", "error": END},
    )
    graph.add_conditional_edges(
        "calibrate",
        gate_router,
        {"proceed": "ideate", "refuse": END, "error": END},
    )
    graph.add_edge("ideate", "critic")
    # "pass" ends the run until Agent 4 exists; it becomes "advise" then, and
    # that is the only line which changes.
    graph.add_conditional_edges(
        "critic",
        route_loop,
        {"pass": END, "revise": "ideate", "escalate": END, "error": END},
    )

    return graph.compile()
