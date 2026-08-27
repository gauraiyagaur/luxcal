"""Run directory management and provenance capture (SPEC §7).

This is a research instrument rather than operational telemetry: every
quantitative claim in the results chapter must be traceable to a file written
here, and a run must be reconstructible months later without access to the
process that produced it. Hence snapshots rather than references, full prompt
and response text rather than token counts alone, and JSONL for the call log so
that a crashed run still yields a readable partial record.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import traceback
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Optional

import yaml
from pydantic import BaseModel

from luxcal.core.state import LuxcalState

# USD per token, derived from the published per-million-token rates. The exact
# figures matter less than that a rate is recorded for every model that ran;
# `finalise` reports any model it could not price rather than silently
# treating it as free.
DEFAULT_COST_PER_TOKEN: Final[dict[str, dict[str, float]]] = {
    "claude-opus-4-6": {"input": 5.00 / 1_000_000, "output": 25.00 / 1_000_000},
    "claude-sonnet-4-6": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
}

_DIR_TIMESTAMP = "%Y-%m-%dT%H-%M-%S"


class RunLogger:
    """Owns one run directory and everything written into it.

    Constructing a `RunLogger` has side effects: it creates the run directory,
    snapshots the config and rubric into it, resolves the git commit, and
    writes an initial manifest. The manifest is written twice — once here, so
    that a run which crashes mid-flight is still identifiable, and again from
    `finalise` with the outcome fields filled in.
    """

    def __init__(
        self,
        case_id: str,
        variant: str,
        replicate_id: int,
        config_path: Path,
        rubric_path: Path,
        runs_dir: Path = Path("runs"),
        allow_dirty: bool = False,
        cost_per_token: Optional[dict[str, dict[str, float]]] = None,
    ) -> None:
        self._cost_per_token = cost_per_token or DEFAULT_COST_PER_TOKEN
        self._seq = 0

        self._run_id = uuid.uuid4().hex[:6]
        started = datetime.now(timezone.utc)
        self._timestamp_utc = started.isoformat()
        self._wall_clock_start = time.monotonic()

        self._run_dir = runs_dir / f"{started.strftime(_DIR_TIMESTAMP)}__{self._run_id}"
        self._run_dir.mkdir(parents=True, exist_ok=False)
        (self._run_dir / "retrieval").mkdir()
        (self._run_dir / "states").mkdir()

        # Snapshot, not reference: a run must stay interpretable after the
        # rubric has been revised, and a hash alone does not permit
        # reconstruction.
        shutil.copy2(config_path, self._run_dir / "config.yaml")
        shutil.copy2(rubric_path, self._run_dir / "rubric.yaml")

        rubric_bytes = rubric_path.read_bytes()
        rubric = yaml.safe_load(rubric_bytes.decode("utf-8-sig"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))

        self._manifest: dict[str, Any] = {
            "run_id": self._run_id,
            "timestamp_utc": self._timestamp_utc,
            "case_id": case_id,
            "variant": variant,
            "replicate_id": replicate_id,
            "git_commit_sha": _git_commit_sha(allow_dirty),
            "rubric_version": rubric["version"],
            "rubric_sha256": hashlib.sha256(rubric_bytes).hexdigest(),
            "model_generation": _required_model(config, "model_generation"),
            "model_judge": _required_model(config, "model_judge"),
        }
        self._write_manifest()

    @property
    def run_dir(self) -> Path:
        """The directory this run writes into."""
        return self._run_dir

    def save_brief(self, brief: str) -> None:
        """Write the input brief verbatim to `input_brief.txt`."""
        (self._run_dir / "input_brief.txt").write_text(brief, encoding="utf-8")

    def log_call(
        self,
        node: str,
        model: str,
        temperature: float,
        prompt: str,
        response: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int,
        schema_valid: bool,
        retry_count: int,
        stop_reason: str,
        iteration: Optional[int] = None,
    ) -> None:
        """Append one LLM call to `calls.jsonl`.

        Full prompt and response text are stored deliberately: token counts
        alone will not answer the questions that arise during analysis, and
        re-running to recover a prompt is not possible once a live retrieval
        source has changed.
        """
        self._seq += 1
        entry = {
            "run_id": self._run_id,
            "seq": self._seq,
            "node": node,
            "iteration": iteration,
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "schema_valid": schema_valid,
            "retry_count": retry_count,
            "stop_reason": stop_reason,
        }
        with (self._run_dir / "calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=_jsonable) + "\n")

    def save_state(
        self,
        node: str,
        state: dict,
        iteration: Optional[int] = None,
    ) -> None:
        """Snapshot the shared state after a graph node has run."""
        name = node if iteration is None else f"{node}_{iteration}"
        _write_json(self._run_dir / "states" / f"{name}.json", state)

    def save_error(self, exc: BaseException) -> None:
        """Record the exception that terminated a run, as `error.json`.

        `terminal_state: ERROR` alone cannot distinguish a transient API
        failure from a response that failed validation three times, and the
        two want opposite treatment on a batch resume — the first should be
        retried, the second is a real result. The class and module are written
        out so that decision reads an artefact rather than inferring one.

        A run that ended ERROR with no `error.json` exhausted its schema
        retries: the agent returned the terminal state without raising.
        """
        _write_json(
            self._run_dir / "error.json",
            {
                "exception_class": type(exc).__name__,
                "exception_module": type(exc).__module__,
                "message": str(exc),
                "traceback": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            },
        )

    def save_retrieval(self, key: str, payload: dict | str) -> None:
        """Cache one external query payload under `retrieval/<key>.json`.

        Snapshotting every retrieved payload is a reproducibility requirement,
        not an optimisation: live sources change, and an unsnapshotted run
        cannot be re-run for the results chapter.
        """
        if "/" in key or "\\" in key or key in {".", ".."}:
            raise ValueError(f"retrieval key must be a bare filename, got {key!r}")
        _write_json(self._run_dir / "retrieval" / f"{key}.json", payload)

    def finalise(self, state: LuxcalState) -> None:
        """Close the run: write the full manifest, `output.json` and `metrics.json`.

        Token totals and cost are summed from `calls.jsonl` rather than tracked
        in memory, so the figures in the manifest are derived from the same
        artefact an examiner would read.
        """
        totals = self._sum_calls()
        calibration = state.get("calibration")

        self._manifest.update(
            {
                "gate_decision": calibration.gate_decision if calibration else None,
                "gate_rationale": calibration.gate_rationale if calibration else None,
                "visibility_band": calibration.visibility_band if calibration else None,
                "intensity_band": calibration.intensity_band if calibration else None,
                "selected_locus": _selected_locus(state),
                "excluded_loci": (
                    [entry.locus for entry in calibration.excluded_loci]
                    if calibration
                    else []
                ),
                "critic_iterations": len(state.get("critic_history") or []),
                "terminal_state": state.get("terminal_state"),
                "total_input_tokens": totals["input_tokens"],
                "total_output_tokens": totals["output_tokens"],
                "est_cost_usd": totals["est_cost_usd"],
                "unpriced_models": totals["unpriced_models"],
                "wall_clock_seconds": round(
                    time.monotonic() - self._wall_clock_start, 3
                ),
            }
        )

        self._write_manifest()
        _write_json(self._run_dir / "output.json", dict(state))
        _write_json(self._run_dir / "metrics.json", self._metrics_row())

    def _metrics_row(self) -> dict[str, Any]:
        """Flatten the manifest into one row for the aggregation script (§7.4)."""
        row: dict[str, Any] = {}
        for field, value in self._manifest.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[field] = value
            elif isinstance(value, list) and len(value) <= 20:
                row[field] = value
        return row

    def _sum_calls(self) -> dict[str, Any]:
        """Sum tokens and estimated cost over `calls.jsonl`."""
        calls_path = self._run_dir / "calls.jsonl"
        input_tokens = 0
        output_tokens = 0
        cost = 0.0
        unpriced: set[str] = set()

        if calls_path.exists():
            for line in calls_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                call = json.loads(line)
                input_tokens += call["input_tokens"]
                output_tokens += call["output_tokens"]

                rates = self._cost_per_token.get(call["model"])
                if rates is None:
                    unpriced.add(call["model"])
                    continue
                cost += call["input_tokens"] * rates["input"]
                cost += call["output_tokens"] * rates["output"]

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "est_cost_usd": round(cost, 6),
            "unpriced_models": sorted(unpriced),
        }

    def _write_manifest(self) -> None:
        _write_json(self._run_dir / "manifest.json", self._manifest)


def _selected_locus(state: LuxcalState) -> Optional[str]:
    """The locus the run recommends: the concept's, else the top-ranked viable one."""
    concept = state.get("concept")
    if concept is not None:
        return concept.locus

    calibration = state.get("calibration")
    if calibration is None or not calibration.viable_loci:
        return None
    return min(calibration.viable_loci, key=lambda entry: entry.rank).locus


def _required_model(config: dict, field: str) -> str:
    """Read a pinned model string from the config, failing if it is absent.

    SPEC §7.5 requires the model version to be recorded; a run whose manifest
    cannot name the model that produced it is not reproducible, so this is a
    hard failure rather than a None.
    """
    value = config.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"config must declare {field} as a pinned model string (SPEC §7.5)"
        )
    return value


def _git_commit_sha(allow_dirty: bool) -> str:
    """Resolve HEAD, refusing to run on a dirty working tree unless overridden.

    Dirtiness is `git status --porcelain` being non-empty rather than
    `git diff-index --quiet HEAD`, because the latter ignores untracked files:
    a new, uncommitted module would let a run record a SHA for a tree that
    never contained the code that produced it. Ignored paths — `runs/`, the
    virtualenv — are excluded by `--porcelain` and so do not trip the guard.

    Batch runs deliberately skip this check. `scripts/run_batch.py` performs
    the same check once, before any cell is dispatched, and then passes
    `allow_dirty=True` into every run it launches: re-checking per run would
    abort the remainder of a batch as soon as the batch's own output made the
    tree dirty. A standalone run has no equivalent pre-flight, so the guard
    stays live there.

    The SHA itself is unaffected either way — it is always `git rev-parse
    HEAD`, the commit actually checked out when the run started, so a run
    stays traceable to a code version regardless of whether the guard ran.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "could not resolve the git working tree state; a run must be traceable "
            "to an exact code state (SPEC §7.5)"
        ) from exc

    if not status:
        return sha
    if not allow_dirty:
        entries = status.splitlines()
        shown = "\n  ".join(entries[:10])
        more = f"\n  ... and {len(entries) - 10} more" if len(entries) > 10 else ""
        raise RuntimeError(
            "the working tree is dirty (modified or untracked files present); "
            "commit the changes or pass allow_dirty=True:\n  "
            f"{shown}{more}"
        )
    return f"{sha}-dirty"


def _write_json(path: Path, payload: object) -> None:
    """Write one JSON document, indented for reading by hand."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_jsonable)


def _jsonable(value: object) -> object:
    """Serialise the non-JSON types that reach the logger from the state object."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__} to JSON")
