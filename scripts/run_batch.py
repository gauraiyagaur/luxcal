"""The overnight experimental batch: 11 cases x 5 variants x 3 replicates.

    python -m scripts.run_batch                 # the full 165
    python -m scripts.run_batch --dry-run       # plan only, no API calls

Sequential by design. One request in flight at a time makes rate limiting a
non-issue, and the job fits an overnight window as it is; parallelism would buy
wall-clock on a run that does not need it, at the cost of interleaved failures.

Resumable: a run is identified by `(case_id, variant, replicate)` read back
from the manifests under `runs/`, never from a directory name — the directory
name carries only a timestamp and a random id. See `_should_run` for how a
transient API failure is distinguished from a genuine one.

Nothing here reimplements a run. `run_single.run` and `run_baseline.run` have
the same signature and both own their whole RunLogger lifecycle; this module
builds their arguments, dispatches, and records what came back.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

from dotenv import load_dotenv

from luxcal.core.config import load_config
from luxcal.logging.run_logger import _git_commit_sha
from scripts import run_baseline, run_single

# The experimental scope, stated explicitly rather than globbed. Cases 001-003
# are development fixtures and are deliberately excluded: a bare glob of
# data/cases/ would silently pull them in and overspend the API budget.
CASE_IDS: list[str] = [f"case_{n:03d}" for n in range(4, 15)]  # case_004..case_014

VARIANT_CONFIGS: dict[str, Path] = {
    "full": Path("luxcal/configs/full.yaml"),
    "minus_critic": Path("luxcal/configs/minus_critic.yaml"),
    "minus_loop": Path("luxcal/configs/minus_loop.yaml"),
    "llm_bands": Path("luxcal/configs/llm_bands.yaml"),
    "baseline": Path("luxcal/configs/baseline.yaml"),
}

REPLICATES: list[int] = [0, 1, 2]

# How many times a cell that failed with an API-class error may be retried
# across resumes before it is left alone. Without a cap, a permanently broken
# cell would be re-attempted on every restart.
MAX_API_ATTEMPTS = 3

# An exception raised from these modules is treated as transient infrastructure
# failure rather than a result. Schema failures never reach here: an agent that
# exhausts its retries returns terminal_state ERROR without raising, and so
# writes no error.json.
API_ERROR_MODULES = ("anthropic", "httpx", "httpcore")
API_ERROR_CLASSES = ("TimeoutError", "ConnectionError", "ConnectionResetError")


class Cell(NamedTuple):
    """One point in the case x variant x replicate design."""

    case_id: str
    variant: str
    replicate: int


class Attempt(NamedTuple):
    """What a previous run of a cell recorded on disk."""

    run_dir: Path
    terminal_state: Optional[str]
    error_kind: Optional[str]  # "api" | "validation" | None


# ---------------------------------------------------------------------------
# Pre-flight — everything that can fail before an API call is spent
# ---------------------------------------------------------------------------


def preflight_git(allow_dirty: bool) -> str:
    """Resolve the commit once, aborting the batch if the tree is dirty.

    `RunLogger` performs this check per run. Left to it, a dirty tree would
    produce 165 individually-caught failures, no API calls, and a batch that
    looks like it completed. Failing here instead costs two seconds.
    """
    return _git_commit_sha(allow_dirty)


def preflight_cases(cases_dir: Path, case_ids: Iterable[str]) -> dict[str, Path]:
    """Validate every targeted case file before the batch starts.

    A typo must fail at second two, not at run 47 with an opaque KeyError.
    """
    resolved: dict[str, Path] = {}
    problems: list[str] = []

    for case_id in case_ids:
        path = cases_dir / f"{case_id}.json"
        if not path.exists():
            problems.append(f"{case_id}: no such file {path}")
            continue
        try:
            case = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            problems.append(f"{case_id}: unparseable JSON — {exc}")
            continue
        for key in ("case_id", "brief"):
            if not case.get(key):
                problems.append(f"{case_id}: missing or empty '{key}'")
        if case.get("case_id") != case_id:
            problems.append(
                f"{case_id}: case_id {case.get('case_id')!r} does not match filename"
            )
        resolved[case_id] = path

    if problems:
        raise SystemExit(
            "case pre-validation failed; no runs attempted:\n  "
            + "\n  ".join(problems)
        )
    return resolved


def preflight_configs(variant_configs: dict[str, Path]) -> None:
    """Check each config exists and declares the variant it is filed under.

    The manifest records the variant *label* passed to `RunLogger`, while
    behaviour follows `config["variant"]`. A config filed as one and declaring
    the other would produce data that is wrong in a way nothing else notices.
    """
    problems: list[str] = []
    for variant, path in variant_configs.items():
        if not path.exists():
            problems.append(f"{variant}: no such config {path}")
            continue
        declared = load_config(path).get("variant")
        if declared != variant:
            problems.append(
                f"{variant}: {path} declares variant {declared!r}, expected {variant!r}"
            )
    if problems:
        raise SystemExit(
            "config pre-validation failed; no runs attempted:\n  " + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------
# Resume — what has already been done, and what may be retried
# ---------------------------------------------------------------------------


def classify_error(run_dir: Path) -> Optional[str]:
    """Classify an ERROR run as transient ("api") or a real result ("validation").

    An `error.json` means an exception escaped the graph; its module says
    whether that was infrastructure. No `error.json` means an agent exhausted
    its schema retries and returned ERROR without raising — a genuine outcome.
    """
    error_path = run_dir / "error.json"
    if not error_path.exists():
        return "validation"
    try:
        record = json.loads(error_path.read_text(encoding="utf-8"))
    except Exception:
        return "validation"

    module = str(record.get("exception_module", ""))
    klass = str(record.get("exception_class", ""))
    if module.split(".")[0] in API_ERROR_MODULES or klass in API_ERROR_CLASSES:
        return "api"
    return "validation"


def scan_previous_runs(runs_dir: Path) -> dict[Cell, list[Attempt]]:
    """Read every manifest under `runs/` and index it by cell."""
    history: dict[Cell, list[Attempt]] = {}
    if not runs_dir.exists():
        return history

    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cell = Cell(
            manifest.get("case_id"),
            manifest.get("variant"),
            manifest.get("replicate_id"),
        )
        run_dir = manifest_path.parent
        terminal = manifest.get("terminal_state")
        kind = classify_error(run_dir) if terminal == "ERROR" else None
        history.setdefault(cell, []).append(Attempt(run_dir, terminal, kind))
    return history


def scan_batch_errors(errors_path: Path) -> dict[Cell, list[Attempt]]:
    """Index failures that never produced a manifest.

    A run that dies before `RunLogger` is constructed — a dirty tree, a bad
    config, an error escaping the entry point — leaves no run directory, so
    `scan_previous_runs` cannot see it. Without this, such a cell looks
    unattempted on every resume and the retry cap never applies to the very
    failures most likely to recur.
    """
    history: dict[Cell, list[Attempt]] = {}
    if not errors_path.exists():
        return history

    for line in errors_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        cell = Cell(
            record.get("case_id"), record.get("variant"), record.get("replicate")
        )
        module = str(record.get("exception_module", "")).split(".")[0]
        klass = str(record.get("exception_class", ""))
        kind = (
            "api"
            if module in API_ERROR_MODULES or klass in API_ERROR_CLASSES
            else "batch"
        )
        history.setdefault(cell, []).append(Attempt(None, "ERROR", kind))
    return history


def merge_history(
    *sources: dict[Cell, list[Attempt]],
) -> dict[Cell, list[Attempt]]:
    """Combine per-cell attempt lists from several sources."""
    merged: dict[Cell, list[Attempt]] = {}
    for source in sources:
        for cell, attempts in source.items():
            merged.setdefault(cell, []).extend(attempts)
    return merged


def should_run(attempts: list[Attempt]) -> tuple[bool, str]:
    """Decide whether a cell still needs running, and say why.

    - never attempted            -> run
    - any non-ERROR outcome      -> done, including GATE_NO and ESCALATE
    - a validation ERROR         -> done; the retries were exhausted, that is
                                    the result and re-running would overwrite it
    - only API-class ERRORs      -> retry, up to MAX_API_ATTEMPTS, so a
                                    transient overload cannot silently punch a
                                    hole in the design
    """
    if not attempts:
        return True, "not yet run"
    if any(a.terminal_state != "ERROR" for a in attempts):
        return False, "already completed"
    if any(a.error_kind == "validation" for a in attempts):
        return False, "completed with a validation ERROR (a real result)"

    batch_attempts = sum(1 for a in attempts if a.error_kind == "batch")
    if batch_attempts >= MAX_API_ATTEMPTS:
        return False, f"batch-level failure {batch_attempts}x, at the cap"

    api_attempts = sum(1 for a in attempts if a.error_kind in ("api", "batch"))
    if api_attempts >= MAX_API_ATTEMPTS:
        return False, f"API-class failure {api_attempts}x, at the cap"
    return True, f"retrying after API-class failure ({api_attempts}/{MAX_API_ATTEMPTS})"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def summary_row(cell: Cell, run_dir: Optional[Path], **extra: Any) -> dict[str, Any]:
    """Build one summary line, reading the manifest the run wrote."""
    row: dict[str, Any] = {
        "case_id": cell.case_id,
        "variant": cell.variant,
        "replicate": cell.replicate,
        "run_dir": str(run_dir) if run_dir else None,
        "terminal_state": None,
        "selected_locus": None,
        "gate_decision": None,
        "visibility_band": None,
        "intensity_band": None,
        "critic_iterations": None,
        "total_input_tokens": None,
        "total_output_tokens": None,
        "est_cost_usd": None,
        "wall_clock_seconds": None,
        "git_commit_sha": None,
        "error_kind": None,
        "error_class": None,
        "error_message": None,
    }
    if run_dir is not None:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in (
                "terminal_state",
                "selected_locus",
                "gate_decision",
                "visibility_band",
                "intensity_band",
                "critic_iterations",
                "total_input_tokens",
                "total_output_tokens",
                "est_cost_usd",
                "wall_clock_seconds",
                "git_commit_sha",
            ):
                row[key] = manifest.get(key)
            # The label the run recorded must match the variant it was filed
            # under, or the data says one thing and the behaviour was another.
            if manifest.get("variant") != cell.variant:
                row["error_kind"] = "label_mismatch"
                row["error_message"] = (
                    f"manifest variant {manifest.get('variant')!r} != {cell.variant!r}"
                )
    row.update(extra)
    return row


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one record and flush it, so a killed batch keeps what it had."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


async def execute(cell: Cell, case_path: Path, args: Namespace) -> Path:
    """Dispatch one cell to the right entry point and return its run directory.

    `baseline` does not go through the graph, so it runs via `run_baseline`;
    both entry points take the same Namespace and return the run directory.

    `allow_dirty=True` is passed unconditionally, and only from here. By the
    time this runs, `preflight_git` has already confirmed the tree was clean
    at batch launch — so the rubric and code the runs execute against are
    pinned. Leaving the per-run guard live would then re-check the tree after
    every run and abort the remainder the moment the batch's own output made
    it dirty, which is what cost Campaign 1 its first 164 cells. A standalone
    run has no such pre-flight, so its guard stays live: this is not a change
    to the default.
    """
    runner = run_baseline.run if cell.variant == "baseline" else run_single.run
    return await runner(
        Namespace(
            case=case_path,
            variant=cell.variant,
            replicate=cell.replicate,
            config=VARIANT_CONFIGS[cell.variant],
            runs_dir=args.runs_dir,
            allow_dirty=True,
        )
    )


async def run_batch(args: Namespace) -> int:
    """Run every outstanding cell. Returns the number of failures recorded."""
    cases = preflight_cases(args.cases_dir, args.case_ids)
    variant_configs = {v: VARIANT_CONFIGS[v] for v in args.variants}
    preflight_configs(variant_configs)
    commit = preflight_git(args.allow_dirty)

    cells = [
        Cell(case_id, variant, replicate)
        for case_id in args.case_ids
        for variant in args.variants
        for replicate in args.replicates
    ]
    history = merge_history(
        scan_previous_runs(args.runs_dir), scan_batch_errors(args.errors)
    )

    planned: list[tuple[Cell, str]] = []
    skipped: list[tuple[Cell, str]] = []
    for cell in cells:
        go, why = should_run(history.get(cell, []))
        (planned if go else skipped).append((cell, why))

    print(f"commit          {commit}")
    print(f"design          {len(args.case_ids)} cases x {len(args.variants)} variants "
          f"x {len(args.replicates)} replicates = {len(cells)} runs")
    print(f"already done    {len(skipped)}")
    print(f"to run          {len(planned)}")
    print(f"summary         {args.summary}")
    if args.dry_run:
        for cell, why in planned[:20]:
            print(f"  would run {cell.case_id} {cell.variant} r{cell.replicate}  ({why})")
        if len(planned) > 20:
            print(f"  ... and {len(planned) - 20} more")
        return 0

    total_cost = 0.0
    failures = 0
    started = time.monotonic()

    for index, (cell, why) in enumerate(planned, start=1):
        label = f"[{index}/{len(planned)}] {cell.case_id} {cell.variant} r{cell.replicate}"
        print(f"\n{label}  ({why})", flush=True)

        try:
            run_dir = await execute(cell, cases[cell.case_id], args)
            row = summary_row(cell, run_dir)
        except Exception as exc:
            # The run never got far enough to write its own manifest — a
            # RunLogger construction failure, or an error escaping the entry
            # point. Record it here and carry on; one cell must never end the
            # batch.
            failures += 1
            module = type(exc).__module__.split(".")[0]
            kind = (
                "api"
                if module in API_ERROR_MODULES or type(exc).__name__ in API_ERROR_CLASSES
                else "batch"
            )
            row = summary_row(
                cell,
                None,
                terminal_state="ERROR",
                error_kind=kind,
                error_class=type(exc).__name__,
                error_message=str(exc)[:500],
            )
            append_jsonl(
                args.errors,
                {
                    "case_id": cell.case_id,
                    "variant": cell.variant,
                    "replicate": cell.replicate,
                    "exception_class": type(exc).__name__,
                    "exception_module": type(exc).__module__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"    FAILED ({kind}) {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        else:
            if row["terminal_state"] == "ERROR":
                failures += 1
                row["error_kind"] = classify_error(run_dir)
            print(
                f"    {row['terminal_state']}  locus={row['selected_locus']}  "
                f"${row['est_cost_usd']}  {row['wall_clock_seconds']}s",
                flush=True,
            )

        append_jsonl(args.summary, row)
        total_cost += row.get("est_cost_usd") or 0.0
        elapsed = time.monotonic() - started
        print(
            f"    running total  ${total_cost:.4f}  |  {elapsed / 60:.1f} min  "
            f"|  {failures} failed",
            flush=True,
        )

    print(f"\nbatch complete: {len(planned)} attempted, {failures} failed, "
          f"${total_cost:.4f} spent")
    return failures


def parse_args(argv: list[str] | None = None) -> Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Run the LUXCAL experimental batch.")
    parser.add_argument("--cases-dir", type=Path, default=Path("data/cases"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "--summary", type=Path, default=Path("analysis/batch_summary.jsonl")
    )
    parser.add_argument(
        "--errors", type=Path, default=Path("analysis/batch_errors.jsonl")
    )
    parser.add_argument("--case-ids", nargs="+", default=CASE_IDS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_CONFIGS))
    parser.add_argument("--replicates", nargs="+", type=int, default=REPLICATES)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--dry-run", action="store_true", help="Plan only; make no API calls."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Non-zero exit if any run failed."""
    load_dotenv()
    return 1 if asyncio.run(run_batch(parse_args(argv))) else 0


if __name__ == "__main__":
    sys.exit(main())
