"""Collect completed runs into one analysable frame, and eyeball the output.

    python -m scripts.aggregate              # summary + parquet
    python -m scripts.aggregate --concepts   # also print sampled concept text

One row per run, read from each run's `manifest.json` — richer than the batch
summary JSONL, and the artefact SPEC §7.4 says the analysis frame is built
from. No KPIs, deltas or rates are computed here: this stage exists to confirm
the data is sound before anything is written on top of it.

The distributions are computed with the standard library so that a missing
analysis dependency cannot stop you inspecting the campaign; only the parquet
write needs pandas.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

# Campaign 1: the eleven experimental briefs. Cases 001-003 are development
# fixtures whose runs share the same runs/ directory, and are excluded by
# case_id rather than by commit — one legitimate cell was run at an earlier
# commit than the rest of the batch, and a commit filter would silently drop
# it. `git_commit_sha` is carried into the frame so that difference stays
# auditable.
CAMPAIGN_CASES: list[str] = [f"case_{n:03d}" for n in range(4, 15)]

VARIANTS = ["full", "minus_critic", "minus_loop", "llm_bands", "baseline"]

SCALAR_FIELDS = [
    "run_id",
    "timestamp_utc",
    "case_id",
    "variant",
    "replicate_id",
    "git_commit_sha",
    "rubric_version",
    "rubric_sha256",
    "model_generation",
    "model_judge",
    "gate_decision",
    "visibility_band",
    "intensity_band",
    "selected_locus",
    "critic_iterations",
    "terminal_state",
    "total_input_tokens",
    "total_output_tokens",
    "est_cost_usd",
    "wall_clock_seconds",
]


def collect_rows(runs_dir: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    """Read every manifest under `runs_dir`, keeping the campaign cases."""
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("case_id") not in case_ids:
            continue
        row = {field: manifest.get(field) for field in SCALAR_FIELDS}
        row["n_excluded_loci"] = len(manifest.get("excluded_loci") or [])
        row["excluded_loci"] = ";".join(manifest.get("excluded_loci") or [])
        row["unpriced_models"] = ";".join(manifest.get("unpriced_models") or [])
        row["gate_rationale"] = manifest.get("gate_rationale")
        row["run_dir"] = str(manifest_path.parent)
        rows.append(row)
    return rows


def dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce to exactly one row per (case_id, variant, replicate).

    A cell can hold more than one run only if it was retried, which happens
    only after a failure. The rule is therefore: prefer a real result over an
    ERROR, and among equals take the most recent by timestamp. Collapsing is
    reported loudly — on a clean campaign this should collapse nothing, and
    silence about a discarded row would hide a retry from the analysis.
    """
    by_cell: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["case_id"], row["variant"], row["replicate_id"])].append(row)

    kept: list[dict[str, Any]] = []
    collapsed: list[tuple] = []
    for cell, candidates in sorted(by_cell.items()):
        if len(candidates) > 1:
            collapsed.append((cell, len(candidates)))
        best = sorted(
            candidates,
            key=lambda r: (r["terminal_state"] != "ERROR", r["timestamp_utc"] or ""),
        )[-1]
        kept.append(best)

    if collapsed:
        print("\n!! DEDUP COLLAPSED CELLS — these cells had more than one run:")
        for cell, n in collapsed:
            print(f"     {cell[0]} {cell[1]} r{cell[2]}: {n} runs -> kept 1")
        print("   Inspect these before trusting the frame.\n")
    else:
        print("dedup: no cell had more than one run; nothing collapsed")
    return kept


def _counts(rows: list[dict], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r[field]) for r in rows).items()))


def print_summary(rows: list[dict[str, Any]], expected: int) -> None:
    """Print raw distributions only — no rates, deltas or derived measures."""
    line = "-" * 78

    print(f"\n{line}\nROW COUNT\n{line}")
    print(f"  rows after dedup : {len(rows)}")
    print(f"  expected         : {expected}")
    if len(rows) != expected:
        print(f"  !! GAP OF {expected - len(rows)} ROWS")

    print(f"\n{line}\nTERMINAL STATE\n{line}")
    for value, n in _counts(rows, "terminal_state").items():
        print(f"  {value:<12} {n}")

    print(f"\n{line}\nGATE DECISION\n{line}")
    for value, n in _counts(rows, "gate_decision").items():
        print(f"  {value:<12} {n}")

    print(f"\n{line}\nSELECTED LOCUS (all runs)\n{line}")
    for value, n in sorted(
        Counter(str(r["selected_locus"]) for r in rows).items(),
        key=lambda kv: (-kv[1], kv[0]),
    ):
        print(f"  {value:<16} {n}")

    print(f"\n{line}\nBANDS\n{line}")
    for field in ("visibility_band", "intensity_band"):
        print(f"  {field}:")
        for value, n in _counts(rows, field).items():
            print(f"     {value:<10} {n}")

    print(f"\n{line}\nROWS PER VARIANT\n{line}")
    for value, n in _counts(rows, "variant").items():
        flag = "" if n == expected // len(VARIANTS) else "   <-- unexpected"
        print(f"  {value:<14} {n}{flag}")

    # case x locus crosstab
    print(f"\n{line}\nCASE x SELECTED_LOCUS\n{line}")
    loci = sorted({str(r["selected_locus"]) for r in rows})
    grid: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        grid[row["case_id"]][str(row["selected_locus"])] += 1
    width = max(len(x) for x in loci) if loci else 10
    header = "  " + "case".ljust(10) + "".join(x[:width].rjust(width + 2) for x in loci)
    print(header)
    for case_id in sorted(grid):
        cells = "".join(
            (str(grid[case_id][x]) if grid[case_id][x] else ".").rjust(width + 2)
            for x in loci
        )
        print(f"  {case_id:<10}{cells}")

    print(f"\n{line}\nNULLS BY VARIANT\n{line}")
    fields = ["selected_locus", "terminal_state", "visibility_band", "intensity_band"]
    print("  " + "variant".ljust(14) + "".join(f.rjust(18) for f in fields))
    for variant in sorted({r["variant"] for r in rows}):
        subset = [r for r in rows if r["variant"] == variant]
        counts = "".join(
            str(sum(1 for r in subset if r[f] is None)).rjust(18) for f in fields
        )
        print(f"  {variant:<14}{counts}")

    for label, state in (("GATE_NO", "GATE_NO"), ("ESCALATE", "ESCALATE"), ("ERROR", "ERROR")):
        hits = [r for r in rows if r["terminal_state"] == state]
        if not hits:
            continue
        print(f"\n{line}\n{label} CELLS ({len(hits)})\n{line}")
        for row in sorted(hits, key=lambda r: (r["case_id"], r["variant"], r["replicate_id"])):
            print(f"  {row['case_id']}  {row['variant']:<14} r{row['replicate_id']}  "
                  f"locus={row['selected_locus']}  bands=({row['visibility_band']},"
                  f"{row['intensity_band']})")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> bool:
    """Write the frame, or explain precisely why it could not be written."""
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError:
        print(
            f"\n!! PARQUET NOT WRITTEN — pandas is not installed.\n"
            f"   SPEC §8 lists pandas in the analysis stack but requirements.txt\n"
            f"   does not pin it, and neither pandas nor pyarrow is in the venv.\n"
            f"   The summary above is complete and needs no dependency.\n"
            f"   To enable {path}:  uv pip install pandas pyarrow  (then re-run)"
        )
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    try:
        frame.to_parquet(path, index=False)
    except ImportError:
        print(
            f"\n!! PARQUET NOT WRITTEN — pandas is present but no parquet engine is.\n"
            f"   uv pip install pyarrow  (then re-run)"
        )
        return False
    print(f"\nwrote {path}  ({len(frame)} rows x {len(frame.columns)} columns)")
    return True


# A purposive sample for the eyeball step, not a random one: three different
# brands under `full` to see whether concepts actually differ by brand, the
# unconstrained baseline floor, an unchecked minus_critic first draft, the one
# capped-out ESCALATE, and one refusal to read the gate's reasoning.
CONCEPT_SAMPLE: list[tuple[str, str, int, str]] = [
    ("case_004", "full", 0, "hospitality brand under the full system"),
    ("case_005", "full", 0, "watchmaker under the full system"),
    ("case_009", "full", 0, "automotive brand under the full system"),
    ("case_004", "baseline", 0, "the unconstrained floor, same brand as above"),
    ("case_005", "minus_critic", 0, "unchecked first draft, same brand as above"),
    ("case_009", "minus_loop", 2, "the one ESCALATE: a capped-out loop"),
    ("case_010", "full", 0, "a refusal: no concept, read the gate rationale"),
]


def brand_names(cases_dir: Path) -> dict[str, str]:
    """Map case_id to the brand named in the case file, for labelling only."""
    names: dict[str, str] = {}
    for path in sorted(cases_dir.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8-sig"))
        names[case["case_id"]] = case.get("brand_name", "(no brand_name field)")
    return names


def print_concepts(
    rows: list[dict[str, Any]],
    sample: list[tuple[str, str, int, str]],
    cases_dir: Path,
) -> None:
    """Print the verbatim concept for each sampled run.

    The concept text is not in the manifest. It lives in the run's
    `output.json` under `concept` — the final concept after any revision — and
    per-iteration in `states/ideate.json` / `states/ideate_<n>.json`. The
    `Concept` schema has no single description field: `touchpoint` and
    `mechanism` are the prose.
    """
    brands = brand_names(cases_dir)
    index = {(r["case_id"], r["variant"], r["replicate_id"]): r for r in rows}
    rule = "=" * 78

    for case_id, variant, replicate, why in sample:
        row = index.get((case_id, variant, replicate))
        print(f"\n{rule}")
        print(f"{case_id}  {variant}  r{replicate}   [{why}]")
        print(f"brand: {brands.get(case_id, '?')}")
        if row is None:
            print("  !! no such run in the frame")
            continue
        print(f"terminal_state={row['terminal_state']}  "
              f"ceilings=({row['visibility_band']},{row['intensity_band']})  "
              f"critic_iterations={row['critic_iterations']}")
        print(rule)

        output = json.loads(
            (Path(row["run_dir"]) / "output.json").read_text(encoding="utf-8")
        )
        concept = output.get("concept")

        if concept is None:
            calibration = output.get("calibration") or {}
            print("  no concept — the gate refused.\n")
            print(f"  gate_decision : {calibration.get('gate_decision')}")
            print("  gate_rationale:")
            print(f"    {calibration.get('gate_rationale')}")
            continue

        print(f"  name                 : {concept['name']}")
        print(f"  locus                : {concept['locus']}")
        print(f"  ai_position          : {concept['ai_position']}")
        print(f"  differentiation_unit : {concept['differentiation_unit']}")
        print(f"  claimed_visibility   : {concept['claimed_visibility']}")
        print(f"  claimed_intensity    : {concept['claimed_intensity']}")
        print(f"  evidence_ids         : {concept['evidence_ids']}")
        print("\n  touchpoint:")
        print(f"    {concept['touchpoint']}")
        print("\n  mechanism:")
        print(f"    {concept['mechanism']}")

        verdict = output.get("verdict")
        if verdict:
            print(f"\n  critic verdict: {verdict['verdict']}  "
                  f"misdeclared={verdict['misdeclared']}  "
                  f"violations={len(verdict['violations'])}")
            if verdict.get("misdeclaration_rationale"):
                print(f"    {verdict['misdeclaration_rationale']}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Aggregate completed LUXCAL runs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("analysis/runs.parquet"))
    parser.add_argument("--case-ids", nargs="+", default=CAMPAIGN_CASES)
    parser.add_argument("--expected", type=int, default=165)
    parser.add_argument("--cases-dir", type=Path, default=Path("data/cases"))
    parser.add_argument(
        "--concepts", action="store_true", help="Print sampled concept text."
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    rows = collect_rows(args.runs_dir, args.case_ids)
    print(f"collected {len(rows)} manifests for cases "
          f"{args.case_ids[0]}..{args.case_ids[-1]}")
    rows = dedup(rows)
    print_summary(rows, args.expected)
    write_parquet(rows, args.out)
    if args.concepts:
        print_concepts(rows, CONCEPT_SAMPLE, args.cases_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
