"""Concept-level extraction and a sameness diagnostic.

    python -m scripts.concepts              # write analysis/concepts.csv + diagnose
    python -m scripts.concepts --no-write   # diagnose only

The concept text is not in the manifest, so it is not in `runs.parquet`. It
lives in each run's `output.json` under `concept`. This module lifts it into
one row per (case_id, variant, replicate) and then asks how often two runs
produced the *same* concept — within a variant across replicates, and across
variants for the same case.

Identity is exact equality of `(name, mechanism)`, the two fields that carry
the substance. A run with no concept (a refusal) is never counted as matching
anything; those cells are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

CAMPAIGN_CASES: list[str] = [f"case_{n:03d}" for n in range(4, 15)]
VARIANTS = ["full", "minus_critic", "minus_loop", "llm_bands", "baseline"]

CONCEPT_FIELDS = [
    "name",
    "ai_position",
    "differentiation_unit",
    "claimed_visibility",
    "claimed_intensity",
    "touchpoint",
    "mechanism",
]

COLUMNS = [
    "case_id",
    "variant",
    "replicate",
    "terminal_state",
    "locus",
    *CONCEPT_FIELDS,
    "evidence_ids",
    "gate_rationale",
    "run_dir",
]


def collect(runs_dir: Path, case_ids: list[str]) -> list[dict[str, Any]]:
    """One row per run, with the concept fields lifted out of output.json.

    A run without a concept gets an explicit marker in every concept field
    rather than a blank, so an empty cell in the CSV always means a genuine
    missing value and never "this run had nothing to produce".
    """
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(runs_dir.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("case_id") not in case_ids:
            continue

        run_dir = manifest_path.parent
        output = json.loads((run_dir / "output.json").read_text(encoding="utf-8"))
        concept = output.get("concept")
        calibration = output.get("calibration") or {}
        terminal = manifest.get("terminal_state")

        row: dict[str, Any] = {
            "case_id": manifest["case_id"],
            "variant": manifest["variant"],
            "replicate": manifest["replicate_id"],
            "terminal_state": terminal,
            "run_dir": str(run_dir),
            "gate_rationale": calibration.get("gate_rationale") or "",
        }
        if concept is None:
            marker = f"<no concept: {terminal}>"
            row["locus"] = marker
            row["evidence_ids"] = marker
            for field in CONCEPT_FIELDS:
                row[field] = marker
        else:
            row["locus"] = concept["locus"]
            row["evidence_ids"] = ";".join(concept.get("evidence_ids") or [])
            for field in CONCEPT_FIELDS:
                row[field] = concept[field]
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the concept table, quoted so embedded newlines survive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows x {len(COLUMNS)} columns)")


def key(row: dict[str, Any]) -> Optional[tuple[str, str]]:
    """The identity of a concept: exact `(name, mechanism)`. None if absent."""
    if str(row["name"]).startswith("<no concept"):
        return None
    return (row["name"], row["mechanism"])


def diagnose(rows: list[dict[str, Any]]) -> None:
    """Report how often runs produced the same concept."""
    rule = "-" * 78
    index: dict[tuple, dict[str, Any]] = {
        (r["case_id"], r["variant"], r["replicate"]): r for r in rows
    }
    cases = sorted({r["case_id"] for r in rows})
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]

    # ---- within-variant: are the three replicates the same? ----------------
    print(f"\n{rule}\nWITHIN-VARIANT: are the 3 replicates identical?\n{rule}")
    print("  (temperature 0.7 at ideation is the only intended source of variation)")
    stable: Counter = Counter()
    varying: Counter = Counter()
    no_concept_cells: Counter = Counter()

    for case_id in cases:
        for variant in variants:
            keys = [
                key(index[(case_id, variant, r)])
                for r in (0, 1, 2)
                if (case_id, variant, r) in index
            ]
            if any(k is None for k in keys):
                no_concept_cells[variant] += 1
                continue
            (stable if len(set(keys)) == 1 else varying)[variant] += 1

    print(f"\n  {'variant':<14}{'all 3 same':>12}{'they vary':>12}{'no concept':>13}")
    for variant in variants:
        print(f"  {variant:<14}{stable[variant]:>12}{varying[variant]:>12}"
              f"{no_concept_cells[variant]:>13}")
    print(f"  {'TOTAL':<14}{sum(stable.values()):>12}{sum(varying.values()):>12}"
          f"{sum(no_concept_cells.values()):>13}")

    # ---- across-variant, per case ------------------------------------------
    print(f"\n{rule}\nACROSS-VARIANT, PER CASE (replicate 0)\n{rule}")
    print("  variants sharing a letter produced the identical concept\n")
    for case_id in cases:
        groups: dict[Optional[tuple], list[str]] = defaultdict(list)
        for variant in variants:
            row = index.get((case_id, variant, 0))
            if row is None:
                continue
            groups[key(row)].append(variant)
        labels: dict[str, str] = {}
        for i, (k, members) in enumerate(sorted(groups.items(), key=lambda kv: str(kv[0]))):
            letter = "-" if k is None else chr(ord("A") + i)
            for m in members:
                labels[m] = letter
        rendered = "  ".join(f"{v}={labels.get(v, '?')}" for v in variants)
        distinct = len({v for v in labels.values() if v != "-"})
        print(f"  {case_id}   {rendered}    ({distinct} distinct concept(s))")

    # ---- pairwise across all cases and replicates --------------------------
    print(f"\n{rule}\nPAIRWISE: same vs different concept, across all case x replicate\n{rule}")
    print(f"  {'pair':<30}{'SAME':>7}{'DIFFERENT':>11}{'n/a':>6}")
    flagged: list[str] = []
    for a, b in combinations(variants, 2):
        same = different = missing = 0
        for case_id in cases:
            for replicate in (0, 1, 2):
                ra = index.get((case_id, a, replicate))
                rb = index.get((case_id, b, replicate))
                if ra is None or rb is None:
                    missing += 1
                    continue
                ka, kb = key(ra), key(rb)
                if ka is None or kb is None:
                    missing += 1
                elif ka == kb:
                    same += 1
                else:
                    different += 1
        pair = f"{a} | {b}"
        mark = ""
        if same and {a, b} in ({"baseline", "full"}, {"llm_bands", "full"}):
            mark = "   <-- WOULD BE A BUG"
            flagged.append(f"{pair}: {same} identical")
        print(f"  {pair:<30}{same:>7}{different:>11}{missing:>6}{mark}")

    print(f"\n{rule}\nBUG FLAGS\n{rule}")
    for label, pair in (
        ("baseline == full", ("baseline", "full")),
        ("llm_bands == full", ("llm_bands", "full")),
    ):
        n = 0
        for case_id in cases:
            for replicate in (0, 1, 2):
                ra, rb = index.get((case_id, pair[0], replicate)), index.get(
                    (case_id, pair[1], replicate)
                )
                if ra and rb and key(ra) and key(ra) == key(rb):
                    n += 1
        verdict = "CLEAN" if n == 0 else f"{n} IDENTICAL — INVESTIGATE"
        print(f"  {label:<22} {verdict}")


def structural_key(row: dict[str, Any]) -> Optional[tuple]:
    """The concept's declared shape, ignoring its prose.

    Two concepts can be worded entirely differently and still occupy the same
    cell of the design space. Exact text identity answers "did the pipeline
    duplicate itself"; this answers "did the variants actually land anywhere
    different", which is what an ablation delta rests on.
    """
    if str(row["name"]).startswith("<no concept"):
        return None
    return (
        row["locus"],
        row["ai_position"],
        row["differentiation_unit"],
        row["claimed_visibility"],
        row["claimed_intensity"],
    )


def diagnose_structure(rows: list[dict[str, Any]]) -> None:
    """Same comparisons again, on the structured fields rather than the text."""
    rule = "-" * 78
    index = {(r["case_id"], r["variant"], r["replicate"]): r for r in rows}
    cases = sorted({r["case_id"] for r in rows})
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]

    print(f"\n{rule}\nSTRUCTURED FIELDS: locus + position + unit + claimed bands\n{rule}")
    print("  Text differs everywhere; this asks whether the declared shape does.\n")

    print(f"  {'pair':<30}{'SAME':>7}{'DIFFERENT':>11}{'n/a':>6}")
    for a, b in combinations(variants, 2):
        same = different = missing = 0
        for case_id in cases:
            for replicate in (0, 1, 2):
                ra, rb = index.get((case_id, a, replicate)), index.get(
                    (case_id, b, replicate)
                )
                ka = structural_key(ra) if ra else None
                kb = structural_key(rb) if rb else None
                if ka is None or kb is None:
                    missing += 1
                elif ka == kb:
                    same += 1
                else:
                    different += 1
        print(f"  {a + ' | ' + b:<30}{same:>7}{different:>11}{missing:>6}")

    print(f"\n  distinct structured shapes per case (across all 15 runs):")
    for case_id in cases:
        shapes = {
            structural_key(index[(case_id, v, r)])
            for v in variants
            for r in (0, 1, 2)
            if (case_id, v, r) in index
        }
        shapes.discard(None)
        print(f"    {case_id}  {len(shapes)} distinct shape(s) from 15 runs")

    print(f"\n  within-variant structured stability (all 3 replicates same shape):")
    print(f"  {'variant':<14}{'all 3 same':>12}{'they vary':>12}")
    for variant in variants:
        stable = varying = 0
        for case_id in cases:
            keys = [
                structural_key(index[(case_id, variant, r)])
                for r in (0, 1, 2)
                if (case_id, variant, r) in index
            ]
            if any(k is None for k in keys):
                continue
            (stable, varying) = (
                (stable + 1, varying) if len(set(keys)) == 1 else (stable, varying + 1)
            )
        print(f"  {variant:<14}{stable:>12}{varying:>12}")


def show_examples(rows: list[dict[str, Any]], n_same: int, n_diff: int) -> None:
    """Print full text for a case where variants agree and one where they differ."""
    rule = "=" * 78
    index = {(r["case_id"], r["variant"], r["replicate"]): r for r in rows}
    cases = sorted({r["case_id"] for r in rows})

    agree, differ = [], []
    for case_id in cases:
        keys = {
            v: key(index[(case_id, v, 0)])
            for v in VARIANTS
            if (case_id, v, 0) in index
        }
        present = {v: k for v, k in keys.items() if k is not None}
        if len(present) < 2:
            continue
        (agree if len(set(present.values())) == 1 else differ).append(case_id)

    def dump(case_id: str, heading: str) -> None:
        print(f"\n{rule}\n{heading}: {case_id}\n{rule}")
        for variant in VARIANTS:
            row = index.get((case_id, variant, 0))
            if row is None:
                continue
            print(f"\n--- {variant} (r0, {row['terminal_state']}) ---")
            if key(row) is None:
                print(f"  {row['name']}")
                if row["gate_rationale"]:
                    print(f"  gate_rationale: {row['gate_rationale'][:300]}")
                continue
            print(f"  name      : {row['name']}")
            print(f"  locus     : {row['locus']}   "
                  f"position={row['ai_position']}  unit={row['differentiation_unit']}  "
                  f"claimed=({row['claimed_visibility']},{row['claimed_intensity']})")
            print(f"  touchpoint: {row['touchpoint']}")
            print(f"  mechanism : {row['mechanism']}")

    for case_id in agree[:n_same]:
        dump(case_id, "VARIANTS PRODUCED THE IDENTICAL CONCEPT")
    for case_id in differ[:n_diff]:
        dump(case_id, "VARIANTS PRODUCED GENUINELY DIFFERENT CONCEPTS")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Extract and compare run concepts.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("analysis/concepts.csv"))
    parser.add_argument("--case-ids", nargs="+", default=CAMPAIGN_CASES)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--examples-same", type=int, default=1)
    parser.add_argument("--examples-diff", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    rows = collect(args.runs_dir, args.case_ids)
    print(f"collected {len(rows)} runs")
    if not args.no_write:
        write_csv(rows, args.out)
    diagnose(rows)
    diagnose_structure(rows)
    show_examples(rows, args.examples_same, args.examples_diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
