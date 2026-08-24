"""Campaign-1 KPIs, grouped by research question.

    python -m scripts.kpi

Reads `analysis/runs.parquet` to enumerate the runs and each run's
`output.json` for the detail the manifest does not carry — the Critic's
verdict, the profile's per-dimension provenance, the ranked and excluded loci.

Writes `analysis/kpis.json` and prints an RQ-grouped report. Every rate carries
its denominator, and every caveat is printed inline with the number rather than
collected at the end, so a figure cannot be quoted without it.

Nothing here is a significance test. Several measures rest on one or two
observations and say so.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from luxcal.core.ceilings import BAND_ORDER
from luxcal.core.locus import AI_POSITION_VISIBILITY, LOCUS_AI_POSITION, LOCUS_GRID

VARIANTS = ["full", "minus_critic", "minus_loop", "llm_bands", "baseline"]

# Variants whose pipeline includes the Critic. Only these can report a
# misdeclaration rate; the others record no verdict at all.
CRITIC_VARIANTS = ["full", "minus_loop", "llm_bands"]

DIMENSIONS = [
    ("D1", "d1_rarity"),
    ("D2", "d2_invisibility"),
    ("D3", "d3_value_orientation"),
    ("D4", "d4_motivation"),
    ("D5", "d5_orchestration"),
]

EVIDENCE_CAVEAT = (
    "CONTAMINATED: inflated by the known evidence-attribution defect "
    "(spans are verbatim but not probative); corrected in campaign 2"
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_runs(parquet: Path) -> list[dict[str, Any]]:
    """One record per run: the manifest row plus its output.json detail."""
    frame = pd.read_parquet(parquet)
    runs: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        output = json.loads(
            (Path(row["run_dir"]) / "output.json").read_text(encoding="utf-8")
        )
        row["profile"] = output.get("profile")
        row["calibration"] = output.get("calibration")
        row["concept"] = output.get("concept")
        row["verdict"] = output.get("verdict")
        row["critic_history"] = output.get("critic_history") or []
        runs.append(row)
    return runs


def present(value: Any) -> bool:
    """Whether a parquet Optional actually holds a value.

    pandas renders a missing string as float('nan'), which is *truthy* — so a
    bare truth test silently counts absent values as present and inflates both
    the numerator and the denominator of any rate built on one.
    """
    return value is not None and not (isinstance(value, float) and pd.isna(value))


def rate(numerator: int, denominator: int) -> dict[str, Any]:
    """A rate that always carries its denominator."""
    return {
        "n": numerator,
        "of": denominator,
        "rate": round(numerator / denominator, 4) if denominator else None,
    }


def fmt(r: dict[str, Any]) -> str:
    """Render a rate as `n/d (pp.p%)`, or `n/a` when the denominator is zero."""
    if not r["of"]:
        return "n/a (denominator 0)"
    return f"{r['n']}/{r['of']} ({100 * r['rate']:.1f}%)"


def within(claimed: str, ceiling: str) -> bool:
    """Whether a claimed band sits inside a ceiling."""
    return BAND_ORDER[claimed] <= BAND_ORDER[ceiling]


def full_targets(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per case, the ceilings and top-ranked locus that `full` derived.

    Used to score `baseline`, which has no calibration of its own. The ceilings
    are replicate-stable; the ranked locus is not guaranteed to be, so the
    modal value is taken and unanimity is recorded alongside it.
    """
    targets: dict[str, dict[str, Any]] = {}
    by_case: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        if run["variant"] == "full":
            by_case[run["case_id"]].append(run)

    for case_id, group in by_case.items():
        bands = {(r["visibility_band"], r["intensity_band"]) for r in group}
        loci = [r["selected_locus"] for r in group if present(r["selected_locus"])]
        modal = Counter(loci).most_common(1)[0][0] if loci else None
        targets[case_id] = {
            "visibility_band": group[0]["visibility_band"],
            "intensity_band": group[0]["intensity_band"],
            "bands_unanimous": len(bands) == 1,
            "locus": modal,
            "locus_unanimous": len(set(loci)) <= 1,
            "locus_values": sorted(set(loci)),
        }
    return targets


# ---------------------------------------------------------------------------
# RQ1 — does the architecture beat a single prompt?
# ---------------------------------------------------------------------------


def rq1(runs: list[dict[str, Any]], targets: dict[str, dict]) -> dict[str, Any]:
    """Coherence and breach measures, per variant. The contrast is full vs baseline."""
    out: dict[str, Any] = {"per_variant": {}}

    for variant in VARIANTS:
        subset = [r for r in runs if r["variant"] == variant]
        concepts = [r for r in subset if r["concept"]]

        # 2.1 locus-position alignment — cites core.locus.LOCUS_AI_POSITION
        grid = [c for c in concepts if c["concept"]["locus"] in LOCUS_AI_POSITION]
        inconsistent = [
            c
            for c in grid
            if c["concept"]["ai_position"] != LOCUS_AI_POSITION[c["concept"]["locus"]]
        ]

        # 2.2 band-position coherence — claimed band vs the DECLARED position
        incoherent = [
            c
            for c in concepts
            if c["concept"]["claimed_visibility"]
            != AI_POSITION_VISIBILITY[c["concept"]["ai_position"]]
        ]

        # 2.3 locus match, and 1.1 ceiling breach
        matched = considered = breached = breach_considered = 0
        for run in concepts:
            concept = run["concept"]
            if variant == "baseline":
                target = targets.get(run["case_id"])
                if not target:
                    continue
                want_locus = target["locus"]
                vis, inten = target["visibility_band"], target["intensity_band"]
            else:
                calibration = run["calibration"] or {}
                viable = calibration.get("viable_loci") or []
                want_locus = viable[0]["locus"] if viable else None
                vis, inten = (
                    calibration.get("visibility_band"),
                    calibration.get("intensity_band"),
                )
            if want_locus:
                considered += 1
                matched += concept["locus"] == want_locus
            if vis and inten:
                breach_considered += 1
                breached += not (
                    within(concept["claimed_visibility"], vis)
                    and within(concept["claimed_intensity"], inten)
                )

        entry = {
            "n_runs": len(subset),
            "n_concepts": len(concepts),
            "locus_position_inconsistency_2_1": rate(len(inconsistent), len(grid)),
            "band_position_incoherence_2_2": rate(len(incoherent), len(concepts)),
            "locus_match_2_3": rate(matched, considered),
            "ceiling_breach_1_1": rate(breached, breach_considered),
        }

        # 1.2 misdeclaration — only variants that ran a Critic
        if variant in CRITIC_VARIANTS:
            judged = [r for r in subset if r["verdict"]]
            entry["misdeclaration_1_2"] = rate(
                sum(1 for r in judged if r["verdict"]["misdeclared"]), len(judged)
            )
        else:
            entry["misdeclaration_1_2"] = {
                "n": None,
                "of": 0,
                "rate": None,
                "note": "N/A — this variant runs no Critic, so no misdeclaration flag exists",
            }
        out["per_variant"][variant] = entry

    out["notes"] = {
        "2_1": (
            "Declared inconsistency rate, not an error rate: a mismatch may be the "
            "concept honestly declaring a higher visibility than its locus name "
            "implies (e.g. AT_BACKSTAGE declared ADVISOR_MEDIATED). Cites "
            "core.locus.LOCUS_AI_POSITION."
        ),
        "2_3_and_1_1_baseline": (
            "baseline is scored against the ceilings and top-ranked locus FULL "
            "derived for the same case_id. This is a cross-variant comparison — "
            "'would the unconstrained concept have breached the constraint the "
            "architecture would have imposed?' — not baseline's own calibration."
        ),
    }
    return out


# ---------------------------------------------------------------------------
# RQ2 — is calibration biased toward invisibility?
# ---------------------------------------------------------------------------


def rq2(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive distributions over every run that produced a calibration."""
    calibrated = [r for r in runs if r["calibration"]]

    survival = [
        len(r["calibration"]["viable_loci"]) / len(LOCUS_GRID)
        for r in calibrated
        if r["calibration"]["gate_decision"] == "PROCEED"
    ]
    reasons: Counter = Counter()
    for run in calibrated:
        for entry in run["calibration"]["excluded_loci"]:
            reasons[entry["reason"]] += 1

    return {
        "band_distribution_5_1": {
            "denominator": len(calibrated),
            "visibility": dict(
                sorted(Counter(r["visibility_band"] for r in calibrated).items())
            ),
            "intensity": dict(
                sorted(Counter(r["intensity_band"] for r in calibrated).items())
            ),
        },
        "locus_frequency_5_4": {
            "denominator": sum(1 for r in runs if present(r["selected_locus"])),
            "counts": dict(
                Counter(
                    r["selected_locus"] for r in runs if present(r["selected_locus"])
                ).most_common()
            ),
            "note": (
                "Denominator excludes runs with no locus (the 12 gate refusals); "
                "baseline contributes its own unconstrained choice."
            ),
        },
        "locus_survival_5_2": {
            "denominator_runs_proceeding": len(survival),
            "grid_size": len(LOCUS_GRID),
            "mean_fraction_surviving": round(statistics.mean(survival), 4)
            if survival
            else None,
            "mean_loci_surviving": round(statistics.mean(survival) * len(LOCUS_GRID), 2)
            if survival
            else None,
        },
        "exclusion_reason_5_3": {
            "denominator_exclusions": sum(reasons.values()),
            "counts": dict(reasons.most_common()),
        },
    }


# ---------------------------------------------------------------------------
# RQ3 — rules vs model judgement
# ---------------------------------------------------------------------------


def rq3(runs: list[dict[str, Any]], rq1_out: dict[str, Any]) -> dict[str, Any]:
    """Per-case band agreement between the arithmetic and the model."""
    by_case: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for run in runs:
        if run["variant"] in ("full", "llm_bands") and run["visibility_band"]:
            by_case[run["case_id"]][run["variant"]].add(
                (run["visibility_band"], run["intensity_band"])
            )

    agree, disagree = [], []
    for case_id in sorted(by_case):
        f = by_case[case_id].get("full", set())
        l = by_case[case_id].get("llm_bands", set())
        if not f or not l:
            continue
        if f == l:
            agree.append(case_id)
            continue
        (fv, fi), (lv, li) = sorted(f)[0], sorted(l)[0]
        direction = []
        if fv != lv:
            direction.append(
                f"visibility {fv}->{lv} "
                f"({'stricter' if BAND_ORDER[lv] < BAND_ORDER[fv] else 'looser'})"
            )
        if fi != li:
            direction.append(
                f"intensity {fi}->{li} "
                f"({'stricter' if BAND_ORDER[li] < BAND_ORDER[fi] else 'looser'})"
            )
        disagree.append(
            {
                "case_id": case_id,
                "full_deterministic": f"({fv},{fi})",
                "llm_bands": f"({lv},{li})",
                "direction": "; ".join(direction),
            }
        )

    return {
        "band_agreement": rate(len(agree), len(agree) + len(disagree)),
        "agreeing_cases": agree,
        "disagreeing_cases": disagree,
        "ceiling_breach_1_1_full": rq1_out["per_variant"]["full"]["ceiling_breach_1_1"],
        "ceiling_breach_1_1_llm_bands": rq1_out["per_variant"]["llm_bands"][
            "ceiling_breach_1_1"
        ],
        "note": (
            "Both variants' bands were replicate-stable, so a disagreement is a "
            "reproducible difference in judgement, not sampling noise. In the "
            "agreeing cases the ideation prompt is byte-identical to full's, so "
            "any downstream concept divergence there is RNG."
        ),
    }


# ---------------------------------------------------------------------------
# RQ4 — does the loop help or over-smooth? (qualitative)
# ---------------------------------------------------------------------------


def rq4(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Loop behaviour. The effective n for any revision effect is 1."""
    per_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        subset = [r for r in runs if r["variant"] == variant]
        if variant not in CRITIC_VARIANTS:
            per_variant[variant] = {
                "note": "N/A — no Critic runs in this variant",
                "iteration_distribution_3_2": dict(
                    sorted(Counter(int(r["critic_iterations"]) for r in subset).items())
                ),
            }
            continue
        judged = [r for r in subset if r["verdict"]]
        per_variant[variant] = {
            "first_pass_acceptance_3_1": rate(
                sum(1 for r in judged if int(r["critic_iterations"]) == 1
                    and r["verdict"]["verdict"] == "PASS"),
                len(judged),
            ),
            "iteration_distribution_3_2": dict(
                sorted(Counter(int(r["critic_iterations"]) for r in subset).items())
            ),
            "escalation_3_3": rate(
                sum(1 for r in subset if r["terminal_state"] == "ESCALATE"), len(subset)
            ),
        }

    revised = [
        r for r in runs if r["variant"] == "full" and int(r["critic_iterations"]) > 1
    ]
    worked = []
    for run in revised:
        history = run["critic_history"]
        worked.append(
            {
                "case_id": run["case_id"],
                "replicate": int(run["replicate_id"]),
                "before": {
                    "verdict": history[0]["verdict"],
                    "misdeclared": history[0]["misdeclared"],
                    "violations": [
                        f"{v['dimension']}/{v['severity']}" for v in history[0]["violations"]
                    ],
                    "directive": (
                        f"{history[0]['revision_directive']['dimension']} "
                        f"{history[0]['revision_directive']['axis']} "
                        f"{history[0]['revision_directive']['direction']}"
                        if history[0].get("revision_directive")
                        else None
                    ),
                },
                "after": {
                    "verdict": history[-1]["verdict"],
                    "misdeclared": history[-1]["misdeclared"],
                    "violations": [
                        f"{v['dimension']}/{v['severity']}" for v in history[-1]["violations"]
                    ],
                },
            }
        )

    return {
        "per_variant": per_variant,
        "quality_improvement_3_5": {
            "effective_n": len(worked),
            "cases": worked,
            "note": (
                "EFFECTIVE n = 1. Reported as a worked before/after example only. "
                "No rate is computed and none should be inferred."
            ),
        },
        "oscillation_3_4": {
            "computable": False,
            "note": (
                "NOT COMPUTABLE: the loop reached at most 2 iterations and fired "
                "once in 33 full runs. Over-smoothing is not observed because the "
                "loop rarely fires, which is not evidence that it does not occur."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def reliability(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate behaviour and Agent 1 stability."""
    gated = [r for r in runs if present(r["gate_decision"])]
    refusals = [r for r in gated if r["gate_decision"] == "REFUSE"]

    profiled = [r for r in runs if r["profile"]]
    agreement: dict[str, Any] = {}
    confidence: dict[str, Any] = {}
    evidence: dict[str, Any] = {}

    for label, field in DIMENSIONS:
        by_case: dict[str, list[str]] = defaultdict(list)
        confidences, stated = [], 0
        for run in profiled:
            position = run["profile"][field]
            by_case[run["case_id"]].append(position["position"])
            confidences.append(position["confidence"])
            stated += bool(position["stated_in_brief"])

        unanimous = sum(1 for v in by_case.values() if len(set(v)) == 1)
        modal_share = statistics.mean(
            Counter(v).most_common(1)[0][1] / len(v) for v in by_case.values()
        )
        agreement[label] = {
            "cases_unanimous": rate(unanimous, len(by_case)),
            "mean_modal_share": round(modal_share, 4),
            "runs_per_case": len(profiled) // max(len(by_case), 1),
        }
        confidence[label] = {
            "n": len(confidences),
            "mean": round(statistics.mean(confidences), 3),
            "median": round(statistics.median(confidences), 3),
            "min": round(min(confidences), 3),
            "max": round(max(confidences), 3),
        }
        evidence[label] = {**rate(stated, len(profiled)), "caveat": EVIDENCE_CAVEAT}

    return {
        "gate_refusal_1_4": {
            **rate(len(refusals), len(gated)),
            "cases_refused": dict(Counter(r["case_id"] for r in refusals)),
            "note": (
            "Denominator is runs that reached the gate — baseline has no gate and "
            "is excluded. Every refusal is case_010; no other case was refused."
        ),
        },
        "agent1_position_agreement_4_1": agreement,
        "agent1_confidence_4_3": confidence,
        "agent1_evidence_rate_4_2": evidence,
        "evidence_caveat": EVIDENCE_CAVEAT,
    }


# ---------------------------------------------------------------------------
# Efficiency and the deferred list
# ---------------------------------------------------------------------------


def efficiency(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Tokens, cost and wall clock per variant."""
    per_variant = {}
    for variant in VARIANTS:
        subset = [r for r in runs if r["variant"] == variant]
        per_variant[variant] = {
            "n_runs": len(subset),
            "input_tokens_7_1": {
                "total": int(sum(r["total_input_tokens"] for r in subset)),
                "mean": round(statistics.mean(r["total_input_tokens"] for r in subset), 1),
            },
            "output_tokens_7_1": {
                "total": int(sum(r["total_output_tokens"] for r in subset)),
                "mean": round(statistics.mean(r["total_output_tokens"] for r in subset), 1),
            },
            "cost_usd_7_2": {
                "total": round(sum(r["est_cost_usd"] for r in subset), 4),
                "mean": round(statistics.mean(r["est_cost_usd"] for r in subset), 5),
            },
            "wall_clock_s_7_3": {
                "total": round(sum(r["wall_clock_seconds"] for r in subset), 1),
                "mean": round(statistics.mean(r["wall_clock_seconds"] for r in subset), 1),
            },
        }

    full = per_variant["full"]["cost_usd_7_2"]["total"]
    minus = per_variant["minus_critic"]["cost_usd_7_2"]["total"]
    return {
        "per_variant": per_variant,
        "cost_per_improvement_7_4": {
            "extra_cost_full_over_minus_critic_usd": round(full - minus, 4),
            "improvements_observed": 1,
            "note": (
                "ILLUSTRATIVE ONLY, n = 1. One revision fired across 33 full runs, "
                "so this divides the whole cost difference by a single observation. "
                "It is not a cost-effectiveness estimate."
            ),
        },
    }


DEFERRED = {
    "6_1_delta_misdeclaration": (
        "not computable this batch — minus_critic records no verdict, so the "
        "comparison is against a structural zero and would measure the Critic's "
        "presence, not its effect; needs a post-hoc critic pass, deferred to campaign 2"
    ),
    "6_2_delta_violations": (
        "not computable this batch — same reason as 6.1; deferred to campaign 2"
    ),
    "6_3_calibration_contribution": (
        "not computable this batch — needs the minus_calibration variant, deferred to campaign 2"
    ),
    "6_4_profiler_contribution": (
        "not computable this batch — needs the minus_profiler variant, deferred to campaign 2"
    ),
    "loop_depth_comparison": (
        "not computable this batch — no loop_depth {1,3,5} variants were run, deferred to campaign 2"
    ),
    "post_hoc_critic_on_minus_critic": (
        "not attempted — would require re-running the Critic over stored minus_critic "
        "concepts, deferred to campaign 2"
    ),
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(k: dict[str, Any]) -> None:
    """Print the RQ-grouped readable report."""
    h1 = "=" * 78
    h2 = "-" * 78

    print(f"\n{h1}\nLUXCAL CAMPAIGN 1 — KPIs BY RESEARCH QUESTION\n{h1}")
    print(f"runs: {k['meta']['n_runs']}  |  cases: {k['meta']['n_cases']}  |  "
          f"variants: {len(k['meta']['variants'])}  |  replicates: {k['meta']['n_replicates']}")

    # RQ1
    print(f"\n{h1}\nRQ1 — Does the architecture beat a single prompt?  (full vs baseline)\n{h1}")
    print(f"  {'variant':<14}{'2.1 locus/pos':>16}{'2.2 band/pos':>16}"
          f"{'2.3 locus match':>18}{'1.1 breach':>14}{'1.2 misdecl':>16}")
    for variant in VARIANTS:
        e = k["rq1"]["per_variant"][variant]
        m = e["misdeclaration_1_2"]
        mtxt = "N/A (no critic)" if m["rate"] is None else fmt(m)
        print(f"  {variant:<14}{fmt(e['locus_position_inconsistency_2_1']):>16}"
              f"{fmt(e['band_position_incoherence_2_2']):>16}"
              f"{fmt(e['locus_match_2_3']):>18}{fmt(e['ceiling_breach_1_1']):>14}{mtxt:>16}")
    print("\n  2.1 = DECLARED INCONSISTENCY rate (lower is more coherent), not an error rate.")
    print(f"      {k['rq1']['notes']['2_1']}")
    print("  2.2 = claimed_visibility vs the DECLARED ai_position's native visibility.")
    print(f"  2.3/1.1 for baseline: {k['rq1']['notes']['2_3_and_1_1_baseline']}")

    # RQ2
    print(f"\n{h1}\nRQ2 — Is calibration biased toward invisibility?  (descriptive)\n{h1}")
    b = k["rq2"]["band_distribution_5_1"]
    print(f"  5.1 band distribution (denominator {b['denominator']} calibrated runs)")
    print(f"      visibility: {b['visibility']}")
    print(f"      intensity : {b['intensity']}")
    s = k["rq2"]["locus_survival_5_2"]
    print(f"\n  5.2 locus survival: mean {s['mean_loci_surviving']} of {s['grid_size']} cells "
          f"({100 * s['mean_fraction_surviving']:.1f}%), denominator "
          f"{s['denominator_runs_proceeding']} runs that proceeded")
    e = k["rq2"]["exclusion_reason_5_3"]
    print(f"\n  5.3 exclusion reasons (denominator {e['denominator_exclusions']} exclusions)")
    for reason, n in e["counts"].items():
        print(f"      {reason:<20}{n:>5}  ({100 * n / e['denominator_exclusions']:.1f}%)")
    f = k["rq2"]["locus_frequency_5_4"]
    print(f"\n  5.4 selected locus (denominator {f['denominator']} runs with a locus)")
    for locus, n in f["counts"].items():
        print(f"      {locus:<18}{n:>5}")

    # RQ3
    print(f"\n{h1}\nRQ3 — Rules vs model judgement for calibration?  (full vs llm_bands)\n{h1}")
    print(f"  band agreement: {fmt(k['rq3']['band_agreement'])} of cases")
    print(f"  agreeing ({len(k['rq3']['agreeing_cases'])}): {', '.join(k['rq3']['agreeing_cases'])}")
    print(f"  disagreeing ({len(k['rq3']['disagreeing_cases'])}):")
    for d in k["rq3"]["disagreeing_cases"]:
        print(f"      {d['case_id']}  full {d['full_deterministic']} -> "
              f"llm {d['llm_bands']}   {d['direction']}")
    print(f"\n  1.1 ceiling breach: full {fmt(k['rq3']['ceiling_breach_1_1_full'])}  |  "
          f"llm_bands {fmt(k['rq3']['ceiling_breach_1_1_llm_bands'])}")
    print(f"  {k['rq3']['note']}")

    # RQ4
    print(f"\n{h1}\nRQ4 — Does the loop help or over-smooth?   *** QUALITATIVE, n IS TINY ***\n{h1}")
    for variant in VARIANTS:
        v = k["rq4"]["per_variant"][variant]
        if "note" in v:
            print(f"  {variant:<14} {v['note']}   iterations={v['iteration_distribution_3_2']}")
            continue
        print(f"  {variant:<14} 3.1 first-pass accept {fmt(v['first_pass_acceptance_3_1']):<16}"
              f"3.3 escalation {fmt(v['escalation_3_3']):<14}"
              f"3.2 iterations {v['iteration_distribution_3_2']}")
    q = k["rq4"]["quality_improvement_3_5"]
    print(f"\n  3.5 quality improvement — EFFECTIVE n = {q['effective_n']}")
    for c in q["cases"]:
        print(f"      {c['case_id']} r{c['replicate']}")
        print(f"        before: {c['before']['verdict']}  misdeclared={c['before']['misdeclared']}  "
              f"violations={c['before']['violations']}  directive={c['before']['directive']}")
        print(f"        after : {c['after']['verdict']}  misdeclared={c['after']['misdeclared']}  "
              f"violations={c['after']['violations']}")
    print(f"      {q['note']}")
    print(f"\n  3.4 oscillation: {k['rq4']['oscillation_3_4']['note']}")

    # Reliability
    print(f"\n{h1}\nRELIABILITY (cross-cutting)\n{h1}")
    g = k["reliability"]["gate_refusal_1_4"]
    print(f"  1.4 gate refusal: {fmt(g)} of runs that reached the gate — {g['cases_refused']}")
    print(f"      {g['note']}")
    print(f"\n  4.1 Agent 1 position agreement across replicates "
          f"({k['reliability']['agent1_position_agreement_4_1']['D1']['runs_per_case']} runs per case)")
    print(f"      {'dim':<6}{'cases unanimous':>20}{'mean modal share':>20}")
    for label, _ in DIMENSIONS:
        a = k["reliability"]["agent1_position_agreement_4_1"][label]
        print(f"      {label:<6}{fmt(a['cases_unanimous']):>20}{a['mean_modal_share']:>20.3f}")
    print(f"\n  4.3 confidence by dimension")
    print(f"      {'dim':<6}{'mean':>8}{'median':>9}{'min':>7}{'max':>7}")
    for label, _ in DIMENSIONS:
        c = k["reliability"]["agent1_confidence_4_3"][label]
        print(f"      {label:<6}{c['mean']:>8}{c['median']:>9}{c['min']:>7}{c['max']:>7}")
    print(f"\n  4.2 evidence rate by dimension  *** {EVIDENCE_CAVEAT} ***")
    for label, _ in DIMENSIONS:
        ev = k["reliability"]["agent1_evidence_rate_4_2"][label]
        print(f"      {label:<6}{fmt(ev):>16}   [{ev['caveat']}]")

    # Efficiency
    print(f"\n{h1}\nEFFICIENCY\n{h1}")
    print(f"  {'variant':<14}{'in tok':>10}{'out tok':>10}{'cost $':>10}"
          f"{'mean $':>10}{'wall s':>10}{'mean s':>9}")
    for variant in VARIANTS:
        e = k["efficiency"]["per_variant"][variant]
        print(f"  {variant:<14}{e['input_tokens_7_1']['total']:>10}"
              f"{e['output_tokens_7_1']['total']:>10}{e['cost_usd_7_2']['total']:>10}"
              f"{e['cost_usd_7_2']['mean']:>10}{e['wall_clock_s_7_3']['total']:>10}"
              f"{e['wall_clock_s_7_3']['mean']:>9}")
    c = k["efficiency"]["cost_per_improvement_7_4"]
    print(f"\n  7.4 cost per improvement: ${c['extra_cost_full_over_minus_critic_usd']} extra "
          f"for {c['improvements_observed']} improvement")
    print(f"      {c['note']}")

    # Deferred
    print(f"\n{h1}\nNOT COMPUTED THIS BATCH\n{h1}")
    for kpi, why in k["deferred"].items():
        print(f"  {kpi}\n      {why}")
    print()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Compute campaign-1 KPIs.")
    parser.add_argument("--parquet", type=Path, default=Path("analysis/runs.parquet"))
    parser.add_argument("--out", type=Path, default=Path("analysis/kpis.json"))
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    runs = load_runs(args.parquet)
    targets = full_targets(runs)

    rq1_out = rq1(runs, targets)
    kpis = {
        "meta": {
            "n_runs": len(runs),
            "n_cases": len({r["case_id"] for r in runs}),
            "variants": VARIANTS,
            "n_replicates": len({int(r["replicate_id"]) for r in runs}),
            "full_targets_for_baseline": targets,
        },
        "rq1": rq1_out,
        "rq2": rq2(runs),
        "rq3": rq3(runs, rq1_out),
        "rq4": rq4(runs),
        "reliability": reliability(runs),
        "efficiency": efficiency(runs),
        "deferred": DEFERRED,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(kpis, indent=2), encoding="utf-8")
    report(kpis)
    print(f"wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
