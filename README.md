# LUXCAL

A calibrated multi-agent system for AI personalisation strategy in the luxury industry.

LUXCAL takes an unstructured brand brief describing a business problem where AI integration is being considered, and returns a calibrated recommendation: whether personalisation is appropriate at all, at what visibility and intensity, in which part of the business, and a baseline concept checked against luxury-specific constraints.

This repository accompanies an MSc dissertation at Warwick Business School. The contribution is the architecture and its evaluation rather than the engineering.

---

## The core idea

Personalisation in luxury fails along two separable axes, not one:

- **Visibility** — how perceptible the AI is to the client
- **Intensity** — how far the offering differentiates per individual

Ceilings for each are computed **deterministically** from five brand dimensions drawn from the luxury literature. LLMs are used only for semantic judgement — the gate, locus ranking, ideation, and the Critic's second check. They never perform the arithmetic.

Positions are placed on an 11-cell grid crossing customer-journey phase (PRE / AT / POST) with client-facing depth (BACKSTAGE / ADVISOR / CLIENT). Loci whose native visibility or intensity exceeds either ceiling are filtered out before generation begins.

## The five dimensions

| Dim | Name | Positions | Drives |
|-----|------|-----------|--------|
| D1 | Rarity & Scarcity Posture | OBJECTIVE / VIRTUAL / DIFFUSE | Intensity |
| D2 | Scenic Invisibility | STRICT / MODERATE / RELAXED | Visibility |
| D3 | Consumption Value Orientation | EMOTIONAL_LED / BALANCED / FUNCTIONAL_LED | Intensity |
| D4 | Motivation Profile | UNM / MIXED / ELT | Visibility |
| D5 | Symbolic Orchestration Control | STRONG / PARTIAL / WEAK | Adjusts both |

Definitions, diagnostic questions and ordinal maps live in `rubric/rubric_v1.yaml`. The rubric is the single source of truth: agent prompts are generated from it rather than restating it, so editing the rubric changes the system's behaviour without touching code.

## Architecture

```
Agent 1 (Brand Profiler)
  → Agent 2 (Gate → Ceilings → Locus filter → Ranking)
    → Agent 3 (Ideation) ⇄ Critic (max 3 iterations)
      → END
```

- **Agent 1** converts free text into a typed five-dimension profile with per-dimension provenance. One schema-constrained call at temperature 0.
- **Agent 2** alternates LLM judgement and deterministic rules: an LLM gate decides whether this is a personalisation problem at all, arithmetic computes both ceilings, a set operation filters the grid, then an LLM ranks the survivors.
- **Agent 3** designs a concept into the top-ranked locus at temperature 0.7 — the only non-zero temperature in the system. It must self-declare its AI position, differentiation unit and claimed bands. It is *asked* to respect the ceilings; it does not enforce them.
- **The Critic** runs a deterministic ceiling comparison first, then an LLM check on whether the concept's description matches its own declarations. Verdicts are PASS, REVISE (with a directive) or ESCALATE.

Constraint lives *around* generation, not inside it. Because Agent 3 can breach a ceiling, breaches are measurable and the Critic has something to detect.

## Repository layout

```
luxcal/
  agents/        profiler, calibration, ideation, critic, shared LLM helpers
  core/          schemas, state, ceilings, locus grid, graph wiring, config
  configs/       one YAML per ablation variant
  logging/       run logger (manifest, JSONL calls, state snapshots)
rubric/          versioned rubric definitions
data/cases/      brand brief case files
scripts/         run_single, run_baseline, run_batch, aggregate, concepts, kpi
tests/           110 tests
analysis/        campaign outputs (parquet, CSV, KPI JSON)
```

## Ablation variants

| Variant | Change relative to `full` |
|---------|---------------------------|
| `full` | Complete system — reference condition |
| `minus_critic` | Critic node removed from the graph; first concept returned unchecked |
| `minus_loop` | Critic runs once, no revision permitted |
| `llm_bands` | Ceilings computed by an LLM instead of the rubric arithmetic |
| `baseline` | Single prompt to a bare model; none of the architecture |

Variant behaviour is driven by an enum read by `build_graph`, not by the CLI label — the two are asserted to agree post-run.

## Running it

Requires Python 3.11 and an Anthropic API key in `.env`.

```bash
uv venv
uv pip install -r requirements.txt
```

Single run:
```bash
python -m scripts.run_single --case data/cases/case_004.json --config luxcal/configs/full.yaml
```

Full batch (cases × variants × replicates):
```bash
python -m scripts.run_batch
```

The batch runner isolates per-run failures, writes results incrementally, and resumes by skipping completed cells. Transient API failures are retried up to a cap and are distinguished from genuine validation failures via an explicit error artefact, so a completed batch has no silent holes.

Analysis:
```bash
python -m scripts.aggregate   # manifests → analysis/runs.parquet
python -m scripts.concepts    # concept text → analysis/concepts.csv
python -m scripts.kpi         # KPIs by research question → analysis/kpis.json
```

## Reproducibility

Every run records its git commit SHA, the rubric version and SHA-256, both model strings, token counts, cost and wall clock. The rubric and config are snapshotted into the run directory. A dirty working tree blocks a run by default.

Raw run directories are excluded from the repository by size. The analysed outputs — `analysis/runs.parquet`, `analysis/concepts.csv`, `analysis/kpis.json` — are committed and version-pinned to the commit that produced them.

## Campaign 1 results

165 runs: 11 brand cases × 5 variants × 3 replicates.

- The unconstrained baseline breached the ceiling the full system derived for the same case in **87.9%** of runs. Every architecture-bearing variant breached **zero** times.
- Visibility was calibrated LOW in **112 of 132** calibrated runs and HIGH in none. Mean locus survival was **3.12 of 11** grid cells.
- LLM-computed ceilings reproduced the deterministic arithmetic in **9 of 11** cases; both divergences were stricter, not looser.
- The critique loop fired **once in 33** full runs. It repaired the concept when it did.

Full figures are in `analysis/kpis.json`, organised by research question with sample-size caveats attached.

## Known limitations

- **Profiler provenance is unverified.** Agent 1 is instructed to quote verbatim evidence and to abstain where the brief is silent, but no code checks that a span is genuinely from the brief or that it bears on the dimension it is attached to. The evidence-rate metric is contaminated and is not reported.
- **The category-prior fallback does not exist.** Unevidenced dimensions fall back to model judgement rather than a declared table.
- **The loop has an effective sample of one.** No quantitative claim about its contribution is made.
- **Temperature 0 is not deterministic.** Identical inputs have produced identical ceilings but a different top-ranked locus. The deterministic layer is stable; the LLM-ranked layer is not.

## What this is not

LUXCAL does not learn. The models are frozen and nothing is trained — the critique-revision cycle is inference-time self-correction, not reinforcement learning. Agency is confined to four bounded semantic decisions inside a deterministic scaffold. The system trades autonomy for auditability deliberately.

## Data

Brand briefs are constructed vignettes derived from public sources (annual reports, trade press), following established vignette methodology. They are not confidential material and do not represent statements by the brands named.

## Licence

See `LICENSE`.
