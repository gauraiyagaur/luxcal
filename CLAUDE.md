# LUXCAL — project instructions

## What this is

LUXCAL is a five-agent system for an MSc dissertation on agentic AI for market
research in the luxury industry. It takes a luxury brand brief describing a
business problem where AI integration is being considered, and returns a
calibrated recommendation: whether personalisation is appropriate at all, at
what visibility and intensity, in which part of the business, and a baseline
concept checked against luxury-specific constraints.

**The full specification is in `SPEC.md`. Read the relevant section before
writing any code. It is the source of truth — if my request contradicts it,
say so rather than silently following one or the other.**

## Non-negotiable design rules

1. **Deterministic where possible, generative where necessary.** Calibration
   arithmetic is plain Python and must never be delegated to an LLM. LLM calls
   are only for semantic judgements that cannot be reduced to rules.
2. **Closed vocabularies.** Every categorical field is a `Literal` or `Enum`.
   Free text appears only in `*_rationale` fields, never in fields I will
   analyse statistically.
3. **Pydantic contracts between agents.** No dicts crossing agent boundaries.
4. **Every LLM call is logged in full** — prompt, response, model, temperature,
   tokens, latency. Nothing is inferred later from token counts alone.
5. **Rubric and config are data, not code.** Ordinal maps, thresholds and
   dimension definitions live in `rubric/rubric_v1.yaml`.
6. **Model version strings are pinned and dated.** Never use a bare alias.
7. **Temperature is always explicit at the call site**, including where the
   default would do.

## How I want you to work

- **Small units.** One module per request. Do not scaffold the whole system in
  one pass — I need to read and understand each piece.
- **Tests before I trust it.** `core/ceilings.py` and `core/locus.py` must have
  pytest coverage with hand-worked cases before I run anything against the API.
- **No invented dependencies.** If something needs a library not in
  `pyproject.toml`, tell me and wait; don't add it.
- **Explain deviations.** If the spec's approach won't work, say why before
  writing something different.
- **No placeholder logic that silently succeeds.** A stub raises
  `NotImplementedError`. It never returns fake data.
- **Comment sparingly.** Code should read cleanly; comments explain *why*, not
  *what*.

## Style

- Python 3.11+, type hints throughout, `from __future__ import annotations`.
- Pydantic v2 (`model_validate`, `model_dump` — not the v1 API).
- Prefer clean minimal readable code over cleverness.
- Prose in prompts, docstrings and rationale fields: plain professional
  English, British spelling. No marketing register.

## Current state

Working on: **Agent 1 (Brand Profiler) and Agent 2 (Calibration & Gate).**
Agents 3, Critic and 4 are stubs raising `NotImplementedError`.

## Layout

```
luxcal/
  agents/     profiler.py calibration.py ideation.py critic.py chat.py
  core/       state.py schemas.py ceilings.py locus.py graph.py config.py
  retrieval/  rubric_index.py market_search.py cache.py
  logging/    run_logger.py manifest.py
  rubric/     rubric_v1.yaml dimensions/*.md
  configs/    full.yaml minus_critic.yaml ...
  data/cases/ case_001.json ...
  scripts/    run_single.py run_batch.py aggregate.py
  tests/      test_ceilings.py test_locus.py test_schemas.py
  runs/       (git-ignored)
```

## The five dimensions

| Dim | Name | Positions |
|-----|------|-----------|
| D1 | Rarity & Scarcity Posture | OBJECTIVE / VIRTUAL / DIFFUSE |
| D2 | Scenic Invisibility Requirement | STRICT / MODERATE / RELAXED |
| D3 | Consumption Value Orientation | EMOTIONAL_LED / BALANCED / FUNCTIONAL_LED |
| D4 | Motivation Profile | UNM / MIXED / ELT |
| D5 | Symbolic Orchestration Control | STRONG / PARTIAL / WEAK |

Visibility ceiling is driven by D2 + D4, adjusted by D5.
Intensity ceiling is driven by D1 + D3, adjusted by D5.
See SPEC.md §5.2 for the arithmetic.

## Things that will waste my time

- Suggesting I train a GNN or any model. Sample size is 25–30 cases.
- Adding a graph database. NetworkX at most, and probably not even that.
- Using a `seed` parameter — the Anthropic API does not have one.
- Rewriting files I did not ask you to touch.
- Notebooks for anything that produces experimental output.
