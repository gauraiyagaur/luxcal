"""Agent 1 — Brand Profiler (SPEC §5.1).

Converts an unstructured brief into a typed five-dimension profile in a single
schema-constrained extraction call. The dimension definitions and diagnostic
questions are injected verbatim from the versioned rubric rather than restated
here, so that revising the rubric revises the prompt.

Abstention is the point of the provenance fields: where the brief does not
evidence a dimension, the agent must say so rather than inferring silently.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, get_args

import anthropic
import yaml
from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from luxcal.core.schemas import BrandProfile, Category
from luxcal.core.state import LuxcalState
from luxcal.logging.run_logger import RunLogger

# Three attempts: the initial call plus the two retries SPEC §5.1 allows.
_MAX_ATTEMPTS = 3

# The profile is a bounded object — five position records plus two short
# strings — so this is roughly four times the expected output. It exists to
# stop a truncated response being mistaken for a malformed one.
_MAX_TOKENS = 8192


async def run_profiler(
    state: LuxcalState,
    config: dict,
    logger: RunLogger,
) -> dict:
    """Extract a `BrandProfile` from the brief. LangGraph node for Agent 1.

    Reads `state["brief"]`, makes up to three extraction calls, and returns
    `{"profile": BrandProfile}` on success. Every attempt is logged in full,
    including the ones that fail validation. If all three attempts fail the
    run is terminated with `{"terminal_state": "ERROR"}` rather than returning
    a partial or fabricated profile.
    """
    rubric = _load_rubric(Path(config["rubric_path"]))
    system_prompt = _build_system_prompt(rubric)
    model = config["model_generation"]
    brief = state["brief"]

    previous_error: Optional[str] = None

    try:
        async with anthropic.AsyncAnthropic() as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(_MAX_ATTEMPTS),
                retry=retry_if_exception_type((ValidationError, json.JSONDecodeError)),
                reraise=True,
            ):
                with attempt:
                    retry_count = attempt.retry_state.attempt_number - 1
                    try:
                        profile = await _extract(
                            client=client,
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=_build_user_prompt(brief, previous_error),
                            logger=logger,
                            retry_count=retry_count,
                        )
                    except (ValidationError, json.JSONDecodeError) as exc:
                        # Tenacity clears `retry_state.outcome` between
                        # attempts, so the error is carried forward here.
                        previous_error = str(exc)
                        raise
    except (ValidationError, json.JSONDecodeError):
        return {"terminal_state": "ERROR"}

    logger.save_state("profiler", {**state, "profile": profile})
    return {"profile": profile}


async def _extract(
    client: anthropic.AsyncAnthropic,
    model: str,
    system_prompt: str,
    user_prompt: str,
    logger: RunLogger,
    retry_count: int,
) -> BrandProfile:
    """Make one extraction call, log it, and validate the response.

    The call is logged whether or not the response validates: a response that
    fails the schema is the more interesting record of the two.
    """
    started = time.perf_counter()
    response = await client.messages.create(
        model=model,
        max_tokens=_MAX_TOKENS,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    text = _response_text(response)

    def record(schema_valid: bool) -> None:
        logger.log_call(
            node="profiler",
            model=model,
            temperature=0.0,
            prompt=f"[system]\n{system_prompt}\n\n[user]\n{user_prompt}",
            response=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            schema_valid=schema_valid,
            retry_count=retry_count,
            stop_reason=response.stop_reason,
        )

    try:
        profile = BrandProfile.model_validate(_parse_json(text))
    except (ValidationError, json.JSONDecodeError):
        record(schema_valid=False)
        raise

    record(schema_valid=True)
    return profile


def _load_rubric(rubric_path: Path) -> dict:
    """Read the versioned rubric, tolerating a byte-order mark."""
    return yaml.safe_load(rubric_path.read_text(encoding="utf-8-sig"))


def _build_system_prompt(rubric: dict) -> str:
    """Assemble the extraction prompt from the rubric.

    Definitions and diagnostic questions are inserted exactly as the rubric
    states them. The `retrieval_confidence` and `grounding` blocks are
    deliberately not injected: they describe the strength of the literature
    behind each dimension, which is not information the extraction should be
    conditioned on.
    """
    sections = [_dimension_section(dimension) for dimension in rubric["dimensions"]]
    categories = " | ".join(get_args(Category))

    return f"""You are a luxury brand analyst performing structured extraction.

You will be given a brand brief: free text describing a luxury brand, its
category, and a business problem for which AI integration is being considered.
Your task is to read the brief and return a five-dimension brand profile as a
single JSON object.

This is an analytical instrument, not a piece of client-facing writing. Use
plain professional English throughout. Do not use marketing register.

# The five dimensions

Each dimension has a closed vocabulary of positions. You must select exactly
one position per dimension. The definitions and diagnostic questions below are
the calibration standard; assess the brief against them rather than against
your own intuitions about the category.

{chr(10).join(sections)}
# Category

Assign exactly one category from this closed vocabulary:

{categories}

# Provenance and abstention

For each dimension, decide whether the brief actually evidences the position
you have chosen.

- If it does, set `stated_in_brief` to true and set `evidence_span` to a
  verbatim quotation from the brief. The quotation must appear in the brief
  character for character. Do not paraphrase, tidy or join separated phrases.
- If it does not, set `stated_in_brief` to false and set `evidence_span` to
  null. Choose the position most typical of the brief's category, and say in
  the rationale that this is a category fallback rather than something the
  brief states. Set `confidence` low to reflect the absence of evidence.

Inferring a position from silence and presenting it as evidenced is the single
most damaging error you can make here. A dimension the brief does not address
is a normal and expected outcome, not a failure.

`confidence` is your confidence in the position, from 0.0 to 1.0.

`rationale` is one line explaining why that position was chosen. It must be a
single sentence.

`problem_statement` restates the brief's business problem in one sentence.

# Response format

Return a single JSON object and nothing else. No preamble, no explanation, no
code fences. The object must have exactly these keys:

{_response_shape(rubric)}"""


def _dimension_section(dimension: dict) -> str:
    """Render one dimension's rubric entry for injection into the prompt."""
    questions = "\n".join(
        f"- {question}" for question in dimension["diagnostic_questions"]
    )
    positions = " | ".join(dimension["positions"])

    return f"""## {dimension["id"].upper()} — {dimension["name"]}

JSON field: `{dimension["profile_field"]}`
Positions: {positions}

Definition:
{dimension["definition"]}

Diagnostic questions:
{questions}
"""


def _response_shape(rubric: dict) -> str:
    """Render the expected JSON object, keyed and constrained from the rubric.

    Written out explicitly rather than generated from `BrandProfile`'s JSON
    schema: the generic `DimensionPosition` produces `$defs` names such as
    `DimensionPosition_Literal__STRICT____MODERATE____RELAXED___`, which would
    go into the prompt verbatim and read as noise to the model.
    """
    shape: dict[str, Any] = {
        "category": "<one of the category vocabulary above>",
        "problem_statement": "<the business problem, in one sentence>",
    }
    for dimension in rubric["dimensions"]:
        shape[dimension["profile_field"]] = {
            "position": "<one of: " + " | ".join(dimension["positions"]) + ">",
            "evidence_span": "<verbatim quotation from the brief, or null>",
            "stated_in_brief": "<true or false>",
            "confidence": "<number from 0.0 to 1.0>",
            "rationale": "<one sentence>",
        }
    return json.dumps(shape, indent=2)


def _build_user_prompt(brief: str, previous_error: Optional[str]) -> str:
    """Assemble the user turn, appending the prior validation error on a retry."""
    prompt = f"Here is the brand brief.\n\n<brief>\n{brief}\n</brief>"
    if previous_error is None:
        return prompt
    return (
        f"{prompt}\n\nYour previous response failed validation: "
        f"{previous_error}. Please correct and return valid JSON."
    )


def _response_text(response: anthropic.types.Message) -> str:
    """Concatenate the text blocks of a response.

    Returns an empty string when the response carries no text, which
    `_parse_json` then surfaces as a decode failure and retries.
    """
    return "".join(block.text for block in response.content if block.type == "text")


def _parse_json(text: str) -> Any:
    """Parse the response body, tolerating a code fence around the JSON.

    The prompt asks for bare JSON, but a fence is a common and harmless
    deviation; stripping it is normalisation rather than repair. Anything else
    raises `json.JSONDecodeError` and is retried.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return json.loads(stripped)
