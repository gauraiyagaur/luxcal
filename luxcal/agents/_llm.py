"""Call and response handling shared by the agent nodes.

Private to `luxcal.agents`. This owns the mechanics every agent repeats — read
the response, decode it, retry a response that cannot be used, log every
attempt — and nothing else. Prompts, model choice and temperature stay at the
call site: they are the experimental instrument, and SPEC §7.5 requires them to
be visible where the call is made rather than buried behind a default.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional, TypeVar

import anthropic
from pydantic import ValidationError
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

from luxcal.logging.run_logger import RunLogger

T = TypeVar("T")

# The initial call plus the two retries the spec allows each agent.
MAX_ATTEMPTS = 3


class ResponseError(ValueError):
    """A response that parsed and validated but is wrong for the request made.

    Distinct from a schema failure: the shape was right and the values were in
    vocabulary, but the content contradicts something the caller asked for —
    a ranking that names an excluded locus, a concept placed at a locus other
    than the selected one. Retried like a schema failure so the model is told.
    """


# What `call_with_retries` retries on, and therefore what a caller should catch
# to detect exhausted retries. Exported so the two cannot drift apart.
RETRYABLE = (ValidationError, json.JSONDecodeError, ResponseError)


def response_text(response: anthropic.types.Message) -> str:
    """Concatenate the text blocks of a response.

    Returns an empty string when the response carries no text, which
    `parse_json` then surfaces as a decode failure and the caller retries.
    """
    return "".join(block.text for block in response.content if block.type == "text")


def parse_json(text: str) -> Any:
    """Parse a response body, tolerating a code fence around the JSON.

    The prompts ask for bare JSON, but a fence is a common and harmless
    deviation; stripping it is normalisation rather than repair. Anything else
    raises `json.JSONDecodeError`, which every agent treats as retryable.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return json.loads(stripped)


async def call_with_retries(
    *,
    client: anthropic.AsyncAnthropic,
    model: str,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    logger: RunLogger,
    node: str,
    parse: Callable[[Any], T],
    iteration: Optional[int] = None,
) -> T:
    """Make one logged LLM call, retrying up to twice on an unusable response.

    `parse` receives the decoded JSON and returns the validated object;
    raising `ValidationError`, `json.JSONDecodeError` or `ResponseError` from
    it triggers a retry with the error text appended to the user turn. Every
    attempt is logged, including the ones that fail — a response that failed
    the schema is the more interesting record of the two.
    """
    previous_error: Optional[str] = None

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        retry=retry_if_exception_type(RETRYABLE),
        reraise=True,
    ):
        with attempt:
            retry_count = attempt.retry_state.attempt_number - 1
            turn = _with_error(user_prompt, previous_error)

            started = time.perf_counter()
            response = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": turn}],
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            text = response_text(response)

            def record(schema_valid: bool) -> None:
                logger.log_call(
                    node=node,
                    model=model,
                    temperature=temperature,
                    prompt=f"[system]\n{system_prompt}\n\n[user]\n{turn}",
                    response=text,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=latency_ms,
                    schema_valid=schema_valid,
                    retry_count=retry_count,
                    stop_reason=response.stop_reason,
                    iteration=iteration,
                )

            try:
                parsed = parse(parse_json(text))
            except RETRYABLE as exc:
                record(schema_valid=False)
                # Tenacity clears `retry_state.outcome` between attempts, so
                # the error is carried forward here.
                previous_error = str(exc)
                raise

            record(schema_valid=True)
            return parsed

    raise AssertionError("unreachable: AsyncRetrying either returns or raises")


def _with_error(user_prompt: str, previous_error: Optional[str]) -> str:
    """Append the prior validation error to the user turn on a retry."""
    if previous_error is None:
        return user_prompt
    return (
        f"{user_prompt}\n\nYour previous response failed validation: "
        f"{previous_error}. Please correct and return valid JSON."
    )
