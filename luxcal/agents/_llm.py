"""Response-handling helpers shared by the agent nodes.

Private to `luxcal.agents`. Deliberately narrow: this reads what came back
from a call, and nothing here builds prompts, chooses models, sets temperature
or performs retries. Those stay at the call site, where the spec requires them
to be visible and where each agent's requirements differ.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic


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
