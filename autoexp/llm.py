"""Thin wrapper around the Anthropic SDK shared by the planner and the narrator.

Keeps three things in one place: credential detection, a daily call cap (the
public demo holds the owner's key) and a cost estimate from the usage block.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

MODEL = os.environ.get("AUTOEXP_MODEL", "claude-opus-5")
DAILY_CALL_CAP = int(os.environ.get("AUTOEXP_DAILY_LLM_CALLS", "60"))
COUNTER_PATH = Path(os.environ.get("AUTOEXP_COUNTER_PATH", Path(tempfile.gettempdir()) / "autoexp_llm_calls.json"))

# USD per million tokens (input, output); used only for the on-screen estimate.
PRICE_PER_MTOK = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class LLMUnavailable(RuntimeError):
    """The model could not be called or refused; callers fall back to rules/templates."""


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from an env file without overriding real env vars.

    The file is .env in the project root, or the path in AUTOEXP_ENV_FILE. A bare
    line that looks like an Anthropic key (sk-ant-...) or a Hugging Face token
    (hf_...) is accepted as ANTHROPIC_API_KEY or HF_TOKEN.
    """
    env_path = path or Path(os.environ.get("AUTOEXP_ENV_FILE") or Path(__file__).resolve().parent.parent / ".env")
    try:
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip().strip('"').strip("'")
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
            elif line.startswith("sk-ant-"):
                key, value = "ANTHROPIC_API_KEY", line
            elif line.startswith("hf_"):
                key, value = "HF_TOKEN", line
            else:
                continue
            if key and value and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


load_dotenv()


def api_key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _load_counter() -> dict:
    try:
        data = json.loads(COUNTER_PATH.read_text(encoding="utf-8"))
        if data.get("date") == date.today().isoformat():
            return data
    except (OSError, ValueError):
        pass
    return {"date": date.today().isoformat(), "calls": 0}


def calls_today() -> int:
    return int(_load_counter().get("calls", 0))


def _consume_call() -> None:
    data = _load_counter()
    if data["calls"] >= DAILY_CALL_CAP:
        raise LLMUnavailable(f"daily cap of {DAILY_CALL_CAP} model calls reached; try again tomorrow")
    data["calls"] += 1
    try:
        COUNTER_PATH.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass


def estimate_cost_usd(usage: dict, model: str = MODEL) -> float:
    price_in, price_out = PRICE_PER_MTOK.get(model, PRICE_PER_MTOK["claude-opus-5"])
    return round(
        usage.get("input_tokens", 0) / 1e6 * price_in + usage.get("output_tokens", 0) / 1e6 * price_out,
        4,
    )


@dataclass
class LLMResponse:
    text: str
    parsed: Any
    usage: dict = field(default_factory=dict)
    model: str = MODEL
    stop_reason: str = ""


def _client():
    if not api_key_present():
        raise LLMUnavailable("no ANTHROPIC_API_KEY in the environment")
    import anthropic

    headers = {}
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    if workspace:  # keys that are not scoped to a workspace must name one per request
        headers["anthropic-workspace-id"] = workspace
    return anthropic.Anthropic(max_retries=2, default_headers=headers or None)


def _usage_dict(response) -> dict:
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "calls": 1,
    }


def _check_stop(response) -> None:
    if getattr(response, "stop_reason", "") == "refusal":
        details = getattr(response, "stop_details", None)
        why = getattr(details, "explanation", None) or "no explanation"
        raise LLMUnavailable(f"model declined the request ({why})")
    if getattr(response, "stop_reason", "") == "max_tokens":
        raise LLMUnavailable("model output was cut off at max_tokens")


def call_structured(system: str, user: str, output_model, max_tokens: int = 16000, effort: str = "high") -> LLMResponse:
    """One structured-output call; returns the validated pydantic object in .parsed.

    max_tokens leaves room for the model's adaptive thinking, which counts
    against the same limit as the answer.
    """
    import anthropic

    client = _client()
    _consume_call()
    try:
        response = client.messages.parse(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_model,
            output_config={"effort": effort},
        )
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable(f"authentication failed: {exc.message}") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable(f"rate limited: {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable(f"connection error: {exc}") from exc
    except ValueError as exc:  # pydantic ValidationError: truncated or mismatched JSON
        raise LLMUnavailable(f"unparseable structured output: {str(exc)[:200]}") from exc
    _check_stop(response)
    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        raise LLMUnavailable("model returned no parseable output")
    text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "")
    return LLMResponse(text=text, parsed=parsed, usage=_usage_dict(response), model=MODEL, stop_reason=response.stop_reason or "")


def call_text(system: str, user: str, max_tokens: int = 16000, effort: str = "high") -> LLMResponse:
    """One plain text call."""
    import anthropic

    client = _client()
    _consume_call()
    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
        )
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable(f"authentication failed: {exc.message}") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable(f"rate limited: {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable(f"connection error: {exc}") from exc
    _check_stop(response)
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    if not text:
        raise LLMUnavailable("model returned no text")
    return LLMResponse(text=text, parsed=None, usage=_usage_dict(response), model=MODEL, stop_reason=response.stop_reason or "")
