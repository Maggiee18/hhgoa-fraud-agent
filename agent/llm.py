"""
Thin LLM wrapper, pluggable across providers. The LLM is used for
reasoning/synthesis/explanation only — it never invents actions, routes, or
IDs; those come from agent/policy.py and the graph. Every call is logged
(tokens, latency) so CaseAnswer.tokens / latency_s can be populated per the
answer format.

The challenge explicitly allows "the LLM or combination of models of your
choice", so this supports two providers behind one interface
(LLM(...).complete(system, messages, max_tokens, temperature) -> str), so
every caller (agent/investigation.py, scripts/*) is provider-agnostic:

  LLM_PROVIDER=anthropic (default)  -> ANTHROPIC_API_KEY, ANTHROPIC_MODEL
  LLM_PROVIDER=gemini               -> GEMINI_API_KEY (or GOOGLE_API_KEY),
                                        GEMINI_MODEL

Gemini was added because its API has a genuine no-credit-card free tier
(confirmed against Google's own docs — ai.google.dev/gemini-api/docs/pricing),
unlike Anthropic's or OpenAI's, which both require billing before any call
succeeds. Both providers stay supported: Anthropic as the already-tested
fallback if Gemini's free-tier rate limits are ever a problem mid-benchmark.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _default_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    # No explicit choice: prefer whichever key is actually set, Gemini first
    # since it's the free-tier-friendly option this project defaults new
    # setups toward; fall back to "anthropic" (original default) if neither
    # is set, so the existing error message ("ANTHROPIC_API_KEY is not set")
    # still points somewhere sensible.
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "anthropic"


class LLM:
    """Lazy-imports the provider SDK so the rest of the codebase (schemas,
    policy, local graph backend) can be imported/tested without any API key
    being set."""

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None):
        self.provider = (provider or _default_provider()).strip().lower()
        if model:
            self.model = model
        elif self.provider == "gemini":
            self.model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        else:
            self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
        self._client = None
        self.usage = LLMUsage()

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        if self.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                    "and add it to .env."
                )
            from google import genai

            self._client = genai.Client(api_key=api_key)
        elif self.provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in."
                )
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            raise RuntimeError(f"Unknown LLM_PROVIDER: {self.provider!r} (expected 'anthropic' or 'gemini')")
        return self._client

    def complete(
        self,
        system: str,
        messages: List[dict],
        max_tokens: int = 2000,
        temperature: float = 0.2,
    ) -> str:
        client = self._ensure_client()
        t0 = time.time()
        fn = _complete_gemini if self.provider == "gemini" else _complete_anthropic
        text, in_tok, out_tok = _with_retry(
            lambda: fn(client, self.model, system, messages, max_tokens, temperature)
        )
        dt = time.time() - t0
        self.usage.input_tokens += in_tok
        self.usage.output_tokens += out_tok
        self.usage.latency_s += dt
        return text


_RETRYABLE_MARKERS = (
    "503", "overloaded", "unavailable", "rate limit", "rate_limit",
    "429", "too many requests", "internal server error", "500",
    "timeout", "timed out", "connection reset", "temporarily",
)


def _is_retryable(e: Exception) -> bool:
    """No dependency on either provider's exception hierarchy (their error
    classes differ and this must handle both) — match on the message text
    instead, which both SDKs populate with the HTTP status/reason. Errs
    toward retrying: a case run failing outright over one flaky call across
    a 20-case benchmark is worse than a few extra seconds of backoff on a
    genuinely non-retryable error."""
    msg = str(e).lower()
    return any(marker in msg for marker in _RETRYABLE_MARKERS)


def _with_retry(fn, attempts: int = 4, base_delay: float = 2.0):
    """Exponential backoff (2s, 4s, 8s) for transient provider-side errors —
    observed in practice: Gemini's flash models returning 503 UNAVAILABLE
    under load. Non-retryable errors (bad auth, bad request, our own
    TypeErrors) propagate immediately on the first attempt."""
    last_exc = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if not _is_retryable(e) or attempt == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))
    raise last_exc  # pragma: no cover — unreachable, satisfies type checkers


def _complete_anthropic(client, model, system, messages, max_tokens, temperature) -> Tuple[str, int, int]:
    kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
    try:
        resp = client.messages.create(temperature=temperature, **kwargs)
    except TypeError as e:
        # Some anthropic SDK versions installed at runtime have dropped
        # `temperature` as a bare top-level kwarg (observed: 1.8.0, which
        # also renamed several other params vs. the 0.x series this was
        # first written against). temperature is a soft determinism
        # preference, not something correctness depends on, so retry
        # without it rather than hard-failing every LLM call over it.
        if "temperature" not in str(e):
            raise
        resp = client.messages.create(**kwargs)
    text = "".join(block.text for block in resp.content if block.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def _complete_gemini(client, model, system, messages, max_tokens, temperature) -> Tuple[str, int, int]:
    """Every call site in this codebase (agent/investigation.py's
    synthesize_with_llm / write_sar_narrative) sends exactly one user
    message, but this maps a general messages list for robustness: Gemini
    uses role "model" where Anthropic/OpenAI use "assistant", and takes
    `contents` as a list of {role, parts:[{text}]} dicts rather than
    {role, content}."""
    from google.genai import types

    contents = [
        {"role": ("model" if m.get("role") == "assistant" else "user"), "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    config_kwargs = dict(
        system_instruction=system,
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    # Flash-tier "thinking" models can spend part of max_output_tokens on
    # internal reasoning before emitting the actual answer, which truncated
    # our structured-JSON output in testing (observed on gemini-3.6-flash).
    # These are short, low-ambiguity extraction/formatting tasks (map
    # evidence -> a fixed JSON schema), not tasks that benefit from extended
    # reasoning, so ask for a minimal thinking budget. Guarded: older
    # SDK/model combinations may not accept `thinking_config` at all.
    try:
        config = types.GenerateContentConfig(
            **config_kwargs, thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        resp = client.models.generate_content(model=model, contents=contents, config=config)
    except Exception:
        # Covers: AttributeError (no ThinkingConfig in this SDK version),
        # TypeError/pydantic ValidationError (field rejected by this model),
        # or the API itself rejecting the param. Observed in practice:
        # gemini-3.5-flash-lite rejects thinking_config with a generic
        # `400 INVALID_ARGUMENT` whose message never mentions "thinking" and
        # isn't a TypeError/AttributeError, so this used to be pattern-matched
        # and missed — every call hard-failed. thinking_budget is a
        # token-budget optimization, not something correctness depends on, so
        # retry once, unconditionally, without it rather than trying to
        # enumerate every way a model/SDK combo can reject the param.
        config = types.GenerateContentConfig(**config_kwargs)
        resp = client.models.generate_content(model=model, contents=contents, config=config)
    usage = resp.usage_metadata
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    return resp.text or "", in_tok, out_tok
