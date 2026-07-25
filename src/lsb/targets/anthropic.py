"""Adapter for a real target behind the Anthropic Messages API.

Two things in here are specific to running a *security* benchmark rather than a
capability one, and both would silently corrupt the numbers if handled the way
an ordinary client handles them.

**A safety refusal is not a successful defense.** The API can decline a request
at the classifier level: HTTP 200, ``stop_reason == "refusal"``, empty or
partial content. That is the platform declining to answer, not the application
resisting the attack, and scoring it as "the attack failed" credits a defense
that never ran. It is recorded as a distinct outcome instead. (It also means
``response.content[0]`` can be an IndexError - the ordinary client bug that
this endpoint's refusal path produces.)

**Errors must not become passes.** Rate limits, overloads and timeouts raise
``TargetError``; the runner turns that into an errored result which the
statistics exclude from the denominator. Swallowing them and returning an empty
string would score every failed request as an attack that didn't work.
"""

from __future__ import annotations

import os
from typing import Any

from . import TargetError, TargetRequest, TargetResponse

#: Sensible default. Override per run with `--target anthropic:claude-sonnet-5`.
DEFAULT_MODEL = "claude-opus-5"

#: Marker recorded in the response when the platform declined the request.
#: Deliberately not something a grader can mistake for leaked content.
REFUSAL_MARKER = "[[LSB: request declined by platform safety classifiers]]"


class AnthropicTarget:
    """Sends probes to a Claude model through the Messages API.

    The application shape modelled here is the common one: a system prompt
    holding the policy, a user turn holding the request, and untrusted content
    fenced into its own content block. That last part matters - it is the
    channel injection actually arrives through, and a benchmark that flattens
    it into the user turn cannot tell you anything about defenses that depend
    on the distinction.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        thinking: bool = False,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise TargetError(
                "the anthropic SDK is not installed; "
                "install it with: pip install 'llm-security-benchmark[anthropic]'"
            ) from exc

        self._sdk = anthropic
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.name = f"anthropic:{self.model}"

        # A bare constructor also resolves an `ant auth login` profile, so an
        # unset ANTHROPIC_API_KEY does not mean there are no credentials.
        kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": max_retries}
        if api_key or os.environ.get("ANTHROPIC_API_KEY"):
            kwargs["api_key"] = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.client = anthropic.Anthropic(**kwargs)

    def send(self, request: TargetRequest) -> TargetResponse:
        content: list[dict[str, Any]] = [{"type": "text", "text": request.user}]
        if request.untrusted:
            content.append({"type": "text", "text": request.untrusted})

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": content}],
        }
        # Thinking is on by default on Opus 5 and shares the max_tokens budget
        # with the reply. Off by default here: a probe response is short, and a
        # long think can eat the budget and truncate the very text being graded.
        if not self.thinking:
            params["thinking"] = {"type": "disabled"}

        try:
            response = self.client.messages.create(**params)
        except self._sdk.NotFoundError as exc:
            raise TargetError(f"unknown model {self.model!r}: {exc}") from exc
        except self._sdk.AuthenticationError as exc:
            raise TargetError(f"authentication failed: {exc}") from exc
        except self._sdk.RateLimitError as exc:
            raise TargetError(f"rate limited: {exc}") from exc
        except self._sdk.APIStatusError as exc:
            raise TargetError(f"API error {exc.status_code}: {exc}") from exc
        except self._sdk.APIConnectionError as exc:
            raise TargetError(f"connection failed: {exc}") from exc

        # Checked before touching content: on a refusal the list can be empty.
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            return TargetResponse(
                text=REFUSAL_MARKER,
                raw={
                    "stop_reason": "refusal",
                    "refusal_category": category,
                    "model": response.model,
                    "platform_refusal": True,
                },
            )

        text = "".join(block.text for block in response.content if block.type == "text")
        tool_calls = tuple(
            block.name for block in response.content if block.type == "tool_use"
        )

        return TargetResponse(
            text=text,
            tool_calls=tool_calls,
            raw={
                "stop_reason": response.stop_reason,
                "model": response.model,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
