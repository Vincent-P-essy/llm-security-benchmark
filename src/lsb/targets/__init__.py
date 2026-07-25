"""Adapters for the system under test.

The unit of measurement here is an *application*, not a model. A raw model
behind a chat endpoint and the same model wrapped in a retrieval pipeline with
a tool loop have completely different exposure, and only the second one is what
anyone actually deploys. So the interface is deliberately small - hand it a
system prompt, a user turn and an optional untrusted blob, get back text and
any tool calls - and anything that can satisfy it can be benchmarked, including
a target that lives behind someone else's HTTP API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class TargetRequest:
    """One turn sent to the system under test.

    ``untrusted`` is kept separate from ``user`` on purpose. Retrieved
    documents, tool output and inbound email are the channels injection
    actually arrives through, and a target that concatenates them into the user
    turn has already lost the distinction a defense would need.
    """

    system: str
    user: str
    untrusted: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetResponse:
    text: str
    tool_calls: tuple[str, ...] = ()
    raw: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class Target(Protocol):
    """Anything that can answer a :class:`TargetRequest`."""

    name: str

    def send(self, request: TargetRequest) -> TargetResponse:
        ...


class TargetError(RuntimeError):
    """Raised when a target cannot answer at all (network, auth, quota)."""


def load(spec: str, **kwargs: object) -> Target:
    """Resolve a target from a CLI-style spec such as ``mock:hardened``.

    Imports are done lazily so that running the offline suite never requires
    the vendor SDKs to be installed.
    """
    kind, _, option = spec.partition(":")
    kind = kind.strip().lower()

    if kind == "mock":
        from .mock import MockTarget

        return MockTarget(profile=option or "naive", **kwargs)  # type: ignore[arg-type]
    if kind == "anthropic":
        from .anthropic import AnthropicTarget

        return AnthropicTarget(model=option or None, **kwargs)  # type: ignore[arg-type]
    if kind == "echo":
        from .mock import EchoTarget

        return EchoTarget()

    raise ValueError(f"unknown target {spec!r}; expected mock:<profile>, anthropic:<model> or echo")


__all__ = [
    "Target",
    "TargetError",
    "TargetRequest",
    "TargetResponse",
    "load",
]
