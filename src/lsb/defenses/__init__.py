"""Mitigations you can switch on to measure their effect.

The reason this package exists is that "we added a guardrail" is normally
unfalsifiable. A defense here is an object the runner wraps around the target,
and the only claim the benchmark makes about it is the difference in
attack-success rate it produced, with a confidence interval.

Defenses split into two kinds, and the distinction matters more than it looks:

**Mechanical** defenses change bytes. An input filter that deletes an injected
payload, or an output screen that redacts a token, has an effect that is fully
observable from outside the model - it holds against a simulated target exactly
as it holds against a real one.

**Prompt-level** defenses ask the model to behave differently. Their effect
cannot be measured against a simulated target at all; you can only declare what
you assume it would be. The runner refuses to present those numbers as
observations - see ``Defense.mechanical`` and the ``simulated`` flag on the run
report.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..targets import TargetRequest, TargetResponse


@runtime_checkable
class Defense(Protocol):
    name: str
    description: str
    #: True when the defense's effect is observable outside the model.
    mechanical: bool

    def guard(self, request: TargetRequest) -> TargetRequest:
        """Transform the request before it reaches the target."""

    def screen(self, request: TargetRequest, response: TargetResponse) -> TargetResponse:
        """Transform the response before it reaches the grader."""


class NoDefense:
    """The baseline every comparison is measured against."""

    name = "none"
    description = "No mitigation. The number every other row is compared to."
    mechanical = True

    def guard(self, request: TargetRequest) -> TargetRequest:
        return request

    def screen(self, request: TargetRequest, response: TargetResponse) -> TargetResponse:
        return response


def load(name: str | None) -> Defense:
    """Resolve a defense by name."""
    key = (name or "none").strip().lower()
    if key in {"none", "off", ""}:
        return NoDefense()
    if key in {"spotlight", "spotlighting"}:
        from .spotlight import Spotlight

        return Spotlight()
    if key in {"input-filter", "filter", "screen"}:
        from .input_filter import InputFilter

        return InputFilter()
    if key in {"output-filter", "canary-filter", "output-screen"}:
        from .output_filter import OutputFilter

        return OutputFilter()
    if key in {"defense-in-depth", "all", "stack"}:
        from .stack import DefenseStack

        return DefenseStack()
    raise ValueError(
        f"unknown defense {name!r}; expected none, spotlight, input-filter, "
        "output-filter or defense-in-depth"
    )


def catalogue() -> tuple[str, ...]:
    return ("none", "spotlight", "input-filter", "output-filter", "defense-in-depth")


__all__ = ["Defense", "NoDefense", "catalogue", "load"]
