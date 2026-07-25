"""All three mitigations at once.

Worth measuring separately because layered defenses do not compose the way the
arithmetic suggests. Two mitigations that each cut attack success in half do not
give you a quarter: they overlap heavily on the blunt probes and both miss the
same obfuscated ones, so the stack's residual is close to the residual of the
strongest layer alone. Seeing that in the numbers is the argument against buying
a fourth guardrail.
"""

from __future__ import annotations

from ..targets import TargetRequest, TargetResponse
from .input_filter import InputFilter
from .output_filter import OutputFilter
from .spotlight import Spotlight


class DefenseStack:
    name = "defense-in-depth"
    description = "Spotlighting + input filtering + output screening, applied in that order."
    mechanical = False  # contains a prompt-level layer

    def __init__(self) -> None:
        self.layers = (InputFilter(), Spotlight(), OutputFilter())

    def guard(self, request: TargetRequest) -> TargetRequest:
        for layer in self.layers:
            request = layer.guard(request)
        return request

    def screen(self, request: TargetRequest, response: TargetResponse) -> TargetResponse:
        for layer in reversed(self.layers):
            response = layer.screen(request, response)
        return response
