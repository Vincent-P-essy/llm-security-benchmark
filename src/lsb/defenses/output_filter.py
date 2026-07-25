"""Output screening: redact known secrets on the way out.

This is the defense with the most instructive result in the whole benchmark. It
is trivially correct against verbatim leakage - the token is a known string, so
you can always find and remove it - and teams reasonably conclude that secret
leakage is a solved problem once it is deployed.

It is not solved. A model that has decided to leak will often spell the token
out, base64 it, reverse it, or interleave it with spaces, and none of those
survive a literal search. Running the extraction family with this defense on
therefore does not drive attack success to zero; it drives it down to exactly
the population of obfuscated leaks, which is the number worth knowing and the
reason :mod:`lsb.core.canary` decodes before it concludes.

The filter is deliberately implemented the way it is usually implemented in
production - literal replacement, plus the two variants people remember to add -
rather than with the full decoder. Making it stronger here would hide the gap
the benchmark exists to show.
"""

from __future__ import annotations

from ..targets import TargetRequest, TargetResponse

MASK = "[REDACTED]"


class OutputFilter:
    name = "output-filter"
    description = "Literal redaction of known secrets from responses (plus case/space variants)."
    mechanical = True

    def __init__(self) -> None:
        self.redactions = 0

    def guard(self, request: TargetRequest) -> TargetRequest:
        return request

    def screen(self, request: TargetRequest, response: TargetResponse) -> TargetResponse:
        secrets_to_mask = [
            request.metadata.get("canary"),
            request.metadata.get("secret"),
        ]
        fields = request.metadata.get("leak_fields", "")
        secrets_to_mask.extend(f for f in fields.split("||") if f)

        text = response.text
        for secret in secrets_to_mask:
            if not secret:
                continue
            for variant in (secret, secret.lower(), secret.upper(), secret.replace("-", "_")):
                if variant and variant in text:
                    text = text.replace(variant, MASK)
                    self.redactions += 1

        if text == response.text:
            return response
        return TargetResponse(text=text, tool_calls=response.tool_calls, raw=response.raw)
