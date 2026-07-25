"""Orchestration: mint tokens, render scenarios, send, grade, collect.

The runner is the only place that knows the secret values in play, which is what
lets grading stay a string comparison. Probes never contain a real secret - they
contain ``{canary}``, ``{secret}`` and ``{sentinel}`` placeholders that get
filled per trial, so the same probe file can be run a thousand times without the
target ever seeing the same token twice.
"""

from __future__ import annotations

import base64
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ..defenses import Defense, NoDefense
from ..targets import Target, TargetRequest, TargetResponse
from . import grader as graders
from .probe import Probe, ProbeResult
from .suite import Suite

UTC = timezone.utc


@dataclass(frozen=True)
class RunConfig:
    """Knobs for one benchmark run."""

    seed: int = 1337
    repeat: int = 1
    workers: int = 8
    response_limit: int = 400
    #: When True, tokens come from os.urandom instead of the seed. Use it against
    #: real targets if you are worried about a token being memorised between
    #: runs; it costs you exact reproducibility.
    fresh_tokens: bool = False

    def __post_init__(self) -> None:
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")


@dataclass
class RunReport:
    """Everything one run produced, plus the caveats attached to it."""

    suite: str
    target: str
    defense: str
    config: RunConfig
    results: list[ProbeResult] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0
    #: True when the numbers depend on a simulated assumption rather than an
    #: observation - a prompt-level defense measured against a mock target.
    #: Every renderer must surface this; a caveat nobody sees is not a caveat.
    simulated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[ProbeResult]:
        return [r for r in self.results if r.error]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "target": self.target,
            "defense": self.defense,
            "simulated": self.simulated,
            "notes": self.notes,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 2),
            "config": {
                "seed": self.config.seed,
                "repeat": self.config.repeat,
                "workers": self.config.workers,
                "fresh_tokens": self.config.fresh_tokens,
            },
            "results": [r.to_dict() for r in self.results],
        }


def _derive(seed: int, probe_id: str, trial: int, kind: str, nbytes: int = 6) -> str:
    """Deterministically derive a token from the run seed.

    Reproducibility is the whole point: given a seed, a suite and a mock
    profile, two people on two machines get identical output, so the numbers in
    a README can be checked rather than believed.
    """
    digest = hashlib.blake2b(
        f"{seed}:{probe_id}:{trial}:{kind}".encode(), digest_size=nbytes
    ).hexdigest()
    return digest


def _mint(config: RunConfig, probe: Probe, trial: int) -> dict[str, str]:
    if config.fresh_tokens:
        import secrets as _secrets

        canary_hex = _secrets.token_hex(6)
        secret_hex = _secrets.token_hex(5)
        sentinel_hex = _secrets.token_hex(5)
    else:
        canary_hex = _derive(config.seed, probe.id, trial, "canary")
        secret_hex = _derive(config.seed, probe.id, trial, "secret", 5)
        sentinel_hex = _derive(config.seed, probe.id, trial, "sentinel", 5)

    sentinel = f"LSB-SENTINEL-{sentinel_hex}"
    return {
        "canary": f"LSB-CANARY-{canary_hex}",
        "secret": f"SEC-{secret_hex.upper()}",
        "sentinel": sentinel,
        "sentinel_b64": base64.b64encode(sentinel.encode()).decode(),
    }


def _render(text: str | None, tokens: dict[str, str]) -> str | None:
    if text is None:
        return None
    out = text
    for key, value in tokens.items():
        out = out.replace("{" + key + "}", value)
    return out


def _build_request(probe: Probe, tokens: dict[str, str], trial: int) -> TargetRequest:
    leak_fields = probe.grader.params.get("fields", []) if probe.grader.name == "record_leak" else []
    expected = probe.grader.params.get("expected_signal", [])

    return TargetRequest(
        system=_render(probe.system, tokens) or "",
        user=_render(probe.task, tokens) or "",
        untrusted=_render(probe.untrusted, tokens),
        metadata={
            "probe_id": probe.id,
            "technique": probe.technique,
            "family": probe.family.value,
            "trial": str(trial),
            "canary": tokens["canary"],
            "secret": tokens["secret"],
            "sentinel": tokens["sentinel"],
            "leak_fields": "||".join(str(f) for f in leak_fields),
            "forbidden_tools": ",".join(probe.grader.params.get("tools", [])),
            "expected_signal": expected[0] if expected else "",
        },
    )


def _grade(
    probe: Probe, tokens: dict[str, str], response: TargetResponse, defense_name: str,
    latency_ms: float, limit: int,
) -> ProbeResult:
    ctx = graders.GradeContext(
        probe=probe,
        response=response.text,
        canary=tokens["canary"],
        secret=tokens["secret"],
        sentinel=tokens["sentinel"],
        tool_calls=response.tool_calls,
    )
    attacked, evidence = graders.run(probe.grader.name, ctx)
    stored = response.text if len(response.text) <= limit else response.text[:limit] + "..."
    return ProbeResult(
        probe_id=probe.id,
        family=probe.family,
        technique=probe.technique,
        severity=probe.severity,
        attacked=attacked,
        evidence=evidence,
        response=stored,
        latency_ms=latency_ms,
        defense=defense_name,
    )


def run_probe(
    probe: Probe,
    trial: int,
    target: Target,
    defense: Defense,
    config: RunConfig,
) -> ProbeResult:
    """Run one probe once. Never raises - target failures become results."""
    tokens = _mint(config, probe, trial)
    request = _build_request(probe, tokens, trial)

    started = time.perf_counter()
    try:
        guarded = defense.guard(request)
        response = target.send(guarded)
        # Screening is graded against the *original* request's tokens: a defense
        # must not be able to hide a leak by rewriting what counts as the secret.
        response = defense.screen(request, response)
    except Exception as exc:  # noqa: BLE001 - a broken target must not abort the run
        latency = (time.perf_counter() - started) * 1000
        return ProbeResult(
            probe_id=probe.id,
            family=probe.family,
            technique=probe.technique,
            severity=probe.severity,
            attacked=False,
            evidence="probe did not complete",
            response="",
            latency_ms=latency,
            error=f"{type(exc).__name__}: {exc}",
            defense=defense.name,
        )

    latency = (time.perf_counter() - started) * 1000
    return _grade(probe, tokens, response, defense.name, latency, config.response_limit)


def run(
    suite: Suite,
    target: Target,
    defense: Defense | None = None,
    config: RunConfig | None = None,
    progress: Any = None,
) -> RunReport:
    """Run a whole suite against a target, optionally behind a defense."""
    defense = defense or NoDefense()
    config = config or RunConfig()

    report = RunReport(
        suite=suite.slug,
        target=target.name,
        defense=defense.name,
        config=config,
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    is_mock = target.name.startswith("mock") or target.name == "echo"
    if is_mock and not getattr(defense, "mechanical", True):
        report.simulated = True
        report.notes.append(
            f"{defense.name} is a prompt-level defense and the target is simulated: its "
            "effect here is an assumption baked into the mock profile, not a measurement. "
            "Run against a real target before quoting this number."
        )
    if is_mock:
        report.notes.append(
            "Target is simulated. These numbers exercise the harness - graders, "
            "statistics, reproducibility - and say nothing about any real model."
        )

    jobs = [(probe, trial) for trial in range(config.repeat) for probe in suite.probes]
    started = time.perf_counter()

    if config.workers == 1:
        for probe, trial in jobs:
            report.results.append(run_probe(probe, trial, target, defense, config))
            if progress is not None:
                progress(len(report.results), len(jobs))
    else:
        with ThreadPoolExecutor(max_workers=config.workers) as pool:
            futures = [
                pool.submit(run_probe, probe, trial, target, defense, config)
                for probe, trial in jobs
            ]
            for done, future in enumerate(futures, start=1):
                report.results.append(future.result())
                if progress is not None:
                    progress(done, len(jobs))

    report.duration_s = time.perf_counter() - started
    report.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
    return report
