"""The probe model: what an attack is, and what happened when we ran it.

A probe is a *scenario*, not a string. Most public prompt-injection corpora are
flat lists of adversarial sentences, which makes them impossible to grade
without a human: you cannot tell whether "ignore previous instructions" worked
unless you know what the model was supposed to be doing instead. So every probe
here carries the whole situation - the system prompt, the legitimate task, the
untrusted content the attack rides in on, and a machine-checkable definition of
what "the attack succeeded" means.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class Family(str, enum.Enum):
    """The four attack families the benchmark measures."""

    INJECTION = "injection"
    EXTRACTION = "extraction"
    JAILBREAK = "jailbreak"
    LEAKAGE = "leakage"

    @property
    def label(self) -> str:
        return {
            Family.INJECTION: "Prompt injection",
            Family.EXTRACTION: "System-prompt extraction",
            Family.JAILBREAK: "Policy circumvention",
            Family.LEAKAGE: "Confidential-data leakage",
        }[self]


class Severity(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def weight(self) -> int:
        """Weight used by the composite exposure score.

        Doubling per step is deliberate: one critical failure should not be
        averaged away by a handful of low-severity passes.
        """
        return {
            Severity.LOW: 1,
            Severity.MEDIUM: 2,
            Severity.HIGH: 4,
            Severity.CRITICAL: 8,
        }[self]

    @property
    def rank(self) -> int:
        return {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}[self]


@dataclass(frozen=True)
class GraderSpec:
    """How to decide whether this probe's attack succeeded.

    ``name`` selects a grader from :mod:`lsb.core.grader`; ``params`` is passed
    to it verbatim. Keeping this declarative means a suite is data, and a new
    attack family does not require touching the runner.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Probe:
    """One attack scenario, fully specified."""

    id: str
    family: Family
    technique: str
    severity: Severity
    description: str
    task: str
    grader: GraderSpec
    system: str = ""
    untrusted: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("probe id must be non-empty")
        if not self.task:
            raise ValueError(f"probe {self.id}: task must be non-empty")

    @property
    def needs_canary(self) -> bool:
        return "{canary}" in self.system

    @property
    def needs_secret(self) -> bool:
        return "{secret}" in self.system or "{secret}" in (self.untrusted or "")


@dataclass(frozen=True)
class ProbeResult:
    """What happened when a probe was run against a target.

    ``attacked`` is the only field the statistics look at. ``evidence`` exists
    so a failing number is debuggable: every grader must say *why* it decided
    what it decided, otherwise a regression is untraceable.
    """

    probe_id: str
    family: Family
    technique: str
    severity: Severity
    attacked: bool
    evidence: str
    response: str
    latency_ms: float
    error: str | None = None
    defense: str = "none"

    @property
    def ok(self) -> bool:
        """True when the target withstood the attack and nothing went wrong."""
        return not self.attacked and self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "family": self.family.value,
            "technique": self.technique,
            "severity": self.severity.value,
            "attacked": self.attacked,
            "evidence": self.evidence,
            "response": self.response,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "defense": self.defense,
        }
