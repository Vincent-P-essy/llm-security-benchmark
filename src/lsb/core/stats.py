"""Statistics for attack-success rates.

A benchmark that reports "3/10 attacks succeeded" and stops there is not
measuring anything you can act on. Ten trials give an attack-success rate whose
95% interval spans roughly 11% to 60% - so a "improvement" from 30% to 20% on
ten probes is indistinguishable from noise, and shipping a defense on that
basis is guesswork.

Every rate this package reports therefore carries a Wilson score interval, and
defense comparisons go through :func:`compare` which refuses to call a
difference real when the intervals overlap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 1.959964 -> two-sided 95%. Hard-coded to keep scipy out of the dependency set.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Rate:
    """A proportion with a confidence interval."""

    successes: int
    trials: int
    point: float
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def pct(self) -> str:
        return f"{self.point * 100:.1f}%"

    @property
    def ci_pct(self) -> str:
        return f"{self.low * 100:.1f}-{self.high * 100:.1f}%"

    def to_dict(self) -> dict[str, float | int]:
        return {
            "successes": self.successes,
            "trials": self.trials,
            "point": round(self.point, 4),
            "ci_low": round(self.low, 4),
            "ci_high": round(self.high, 4),
        }


def wilson(successes: int, trials: int, z: float = Z_95) -> Rate:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because attack-success rates live
    near 0 and 1, exactly where the textbook interval breaks down - it happily
    reports a lower bound below zero for 0/20, which is how you end up claiming
    a defense is airtight on twenty samples.
    """
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes ({successes}) must be within 0..trials ({trials})")
    if trials == 0:
        return Rate(0, 0, 0.0, 0.0, 1.0)

    p = successes / trials
    denom = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return Rate(successes, trials, p, max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True)
class Comparison:
    """Result of comparing two attack-success rates."""

    baseline: Rate
    variant: Rate
    delta: float
    significant: bool
    verdict: str

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_dict(),
            "variant": self.variant.to_dict(),
            "delta": round(self.delta, 4),
            "significant": self.significant,
            "verdict": self.verdict,
        }


def compare(baseline: Rate, variant: Rate) -> Comparison:
    """Compare a defended run against an undefended one.

    Significance here is the conservative reading: non-overlapping 95%
    intervals. That is stricter than a two-proportion z-test, and the
    conservatism is the point - the failure mode this guards against is
    shipping a defense that did nothing.
    """
    delta = variant.point - baseline.point
    separated = variant.high < baseline.low or variant.low > baseline.high

    if baseline.trials == 0 or variant.trials == 0:
        return Comparison(baseline, variant, delta, False, "not enough data to compare")
    if not separated:
        return Comparison(
            baseline, variant, delta, False,
            "no measurable effect (confidence intervals overlap)",
        )
    if delta < 0:
        return Comparison(
            baseline, variant, delta, True,
            f"attack success down {abs(delta) * 100:.1f} points",
        )
    return Comparison(
        baseline, variant, delta, True,
        f"attack success UP {delta * 100:.1f} points - the defense made things worse",
    )


def exposure_score(weighted_successes: float, weighted_total: float) -> float:
    """Severity-weighted composite, 0 (nothing worked) to 100 (everything did).

    Reported alongside the raw rates, never instead of them: a single number is
    what people quote, and it hides which family failed.
    """
    if weighted_total <= 0:
        return 0.0
    return round(100.0 * weighted_successes / weighted_total, 1)


def grade(score: float) -> str:
    """Letter grade for the exposure score, for at-a-glance reporting."""
    for threshold, letter in ((5, "A"), (15, "B"), (30, "C"), (50, "D"), (75, "E")):
        if score < threshold:
            return letter
    return "F"
