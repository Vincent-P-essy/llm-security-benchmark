"""Aggregation and rendering.

The ordering choices here are the opinionated part. Tables are sorted by attack
success descending, so the thing that failed most is the first thing read; the
composite exposure score is printed *after* the per-family breakdown, never
instead of it; and any caveat attached to the run is printed before the numbers
rather than in a footnote, because a caveat below the table is a caveat nobody
reads.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .probe import Family, ProbeResult, Severity
from .runner import RunReport
from .stats import Comparison, Rate, compare, exposure_score, grade, wilson

SEVERITY_COLOUR = {
    Severity.CRITICAL: "bright_red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "cyan",
}


def _rate_colour(point: float) -> str:
    if point >= 0.5:
        return "bright_red"
    if point >= 0.25:
        return "red"
    if point >= 0.10:
        return "yellow"
    if point > 0:
        return "green"
    return "bright_green"


@dataclass(frozen=True)
class Aggregate:
    """Rates sliced every way that turned out to be worth looking at."""

    overall: Rate
    by_family: dict[Family, Rate]
    by_technique: dict[str, Rate]
    by_severity: dict[Severity, Rate]
    exposure: float
    letter: str
    errors: int
    mean_latency_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": self.overall.to_dict(),
            "exposure_score": self.exposure,
            "grade": self.letter,
            "errors": self.errors,
            "mean_latency_ms": round(self.mean_latency_ms, 1),
            "by_family": {f.value: r.to_dict() for f, r in self.by_family.items()},
            "by_technique": {t: r.to_dict() for t, r in self.by_technique.items()},
            "by_severity": {s.value: r.to_dict() for s, r in self.by_severity.items()},
        }


def aggregate(results: list[ProbeResult]) -> Aggregate:
    """Turn a flat result list into every rate the report needs.

    Errored probes are excluded from the denominators rather than counted as
    passes. Counting a timeout as "the attack failed" is how a rate-limited run
    turns into a clean bill of health.
    """
    graded = [r for r in results if r.error is None]
    errors = len(results) - len(graded)

    fam_hits: dict[Family, list[bool]] = defaultdict(list)
    tech_hits: dict[str, list[bool]] = defaultdict(list)
    sev_hits: dict[Severity, list[bool]] = defaultdict(list)

    weighted_success = 0.0
    weighted_total = 0.0

    for r in graded:
        fam_hits[r.family].append(r.attacked)
        tech_hits[r.technique].append(r.attacked)
        sev_hits[r.severity].append(r.attacked)
        weighted_total += r.severity.weight
        if r.attacked:
            weighted_success += r.severity.weight

    def rate(hits: list[bool]) -> Rate:
        return wilson(sum(hits), len(hits))

    latencies = [r.latency_ms for r in graded] or [0.0]

    return Aggregate(
        overall=rate([r.attacked for r in graded]),
        by_family={f: rate(h) for f, h in fam_hits.items()},
        by_technique={t: rate(h) for t, h in tech_hits.items()},
        by_severity={s: rate(h) for s, h in sev_hits.items()},
        exposure=exposure_score(weighted_success, weighted_total),
        letter=grade(exposure_score(weighted_success, weighted_total)),
        errors=errors,
        mean_latency_ms=sum(latencies) / len(latencies),
    )


# --------------------------------------------------------------------------
# terminal rendering
# --------------------------------------------------------------------------


def _rate_cells(rate: Rate) -> tuple[Text, str, str]:
    return (
        Text(rate.pct, style=f"bold {_rate_colour(rate.point)}"),
        rate.ci_pct,
        f"{rate.successes}/{rate.trials}",
    )


def render_terminal(report: RunReport, console: Console | None = None) -> None:
    """Print the full report to a terminal."""
    console = console or Console()
    agg = aggregate(report.results)

    header = Text()
    header.append("target   ", style="dim")
    header.append(f"{report.target}\n", style="bold")
    header.append("suite    ", style="dim")
    header.append(f"{report.suite}  ", style="bold")
    header.append(f"({len(report.results)} trials, seed {report.config.seed}, ", style="dim")
    header.append(f"repeat {report.config.repeat})\n", style="dim")
    header.append("defense  ", style="dim")
    header.append(report.defense, style="bold")
    console.print(Panel(header, title="llm-security-benchmark", border_style="blue", expand=False))

    for note in report.notes:
        style = "bold yellow" if report.simulated else "yellow"
        console.print(Text(f"  ! {note}", style=style))
    if report.notes:
        console.print()

    fam_table = Table(title="Attack success by family", title_style="bold", header_style="dim")
    fam_table.add_column("Family", style="bold")
    fam_table.add_column("ASR", justify="right")
    fam_table.add_column("95% CI", justify="right", style="dim")
    fam_table.add_column("hits", justify="right", style="dim")
    for family in sorted(agg.by_family, key=lambda f: -agg.by_family[f].point):
        pct, ci, hits = _rate_cells(agg.by_family[family])
        fam_table.add_row(family.label, pct, ci, hits)
    console.print(fam_table)
    console.print()

    tech_table = Table(title="Attack success by technique", title_style="bold", header_style="dim")
    tech_table.add_column("Technique", style="bold")
    tech_table.add_column("ASR", justify="right")
    tech_table.add_column("95% CI", justify="right", style="dim")
    tech_table.add_column("hits", justify="right", style="dim")
    ranked = sorted(agg.by_technique, key=lambda t: -agg.by_technique[t].point)
    for technique in ranked[:8]:
        pct, ci, hits = _rate_cells(agg.by_technique[technique])
        tech_table.add_row(technique, pct, ci, hits)
    console.print(tech_table)
    console.print()

    sev_table = Table(title="Attack success by severity", title_style="bold", header_style="dim")
    sev_table.add_column("Severity", style="bold")
    sev_table.add_column("ASR", justify="right")
    sev_table.add_column("hits", justify="right", style="dim")
    for severity in sorted(agg.by_severity, key=lambda s: -s.rank):
        pct, _, hits = _rate_cells(agg.by_severity[severity])
        sev_table.add_row(
            Text(severity.value, style=SEVERITY_COLOUR[severity]), pct, hits
        )
    console.print(sev_table)
    console.print()

    verdict = Text()
    verdict.append("exposure score  ", style="dim")
    verdict.append(f"{agg.exposure:.1f}/100", style=f"bold {_rate_colour(agg.exposure / 100)}")
    verdict.append("   grade  ", style="dim")
    verdict.append(agg.letter, style=f"bold {_rate_colour(agg.exposure / 100)}")
    verdict.append("\noverall ASR     ", style="dim")
    verdict.append(f"{agg.overall.pct}", style="bold")
    verdict.append(f"  (95% CI {agg.overall.ci_pct})", style="dim")
    if agg.errors:
        verdict.append(f"\nerrors          {agg.errors} probe(s) did not complete", style="red")
    console.print(Panel(verdict, border_style="blue", expand=False))


def render_failures(report: RunReport, console: Console | None = None, limit: int = 12) -> None:
    """Print the individual attacks that landed, worst severity first."""
    console = console or Console()
    hits = [r for r in report.results if r.attacked]
    if not hits:
        console.print("[bold green]No attack succeeded.[/]")
        return

    hits.sort(key=lambda r: (-r.severity.rank, r.probe_id))
    table = Table(
        title=f"Successful attacks ({len(hits)} of {len(report.results)} trials)",
        title_style="bold",
        header_style="dim",
        show_lines=False,
    )
    table.add_column("probe", style="bold")
    table.add_column("sev")
    table.add_column("technique", style="cyan")
    table.add_column("why the grader scored it a hit", style="dim", overflow="fold")

    seen: set[str] = set()
    for r in hits:
        if r.probe_id in seen:
            continue
        seen.add(r.probe_id)
        if len(seen) > limit:
            break
        table.add_row(
            r.probe_id,
            Text(r.severity.value, style=SEVERITY_COLOUR[r.severity]),
            r.technique,
            r.evidence[:88],
        )
    console.print(table)
    if len(seen) > limit:
        console.print(f"[dim]  ... and {len(seen) - limit} more distinct probes[/]")


def render_comparison(
    baseline: RunReport, variants: list[RunReport], console: Console | None = None
) -> list[tuple[str, Comparison]]:
    """Print the defense comparison table and return the comparisons."""
    console = console or Console()
    base_agg = aggregate(baseline.results)

    table = Table(
        title="Does the defense actually move the number?",
        title_style="bold",
        header_style="dim",
    )
    table.add_column("Defense", style="bold")
    table.add_column("ASR", justify="right")
    table.add_column("95% CI", justify="right", style="dim")
    table.add_column("Δ vs none", justify="right")
    table.add_column("Verdict", overflow="fold")

    pct, ci, _ = _rate_cells(base_agg.overall)
    table.add_row("none (baseline)", pct, ci, "-", "[dim]reference[/]")

    out: list[tuple[str, Comparison]] = []
    for variant in variants:
        agg = aggregate(variant.results)
        cmp = compare(base_agg.overall, agg.overall)
        out.append((variant.defense, cmp))
        delta = Text(
            f"{cmp.delta * 100:+.1f} pts",
            style="bright_green" if cmp.delta < 0 and cmp.significant else
                  ("red" if cmp.delta > 0 else "dim"),
        )
        verdict = cmp.verdict
        if variant.simulated:
            verdict += "  [yellow](simulated)[/]"
        pct, ci, _ = _rate_cells(agg.overall)
        table.add_row(variant.defense, pct, ci, delta, verdict)

    console.print(table)
    return out


# --------------------------------------------------------------------------
# file output
# --------------------------------------------------------------------------


def to_json(report: RunReport, path: str | Path) -> Path:
    agg = aggregate(report.results)
    payload = {**report.to_dict(), "aggregate": agg.to_dict()}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def to_markdown(report: RunReport, path: str | Path) -> Path:
    """Write a report suitable for pasting into a PR or a wiki page."""
    agg = aggregate(report.results)
    lines = [
        f"# Benchmark run - {report.target}",
        "",
        f"- **suite**: `{report.suite}`",
        f"- **defense**: `{report.defense}`",
        f"- **trials**: {len(report.results)} "
        f"(seed `{report.config.seed}`, repeat {report.config.repeat})",
        f"- **finished**: {report.finished_at}",
        f"- **exposure score**: **{agg.exposure:.1f}/100** (grade {agg.letter})",
        f"- **overall attack success**: **{agg.overall.pct}** (95% CI {agg.overall.ci_pct})",
        "",
    ]
    if report.notes:
        lines += ["> **Caveats**", ""]
        lines += [f"> - {note}" for note in report.notes]
        lines.append("")

    lines += ["## By family", "", "| Family | ASR | 95% CI | Hits |", "| --- | --- | --- | --- |"]
    for family in sorted(agg.by_family, key=lambda f: -agg.by_family[f].point):
        r = agg.by_family[family]
        lines.append(f"| {family.label} | {r.pct} | {r.ci_pct} | {r.successes}/{r.trials} |")

    lines += [
        "", "## By technique", "",
        "| Technique | ASR | 95% CI | Hits |", "| --- | --- | --- | --- |",
    ]
    for technique in sorted(agg.by_technique, key=lambda t: -agg.by_technique[t].point):
        r = agg.by_technique[technique]
        lines.append(f"| `{technique}` | {r.pct} | {r.ci_pct} | {r.successes}/{r.trials} |")

    hits = [r for r in report.results if r.attacked]
    if hits:
        seen: set[str] = set()
        lines += ["", "## Attacks that landed", "", "| Probe | Severity | Technique | Evidence |",
                  "| --- | --- | --- | --- |"]
        for r in sorted(hits, key=lambda r: (-r.severity.rank, r.probe_id)):
            if r.probe_id in seen:
                continue
            seen.add(r.probe_id)
            lines.append(
                f"| `{r.probe_id}` | {r.severity.value} | `{r.technique}` | {r.evidence} |"
            )

    lines.append("")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
