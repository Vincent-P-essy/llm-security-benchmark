"""Command-line interface.

Five commands, chosen so that each answers one question:

  ``run``      how exposed is this target?
  ``compare``  did that defense actually change anything?
  ``list``     what is in the corpus?
  ``validate`` is the corpus itself sound?
  ``graders``  what can a probe be graded with?

``run`` takes ``--fail-over`` so it can sit in a pipeline as a gate. The
default is off: a security tool that starts failing builds the day it is
installed gets uninstalled the same day.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .core import grader as graders
from .core import report as reporting
from .core.runner import RunConfig, run
from .core.suite import SuiteError, load as load_suite
from .defenses import catalogue, load as load_defense
from .targets import load as load_target


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _select(args: argparse.Namespace):
    suite = load_suite(args.suite)
    if any([args.family, args.technique, args.severity, args.probe, args.tag]):
        suite = suite.select(
            families=_split(args.family),
            techniques=_split(args.technique),
            severities=_split(args.severity),
            ids=_split(args.probe),
            tags=_split(args.tag),
        )
    return suite


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", help="suite file or directory (default: bundled corpus)")
    parser.add_argument("--family", help="comma-separated: injection,extraction,jailbreak,leakage")
    parser.add_argument("--technique", help="comma-separated technique filter")
    parser.add_argument("--severity", help="comma-separated: low,medium,high,critical")
    parser.add_argument("--probe", help="comma-separated probe ids")
    parser.add_argument("--tag", help="comma-separated tag filter")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target", default="mock:naive",
        help="mock:naive | mock:guarded | mock:hardened | echo | anthropic:<model>",
    )
    parser.add_argument("--repeat", type=int, default=1,
                        help="trials per probe; the only lever on interval width")
    parser.add_argument("--seed", type=int, default=1337, help="token derivation seed")
    parser.add_argument("--workers", type=int, default=8, help="concurrent requests")
    parser.add_argument("--fresh-tokens", action="store_true",
                        help="draw tokens randomly instead of from the seed (loses reproducibility)")


def cmd_run(args: argparse.Namespace, console: Console) -> int:
    suite = _select(args)
    target = load_target(args.target)
    defense = load_defense(args.defense)
    config = RunConfig(
        seed=args.seed, repeat=args.repeat, workers=args.workers, fresh_tokens=args.fresh_tokens
    )

    total = len(suite) * args.repeat
    with console.status(f"running {total} trials against {target.name} ...") as status:
        def progress(done: int, all_: int) -> None:
            status.update(f"running {done}/{all_} trials against {target.name} ...")

        report = run(suite, target, defense, config, progress=progress)

    reporting.render_terminal(report, console)
    if args.failures:
        console.print()
        reporting.render_failures(report, console)

    if args.json:
        path = reporting.to_json(report, args.json)
        console.print(f"[dim]wrote {path}[/]")
    if args.markdown:
        path = reporting.to_markdown(report, args.markdown)
        console.print(f"[dim]wrote {path}[/]")

    agg = reporting.aggregate(report.results)
    if args.fail_over is not None and agg.exposure > args.fail_over:
        console.print(
            f"[bold red]FAIL[/] exposure {agg.exposure:.1f} exceeds threshold {args.fail_over:.1f}"
        )
        return 1
    return 0


def cmd_compare(args: argparse.Namespace, console: Console) -> int:
    suite = _select(args)
    config = RunConfig(
        seed=args.seed, repeat=args.repeat, workers=args.workers, fresh_tokens=args.fresh_tokens
    )
    names = _split(args.defenses) or list(catalogue())
    if "none" not in names:
        names.insert(0, "none")

    reports = []
    for name in names:
        target = load_target(args.target)  # fresh instance: defenses keep counters
        with console.status(f"running {len(suite) * args.repeat} trials with defense={name} ..."):
            reports.append(run(suite, target, load_defense(name), config))

    baseline = reports[0]
    console.print()
    comparisons = reporting.render_comparison(baseline, reports[1:], console)
    console.print()

    real = [(n, c) for n, c in comparisons if c.significant and c.delta < 0]
    if real:
        best = min(real, key=lambda pair: pair[1].delta)
        console.print(
            f"[bold green]Best measured mitigation:[/] {best[0]} "
            f"({best[1].delta * 100:+.1f} points, intervals separated)"
        )
    else:
        console.print(
            "[bold yellow]No defense produced a statistically separable improvement.[/] "
            "Raise --repeat before concluding anything: narrow intervals need trials."
        )
    return 0


def cmd_list(args: argparse.Namespace, console: Console) -> int:
    suite = _select(args)
    table = Table(
        title=f"{suite.slug} - {len(suite)} probes", title_style="bold", header_style="dim"
    )
    table.add_column("id", style="bold")
    table.add_column("family", style="magenta")
    table.add_column("technique", style="cyan")
    table.add_column("sev")
    table.add_column("grader", style="dim")
    table.add_column("description", overflow="fold")
    for probe in suite.probes:
        table.add_row(
            probe.id,
            probe.family.value,
            probe.technique,
            probe.severity.value,
            probe.grader.name,
            " ".join(probe.description.split())[:70],
        )
    console.print(table)
    return 0


def cmd_validate(args: argparse.Namespace, console: Console) -> int:
    """Check the corpus is sound before trusting any number it produces.

    The interesting check is the last one: every probe is run against a target
    that leaks everything. A probe that does not register as a hit there has a
    grader that can never fire, so it silently lowers every exposure score it
    appears in. That is a bug in the corpus, and it should be loud.
    """
    try:
        suite = load_suite(args.suite)
    except SuiteError as exc:
        console.print(f"[bold red]invalid suite:[/] {exc}")
        return 1

    console.print(f"[green]OK[/]   parsed {len(suite)} probes from {suite.slug}")

    unknown = sorted({p.grader.name for p in suite.probes} - set(graders.available()))
    if unknown:
        console.print(f"[bold red]FAIL[/] unknown graders: {', '.join(unknown)}")
        return 1
    console.print(f"[green]OK[/]   all graders resolve ({len(graders.available())} registered)")

    missing_desc = [p.id for p in suite.probes if not p.description]
    if missing_desc:
        console.print(f"[yellow]WARN[/] probes without a description: {', '.join(missing_desc)}")

    report = run(suite, load_target("echo"), config=RunConfig(workers=args.workers))
    dead = [r.probe_id for r in report.results if not r.attacked]
    if dead:
        console.print(
            f"[bold red]FAIL[/] {len(dead)} probe(s) not detected even against a "
            f"target that leaks everything: {', '.join(dead)}"
        )
        console.print("[dim]     their graders can never fire, so they only dilute the score[/]")
        return 1
    console.print(f"[green]OK[/]   all {len(suite)} probes fire against the echo lower bound")
    console.print("\n[bold green]corpus is sound[/]")
    return 0


def cmd_graders(args: argparse.Namespace, console: Console) -> int:
    table = Table(title="Registered graders", title_style="bold", header_style="dim")
    table.add_column("name", style="bold")
    table.add_column("what counts as a successful attack", overflow="fold")
    for name in graders.available():
        doc = (graders.get(name).__doc__ or "").strip().split("\n")[0]
        table.add_row(name, doc)
    console.print(table)
    console.print(
        "\n[dim]None of these calls a language model. See docs/METHODOLOGY.md for why.[/]"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsb",
        description="Measure how an LLM application holds up under attack.",
    )
    parser.add_argument("--version", action="version", version=f"lsb {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a suite against a target")
    _add_selection_args(p_run)
    _add_run_args(p_run)
    p_run.add_argument("--defense", default="none", help=f"one of: {', '.join(catalogue())}")
    p_run.add_argument("--failures", action="store_true", help="list the attacks that landed")
    p_run.add_argument("--json", help="write the full run to this JSON path")
    p_run.add_argument("--markdown", help="write a markdown report to this path")
    p_run.add_argument("--fail-over", type=float, default=None,
                       help="exit 1 when the exposure score exceeds this (for CI gates)")
    p_run.set_defaults(func=cmd_run)

    p_cmp = sub.add_parser("compare", help="measure defenses against an undefended baseline")
    _add_selection_args(p_cmp)
    _add_run_args(p_cmp)
    p_cmp.add_argument("--defenses", help=f"comma-separated (default: all of {','.join(catalogue())})")
    p_cmp.set_defaults(func=cmd_compare)

    p_list = sub.add_parser("list", help="show the probe corpus")
    _add_selection_args(p_list)
    p_list.set_defaults(func=cmd_list)

    p_val = sub.add_parser("validate", help="check the corpus is sound")
    p_val.add_argument("--suite", help="suite file or directory")
    p_val.add_argument("--workers", type=int, default=8)
    p_val.set_defaults(func=cmd_validate)

    p_gra = sub.add_parser("graders", help="list the grading rules")
    p_gra.set_defaults(func=cmd_graders)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()
    try:
        return int(args.func(args, console))
    except SuiteError as exc:
        console.print(f"[bold red]suite error:[/] {exc}")
        return 2
    except (ValueError, KeyError, FileNotFoundError) as exc:
        console.print(f"[bold red]error:[/] {exc}")
        return 2
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
