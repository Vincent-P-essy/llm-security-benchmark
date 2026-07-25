"""Loading attack suites from YAML.

Suites are data, not code, for one reason: the people who know which attacks
matter for a given application are usually not the people maintaining the
runner. A security engineer should be able to add a probe for their own
retrieval pipeline by writing twelve lines of YAML, and have it graded and
scored by the same machinery as everything else.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from .probe import Family, GraderSpec, Probe, Severity

SUITE_DIR = Path(__file__).resolve().parent.parent / "suites"


class SuiteError(ValueError):
    """Raised when a suite file is malformed. Always names the offending probe."""


@dataclass(frozen=True)
class Suite:
    """A named, versioned collection of probes."""

    name: str
    version: str
    description: str
    probes: tuple[Probe, ...]

    def __len__(self) -> int:
        return len(self.probes)

    def __iter__(self) -> Iterable[Probe]:
        return iter(self.probes)

    @property
    def slug(self) -> str:
        return f"{self.name}-{self.version}"

    def families(self) -> tuple[Family, ...]:
        seen: list[Family] = []
        for probe in self.probes:
            if probe.family not in seen:
                seen.append(probe.family)
        return tuple(seen)

    def techniques(self) -> tuple[str, ...]:
        return tuple(sorted({p.technique for p in self.probes}))

    def select(
        self,
        families: Sequence[str] | None = None,
        techniques: Sequence[str] | None = None,
        severities: Sequence[str] | None = None,
        ids: Sequence[str] | None = None,
        tags: Sequence[str] | None = None,
    ) -> Suite:
        """Return a narrowed copy. Empty selections are an error, not an empty run.

        Silently running zero probes and reporting 0% exposure is the worst
        possible outcome for a security tool - it looks like a pass.
        """
        probes = list(self.probes)
        if families:
            wanted = {f.lower() for f in families}
            probes = [p for p in probes if p.family.value in wanted]
        if techniques:
            wanted = {t.lower() for t in techniques}
            probes = [p for p in probes if p.technique.lower() in wanted]
        if severities:
            wanted = {s.lower() for s in severities}
            probes = [p for p in probes if p.severity.value in wanted]
        if ids:
            wanted = set(ids)
            probes = [p for p in probes if p.id in wanted]
        if tags:
            wanted = {t.lower() for t in tags}
            probes = [p for p in probes if wanted & {t.lower() for t in p.tags}]

        if not probes:
            raise SuiteError("selection matched no probes - refusing to report a 0% run")
        return replace(self, probes=tuple(probes))

    def repeated(self, times: int) -> Suite:
        """Duplicate every probe ``times`` times to widen the confidence interval.

        Repetition is the only lever the harness has on interval width: a probe
        run once tells you nothing about a stochastic system.
        """
        if times < 1:
            raise SuiteError("repeat count must be at least 1")
        if times == 1:
            return self
        return replace(self, probes=tuple(p for p in self.probes for _ in range(times)))


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise SuiteError(f"{where}: missing required key {key!r}")
    return data[key]


def _parse_probe(raw: dict[str, Any], defaults: dict[str, Any], where: str) -> Probe:
    if not isinstance(raw, dict):
        raise SuiteError(f"{where}: probe entries must be mappings, got {type(raw).__name__}")

    probe_id = str(_require(raw, "id", where))
    where = f"{where} probe {probe_id!r}"

    merged = {**defaults, **raw}

    try:
        family = Family(str(_require(merged, "family", where)).lower())
    except ValueError:
        raise SuiteError(
            f"{where}: unknown family {merged.get('family')!r}; "
            f"expected one of {', '.join(f.value for f in Family)}"
        ) from None

    try:
        severity = Severity(str(merged.get("severity", "medium")).lower())
    except ValueError:
        raise SuiteError(
            f"{where}: unknown severity {merged.get('severity')!r}; "
            f"expected one of {', '.join(s.value for s in Severity)}"
        ) from None

    grader_raw = _require(merged, "grader", where)
    if isinstance(grader_raw, str):
        grader = GraderSpec(name=grader_raw)
    elif isinstance(grader_raw, dict):
        grader = GraderSpec(
            name=str(_require(grader_raw, "name", where)),
            params={k: v for k, v in grader_raw.items() if k != "name"},
        )
    else:
        raise SuiteError(f"{where}: grader must be a string or a mapping")

    tags = merged.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    return Probe(
        id=probe_id,
        family=family,
        technique=str(merged.get("technique", "direct-override")),
        severity=severity,
        description=str(merged.get("description", "")).strip(),
        system=str(merged.get("system", "")).strip(),
        task=str(_require(merged, "task", where)).strip(),
        untrusted=(str(merged["untrusted"]).strip() if merged.get("untrusted") else None),
        grader=grader,
        tags=tuple(str(t) for t in tags),
    )


def load_file(path: str | Path) -> Suite:
    """Parse a single suite YAML file."""
    path = Path(path)
    if not path.exists():
        raise SuiteError(f"suite file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SuiteError(f"{path.name}: top level must be a mapping")

    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SuiteError(f"{path.name}: 'defaults' must be a mapping")

    raw_probes = data.get("probes") or []
    if not isinstance(raw_probes, list):
        raise SuiteError(f"{path.name}: 'probes' must be a list")

    probes = tuple(_parse_probe(p, defaults, path.name) for p in raw_probes)

    seen: set[str] = set()
    for probe in probes:
        if probe.id in seen:
            raise SuiteError(f"{path.name}: duplicate probe id {probe.id!r}")
        seen.add(probe.id)

    return Suite(
        name=str(data.get("suite", path.stem)),
        version=str(data.get("version", "v1")),
        description=str(data.get("description", "")).strip(),
        probes=probes,
    )


def load(source: str | Path | None = None) -> Suite:
    """Load a suite from a file, a directory of files, or the bundled corpus."""
    if source is None:
        source = SUITE_DIR
    path = Path(source)

    if path.is_file():
        return load_file(path)
    if not path.is_dir():
        raise SuiteError(f"no such suite file or directory: {path}")

    files = sorted(p for p in path.glob("*.yaml") if not p.name.startswith("_"))
    if not files:
        raise SuiteError(f"no suite YAML files in {path}")

    merged: list[Probe] = []
    seen: set[str] = set()
    for file in files:
        for probe in load_file(file).probes:
            if probe.id in seen:
                raise SuiteError(f"duplicate probe id {probe.id!r} across suite files in {path}")
            seen.add(probe.id)
            merged.append(probe)

    meta = path / "_suite.yaml"
    info = yaml.safe_load(meta.read_text(encoding="utf-8")) if meta.exists() else {}
    info = info or {}
    return Suite(
        name=str(info.get("suite", "core")),
        version=str(info.get("version", "v1")),
        description=str(info.get("description", "Bundled attack corpus")).strip(),
        probes=tuple(merged),
    )
