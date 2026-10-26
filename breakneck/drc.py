"""Verify planned vias with kicad-cli DRC and prune the ones it implicates."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeVar

from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from loguru import logger
from shapely.geometry import LineString, Point

from breakneck.plan import Plan, PlannedUnit, Planner
from breakneck.snapshot import Point2, Snapshot

KICAD_CLI = "kicad-cli"
KNOWN_KICAD_CLI_PATHS: tuple[str, ...] = (
    "/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli",
    "/usr/bin/kicad-cli",
    "/usr/local/bin/kicad-cli",
    r"C:\Program Files\KiCad\10.0\bin\kicad-cli.exe",
    r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe",
)

# Violation keys and attribution compare positions at 1 µm.
POSITION_DECIMALS = 3
POSITION_TOLERANCE = 0.002
COORD_DECIMALS = 6

DEFAULT_MAX_ROUNDS = 5
# The prototype used 1.5 mm; one HALSER starved thermal needed 2.5 mm.
DEFAULT_THERMAL_RADIUS = 2.5

CLEARANCE_TYPES = frozenset(
    {"clearance", "hole_clearance", "hole_to_hole", "copper_edge_clearance"}
)
STARVED_THERMAL = "starved_thermal"
THROUGH_VIA_LAYERS = ("F.Cu", "B.Cu")
FONTS_MARKER = "\n\t(embedded_fonts"
DRC_TIMEOUT_S = 900
VERSION_TIMEOUT_S = 30


class KiCadCliNotFound(RuntimeError):
    pass


class ProjectFilesError(RuntimeError):
    pass


class DrcError(RuntimeError):
    pass


# ---- kicad-cli and project files ----------------------------------------------


class BinaryPathSource(Protocol):
    def get_kicad_binary_path(self, binary_name: str) -> str: ...


def _executable(path: str | os.PathLike[str] | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_file() and os.access(p, os.X_OK) else None


def locate_kicad_cli(explicit: str | None, kicad: BinaryPathSource | None) -> Path:
    """Find kicad-cli: explicit path, the running KiCad, PATH, known installs."""
    if explicit:
        found = _executable(explicit)
        if found is None:
            raise KiCadCliNotFound(f"--kicad-cli {explicit} is not an executable file")
        return found

    if kicad is not None:
        try:
            found = _executable(kicad.get_kicad_binary_path(KICAD_CLI))
        except (ApiError, KiCadConnectionError) as e:  # older KiCad lacks the call
            logger.debug(f"KiCad did not report a kicad-cli path: {e}")
            found = None
        if found:
            return found

    found = _executable(shutil.which(KICAD_CLI))
    if found:
        return found

    for candidate in KNOWN_KICAD_CLI_PATHS:
        found = _executable(candidate)
        if found:
            return found

    raise KiCadCliNotFound(
        "kicad-cli not found; pass --kicad-cli PATH or put kicad-cli on PATH"
    )


def kicad_cli_version(kicad_cli: Path) -> str:
    try:
        out = subprocess.run(
            [str(kicad_cli), "version"],
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "unknown (version probe timed out)"
    return out.stdout.strip() or out.stderr.strip() or "unknown"


@dataclass(frozen=True)
class ProjectFiles:
    pro: Path
    dru: Path | None


class ProjectBoard(Protocol):
    @property
    def name(self) -> str: ...

    def get_project(self) -> object: ...


def project_files(board: ProjectBoard) -> ProjectFiles:
    """On-disk project and rule files of the open board.

    Unsaved rule or netclass edits made in KiCad are not in these files.
    """
    directory = getattr(board.get_project(), "path", "")
    pcb = Path(directory) / board.name if directory else None
    if pcb is None or not pcb.is_file():
        raise ProjectFilesError(
            "The board has not been saved to disk; save it before placing vias"
        )
    pro = pcb.with_suffix(".kicad_pro")
    if not pro.is_file():
        raise ProjectFilesError(f"Project file {pro} not found")
    dru = pcb.with_suffix(".kicad_dru")
    return ProjectFiles(pro=pro, dru=dru if dru.is_file() else None)


# ---- s-expression injection -----------------------------------------------------


def _num(v: float) -> str:
    s = f"{v:.{COORD_DECIMALS}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _via_sexpr(u: PlannedUnit) -> str:
    x, y = u.position
    layers = " ".join(_quote(L) for L in THROUGH_VIA_LAYERS)
    return (
        f"\t(via\n\t\t(at {_num(x)} {_num(y)})\n"
        f"\t\t(size {_num(u.size.diameter)})\n\t\t(drill {_num(u.size.drill)})\n"
        f"\t\t(layers {layers})\n\t\t(net {_quote(u.net)})\n"
        f'\t\t(uuid "{uuid.uuid4()}")\n\t)\n'
    )


def _segment_sexpr(u: PlannedUnit) -> str:
    assert u.stub is not None
    s = u.stub
    return (
        f"\t(segment\n\t\t(start {_num(s.start[0])} {_num(s.start[1])})\n"
        f"\t\t(end {_num(s.end[0])} {_num(s.end[1])})\n"
        f"\t\t(width {_num(s.width)})\n\t\t(layer {_quote(s.layer)})\n"
        f"\t\t(net {_quote(u.net)})\n"
        f'\t\t(uuid "{uuid.uuid4()}")\n\t)\n'
    )


def inject(board_text: str, units: Sequence[PlannedUnit]) -> str:
    """Board file text with the units added as non-free vias and segments."""
    if not units:
        return board_text
    items = "".join(
        _via_sexpr(u) + (_segment_sexpr(u) if u.stub else "") for u in units
    )
    if FONTS_MARKER in board_text:
        at = board_text.rindex(FONTS_MARKER) + 1
    else:
        at = board_text.rstrip().rindex(")")
        if board_text[at - 1] != "\n":
            items = "\n" + items
    return board_text[:at] + items + board_text[at:]


# ---- reports -------------------------------------------------------------------


@dataclass(frozen=True)
class ViolationItem:
    description: str
    position: Point2


@dataclass(frozen=True)
class Violation:
    type: str
    description: str
    severity: str
    items: tuple[ViolationItem, ...]

    @property
    def key(self) -> tuple[str, tuple[tuple[str, float, float], ...]]:
        return (
            self.type,
            tuple(
                sorted(
                    (
                        i.description,
                        round(i.position[0], POSITION_DECIMALS),
                        round(i.position[1], POSITION_DECIMALS),
                    )
                    for i in self.items
                )
            ),
        )

    def __str__(self) -> str:
        where = ", ".join(
            f"{i.description} @ ({i.position[0]:.3f}, {i.position[1]:.3f})"
            for i in self.items
        )
        return f"{self.type}: {self.description} [{where}]"


def parse_report(report: dict) -> list[Violation]:
    out = []
    for v in report.get("violations", []) + report.get("unconnected_items", []):
        if v.get("excluded"):
            continue
        out.append(
            Violation(
                type=v.get("type", "unconnected_items"),
                description=v.get("description", ""),
                severity=v.get("severity", ""),
                items=tuple(
                    ViolationItem(i["description"], (i["pos"]["x"], i["pos"]["y"]))
                    for i in v.get("items", [])
                ),
            )
        )
    return out


def new_violations(
    baseline: Iterable[Violation], planned: Iterable[Violation]
) -> list[Violation]:
    """Planned violations not in the baseline, compared as multisets."""
    remaining = Counter(v.key for v in baseline)
    out = []
    for v in planned:
        if remaining[v.key] > 0:
            remaining[v.key] -= 1
        else:
            out.append(v)
    return out


class DrcRunner:
    """Runs kicad-cli DRC on copies of the board in a scratch directory."""

    def __init__(
        self,
        kicad_cli: Path,
        project: ProjectFiles,
        board_text: str,
        workdir: Path,
    ):
        self._cli = kicad_cli
        self._project = project
        self._board_text = board_text
        self._workdir = workdir
        self._baseline: list[Violation] | None = None

    def _run(self, stem: str, board_text: str) -> list[Violation]:
        pcb = self._workdir / f"{stem}.kicad_pcb"
        pcb.write_text(board_text, encoding="utf-8")
        shutil.copyfile(self._project.pro, pcb.with_suffix(".kicad_pro"))
        if self._project.dru is not None:
            shutil.copyfile(self._project.dru, pcb.with_suffix(".kicad_dru"))
        out = pcb.with_suffix(".json")
        cmd = [
            str(self._cli),
            "pcb",
            "drc",
            "--refill-zones",
            "--format",
            "json",
            "-o",
            str(out),
            str(pcb),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=DRC_TIMEOUT_S, check=False
            )
        except subprocess.TimeoutExpired:
            raise DrcError(f"kicad-cli DRC timed out after {DRC_TIMEOUT_S} s") from None
        if proc.returncode != 0 or not out.is_file():
            raise DrcError(
                f"kicad-cli DRC failed ({proc.returncode}): {proc.stderr.strip()}"
            )
        return parse_report(json.loads(out.read_text(encoding="utf-8")))

    def baseline(self) -> list[Violation]:
        if self._baseline is None:
            logger.info("Running baseline DRC")
            self._baseline = self._run("baseline", self._board_text)
            logger.info(f"Baseline DRC: {len(self._baseline)} violations")
        return self._baseline

    def new_violations(self, plan: Plan) -> list[Violation]:
        base = self.baseline()
        logger.info(f"Running DRC with {len(plan.units)} planned vias")
        planned = self._run("planned", inject(self._board_text, plan.units))
        return new_violations(base, planned)


# ---- attribution -----------------------------------------------------------------


@dataclass(frozen=True)
class ForcedViolation:
    unit: int
    violation: Violation


@dataclass
class Attribution:
    dropped: dict[int, list[Violation]] = field(default_factory=dict)
    forced: list[ForcedViolation] = field(default_factory=list)
    unattributed: list[Violation] = field(default_factory=list)


def _is_zone(item: ViolationItem) -> bool:
    return item.description.startswith("Zone")


def _hits(item: ViolationItem, u: PlannedUnit) -> bool:
    if item.description.startswith("Via"):
        return math.dist(item.position, u.position) <= POSITION_TOLERANCE
    # DRC reports a track at one of its endpoints, so match anywhere on the
    # stub, on the stub's layer.
    if (
        item.description.startswith("Track")
        and u.stub is not None
        and f" on {u.stub.layer}" in item.description
    ):
        stub = LineString([u.stub.start, u.stub.end])
        reach = u.stub.width / 2 + POSITION_TOLERANCE
        return stub.distance(Point(item.position)) <= reach
    return False


def _implicated(
    v: Violation, units: Sequence[PlannedUnit], thermal_radius: float
) -> set[int]:
    hit = {j for j, u in enumerate(units) for i in v.items if _hits(i, u)}
    if v.type == STARVED_THERMAL:
        pads = [i.position for i in v.items if not _is_zone(i)]
        hit |= {
            j
            for j, u in enumerate(units)
            if any(math.dist(p, u.position) <= thermal_radius for p in pads)
        }
    return hit


def attribute(
    violations: Iterable[Violation],
    units: Sequence[PlannedUnit],
    thermal_radius: float = DEFAULT_THERMAL_RADIUS,
) -> Attribution:
    """Pin new violations on planned units.

    Forced units are exempt from clearance-type violations, which are
    reported instead.
    """
    result = Attribution()
    for v in violations:
        hit = _implicated(v, units, thermal_radius)
        if not hit:
            result.unattributed.append(v)
            continue
        for j in sorted(hit):
            if units[j].forced and v.type in CLEARANCE_TYPES:
                result.forced.append(ForcedViolation(j, v))
            else:
                result.dropped.setdefault(j, []).append(v)
    return result


# ---- prune loop --------------------------------------------------------------------


@dataclass(frozen=True)
class DroppedUnit:
    unit: PlannedUnit
    violations: tuple[Violation, ...]


@dataclass(frozen=True)
class PruneResult:
    plan: Plan
    dropped: tuple[DroppedUnit, ...]
    forced: tuple[ForcedViolation, ...]
    unattributed: tuple[Violation, ...]
    rounds: int
    round_limit_hit: bool = False


OptionsT = TypeVar("OptionsT")
Check = Callable[[Plan], list[Violation]]


def prune(
    planner: Planner[OptionsT],
    snapshot: Snapshot,
    options: OptionsT,
    check: Check,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    thermal_radius: float = DEFAULT_THERMAL_RADIUS,
) -> PruneResult:
    """Plan, verify, drop implicated units and re-plan with them forbidden.

    Stops after a round that drops nothing. At the round limit the units
    still implicated are dropped without another DRC pass.
    """
    forbidden: list[Point2] = []
    dropped: list[DroppedUnit] = []
    rounds = 0
    while True:
        rounds += 1
        plan = planner(snapshot, options, list(forbidden))
        found = check(plan) if plan.units else []
        attr = attribute(found, plan.units, thermal_radius)
        for j, vs in sorted(attr.dropped.items()):
            dropped.append(DroppedUnit(plan.units[j], tuple(vs)))
            logger.info(
                f"Round {rounds}: dropping {plan.units[j].net} via at "
                f"({plan.units[j].position[0]:.3f}, {plan.units[j].position[1]:.3f}): "
                f"{', '.join(sorted({v.type for v in vs}))}"
            )
        if not attr.dropped:
            return PruneResult(
                plan,
                tuple(dropped),
                tuple(attr.forced),
                tuple(attr.unattributed),
                rounds,
            )
        if rounds >= max_rounds:
            logger.warning(
                f"DRC round limit ({max_rounds}) reached; dropping the remaining "
                "implicated vias without re-verifying"
            )
            kept = tuple(u for j, u in enumerate(plan.units) if j not in attr.dropped)
            forced = tuple(
                ForcedViolation(kept.index(plan.units[f.unit]), f.violation)
                for f in attr.forced
                if f.unit not in attr.dropped
            )
            return PruneResult(
                Plan(units=kept, unfixable=plan.unfixable),
                tuple(dropped),
                forced,
                tuple(attr.unattributed),
                rounds,
                round_limit_hit=True,
            )
        forbidden.extend(plan.units[j].position for j in attr.dropped)
