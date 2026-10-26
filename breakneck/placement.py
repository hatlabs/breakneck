"""Shared run of a via-placing command: snapshot, plan, verify, commit."""

from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import kipy
from loguru import logger

from breakneck import drc
from breakneck.commit import Marker, MarkerKind, commit_units, refill_tolerant
from breakneck.plan import Planner
from breakneck.snapshot import Snapshot

OptionsT = TypeVar("OptionsT")


@dataclass(frozen=True)
class RunSettings:
    command: str
    dry_run: bool = False
    kicad_cli: str | None = None
    max_rounds: int = drc.DEFAULT_MAX_ROUNDS
    thermal_radius: float = drc.DEFAULT_THERMAL_RADIUS
    # Off where the command reports uncovered areas another way (a heatmap)
    # and per-spot markers and warnings would only be noise.
    mark_unfixable: bool = True


@dataclass(frozen=True)
class RunResult:
    snapshot: Snapshot
    prune: drc.PruneResult
    markers: tuple[Marker, ...]
    committed: bool


def markers_for(
    result: drc.PruneResult, mark_unfixable: bool = True
) -> tuple[Marker, ...]:
    out = (
        [Marker(s.position, MarkerKind.UNFIXABLE) for s in result.plan.unfixable]
        if mark_unfixable
        else []
    )
    out += [
        Marker(u.position, MarkerKind.FORCED) for u in result.plan.units if u.forced
    ]
    out += [
        Marker(v.items[0].position, MarkerKind.UNATTRIBUTED)
        for v in result.unattributed
        if v.items
    ]
    return tuple(out)


def log_summary(
    command: str, result: drc.PruneResult, list_unfixable: bool = True
) -> None:
    units = result.plan.units
    by_net = Counter(u.net for u in units)
    nets = ", ".join(f"{n}: {c}" for n, c in sorted(by_net.items())) or "none"
    logger.info(
        f"{command}: {len(units)} vias planned ({nets}) after {result.rounds} DRC rounds"
    )
    for d in result.dropped:
        x, y = d.unit.position
        rules = ", ".join(sorted({v.type for v in d.violations}))
        logger.info(f"  dropped by DRC: {d.unit.net} at ({x:.3f}, {y:.3f}): {rules}")
    if list_unfixable:
        for s in result.plan.unfixable:
            x, y = s.position
            layer = f" on {s.layer}" if s.layer else ""
            logger.warning(
                f"  unfixable: {s.net}{layer} at ({x:.3f}, {y:.3f}): {s.reason}"
            )
    forced = [u for u in units if u.forced]
    if forced:
        logger.warning(f"  {len(forced)} forced vias")
    for f in result.forced:
        logger.warning(f"  forced via violation: {f.violation}")
    for v in result.unattributed:
        logger.warning(f"  unattributed: {v}")


def run_placement(
    planner: Planner[OptionsT],
    options: OptionsT,
    settings: RunSettings,
    kicad: kipy.KiCad | None = None,
) -> RunResult:
    """Run a planner against the open board.

    kicad-cli and the project files are checked before anything touches the
    board. Zones are refilled before the snapshot even in a dry run, because
    stale fills would drive the placement.
    """
    kicad = kicad or kipy.KiCad()
    board = kicad.get_board()

    kicad_cli = drc.locate_kicad_cli(settings.kicad_cli, kicad)
    logger.info(f"Using {kicad_cli} ({drc.kicad_cli_version(kicad_cli)})")
    project = drc.project_files(board)
    logger.info(
        f"Using project files from disk: {project.pro}"
        + (f", {project.dru}" if project.dru else " (no .kicad_dru)")
    )

    if not refill_tolerant(kicad, board):
        logger.warning("Zone refill not confirmed; fills in the snapshot may be stale")
    snapshot = Snapshot.from_board(board)

    with tempfile.TemporaryDirectory(prefix="breakneck-drc-") as tmp:
        runner = drc.DrcRunner(kicad_cli, project, board.get_as_string(), Path(tmp))
        result = drc.prune(
            planner,
            snapshot,
            options,
            runner.new_violations,
            max_rounds=settings.max_rounds,
            thermal_radius=settings.thermal_radius,
        )

    log_summary(settings.command, result, settings.mark_unfixable)
    markers = markers_for(result, settings.mark_unfixable)

    if settings.dry_run:
        logger.info("Dry run: nothing committed")
        return RunResult(snapshot, result, markers, committed=False)
    if not result.plan.units and not markers:
        logger.info("Nothing to commit")
        return RunResult(snapshot, result, markers, committed=False)

    commit_units(kicad, board, settings.command, result.plan.units, markers)
    return RunResult(snapshot, result, markers, committed=True)
