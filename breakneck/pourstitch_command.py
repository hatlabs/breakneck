"""The `pourstitch` command: plan, verify and commit, then render and overlay."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import kipy
import kipy.board
from loguru import logger

from breakneck import overlay, render
from breakneck.commit import refill_tolerant
from breakneck.placement import RunSettings, run_placement
from breakneck.plan import Plan
from breakneck.pourstitch import CoverageMap, Grid, Options, PourStitcher, summarise
from breakneck.snapshot import Snapshot

COMMAND = "pourstitch"


@dataclass(frozen=True)
class Outputs:
    render: Path | None = None
    overlay_dir: Path | None = None
    overlay_layers: tuple[str, ...] = overlay.DEFAULT_OVERLAY_LAYERS


class OverlayLayersError(RuntimeError):
    pass


def _overlay_ready(board: kipy.board.Board, outputs: Outputs, dry_run: bool) -> bool:
    """Whether overlays can go on the board; raises before any board change if not."""
    count = board.get_copper_layer_count()
    if count > len(outputs.overlay_layers):
        raise OverlayLayersError(
            f"{count} copper layers need {count} overlay layers; got "
            + ", ".join(outputs.overlay_layers)
        )
    needed = outputs.overlay_layers[:count]
    missing = overlay.missing_layers(board, needed)
    if not missing:
        return True
    if dry_run:
        logger.warning(overlay.enable_hint(missing) + "; writing the PNGs only")
        return False
    raise OverlayLayersError(overlay.enable_hint(missing))


def _log_coverage(
    stitcher: PourStitcher,
    snapshot: Snapshot,
    options: Options,
    maps: Sequence[CoverageMap],
    grid: Grid,
) -> None:
    counts = stitcher.stitch_counts(snapshot, options)
    for c in summarise(maps, options.distance, grid.pixel):
        unstitchable = (
            f"; {c.unstitchable_mm2:.2f} mm² can't be stitched (fill on one layer)"
            if c.unstitchable_mm2
            else ""
        )
        logger.info(
            f"  {c.net} on {c.layer}: {c.uncovered_mm2:.2f} mm² ({c.uncovered_pixels} px) "
            f"beyond {options.distance:g} mm; {counts[c.net]} existing stitches"
            + unstitchable
        )


def execute(
    options: Options,
    outputs: Outputs,
    *,
    dry_run: bool = False,
    check: bool = False,
    kicad_cli: str | None = None,
    kicad: kipy.KiCad | None = None,
) -> None:
    """Run pourstitch against the open board.

    `check` only measures the existing stitching: no planning, no kicad-cli,
    no vias or markers. Overlays are placed in their own commit, also in a
    dry run.
    """
    kicad = kicad or kipy.KiCad()
    board = kicad.get_board()
    place_overlay = outputs.overlay_dir is not None and _overlay_ready(
        board, outputs, dry_run
    )

    stitcher = PourStitcher()
    if check:
        if not refill_tolerant(kicad, board):
            logger.warning("Zone refill not confirmed; fills may be stale")
        snapshot = Snapshot.from_board(board)
        plan = Plan()
    else:
        result = run_placement(
            stitcher,
            options,
            RunSettings(COMMAND, dry_run, kicad_cli, mark_unfixable=False),
            kicad,
        )
        snapshot, plan = result.snapshot, result.prune.plan

    grid = stitcher.grid(snapshot, options)
    maps = stitcher.coverage(snapshot, options, plan.units)
    logger.info(f"{COMMAND} coverage" + (" (check only)" if check else ""))
    _log_coverage(stitcher, snapshot, options, maps, grid)

    if outputs.render is not None:
        render.render_panels(
            outputs.render, snapshot, grid, maps, plan, options.distance
        )
        logger.info(f"Wrote {outputs.render}")

    if outputs.overlay_dir is not None:
        pngs = render.write_overlays(outputs.overlay_dir, maps, options.distance)
        layers = overlay.layer_map(snapshot.copper_layers, outputs.overlay_layers)
        placement = overlay.image_placement(
            (grid.x0, grid.y0), (grid.width, grid.height), grid.pixel
        )
        images = [
            overlay.OverlayImage(L, layers[L], path, placement)
            for L, path in pngs.items()
        ]
        if place_overlay and images:
            overlay.place_overlays(board, images, outputs.overlay_layers)
        else:
            for i in images:
                logger.info(f"Overlay not placed: {i.describe()}")
