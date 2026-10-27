"""The `pourstitch` command: plan, verify and commit, then render and overlay."""

from __future__ import annotations

from collections.abc import Sequence

import kipy
import kipy.board
from loguru import logger

from breakneck import overlay, render
from breakneck.placement import RunSettings, measure_only, run_placement
from breakneck.plan import Plan
from breakneck.pourstitch import Options, PourStitcher, summarise
from breakneck.raster import CoverageMap, Grid
from breakneck.snapshot import Snapshot

COMMAND = "pourstitch"
OVERLAY_COMMIT = "breakneck pourstitch overlay"
OVER_COLOUR = "#ff2a6d"
STYLE = render.HeatmapStyle(
    title=(
        "Pour stitching: filled = planned via, ring = existing via, "
        "cyan x = no via fits, grey = out of reach of any two-layer spot"
    ),
    colourbar_label=(
        "distance to nearest stitch through the copper (mm); pink > {distance:g} mm"
    ),
    png_prefix="pourstitch",
    colormap="viridis",
    over_colour=OVER_COLOUR,
    unreachable_colour=OVER_COLOUR,
)


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
            f"; {c.unstitchable_mm2:.2f} mm² can't be stitched (no two-layer spot within reach)"
            if c.unstitchable_mm2
            else ""
        )
        logger.info(
            f"  {c.net} on {c.layer}: {c.uncovered_mm2:.2f} mm² ({c.uncovered_pixels} px) "
            f"beyond {options.distance:g} mm; {counts[c.net]} existing stitches"
            + unstitchable
        )


def markers(
    snapshot: Snapshot, plan: Plan, maps: Sequence[CoverageMap]
) -> dict[render.MapKey, render.Markers]:
    """Per map: the net's existing vias as rings, its planned vias as dots
    and its unfixable spots, in plan order, as crosses."""
    return {
        (m.net, m.layer): render.Markers(
            rings=tuple(
                (v.position, v.diameter / 2) for v in snapshot.vias if v.net == m.net
            ),
            dots=tuple(
                (u.position, u.size.diameter / 2) for u in plan.units if u.net == m.net
            ),
            crosses=tuple(
                (s.layer, s.position) for s in plan.unfixable if s.net == m.net
            ),
        )
        for m in maps
    }


def execute(
    options: Options,
    outputs: overlay.Outputs,
    *,
    dry_run: bool = False,
    check: bool = False,
    kicad_cli: str | None = None,
    allow_splits: bool = False,
    kicad: kipy.KiCad | None = None,
) -> None:
    """Run pourstitch against the open board.

    `check` only measures the existing stitching: no planning, no kicad-cli,
    no vias or markers. Overlays are placed in their own commit, also in a
    dry run.
    """
    kicad = kicad or kipy.KiCad()
    board = kicad.get_board()
    place_overlay = outputs.overlay and overlay.overlay_ready(board, outputs, dry_run)

    stitcher = PourStitcher()
    if check:
        snapshot, plan = measure_only(kicad, board)
    else:
        result = run_placement(
            stitcher,
            options,
            RunSettings(
                COMMAND,
                dry_run,
                kicad_cli,
                mark_unfixable=False,
                allow_splits=allow_splits,
            ),
            kicad,
        )
        snapshot, plan = result.snapshot, result.prune.plan

    grid = stitcher.grid(snapshot, options)
    maps = stitcher.coverage(snapshot, options, plan.units)
    logger.info(f"{COMMAND} coverage" + (" (check only)" if check else ""))
    _log_coverage(stitcher, snapshot, options, maps, grid)

    if outputs.render is not None:
        render.render_panels(
            outputs.render,
            snapshot,
            grid,
            maps,
            markers(snapshot, plan, maps),
            options.distance,
            STYLE,
        )
        logger.info(f"Wrote {outputs.render}")

    if outputs.overlay:
        # On the board, existing and planned vias show as themselves.
        overlay.show_overlays(
            board,
            (
                lambda out_dir: render.write_overlays(
                    out_dir, grid, maps, {}, options.distance, STYLE
                )
            )
            if maps
            else None,
            snapshot,
            outputs,
            place_overlay,
            OVERLAY_COMMIT,
        )
