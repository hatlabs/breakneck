"""The `returnvia` command: plan, verify and commit, then measure, render
and overlay the return distance under each signal layer's copper."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import kipy
import kipy.board
import matplotlib.colors
import numpy as np
from loguru import logger
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from breakneck import overlay, render
from breakneck.placement import RunSettings, measure_only, run_placement
from breakneck.plan import PlannedUnit
from breakneck.raster import CoverageMap
from breakneck.returnpath import NoBoardOutlineError, PlaneReturn, ReturnFields
from breakneck.returnvia import (
    ReturnViaOptions,
    ViaReferences,
    plan_return_vias,
    reference_outlines,
    resolve_ground_nets,
    via_references,
)
from breakneck.snapshot import Snapshot, Via, outline_from_edge_paths, read_edge_paths

COMMAND = "returnvia"
OVERLAY_COMMIT = "breakneck returnvia overlay"
# The colour scale runs to this multiple of --distance and saturates beyond.
SCALE = 3.0
SERVED_COLOUR = "#24323f"
STILL_SERVED_COLOUR = "#6b7d8c"
UNSERVED_COLORMAP = "plasma_r"
UNSERVED_SAMPLES = 64
UNREACHABLE_COLOUR = "#b44dff"
SIGNAL_MAP_NET = "signal copper"
# Whole steps per --distance, so the step at --distance falls between two.
PALETTE_STEPS_PER_DISTANCE = 256
THRESHOLD = 1 / SCALE
_UNSERVED = matplotlib.colormaps[UNSERVED_COLORMAP]
# Quiet up to --distance, then a hard step into plasma: served copper recedes
# and the distance beyond it reads with its full contrast.
PALETTE = matplotlib.colors.LinearSegmentedColormap.from_list(
    "returnvia",
    [
        (0.0, SERVED_COLOUR),
        (THRESHOLD, STILL_SERVED_COLOUR),
        *(
            (THRESHOLD + (1 - THRESHOLD) * p, _UNSERVED(p))
            for p in np.linspace(0.0, 1.0, UNSERVED_SAMPLES)
        ),
    ],
    N=round(SCALE * PALETTE_STEPS_PER_DISTANCE),
)
STYLE = render.HeatmapStyle(
    title=(
        "Return vias: signal copper coloured by the distance to the nearest "
        "ground return through its reference plane; ring = unserved via end, "
        "yellow to dark blue = beyond the "
        "distance, purple = no path to a return"
    ),
    colourbar_label=(
        "distance to nearest return through the plane (mm); "
        "served within {distance:g} mm"
    ),
    png_prefix="returnvia",
    colormap=PALETTE,
    over_colour=None,
    unreachable_colour=UNREACHABLE_COLOUR,
    scale=SCALE,
)


def check_outline(board: kipy.board.Board) -> None:
    """Fail before any board change on a board without an outline."""
    if outline_from_edge_paths(read_edge_paths(board.get_shapes())).is_empty:
        raise NoBoardOutlineError("Board has no closed Edge.Cuts outline")


def with_returns(snapshot: Snapshot, units: Sequence[PlannedUnit]) -> Snapshot:
    """The snapshot with the planned vias added as returns.

    Forced vias are placeholders for a via placed by hand and lie outside
    the current fill, so they serve nothing.
    """
    planned = [
        Via(f"planned-{i}", u.net, u.position, u.size.diameter, u.size.drill)
        for i, u in enumerate(units)
        if not u.forced
    ]
    return dataclasses.replace(snapshot, vias=(*snapshot.vias, *planned))


def signal_copper(
    snapshot: Snapshot, layer: str, grounds: Sequence[str]
) -> BaseGeometry:
    """Tracks, pads and zone fills of the non-ground nets on `layer`."""
    shapes = [
        t.geometry for t in snapshot.tracks if t.layer == layer and t.net not in grounds
    ]
    shapes += [
        p.shapes[layer]
        for p in snapshot.pads
        if p.net not in grounds and layer in p.shapes
    ]
    shapes += [
        z.fills[layer]
        for z in snapshot.copper_zones
        if z.net not in grounds and layer in z.fills
    ]
    return unary_union(shapes)


def signal_map(fields: ReturnFields, layer: str, grounds: Sequence[str]) -> CoverageMap:
    """The return field of each pixel's reference plane, on `layer`'s
    non-ground copper.

    A pixel's reference is chosen as for a via end there. Copper over a void
    in that plane, or outside every ground outline, has no return path.
    """
    grid = fields.grid
    # Every pixel a track touches, so thin tracks stay unbroken.
    copper = grid.touching(signal_copper(fields.snapshot, layer, grounds))
    dist = np.where(copper, np.inf, np.nan)
    unassigned = copper.copy()
    for plane, outline in reference_outlines(fields.snapshot, layer, grounds):
        here = unassigned & grid.mask(outline)
        on_fill = here & fields.fill(plane)
        dist[on_fill] = fields.field(plane)[on_fill]
        unassigned &= ~here
    return CoverageMap(SIGNAL_MAP_NET, layer, dist, np.zeros(grid.shape, bool))


def signal_maps(
    fields: ReturnFields, references: Sequence[ViaReferences], grounds: Sequence[str]
) -> list[CoverageMap]:
    """One map per copper layer a non-exempt via end connects on."""
    layers = {c.layer for r in references if not r.exempt for c in r.layers}
    return [
        signal_map(fields, L, grounds)
        for L in fields.snapshot.copper_layers
        if L in layers
    ]


def markers(
    results: Sequence[PlaneReturn], maps: Sequence[CoverageMap], distance: float
) -> dict[render.MapKey, render.Markers]:
    """Per signal map: a ring of radius `distance` around each via end
    unserved on that layer, showing the reach its return needs."""
    rings: dict[str, list[render.Disc]] = {}
    for r in results:
        if not r.served:
            for layer in r.layers:
                rings.setdefault(layer, []).append((r.position, distance))
    return {
        (m.net, m.layer): render.Markers(rings=tuple(rings.get(m.layer, ())))
        for m in maps
    }


def log_results(
    references: Sequence[ViaReferences],
    results: Sequence[PlaneReturn],
    distance: float,
) -> None:
    exempt = sum(r.exempt for r in references)
    logger.info(f"  {exempt} vias exempt: connected to copper on fewer than two layers")
    unserved = [r for r in results if not r.served]
    for r in unserved:
        x, y = r.position
        where = f"  unserved: {r.net} at ({x:.3f}, {y:.3f})"
        if r.reason is not None:
            logger.debug(f"{where}: {r.reason}")
        else:
            assert r.plane is not None
            logger.debug(
                f"{where} on {r.plane.layer} ({r.plane.ground}): "
                f"{r.distance:.2f} mm to a return"
            )
    vias = {(r.position, r.net) for r in results}
    failing = {(r.position, r.net) for r in unserved}
    logger.info(
        f"{COMMAND}: {len(failing)} of {len(vias)} vias unserved within "
        f"{distance:g} mm; {len(unserved)} of {len(results)} via-plane returns"
    )


def execute(
    options: ReturnViaOptions,
    outputs: overlay.Outputs,
    *,
    dry_run: bool = False,
    check: bool = False,
    kicad_cli: str | None = None,
    allow_splits: bool = False,
    kicad: kipy.KiCad | None = None,
) -> None:
    """Run returnvia against the open board.

    `check` only measures the existing returns: no planning, no kicad-cli,
    no vias or markers. Planned vias count as returns in the measurement,
    except forced ones. Overlays are placed in their own commit, also in a
    dry run.
    """
    kicad = kicad or kipy.KiCad()
    board = kicad.get_board()
    check_outline(board)
    place_overlay = outputs.overlay_dir is not None and overlay.overlay_ready(
        board, outputs, dry_run
    )

    if check:
        snapshot, plan = measure_only(kicad, board)
    else:
        result = run_placement(
            plan_return_vias,
            options,
            RunSettings(COMMAND, dry_run, kicad_cli, allow_splits=allow_splits),
            kicad,
        )
        snapshot, plan = result.snapshot, result.prune.plan

    grounds = resolve_ground_nets(snapshot, options.ground_nets)
    fields = ReturnFields(with_returns(snapshot, plan.units))
    references = via_references(fields.snapshot, grounds)
    results = fields.measure(references, options.distance)
    logger.info(f"{COMMAND} returns" + (" (check only)" if check else ""))
    log_results(references, results, options.distance)

    maps = signal_maps(fields, references, grounds)
    marks = markers(results, maps, options.distance)
    if not maps:
        logger.info("No signal via changes layers: no heatmaps")
    elif outputs.render is not None:
        render.render_panels(
            outputs.render,
            fields.snapshot,
            fields.grid,
            maps,
            marks,
            options.distance,
            STYLE,
        )
        logger.info(f"Wrote {outputs.render}")

    if outputs.overlay_dir is not None:
        pngs = render.write_overlays(
            outputs.overlay_dir, fields.grid, maps, marks, options.distance, STYLE
        )
        overlay.show_overlays(
            board, pngs, snapshot, outputs, place_overlay, OVERLAY_COMMIT
        )
