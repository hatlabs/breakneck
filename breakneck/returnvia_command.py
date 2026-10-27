"""The `returnvia` command: plan, verify and commit, then measure, render
and overlay the return distance under the copper of each via end layer."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import kipy
import kipy.board
import matplotlib.colors
import numpy as np
from loguru import logger
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from breakneck import overlay, render
from breakneck.placement import RunSettings, measure_only, run_placement
from breakneck.plan import QUANTISATION, PlannedUnit
from breakneck.raster import CoverageMap, Grid
from breakneck.returnpath import (
    NoBoardOutlineError,
    Outcome,
    PlaneReturn,
    ReturnFields,
)
from breakneck.returnvia import (
    ReferencePlane,
    ReturnViaOptions,
    ViaReferences,
    plan_return_vias,
    reference_outlines,
    resolve_ground_nets,
    via_references,
)
from breakneck.snapshot import (
    PadKind,
    Snapshot,
    Via,
    outline_from_edge_paths,
    read_edge_paths,
)

COMMAND = "returnvia"
OVERLAY_COMMIT = "breakneck returnvia overlay"
# The colour scale runs to this multiple of the served threshold, --distance
# plus the quantisation margin, and saturates beyond.
SCALE = 3.0
SERVED_COLOUR = "#24323f"
STILL_SERVED_COLOUR = "#6b7d8c"
UNSERVED_COLORMAP = "plasma_r"
UNSERVED_SAMPLES = 64
UNREACHABLE_COLOUR = "#b44dff"
# Pink stands out on both ends of the palette and on the board.
RING_COLOUR = "#ff2a6d"
COPPER_MAP_NET = "copper"
SERVED_TITLE = "Return vias: every via end served within {distance:g} mm"
# Whole steps per threshold, so the step at the threshold falls between two.
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
        "Return vias: copper coloured by the distance to the nearest ground "
        "return through its reference plane; pink ring = unserved via end, yellow "
        "to dark blue = beyond the distance, purple = no path to a return"
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
    margin=QUANTISATION,
    ring_colour=RING_COLOUR,
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


def layer_copper(
    snapshot: Snapshot, layer: str, grounds: Sequence[str]
) -> BaseGeometry:
    """Every net's tracks, pads and zone fills on `layer`, and the ground vias:
    a via placed anywhere in that copper needs its own return, ground pour
    included. Other vias show only where they have a measured end."""
    shapes = [t.geometry for t in snapshot.tracks if t.layer == layer]
    shapes += [p.shapes[layer] for p in snapshot.pads if layer in p.shapes]
    shapes += [
        Point(v.position).buffer(v.diameter / 2)
        for v in snapshot.vias
        if v.net in grounds
    ]
    shapes += [z.fills[layer] for z in snapshot.copper_zones if layer in z.fills]
    return unary_union(shapes)


def copper_map(fields: ReturnFields, layer: str, grounds: Sequence[str]) -> CoverageMap:
    """The return field of each pixel's reference plane, on `layer`'s copper.

    A pixel's reference is chosen as for a via end there. Copper over a void
    in that plane has no return path, unless it is a return's own copper. Copper outside every other layer's
    ground outline has no reference plane to measure and is left blank, as
    on a two-layer board's ground layer.
    """
    grid = fields.grid
    # Every pixel a track touches, so thin tracks stay unbroken.
    copper = grid.touching(layer_copper(fields.snapshot, layer, grounds))
    dist = np.full(grid.shape, np.nan)
    unassigned = copper.copy()
    for plane, outline in reference_outlines(fields.snapshot, layer, grounds):
        here = unassigned & grid.mask(outline)
        dist[here] = fields.field(plane)[here]
        unassigned &= ~here
    return CoverageMap(COPPER_MAP_NET, layer, dist, np.zeros(grid.shape, bool))


def copper_maps(
    fields: ReturnFields, references: Sequence[ViaReferences], grounds: Sequence[str]
) -> list[CoverageMap]:
    """One map per copper layer a non-exempt via end connects on."""
    layers = {c.layer for r in references if not r.exempt for c in r.layers}
    return [
        copper_map(fields, L, grounds)
        for L in fields.snapshot.copper_layers
        if L in layers
    ]


def antipad(
    fields: ReturnFields, copper: BaseGeometry, plane: ReferencePlane
) -> BaseGeometry:
    """`copper` grown to the edge of its clearance hole in `plane`'s fill,
    which the plane's field leaves as a void."""
    fill = fields.snapshot.fill(plane.ground, plane.layer)
    return copper.buffer(fill.distance(copper))


@dataclasses.dataclass(frozen=True)
class End:
    """Copper that changes layers, its clearance hole in the reference
    plane and its return distance."""

    copper: BaseGeometry
    hole: BaseGeometry
    distance: float


def via_ends(
    layer: str, results: Sequence[PlaneReturn], fields: ReturnFields
) -> list[End]:
    """The measured via ends on `layer`."""
    diameters = {(v.position, v.net): v.diameter for v in fields.snapshot.vias}
    out = []
    for r in results:
        if layer not in r.layers or r.outcome is not Outcome.MEASURED:
            continue
        assert r.plane is not None
        via = Point(r.position).buffer(diameters[(r.position, r.net)] / 2)
        out.append(End(via, antipad(fields, via, r.plane), r.distance))
    return out


def pad_ends(
    layer: str, fields: ReturnFields, grounds: Sequence[str], distance: float
) -> list[End]:
    """Signal PTH pads on `layer`, measured as a via end there would be.
    They are shown but never reported unserved."""
    snap = fields.snapshot
    outlines = list(reference_outlines(snap, layer, grounds))
    out = []
    for p in snap.pads:
        if p.kind is not PadKind.PTH or not p.net or p.net in grounds:
            continue
        if layer not in p.shapes:
            continue
        at = Point(p.position)
        plane = next((pl for pl, o in outlines if o.contains(at)), None)
        found = plane and fields.distance_at(p.position, plane, distance)
        if plane is None or found is None:
            continue
        copper = p.shapes[layer]
        out.append(End(copper, antipad(fields, copper, plane), found))
    return out


def paint_ends(m: CoverageMap, grid: Grid, ends: Sequence[End]) -> None:
    """Show each end's distance over its copper and over the layer's copper
    in its hole, which the plane's field leaves as a void. The copper goes
    on top, so a neighbour's hole never covers it."""
    for e in ends:
        win = grid.bounds_window(e.hole.bounds)
        on_copper = grid.mask(e.hole, win) & ~np.isnan(m.distance[win])
        m.distance[win][on_copper] = e.distance
    for e in ends:
        win = grid.bounds_window(e.copper.bounds)
        m.distance[win][grid.mask(e.copper, win)] = e.distance


def markers(
    results: Sequence[PlaneReturn], maps: Sequence[CoverageMap], distance: float
) -> dict[render.MapKey, render.Markers]:
    """Per map: a ring of radius `distance` around each via end
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


def ring_maps(
    maps: Sequence[CoverageMap],
    marks: Mapping[render.MapKey, render.Markers],
    grid: Grid,
) -> list[CoverageMap]:
    """The maps cut to the copper inside their rings, which picks out the
    unserved via ends; a map with no ring is dropped."""
    out = []
    for m in maps:
        rings = marks[(m.net, m.layer)].rings
        if not rings:
            continue
        inside = grid.mask(unary_union([Point(c).buffer(r) for c, r in rings]))
        m.distance[~inside] = np.nan
        out.append(m)
    return out


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
    full_map: bool = False,
    kicad: kipy.KiCad | None = None,
) -> None:
    """Run returnvia against the open board.

    `check` only measures the existing returns: no planning, no kicad-cli,
    no vias or markers. Planned vias count as returns in the measurement,
    except forced ones. The heatmaps show only the copper around unserved
    via ends, or with `full_map` all copper. Overlays are placed in their
    own commit, also in a dry run.
    """
    kicad = kicad or kipy.KiCad()
    board = kicad.get_board()
    check_outline(board)
    place_overlay = outputs.overlay and overlay.overlay_ready(board, outputs, dry_run)

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

    maps = copper_maps(fields, references, grounds)
    for m in maps:
        ends = [
            *pad_ends(m.layer, fields, grounds, options.distance),
            *via_ends(m.layer, results, fields),
        ]
        paint_ends(m, fields.grid, ends)
    marks = markers(results, maps, options.distance)
    if not maps:
        logger.info("No signal via changes layers: no heatmaps")
    elif not full_map:
        maps = ring_maps(maps, marks, fields.grid)
        if not maps:
            logger.info("No unserved via ends: no heatmaps")
    if outputs.render is not None:
        if maps:
            render.render_panels(
                outputs.render,
                fields.snapshot,
                fields.grid,
                maps,
                marks,
                options.distance,
                STYLE,
            )
        else:
            # Replaces the previous run's heatmap, which would show stale rings.
            render.render_outline(
                outputs.render,
                fields.snapshot,
                fields.grid,
                SERVED_TITLE.format(distance=options.distance),
            )
        logger.info(f"Wrote {outputs.render}")

    if outputs.overlay:
        overlay.show_overlays(
            board,
            (
                lambda out_dir: render.write_overlays(
                    out_dir, fields.grid, maps, marks, options.distance, STYLE
                )
            )
            if maps
            else None,
            snapshot,
            outputs,
            place_overlay,
            OVERLAY_COMMIT,
        )
