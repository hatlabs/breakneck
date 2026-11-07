"""Plan a via plus stub for every unserved SMD pad of the stitched nets.

A pad is a candidate when its own net has fill on another copper layer that a
via of the net's size fits in within the planner's reach of the pad; explicit
nets narrow that selection.

A pad is served when a same-net via or through-hole pad lies within
`SERVED_DISTANCE` of it and is reached through the pad's same-net tracks, pads
or pour on the pad's layer. Both distances run copper edge to copper edge on
the pad's layer: the via's disk, the through-hole pad's copper there.
For an unserved pad the planner scans `DIRECTIONS` rays from the pad centre,
takes the first legal spot on each and keeps the one with the shortest stub,
penalising rays that point back toward the footprint centre. The stub runs in
45° steps: a diagonal and a straight leg, diagonal first where it is clear.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import shapely
from loguru import logger
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry

from breakneck.obstacles import Obstacles
from breakneck.plan import Plan, PlannedUnit, Stub, UnfixableSpot
from breakneck.snapshot import Pad, PadKind, Point2, Snapshot, ViaSize

OWN_PAD_GAP = 0.15

SERVED_DISTANCE = 1.0
DIRECTIONS = 48
CLEARING_STEP = 0.025
SEARCH_STEP = 0.05
MAX_STUB = 1.2
INWARD_PENALTY = 0.4
MIN_STUB_WIDTH = 0.2
MAX_STUB_WIDTH = 0.4
MIN_OUTWARD_LENGTH = 1e-6
# How far past its own radius from the pad edge the planner can put a via
# centre: it clears the pad by the gap, possibly overshooting by one step, then
# searches a stub's length.
FILL_REACH = OWN_PAD_GAP + CLEARING_STEP + MAX_STUB
POSITION_DECIMALS = 3

RoomCache = dict[tuple[str, str], BaseGeometry]


@dataclass(frozen=True)
class PadviaOptions:
    nets: tuple[str, ...] | None = None
    via: ViaSize | None = None


def _minor_dimension(geom: BaseGeometry) -> float:
    corners = shapely.get_coordinates(shapely.oriented_envelope(geom))
    return min(math.dist(corners[0], corners[1]), math.dist(corners[1], corners[2]))


def _rounded(p: Point2) -> Point2:
    return (round(p[0], POSITION_DECIMALS), round(p[1], POSITION_DECIMALS))


def _doglegs(start: Point2, via: Point2) -> list[tuple[Point2, ...]]:
    """Paths from `start` to `via` with legs at multiples of 45°, preferred
    first: diagonal-first, then straight-first, or the one segment when the
    offset is itself at a multiple of 45°.

    Both points sit on the position grid, so the bend does too and every leg
    stays exactly at its angle once committed.
    """
    dx = round(via[0] - start[0], POSITION_DECIMALS)
    dy = round(via[1] - start[1], POSITION_DECIMALS)
    m = min(abs(dx), abs(dy))
    if m == 0 or abs(dx) == abs(dy):
        return [(start, via)]
    sx, sy = math.copysign(m, dx), math.copysign(m, dy)
    diagonal_first = _rounded((start[0] + sx, start[1] + sy))
    straight_first = _rounded((via[0] - sx, via[1] - sy))
    return [(start, diagonal_first, via), (start, straight_first, via)]


def _detour(path: Sequence[Point2]) -> float:
    """How much longer `path` runs than the straight line between its ends."""
    legs = sum(math.dist(a, b) for a, b in pairwise(path))
    return legs - math.dist(path[0], path[-1])


def _via_ok(
    obstacles: Obstacles, pos: Point2, net: str, size: ViaSize, own_pad: Pad
) -> bool:
    return (
        obstacles.in_bounds(pos, size)
        and obstacles.via_is_clear(
            pos, net, size, polygon_disk=False, exempt_pad=own_pad.id
        )
        and obstacles.reaches_zones(
            pos, net, size, filled=True, exempt_layer=_own_layer(own_pad)
        )
    )


def _stub_ok(
    obstacles: Obstacles, line: LineString, width: float, net: str, layer: str
) -> bool:
    half = width / 2
    return not obstacles.clashes_with_other_nets(
        line, half, net, layer
    ) and not obstacles.clashes_with_unplated(line, half)


def _clear_dogleg(
    obstacles: Obstacles,
    start: Point2,
    via: Point2,
    width: float,
    net: str,
    layer: str,
) -> tuple[Point2, ...] | None:
    """The preferred dogleg from `start` to `via` that keeps clear, if any."""
    return next(
        (
            p
            for p in _doglegs(start, via)
            if _stub_ok(obstacles, LineString(p), width, net, layer)
        ),
        None,
    )


def _connected_region(obstacles: Obstacles, pad: Pad, layer: str) -> BaseGeometry:
    """Pad copper plus same-net tracks, pads and pour it reaches on `layer`."""
    region = pad.shapes[layer]
    neighbourhood = region.buffer(SERVED_DISTANCE + MAX_STUB)
    pour = obstacles.snapshot.fill(pad.net, layer).intersection(neighbourhood)
    same = [
        g
        for g in obstacles.nearby_copper(neighbourhood, 0.0, pad.net, layer)
        if g.intersects(neighbourhood)
    ] + list(shapely.get_parts(pour))
    while True:
        touching = [g for g in same if g.intersects(region)]
        grown = shapely.union_all([region, *touching])
        same = [g for g in same if not g.intersects(region)]
        if not touching:
            return region
        region = grown


def _through_copper(obstacles: Obstacles, pad: Pad, layer: str) -> list[BaseGeometry]:
    """Same-net vias, and same-net PTH pads' copper on `layer`: both join
    every copper layer."""
    shape = pad.shapes[layer]
    vias = obstacles.nearby_copper(shape, SERVED_DISTANCE, pad.net, None)
    pins = [
        p.shapes[layer]
        for p in obstacles.snapshot.pads
        if p.kind is PadKind.PTH and p.net == pad.net and layer in p.shapes
    ]
    return vias + pins


def _is_served(obstacles: Obstacles, pad: Pad, layer: str) -> bool:
    shape = pad.shapes[layer]
    through = _through_copper(obstacles, pad, layer)
    near = [g for g in through if g.distance(shape) <= SERVED_DISTANCE]
    if not near:
        return False
    region = _connected_region(obstacles, pad, layer)
    return any(g.intersects(region) for g in near)


def _own_layer(pad: Pad) -> str:
    (layer,) = pad.shapes
    return layer


def _via_size(snapshot: Snapshot, options: PadviaOptions, net: str) -> ViaSize:
    return options.via or snapshot.via_size(net)


def _room_for_via(
    snapshot: Snapshot, net: str, layer: str, r: float, cache: RoomCache
) -> BaseGeometry:
    """Where a via centre of radius `r` can sit with its disk inside the fill."""
    key = (net, layer)
    if key not in cache:
        cache[key] = snapshot.fill(net, layer).buffer(-r)
    return cache[key]


def _via_fits_own_fill_elsewhere(
    snapshot: Snapshot, pad: Pad, size: ViaSize, cache: RoomCache
) -> bool:
    """A via within the planner's reach of the pad can sit in the pad's net's
    fill on another copper layer.

    The planner puts the via centre at most `r + FILL_REACH` from the pad
    copper, and the via disk lies inside the fill exactly when its centre lies
    inside the fill eroded by `r`; so the eroded fill must come within
    `r + FILL_REACH` of the pad copper. Fill narrower than the via, such as a
    sliver left under a same-net track, never qualifies.

    The fill, not the zone outline: a higher-priority zone of another net can
    own the copper under the pad, and no via there would reach the net.
    """
    layer = _own_layer(pad)
    copper = pad.shapes[layer]
    r = size.diameter / 2
    for other in snapshot.copper_layers:
        if other == layer:
            continue
        room = _room_for_via(snapshot, pad.net, other, r, cache)
        if not room.is_empty and room.distance(copper) <= r + FILL_REACH:
            return True
    return False


def _candidates(snapshot: Snapshot, options: PadviaOptions) -> list[Pad]:
    cache: RoomCache = {}
    pads = [
        p
        for p in snapshot.pads
        if p.kind == PadKind.SMD
        and len(p.shapes) == 1
        and _via_fits_own_fill_elsewhere(
            snapshot, p, _via_size(snapshot, options, p.net), cache
        )
    ]
    nets = options.nets
    derived = nets is None
    if derived:
        # Single-layer pours are expected here, not worth a warning each.
        stitchable = [n for n in {p.net for p in pads} if snapshot.is_stitchable(n)]
    else:
        stitchable, _ = snapshot.split_stitchable(nets)
    if derived:
        logger.info(
            "padvia: nets with via-sized fill near their pads on another layer: "
            + (", ".join(sorted(stitchable)) or "none")
        )
    return sorted((p for p in pads if p.net in stitchable), key=lambda p: p.position)


def _plan_pad(
    obstacles: Obstacles,
    pad: Pad,
    size: ViaSize,
    outward: tuple[float, float] | None,
) -> PlannedUnit | None:
    layer = _own_layer(pad)
    shape = pad.shapes[layer]
    centre = shape.centroid
    start = _rounded((centre.x, centre.y))
    x0, y0 = start
    width = max(MIN_STUB_WIDTH, min(MAX_STUB_WIDTH, _minor_dimension(shape)))
    r = size.diameter / 2
    best: tuple[float, tuple[Point2, ...]] | None = None
    for k in range(DIRECTIONS):
        a = 2 * math.pi * k / DIRECTIONS
        dx, dy = math.cos(a), math.sin(a)
        t = 0.0
        while Point(x0 + dx * t, y0 + dy * t).distance(shape) < r + OWN_PAD_GAP:
            t += CLEARING_STEP
        t0 = t
        while t <= t0 + MAX_STUB:
            pos = _rounded((x0 + dx * t, y0 + dy * t))
            path = (
                _clear_dogleg(obstacles, start, pos, width, pad.net, layer)
                if _via_ok(obstacles, pos, pad.net, size, pad)
                else None
            )
            if path is not None:
                alignment = dx * outward[0] + dy * outward[1] if outward else 0.0
                # The detour, not the length between the rounded ends, so
                # sub-µm rounding can't break ties between straight stubs.
                score = (t - t0) + _detour(path) + INWARD_PENALTY * (1 - alignment)
                if best is None or score < best[0]:
                    best = (score, path)
                break
            t += SEARCH_STEP
    if best is None:
        return None
    path = best[1]
    return PlannedUnit(
        position=path[-1],
        net=pad.net,
        size=size,
        stub=Stub(path=path, width=width, layer=layer),
    )


def _outward(snapshot: Snapshot, pad: Pad) -> tuple[float, float] | None:
    """Unit vector from the pad's footprint centre to the pad, if defined."""
    fp = next((f for f in snapshot.footprints if f.id == pad.footprint_id), None)
    if fp is None:
        return None
    centre = pad.shapes[_own_layer(pad)].centroid
    ox, oy = centre.x - fp.position[0], centre.y - fp.position[1]
    length = math.hypot(ox, oy)
    if length < MIN_OUTWARD_LENGTH:
        return None
    return (ox / length, oy / length)


def plan_padvia(
    snapshot: Snapshot, options: PadviaOptions, forbidden: Sequence[Point2]
) -> Plan:
    obstacles = Obstacles(snapshot, forbidden)
    units: list[PlannedUnit] = []
    unfixable: list[UnfixableSpot] = []
    candidates = _candidates(snapshot, options)
    for pad in candidates:
        layer = _own_layer(pad)
        if _is_served(obstacles, pad, layer):
            continue
        size = _via_size(snapshot, options, pad.net)
        unit = _plan_pad(obstacles, pad, size, _outward(snapshot, pad))
        if unit is None:
            shape = pad.shapes[layer]
            c = shape.centroid
            unfixable.append(
                UnfixableSpot(
                    position=_rounded((c.x, c.y)),
                    net=pad.net,
                    reason=f"pad {_pad_label(snapshot, pad)}: no via spot "
                    f"within {MAX_STUB} mm",
                    layer=layer,
                    extent=_extent(shape, c),
                )
            )
            continue
        units.append(unit)
        obstacles.add_unit(unit)
    logger.info(
        f"padvia: {len(candidates)} candidate pads, {len(units)} need a via, "
        f"{len(unfixable)} unfixable"
    )
    return Plan(units=tuple(units), unfixable=tuple(unfixable))


def _extent(shape: BaseGeometry, centre: Point) -> float:
    """Distance from `centre` to the farthest corner of `shape`'s bounding box."""
    x0, y0, x1, y1 = shape.bounds
    return max(
        math.dist((centre.x, centre.y), q)
        for q in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
    )


def _pad_label(snapshot: Snapshot, pad: Pad) -> str:
    ref = next((f.ref for f in snapshot.footprints if f.id == pad.footprint_id), "?")
    return f"{ref}.{pad.number}"
