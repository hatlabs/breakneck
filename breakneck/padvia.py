"""Plan a via plus stub for every unserved SMD pad of the stitched nets.

A pad is served when a same-net via lies within `SERVED_DISTANCE` of it and
is reached through the pad's same-net tracks, pads or pour on the pad's layer.
For an unserved pad the planner scans `DIRECTIONS` rays from the pad centre,
takes the first legal spot on each and keeps the one with the shortest stub,
penalising rays that point back toward the footprint centre.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

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
POSITION_DECIMALS = 3


@dataclass(frozen=True)
class PadviaOptions:
    nets: tuple[str, ...]
    via: ViaSize | None = None


def _minor_dimension(geom: BaseGeometry) -> float:
    corners = shapely.get_coordinates(shapely.oriented_envelope(geom))
    return min(math.dist(corners[0], corners[1]), math.dist(corners[1], corners[2]))


def _rounded(p: Point2) -> Point2:
    return (round(p[0], POSITION_DECIMALS), round(p[1], POSITION_DECIMALS))


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


def _is_served(obstacles: Obstacles, pad: Pad, layer: str) -> bool:
    shape = pad.shapes[layer]
    vias = obstacles.nearby_copper(shape, SERVED_DISTANCE, pad.net, None)
    near = [g for g in vias if g.distance(shape) <= SERVED_DISTANCE]
    if not near:
        return False
    region = _connected_region(obstacles, pad, layer)
    return any(g.intersects(region) for g in near)


def _own_layer(pad: Pad) -> str:
    (layer,) = pad.shapes
    return layer


def _candidates(snapshot: Snapshot, nets: Iterable[str]) -> list[Pad]:
    wanted = set(nets)
    pads = [
        p
        for p in snapshot.pads
        if p.net in wanted and p.kind == PadKind.SMD and len(p.shapes) == 1
    ]
    return sorted(pads, key=lambda p: p.position)


def _plan_pad(
    obstacles: Obstacles,
    pad: Pad,
    size: ViaSize,
    outward: tuple[float, float] | None,
) -> PlannedUnit | None:
    layer = _own_layer(pad)
    shape = pad.shapes[layer]
    centre = shape.centroid
    cx, cy = centre.x, centre.y
    width = max(MIN_STUB_WIDTH, min(MAX_STUB_WIDTH, _minor_dimension(shape)))
    r = size.diameter / 2
    best: tuple[float, Point2] | None = None
    for k in range(DIRECTIONS):
        a = 2 * math.pi * k / DIRECTIONS
        dx, dy = math.cos(a), math.sin(a)
        t = 0.0
        while Point(cx + dx * t, cy + dy * t).distance(shape) < r + OWN_PAD_GAP:
            t += CLEARING_STEP
        t0 = t
        while t <= t0 + MAX_STUB:
            pos = _rounded((cx + dx * t, cy + dy * t))
            stub = LineString([(cx, cy), pos])
            if _via_ok(obstacles, pos, pad.net, size, pad) and _stub_ok(
                obstacles, stub, width, pad.net, layer
            ):
                alignment = dx * outward[0] + dy * outward[1] if outward else 0.0
                score = (t - t0) + INWARD_PENALTY * (1 - alignment)
                if best is None or score < best[0]:
                    best = (score, pos)
                break
            t += SEARCH_STEP
    if best is None:
        return None
    pos = best[1]
    return PlannedUnit(
        position=pos,
        net=pad.net,
        size=size,
        stub=Stub(start=(cx, cy), end=pos, width=width, layer=layer),
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
    nets, _ = snapshot.split_stitchable(options.nets)
    obstacles = Obstacles(snapshot, forbidden)
    units: list[PlannedUnit] = []
    unfixable: list[UnfixableSpot] = []
    candidates = _candidates(snapshot, nets)
    for pad in candidates:
        layer = _own_layer(pad)
        if _is_served(obstacles, pad, layer):
            continue
        size = options.via or snapshot.via_size(pad.net)
        unit = _plan_pad(obstacles, pad, size, _outward(snapshot, pad))
        if unit is None:
            c = pad.shapes[layer].centroid
            unfixable.append(
                UnfixableSpot(
                    position=_rounded((c.x, c.y)),
                    net=pad.net,
                    reason=f"pad {_pad_label(snapshot, pad)}: no via spot "
                    f"within {MAX_STUB} mm",
                    layer=layer,
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


def _pad_label(snapshot: Snapshot, pad: Pad) -> str:
    ref = next((f.ref for f in snapshot.footprints if f.id == pad.footprint_id), "?")
    return f"{ref}.{pad.number}"
