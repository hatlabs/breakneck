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

import numpy as np
import shapely
from loguru import logger
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from breakneck.plan import FORBIDDEN_RADIUS, Plan, PlannedUnit, Stub, UnfixableSpot
from breakneck.snapshot import Pad, PadKind, Point2, Snapshot, ViaSize

# Other-net copper clearance on every layer the planned copper touches.
OTHER_NET_CLEARANCE = 0.3
# Added to a pad's clearance override; HALSER mounting holes carry 1.6 mm.
OVERRIDE_MARGIN = 0.1
OWN_PAD_GAP = 0.15
SAME_NET_PAD_GAP = 0.15
HOLE_TO_HOLE_GAP = 0.3
EDGE_GAP = 0.5

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


@dataclass(frozen=True)
class _Copper:
    """Copper on one layer, or on every layer when `layer` is None."""

    net: str
    layer: str | None
    geometry: BaseGeometry
    clearance: float
    pad_id: str | None = None


@dataclass(frozen=True)
class _UnplatedHole:
    centre: Point
    radius: float
    copper_gap: float


def parse_via_size(text: str) -> ViaSize:
    """Parse a `DIAMETER/DRILL` via size in mm, as given to `--via`."""
    try:
        diameter, drill = (float(v) for v in text.split("/"))
    except ValueError:
        raise ValueError(f"expected DIAMETER/DRILL in mm, got {text!r}") from None
    if not 0 < drill < diameter:
        raise ValueError(f"drill must be positive and below the diameter: {text!r}")
    return ViaSize(diameter, drill)


def _clearance(override: float | None) -> float:
    if override is None:
        return OTHER_NET_CLEARANCE
    return max(OTHER_NET_CLEARANCE, override + OVERRIDE_MARGIN)


def _minor_dimension(geom: BaseGeometry) -> float:
    corners = shapely.get_coordinates(shapely.oriented_envelope(geom))
    return min(math.dist(corners[0], corners[1]), math.dist(corners[1], corners[2]))


def _rounded(p: Point2) -> Point2:
    return (round(p[0], POSITION_DECIMALS), round(p[1], POSITION_DECIMALS))


class _Board:
    """Obstacles for the planner, grown with every unit it places."""

    def __init__(self, snapshot: Snapshot, forbidden: Sequence[Point2]) -> None:
        self.snapshot = snapshot
        self.forbidden = [Point(p) for p in forbidden]
        self.copper: list[_Copper] = []
        self.unplated: list[_UnplatedHole] = []
        drills: list[tuple[float, float, float]] = []
        for pad in snapshot.pads:
            clearance = _clearance(pad.clearance_override)
            for layer, geom in pad.shapes.items():
                self.copper.append(_Copper(pad.net, layer, geom, clearance, pad.id))
            if pad.drill > 0:
                drills.append((*pad.position, pad.drill / 2))
                if pad.kind == PadKind.NPTH:
                    self.unplated.append(
                        _UnplatedHole(Point(pad.position), pad.drill / 2, clearance)
                    )
        for t in snapshot.tracks:
            self.copper.append(_Copper(t.net, t.layer, t.geometry, OTHER_NET_CLEARANCE))
        for v in snapshot.vias:
            disk = Point(v.position).buffer(v.diameter / 2)
            self.copper.append(_Copper(v.net, None, disk, OTHER_NET_CLEARANCE))
            drills.append((*v.position, v.drill / 2))
        self.tree = STRtree([c.geometry for c in self.copper])
        self.planned_copper: list[_Copper] = []
        # Columns: x, y, radius.
        self.drills = np.array(drills, dtype=float).reshape(-1, 3)
        self.max_clearance = max(
            [OTHER_NET_CLEARANCE, *(c.clearance for c in self.copper)]
        )

        self.via_area = snapshot.board_outline
        self.keepouts = shapely.union_all(
            [z.outline for z in snapshot.zones if z.keepout_vias]
        )

    def place(self, unit: PlannedUnit) -> None:
        disk = Point(unit.position).buffer(unit.size.diameter / 2)
        self.planned_copper.append(_Copper(unit.net, None, disk, OTHER_NET_CLEARANCE))
        hole = [*unit.position, unit.size.drill / 2]
        self.drills = np.vstack([self.drills, hole])
        if unit.stub is not None:
            s = unit.stub
            line = LineString([s.start, s.end]).buffer(s.width / 2)
            self.planned_copper.append(
                _Copper(unit.net, s.layer, line, OTHER_NET_CLEARANCE)
            )

    def _nearby(self, geom: BaseGeometry, reach: float) -> Iterable[_Copper]:
        for i in self.tree.query(geom, predicate="dwithin", distance=reach):
            yield self.copper[int(i)]
        yield from self.planned_copper

    def clashes_with_other_nets(
        self, geom: BaseGeometry, half_width: float, net: str, layer: str | None
    ) -> bool:
        """Other-net copper within clearance of `geom` buffered by `half_width`.

        `layer` None means the copper spans every layer, like a via.
        """
        for c in self._nearby(geom, half_width + self.max_clearance):
            if c.net == net:
                continue
            if layer is not None and c.layer is not None and c.layer != layer:
                continue
            if geom.distance(c.geometry) < half_width + c.clearance:
                return True
        return False

    def clashes_with_unplated(self, geom: BaseGeometry, half_width: float) -> bool:
        return any(
            geom.distance(h.centre) < h.radius + half_width + h.copper_gap
            for h in self.unplated
        )

    def clashes_with_drills(self, pos: Point2, drill_radius: float) -> bool:
        gaps = np.hypot(self.drills[:, 0] - pos[0], self.drills[:, 1] - pos[1])
        gaps -= self.drills[:, 2] + drill_radius
        return bool((gaps < HOLE_TO_HOLE_GAP).any())

    def via_ok(self, pos: Point2, net: str, size: ViaSize, own_pad: Pad) -> bool:
        c = Point(pos)
        r = size.diameter / 2
        if not self.via_area.contains(c) or c.distance(self.via_area.boundary) < (
            r + EDGE_GAP
        ):
            return False
        if not self.keepouts.is_empty and c.distance(self.keepouts) < r:
            return False
        if any(c.distance(f) < FORBIDDEN_RADIUS for f in self.forbidden):
            return False
        if self.clashes_with_drills(pos, size.drill / 2):
            return False
        if self.clashes_with_unplated(c, r):
            return False
        if self.clashes_with_other_nets(c, r, net, None):
            return False
        if any(
            other.pad_id not in (None, own_pad.id)
            and other.net == net
            and c.distance(other.geometry) < r + SAME_NET_PAD_GAP
            for other in self._nearby(c, r + SAME_NET_PAD_GAP)
        ):
            return False
        return self.reaches_zones(c.buffer(r), net, pos, _own_layer(own_pad))

    def reaches_zones(
        self, disk: BaseGeometry, net: str, pos: Point2, pad_layer: str
    ) -> bool:
        """The zone-layer rule.

        The pad's own layer is exempt from the fill check: the stub connects
        the via there, and the fill around the pad is cut by its thermal gap.
        """
        required = self.snapshot.required_layers(net, pos)
        if len(required) < 2:
            return False
        return all(
            self.snapshot.fill(net, layer).contains(disk)
            for layer in required
            if layer != pad_layer
        )

    def stub_ok(self, line: LineString, width: float, net: str, layer: str) -> bool:
        half = width / 2
        return not self.clashes_with_other_nets(
            line, half, net, layer
        ) and not self.clashes_with_unplated(line, half)

    def connected_region(self, pad: Pad, layer: str) -> BaseGeometry:
        """Pad copper plus same-net tracks, pads and pour it reaches on `layer`."""
        region = pad.shapes[layer]
        neighbourhood = region.buffer(SERVED_DISTANCE + MAX_STUB)
        pour = self.snapshot.fill(pad.net, layer).intersection(neighbourhood)
        same = [
            c.geometry
            for c in self._nearby(neighbourhood, 0.0)
            if c.net == pad.net
            and c.layer == layer
            and c.geometry.intersects(neighbourhood)
        ] + list(shapely.get_parts(pour))
        while True:
            touching = [g for g in same if g.intersects(region)]
            grown = shapely.union_all([region, *touching])
            same = [g for g in same if not g.intersects(region)]
            if not touching:
                return region
            region = grown

    def is_served(self, pad: Pad, layer: str) -> bool:
        shape = pad.shapes[layer]
        vias = [
            c.geometry
            for c in self._nearby(shape, SERVED_DISTANCE)
            if c.layer is None and c.net == pad.net
        ]
        near = [g for g in vias if g.distance(shape) <= SERVED_DISTANCE]
        if not near:
            return False
        region = self.connected_region(pad, layer)
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
    board: _Board, pad: Pad, size: ViaSize, outward: tuple[float, float] | None
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
            if board.via_ok(pos, pad.net, size, pad) and board.stub_ok(
                stub, width, pad.net, layer
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
    board = _Board(snapshot, forbidden)
    units: list[PlannedUnit] = []
    unfixable: list[UnfixableSpot] = []
    candidates = _candidates(snapshot, nets)
    for pad in candidates:
        layer = _own_layer(pad)
        if board.is_served(pad, layer):
            continue
        size = options.via or snapshot.via_size(pad.net)
        unit = _plan_pad(board, pad, size, _outward(snapshot, pad))
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
        board.place(unit)
    logger.info(
        f"padvia: {len(candidates)} candidate pads, {len(units)} need a via, "
        f"{len(unfixable)} unfixable"
    )
    return Plan(units=tuple(units), unfixable=tuple(unfixable))


def _pad_label(snapshot: Snapshot, pad: Pad) -> str:
    ref = next((f.ref for f in snapshot.footprints if f.id == pad.footprint_id), "?")
    return f"{ref}.{pad.number}"
