"""Return vias: a ground via near every via on a non-ground net.

The same check drives `returnvia`, which plans the missing vias, and
`gndvia`, which only draws lines from each unserved via to its nearest
return. Geometry is shapely only, so neither needs the `place` extra.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

import shapely
from loguru import logger
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.prepared import PreparedGeometry, prep
from shapely.strtree import STRtree

from breakneck.plan import FORBIDDEN_RADIUS, Plan, PlannedUnit, UnfixableSpot
from breakneck.snapshot import (
    MIN_STITCH_LAYERS,
    PadKind,
    Point2,
    Snapshot,
    ViaSize,
)

DEFAULT_DISTANCE = 1.5
# Spacing rules the planner keeps; kicad-cli DRC remains the authority.
CLEARANCE = 0.3
EDGE_CLEARANCE = 0.5
HOLE_TO_HOLE = 0.3
SAME_NET_PAD_GAP = 0.15
# Candidate rings around each unserved via.
RING_STEP = 0.05
RING_ANGLES = 36
FORCE_RING_ANGLES = 72
POSITION_DECIMALS = 3
# Rounding a ring point moves it outward by less than this; rings stop short by it.
ROUNDING_MARGIN = 10**-POSITION_DECIMALS
DISTANCE_EPSILON = 1e-9

OUTSIDE_GROUND = "outside every ground zone"
NO_ROOM = "no room for a return via"
WOULD_OVERLAP = "every spot overlaps other-net copper"


@dataclass(frozen=True)
class ReturnViaOptions:
    distance: float = DEFAULT_DISTANCE
    ground_nets: tuple[str, ...] | None = None
    via: ViaSize | None = None
    force: bool = False


@dataclass(frozen=True)
class Unserved:
    """A via on a non-ground net with no return of its ground within reach."""

    position: Point2
    net: str
    diameter: float
    ground: str | None
    nearest_return: Point2 | None


def resolve_ground_nets(
    snapshot: Snapshot, ground_nets: Iterable[str] | None
) -> list[str]:
    grounds = sorted(ground_nets) if ground_nets else snapshot.default_ground_nets()
    if not grounds:
        logger.warning("No ground nets: no net has zones on two inner layers")
    return grounds


def _returns(snapshot: Snapshot, grounds: Iterable[str]) -> dict[str, list[Point2]]:
    """Existing return positions per ground: its vias and through-hole pads."""
    out: dict[str, list[Point2]] = {g: [] for g in grounds}
    for v in snapshot.vias:
        if v.net in out:
            out[v.net].append(v.position)
    for p in snapshot.pads:
        if p.net in out and p.kind is PadKind.PTH:
            out[p.net].append(p.position)
    return out


def _nearest(position: Point2, candidates: Sequence[Point2]) -> Point2 | None:
    return min(candidates, key=lambda c: math.dist(c, position), default=None)


def unserved_vias(
    snapshot: Snapshot, ground_nets: Iterable[str] | None, distance: float
) -> list[Unserved]:
    """Vias that lack a return of the ground at their location within `distance`.

    Vias on ground nets, and vias with no net, never need a return. A via
    outside every ground zone is unserved with `ground` None.
    """
    grounds = resolve_ground_nets(snapshot, ground_nets)
    returns = _returns(snapshot, grounds)
    out = []
    for v in sorted(snapshot.vias, key=lambda v: v.position):
        if not v.net or v.net in returns:
            continue
        ground = snapshot.ground_net_at(v.position, grounds)
        nearest = _nearest(v.position, returns[ground]) if ground else None
        if (
            nearest is not None
            and math.dist(nearest, v.position) <= distance + DISTANCE_EPSILON
        ):
            continue
        out.append(Unserved(v.position, v.net, v.diameter, ground, nearest))
    return out


def return_lines(
    snapshot: Snapshot, ground_nets: Iterable[str] | None, distance: float
) -> tuple[list[tuple[Point2, Point2]], list[Point2]]:
    """`gndvia` output: lines from unserved vias to their nearest return.

    Vias outside every ground zone, or whose ground has no return at all,
    get a marker position instead of a line.
    """
    lines, markers = [], []
    for u in unserved_vias(snapshot, ground_nets, distance):
        if u.nearest_return is None:
            markers.append(u.position)
        else:
            lines.append((u.position, u.nearest_return))
    return lines, markers


# ---- obstacles ---------------------------------------------------------------


@dataclass(frozen=True)
class _Obstacle:
    """Copper a through via must keep clear of on every layer.

    `same_net_gap` None lets a via of the same net touch it.
    """

    net: str
    geometry: BaseGeometry
    clearance: float
    same_net_gap: float | None = None

    def required_gap(self, net: str) -> float | None:
        return self.clearance if self.net != net else self.same_net_gap


@dataclass(frozen=True)
class _Circle:
    """A planned via, or a mounting hole whose keep-off binds every net."""

    centre: Point2
    radius: float
    net: str | None = None
    keep_off: float = 0.0

    def gap(self, position: Point2, radius: float) -> float:
        return math.dist(position, self.centre) - self.radius - radius


@dataclass(frozen=True)
class _Hole:
    position: Point2
    radius: float


class _Board:
    """Spatial index of everything a new via must respect.

    Vias placed during planning are added as they are placed, so later
    candidates keep clear of them.
    """

    def __init__(self, snapshot: Snapshot, forbidden: Sequence[Point2]):
        self.snapshot = snapshot
        self.forbidden = list(forbidden)
        self.obstacles = _static_obstacles(snapshot)
        self.tree = STRtree([o.geometry for o in self.obstacles])
        self.max_gap = max((o.clearance for o in self.obstacles), default=CLEARANCE)
        self.keep_offs = [
            _Circle(
                p.position,
                p.drill / 2,
                None,
                max(HOLE_TO_HOLE, p.clearance_override or 0.0),
            )
            for p in snapshot.pads
            if p.kind is PadKind.NPTH and p.drill > 0
        ]
        self.holes = [
            _Hole(p.position, p.drill / 2) for p in snapshot.pads if p.drill > 0
        ]
        self.holes += [_Hole(v.position, v.drill / 2) for v in snapshot.vias]
        self.added: list[_Circle] = []
        self.added_holes: list[_Hole] = []
        self._fills: dict[tuple[str, str], PreparedGeometry] = {}

    @cached_property
    def placeable(self) -> PreparedGeometry:
        return prep(self.snapshot.board_outline)

    @cached_property
    def edge(self) -> BaseGeometry:
        return self.snapshot.board_outline.boundary

    @cached_property
    def keepouts(self) -> BaseGeometry:
        return shapely.union_all(
            [
                z.outline
                for z in self.snapshot.zones
                if z.is_rule_area and z.keepout_vias
            ]
        )

    def add_via(self, position: Point2, net: str, size: ViaSize) -> None:
        self.added.append(_Circle(position, size.diameter / 2, net))
        self.added_holes.append(_Hole(position, size.drill / 2))

    def _nearby(self, geom: BaseGeometry, margin: float) -> list[_Obstacle]:
        minx, miny, maxx, maxy = geom.bounds
        area = shapely.box(minx - margin, miny - margin, maxx + margin, maxy + margin)
        return [self.obstacles[i] for i in self.tree.query(area)]

    def in_bounds(self, position: Point2, size: ViaSize) -> bool:
        """Board edge, keepout and forbidden-position rules."""
        pt = Point(position)
        r = size.diameter / 2
        if self.snapshot.board_outline.is_empty:
            return False
        if (
            not self.placeable.contains(pt)
            or pt.distance(self.edge) < EDGE_CLEARANCE + r
        ):
            return False
        if not self.keepouts.is_empty and pt.buffer(r).intersects(self.keepouts):
            return False
        return all(math.dist(position, f) >= FORBIDDEN_RADIUS for f in self.forbidden)

    def clear(self, position: Point2, net: str, size: ViaSize) -> bool:
        """Full clearance to copper and holes on every layer."""
        radius = size.diameter / 2
        if any(k.gap(position, radius) < k.keep_off for k in self.keep_offs):
            return False
        if any(
            c.net != net and c.gap(position, radius) < CLEARANCE for c in self.added
        ):
            return False
        disk = Point(position).buffer(radius)
        for o in self._nearby(disk, self.max_gap):
            gap = o.required_gap(net)
            if gap is not None and disk.distance(o.geometry) < gap:
                return False
        r = size.drill / 2
        return all(
            math.dist(position, h.position) - r - h.radius >= HOLE_TO_HOLE
            for h in (*self.holes, *self.added_holes)
        )

    def clearance_to_other_nets(
        self, position: Point2, net: str, size: ViaSize
    ) -> float:
        """Smallest gap to other-net copper, capped at CLEARANCE.

        A mounting-hole keep-off counts as copper; zero or less overlaps.
        """
        radius = size.diameter / 2
        disk = Point(position).buffer(radius)
        gaps = [
            disk.distance(o.geometry)
            for o in self._nearby(disk, CLEARANCE)
            if o.net != net
        ]
        gaps += [c.gap(position, radius) for c in self.added if c.net != net]
        gaps += [k.gap(position, radius) - k.keep_off for k in self.keep_offs]
        return min([CLEARANCE, *gaps])

    def connects(
        self, position: Point2, net: str, size: ViaSize, *, filled: bool
    ) -> bool:
        """The zone-layer rule: inside `net`'s zones on at least two layers.

        With `filled`, the via must also sit inside the fill on each of those
        layers; the forced pass only asks for the zone outlines.
        """
        layers = self.snapshot.required_layers(net, position)
        if len(layers) < MIN_STITCH_LAYERS:
            return False
        if not filled:
            return True
        disk = Point(position).buffer(size.diameter / 2)
        return all(self._fill(net, L).contains(disk) for L in layers)

    def _fill(self, net: str, layer: str) -> PreparedGeometry:
        key = (net, layer)
        if key not in self._fills:
            self._fills[key] = prep(self.snapshot.fill(net, layer))
        return self._fills[key]


def _static_obstacles(snapshot: Snapshot) -> list[_Obstacle]:
    out: list[_Obstacle] = []
    for p in snapshot.pads:
        clearance = max(CLEARANCE, p.clearance_override or 0.0)
        if p.shapes:
            copper = shapely.union_all(list(p.shapes.values()))
            out.append(_Obstacle(p.net, copper, clearance, SAME_NET_PAD_GAP))
    out += [_Obstacle(t.net, t.geometry, CLEARANCE) for t in snapshot.tracks]
    out += [
        _Obstacle(v.net, Point(v.position).buffer(v.diameter / 2), CLEARANCE)
        for v in snapshot.vias
    ]
    return out


# ---- planner -----------------------------------------------------------------


def _ring_point(centre: Point2, radius: float, angle: float) -> Point2:
    return (
        round(centre[0] + radius * math.cos(angle), POSITION_DECIMALS),
        round(centre[1] + radius * math.sin(angle), POSITION_DECIMALS),
    )


def _angles(count: int) -> list[float]:
    return [2 * math.pi * k / count for k in range(count)]


def _radii(start: float, stop: float) -> list[float]:
    if start > stop:
        return []
    steps = math.floor((stop - start) / RING_STEP + DISTANCE_EPSILON)
    return [start + i * RING_STEP for i in range(steps + 1)]


def _served(
    position: Point2, ground: str, returns: dict[str, list[Point2]], distance: float
) -> bool:
    return any(
        math.dist(position, r) <= distance + DISTANCE_EPSILON for r in returns[ground]
    )


def _find_spot(
    board: _Board, u: Unserved, ground: str, size: ViaSize, distance: float
) -> Point2 | None:
    """Nearest spot on rings around `u` with full clearance."""
    start = u.diameter / 2 + size.diameter / 2 + CLEARANCE
    for r in _radii(start, distance - ROUNDING_MARGIN):
        for a in _angles(RING_ANGLES):
            pos = _ring_point(u.position, r, a)
            if (
                board.in_bounds(pos, size)
                and board.connects(pos, ground, size, filled=True)
                and board.clear(pos, ground, size)
            ):
                return pos
    return None


def _force_spot(
    board: _Board, u: Unserved, ground: str, size: ViaSize, distance: float
) -> tuple[Point2, float] | None:
    """Spot with the most clearance to other-net copper, nearest first on ties.

    Clearance is capped at the normal rule, so a closer spot wins once the
    rule is met.
    """
    best: tuple[tuple[float, float], Point2, float] | None = None
    start = u.diameter / 2 + size.diameter / 2
    for a in _angles(FORCE_RING_ANGLES):
        for r in _radii(start, distance - ROUNDING_MARGIN):
            pos = _ring_point(u.position, r, a)
            if not board.in_bounds(pos, size) or not board.connects(
                pos, ground, size, filled=False
            ):
                continue
            gap = board.clearance_to_other_nets(pos, ground, size)
            score = (gap, -r)
            if best is None or score > best[0]:
                best = (score, pos, gap)
    return None if best is None else (best[1], best[2])


def plan_return_vias(
    snapshot: Snapshot, options: ReturnViaOptions, forbidden: Sequence[Point2]
) -> Plan:
    """Planner for `returnvia`; see `breakneck.plan.Planner`.

    A first pass places full-clearance vias. With `force`, a second pass
    gives each via still unserved the spot with the most clearance, unless
    that spot would overlap other-net copper.
    """
    grounds = resolve_ground_nets(snapshot, options.ground_nets)
    unserved = unserved_vias(snapshot, grounds, options.distance)
    board = _Board(snapshot, forbidden)
    returns = _returns(snapshot, grounds)
    units: list[PlannedUnit] = []
    unfixable: list[UnfixableSpot] = []
    pending: list[tuple[Unserved, str, ViaSize]] = []

    for u in unserved:
        if u.ground is None:
            unfixable.append(UnfixableSpot(u.position, u.net, OUTSIDE_GROUND))
            continue
        if _served(u.position, u.ground, returns, options.distance):
            continue
        size = options.via or snapshot.via_size(u.ground)
        pos = _find_spot(board, u, u.ground, size, options.distance)
        if pos is None:
            pending.append((u, u.ground, size))
            continue
        units.append(PlannedUnit(pos, u.ground, size))
        board.add_via(pos, u.ground, size)
        returns[u.ground].append(pos)

    for u, ground, size in pending:
        if _served(u.position, ground, returns, options.distance):
            continue
        if not options.force:
            unfixable.append(UnfixableSpot(u.position, u.net, NO_ROOM))
            continue
        spot = _force_spot(board, u, ground, size, options.distance)
        if spot is None or spot[1] <= 0:
            reason = NO_ROOM if spot is None else WOULD_OVERLAP
            unfixable.append(UnfixableSpot(u.position, u.net, reason))
            continue
        pos, _ = spot
        units.append(PlannedUnit(pos, ground, size, forced=True))
        board.add_via(pos, ground, size)
        returns[ground].append(pos)

    return Plan(units=tuple(units), unfixable=tuple(unfixable))
