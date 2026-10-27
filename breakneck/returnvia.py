"""Return vias: a ground via near every via on a non-ground net.

The planner places the missing vias by straight-line distance. Each via's
reference planes are resolved here; the `returnvia` command measures the
distance through them (see `returnpath`), which needs the `place` extra.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from loguru import logger
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from breakneck.obstacles import Obstacles
from breakneck.plan import OTHER_NET_CLEARANCE, Plan, PlannedUnit, UnfixableSpot
from breakneck.snapshot import PadKind, Point2, Snapshot, Via, ViaSize

DEFAULT_DISTANCE = 1.5
FALLBACK_GROUND = "GND"
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


class NoGroundNetsError(ValueError):
    pass


def resolve_ground_nets(
    snapshot: Snapshot, ground_nets: Iterable[str] | None
) -> list[str]:
    """The given ground nets, else the detected ones, else the net named GND.

    Detection needs zones on two inner layers, which two-layer boards and
    single-plane stackups never have; their ground is the net named GND.
    """
    if ground_nets:
        return sorted(ground_nets)
    grounds = snapshot.default_ground_nets()
    if grounds:
        return grounds
    if _has_copper(snapshot, FALLBACK_GROUND):
        logger.info(
            f"No net has zones on two inner layers; using {FALLBACK_GROUND} as ground"
        )
        return [FALLBACK_GROUND]
    raise NoGroundNetsError(
        f"No ground net found: no net has zones on two inner layers and there is "
        f"no {FALLBACK_GROUND} net; name the ground nets with --ground-nets"
    )


def _has_copper(snapshot: Snapshot, net: str) -> bool:
    return any(
        item.net == net
        for item in (*snapshot.copper_zones, *snapshot.vias, *snapshot.pads)
    )


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
        out.append(Unserved(v.position, v.net, v.diameter, ground))
    return out


def _spot_ok(
    obstacles: Obstacles, position: Point2, ground: str, size: ViaSize
) -> bool:
    """A full-clearance return via spot."""
    return (
        obstacles.in_bounds(position, size)
        and obstacles.reaches_zones(position, ground, size, filled=True)
        and obstacles.via_is_clear(position, ground, size, polygon_disk=True)
    )


# ---- reference planes --------------------------------------------------------

MIN_CONNECTED_LAYERS = 2


@dataclass(frozen=True)
class ReferencePlane:
    layer: str
    ground: str


@dataclass(frozen=True)
class LayerReference:
    """A layer a via connects on; `plane` None means outside every ground zone."""

    layer: str
    plane: ReferencePlane | None


@dataclass(frozen=True)
class ViaReferences:
    via: Via
    # In stack order.
    layers: tuple[LayerReference, ...]

    @property
    def exempt(self) -> bool:
        """A via connecting on fewer than two layers carries no layer change."""
        return len(self.layers) < MIN_CONNECTED_LAYERS

    @property
    def planes(self) -> tuple[ReferencePlane, ...]:
        """Distinct reference planes: each is one return measurement."""
        return tuple(dict.fromkeys(c.plane for c in self.layers if c.plane))


def via_references(
    snapshot: Snapshot, ground_nets: Iterable[str] | None
) -> list[ViaReferences]:
    """Connected layers and their reference planes for every non-ground via."""
    grounds = resolve_ground_nets(snapshot, ground_nets)
    return [
        ViaReferences(
            v,
            tuple(
                LayerReference(L, _reference_plane(snapshot, v.position, L, grounds))
                for L in _connected_layers(snapshot, v)
            ),
        )
        for v in sorted(snapshot.vias, key=lambda v: v.position)
        if v.net and v.net not in grounds
    ]


def reference_outlines(
    snapshot: Snapshot, layer: str, grounds: Sequence[str]
) -> Iterator[tuple[ReferencePlane, BaseGeometry]]:
    """Ground zone outlines that can be `layer`'s reference, in precedence order.

    The nearest other layer in stack order comes first, the upper one of two
    equidistant layers, then on a layer the highest zone priority.
    """
    stack = snapshot.copper_layers
    i = stack.index(layer)
    for step in range(1, len(stack)):
        for j in (i - step, i + step):
            if not 0 <= j < len(stack):
                continue
            zones = [
                z
                for z in snapshot.copper_zones
                if z.net in grounds and stack[j] in z.layers
            ]
            for z in sorted(zones, key=lambda z: -z.priority):
                assert z.net is not None
                yield ReferencePlane(stack[j], z.net), z.outline


def _reference_plane(
    snapshot: Snapshot, position: Point2, layer: str, grounds: Sequence[str]
) -> ReferencePlane | None:
    """The first reference outline containing `position`, as `ground_net_at`
    resolves overlapping outlines by zone priority."""
    pt = Point(position)
    return next(
        (
            plane
            for plane, outline in reference_outlines(snapshot, layer, grounds)
            if outline.contains(pt)
        ),
        None,
    )


def _touches(v: Via, geometry: BaseGeometry, reach: float = 0.0) -> bool:
    return (
        not geometry.is_empty
        and geometry.distance(Point(v.position))
        <= v.diameter / 2 + reach + DISTANCE_EPSILON
    )


def tied_layers(snapshot: Snapshot, v: Via) -> set[str]:
    """Layers where a same-net track or pad touches the via."""
    layers = {
        t.layer
        for t in snapshot.tracks
        if t.net == v.net and _touches(v, t.centreline, t.width / 2)
    }
    return layers | {
        L
        for p in snapshot.pads
        if p.net == v.net
        for L, shape in p.shapes.items()
        if _touches(v, shape)
    }


def filled_layers(snapshot: Snapshot, v: Via) -> set[str]:
    """Layers where a fill of the via's own net touches it."""
    return {L for L in snapshot.copper_layers if _touches(v, snapshot.fill(v.net, L))}


def _connected_layers(snapshot: Snapshot, v: Via) -> list[str]:
    """Layers where same-net copper (track, pad or fill) touches the via."""
    layers = tied_layers(snapshot, v) | filled_layers(snapshot, v)
    return [L for L in snapshot.copper_layers if L in layers]


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
    obstacles: Obstacles, u: Unserved, ground: str, size: ViaSize, distance: float
) -> Point2 | None:
    """Nearest spot on rings around `u` with full clearance."""
    start = u.diameter / 2 + size.diameter / 2 + OTHER_NET_CLEARANCE
    for r in _radii(start, distance - ROUNDING_MARGIN):
        for a in _angles(RING_ANGLES):
            pos = _ring_point(u.position, r, a)
            if _spot_ok(obstacles, pos, ground, size):
                return pos
    return None


def _force_spot(
    obstacles: Obstacles, u: Unserved, ground: str, size: ViaSize, distance: float
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
            if not (
                obstacles.in_bounds(pos, size)
                and obstacles.reaches_zones(pos, ground, size, filled=False)
                and obstacles.keeps_unrelaxed_rules(pos, ground, size)
            ):
                continue
            gap = obstacles.clearance_to_other_nets(pos, ground, size)
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
    obstacles = Obstacles(snapshot, forbidden)
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
        pos = _find_spot(obstacles, u, u.ground, size, options.distance)
        if pos is None:
            pending.append((u, u.ground, size))
            continue
        unit = PlannedUnit(pos, u.ground, size)
        units.append(unit)
        obstacles.add_unit(unit)
        returns[u.ground].append(pos)

    for u, ground, size in pending:
        if _served(u.position, ground, returns, options.distance):
            continue
        if not options.force:
            unfixable.append(UnfixableSpot(u.position, u.net, NO_ROOM))
            continue
        spot = _force_spot(obstacles, u, ground, size, options.distance)
        if spot is None or spot[1] <= 0:
            reason = NO_ROOM if spot is None else WOULD_OVERLAP
            unfixable.append(UnfixableSpot(u.position, u.net, reason))
            continue
        pos, _ = spot
        unit = PlannedUnit(pos, ground, size, forced=True)
        units.append(unit)
        obstacles.add_unit(unit)
        returns[ground].append(pos)

    return Plan(units=tuple(units), unfixable=tuple(unfixable))
