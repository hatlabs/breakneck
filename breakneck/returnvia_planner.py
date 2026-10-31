"""The `returnvia` planner: return vias placed by the through-plane distance.

Every candidate spot is measured with the check's own code
(`ReturnFields.distance_from`), so a planned run passes its own check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from breakneck.obstacles import Obstacles
from breakneck.plan import OTHER_NET_CLEARANCE, Plan, PlannedUnit, UnfixableSpot
from breakneck.returnpath import Outcome, ReturnFields, served_within
from breakneck.returnvia import (
    DISTANCE_EPSILON,
    ReferencePlane,
    ReturnViaOptions,
    resolve_ground_nets,
    via_references,
)
from breakneck.snapshot import Point2, Snapshot, Via, ViaSize

# Candidate rings around each unserved via.
RING_STEP = 0.05
RING_ANGLES = 36
FORCE_RING_ANGLES = 72
POSITION_DECIMALS = 3
# Rounding a ring point moves it outward by less than this; rings stop short by it.
ROUNDING_MARGIN = 10**-POSITION_DECIMALS

NO_ROOM = "no room for a return via"
NO_REACH = "no clear spot reaches the plane within {distance:g} mm"
RELIES_ON = "relies on forced via at ({x:.3f}, {y:.3f})"
WOULD_OVERLAP = "every spot overlaps other-net copper"


@dataclass(frozen=True)
class Target:
    """A signal via and the planes of one ground net where it is unserved."""

    via: Via
    ground: str
    planes: tuple[ReferencePlane, ...]


def _spot_ok(
    obstacles: Obstacles, position: Point2, ground: str, size: ViaSize
) -> bool:
    """A full-clearance return via spot."""
    return (
        obstacles.in_bounds(position, size)
        and obstacles.reaches_zones(position, ground, size, filled=True)
        and obstacles.via_is_clear(position, ground, size, polygon_disk=True)
    )


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


def _force_spot(
    obstacles: Obstacles, v: Via, ground: str, size: ViaSize, distance: float
) -> tuple[Point2, float] | None:
    """Spot with the most clearance to other-net copper, nearest first on ties.

    Clearance is capped at the normal rule, so a closer spot wins once the
    rule is met.
    """
    best: tuple[tuple[float, float], Point2, float] | None = None
    start = v.diameter / 2 + size.diameter / 2
    for a in _angles(FORCE_RING_ANGLES):
        for r in _radii(start, distance - ROUNDING_MARGIN):
            pos = _ring_point(v.position, r, a)
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


@dataclass(frozen=True)
class _Context:
    """What the planner reads from one snapshot; never changed by a round."""

    snapshot: Snapshot
    options: ReturnViaOptions
    grounds: tuple[str, ...]
    fields: ReturnFields
    targets: tuple[Target, ...]
    unfixable: tuple[UnfixableSpot, ...]

    @classmethod
    def measure(cls, snapshot: Snapshot, options: ReturnViaOptions) -> _Context:
        grounds = resolve_ground_nets(snapshot, options.ground_nets)
        fields = ReturnFields(snapshot)
        targets: list[Target] = []
        unfixable: list[UnfixableSpot] = []
        for refs in via_references(snapshot, grounds):
            failing = [
                r for r in fields.measure([refs], options.distance) if not r.served
            ]
            by_ground: dict[str, list[ReferencePlane]] = {}
            for r in failing:
                if r.outcome is Outcome.MEASURED:
                    assert r.plane is not None
                    by_ground.setdefault(r.plane.ground, []).append(r.plane)
                else:
                    assert r.reason is not None
                    unfixable.append(UnfixableSpot(r.position, r.net, r.reason))
            targets += [Target(refs.via, g, tuple(p)) for g, p in by_ground.items()]
        return cls(
            snapshot,
            options,
            tuple(grounds),
            fields,
            tuple(targets),
            tuple(unfixable),
        )


class _Round:
    """One planning pass over the targets, with the round's forbidden spots."""

    def __init__(self, ctx: _Context, forbidden: Sequence[Point2]) -> None:
        self.ctx = ctx
        self.distance = ctx.options.distance
        self.obstacles = Obstacles(ctx.snapshot, forbidden)
        self.units: list[PlannedUnit] = []

    def plan(self) -> Plan:
        options = self.ctx.options
        unfixable = list(self.ctx.unfixable)
        pending: list[tuple[Target, ViaSize, str]] = []
        for t in self.ctx.targets:
            planes = self.unserved(t)
            if not planes:
                continue
            size = options.via or self.ctx.snapshot.via_size(t.ground)
            pos, clear = self.find_spot(t.via, t.ground, planes, size)
            if pos is not None:
                self.add(PlannedUnit(pos, t.ground, size))
            else:
                reason = NO_REACH.format(distance=self.distance) if clear else NO_ROOM
                pending.append((t, size, reason))

        for t, size, reason in pending:
            planes = self.unserved(t)
            if not planes:
                continue
            stand_in = self.stand_in(t)
            if stand_in is not None:
                x, y = stand_in.position
                reason = RELIES_ON.format(x=x, y=y)
            elif options.force:
                spot = _force_spot(self.obstacles, t.via, t.ground, size, self.distance)
                if spot is not None and spot[1] > 0:
                    self.add(PlannedUnit(spot[0], t.ground, size, forced=True))
                    continue
                reason = NO_ROOM if spot is None else WOULD_OVERLAP
            where = ", ".join(p.layer for p in planes)
            unfixable.append(UnfixableSpot(t.via.position, t.via.net, reason, where))

        return Plan(units=tuple(self.units), unfixable=tuple(unfixable))

    def add(self, unit: PlannedUnit) -> None:
        self.units.append(unit)
        self.obstacles.add_unit(unit)

    def serves(
        self, position: Point2, size: ViaSize, v: Via, plane: ReferencePlane
    ) -> bool:
        found = self.ctx.fields.distance_from(
            position, size.diameter, v.position, plane, self.distance
        )
        return found <= served_within(self.distance)

    def unserved(self, t: Target) -> tuple[ReferencePlane, ...]:
        """`t`'s planes that no via planned earlier in this round serves.

        A seed sits within a pixel of its via's centre and no path is shorter
        than the straight line, so farther vias can't serve.
        """
        reach = served_within(self.distance) + self.ctx.fields.grid.pixel
        near = [
            u
            for u in self.units
            if u.net == t.ground
            and not u.forced
            and math.dist(u.position, t.via.position) <= reach
        ]
        return tuple(
            p
            for p in t.planes
            if not any(self.serves(u.position, u.size, t.via, p) for u in near)
        )

    def stand_in(self, t: Target) -> PlannedUnit | None:
        """A forced via of `t`'s ground within a straight `distance`: a via
        placed by hand in its stead may serve `t` too, so `t` gets no forced
        via of its own."""
        reach = self.distance + DISTANCE_EPSILON
        return next(
            (
                u
                for u in self.units
                if u.forced
                and u.net == t.ground
                and math.dist(u.position, t.via.position) <= reach
            ),
            None,
        )

    def find_spot(
        self,
        v: Via,
        ground: str,
        planes: Sequence[ReferencePlane],
        size: ViaSize,
    ) -> tuple[Point2 | None, bool]:
        """Nearest spot on rings around `v` with full clearance that serves
        every one of `planes`, and whether any spot had full clearance."""
        start = v.diameter / 2 + size.diameter / 2 + OTHER_NET_CLEARANCE
        stop = served_within(self.distance) - ROUNDING_MARGIN
        clear = False
        for r in _radii(start, stop):
            for a in _angles(RING_ANGLES):
                pos = _ring_point(v.position, r, a)
                if not _spot_ok(self.obstacles, pos, ground, size):
                    continue
                clear = True
                if all(self.serves(pos, size, v, p) for p in planes):
                    return pos, True
        return None, clear


class ReturnViaPlanner:
    """Planner for `run_placement`; measures each snapshot once across rounds.

    A first pass places full-clearance vias. With `force`, a second pass
    gives each via still unserved the spot with the most clearance, unless
    that spot would overlap other-net copper.
    """

    def __init__(self) -> None:
        self._ctx: _Context | None = None

    def __call__(
        self,
        snapshot: Snapshot,
        options: ReturnViaOptions,
        forbidden: Sequence[Point2],
    ) -> Plan:
        ctx = self._cached(snapshot, options)
        if ctx is None:
            ctx = self._ctx = _Context.measure(snapshot, options)
        return _Round(ctx, forbidden).plan()

    def grounds(self, snapshot: Snapshot, options: ReturnViaOptions) -> list[str]:
        """The ground nets, resolved again only if this snapshot was never planned."""
        ctx = self._cached(snapshot, options)
        if ctx is None:
            return resolve_ground_nets(snapshot, options.ground_nets)
        return list(ctx.grounds)

    def _cached(self, snapshot: Snapshot, options: ReturnViaOptions) -> _Context | None:
        ctx = self._ctx
        if ctx is None or ctx.snapshot is not snapshot or ctx.options != options:
            return None
        return ctx
