"""Everything a new via or stub must respect, shared by the via planners.

Built once from a snapshot and grown with every unit a planner places, so
padvia and returnvia judge spots against the same rules. The rule values
live in `breakneck.plan`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from shapely.geometry import LineString, Point
from shapely.geometry.base import BaseGeometry
from shapely.prepared import PreparedGeometry, prep
from shapely.strtree import STRtree

from breakneck.plan import (
    EDGE_GAP,
    FORBIDDEN_RADIUS,
    HOLE_TO_HOLE_GAP,
    OTHER_NET_CLEARANCE,
    SAME_NET_PAD_GAP,
    PlannedUnit,
    pad_clearance,
)
from breakneck.snapshot import MIN_STITCH_LAYERS, PadKind, Point2, Snapshot, ViaSize


@dataclass(frozen=True)
class _Copper:
    """Copper on one layer, or on every layer when `layer` is None.

    `pad_id` is set for pad copper, which same-net vias must also keep off.
    """

    net: str
    layer: str | None
    geometry: BaseGeometry
    clearance: float
    pad_id: str | None = None


@dataclass(frozen=True)
class _PlannedVia:
    """A via planned this run, measured as an exact circle by via checks."""

    centre: Point2
    radius: float
    net: str

    def gap(self, position: Point2, radius: float) -> float:
        return math.dist(position, self.centre) - self.radius - radius


@dataclass(frozen=True)
class _UnplatedHole:
    """An NPTH whose keep-off binds copper of every net."""

    centre: Point
    radius: float
    copper_gap: float


class Obstacles:
    """Copper, holes and areas a new via or stub must keep clear of.

    Units planned during a run are added with `add_unit`, so later
    candidates keep clear of them.

    Via checks measure copper gaps in one of two forms: from the via's
    centre point less its radius (padvia) or from its 64-segment polygon
    disk (returnvia). The forms differ by under a micron; each planner keeps
    its own so its planned output does not shift. Via checks also measure
    planned vias as exact circles, as KiCad's DRC does; the polygon disk lies
    inside the circle, so the exact check is the stricter one.
    """

    def __init__(self, snapshot: Snapshot, forbidden: Sequence[Point2]) -> None:
        self.snapshot = snapshot
        self.forbidden = list(forbidden)
        self.copper: list[_Copper] = []
        self.unplated: list[_UnplatedHole] = []
        drills: list[tuple[float, float, float]] = []
        for pad in snapshot.pads:
            clearance = pad_clearance(pad.clearance_override)
            for layer, geom in pad.shapes.items():
                self.copper.append(_Copper(pad.net, layer, geom, clearance, pad.id))
            if pad.drill > 0:
                drills.append((*pad.position, pad.drill / 2))
                if pad.kind is PadKind.NPTH:
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
        # Planned stubs and via disks, in placement order.
        self.planned: list[_Copper] = []
        self.planned_vias: list[_PlannedVia] = []
        # Columns: x, y, radius.
        self.drills = np.array(drills, dtype=float).reshape(-1, 3)
        self.max_clearance = max(
            [OTHER_NET_CLEARANCE, *(c.clearance for c in self.copper)]
        )
        self._fills: dict[tuple[str, str], PreparedGeometry] = {}

    @cached_property
    def _placeable(self) -> PreparedGeometry:
        return prep(self.snapshot.board_outline)

    @cached_property
    def _edge(self) -> BaseGeometry:
        return self.snapshot.board_outline.boundary

    def add_unit(self, unit: PlannedUnit) -> None:
        """Add a planned via, its drill and its stub as obstacles."""
        radius = unit.size.diameter / 2
        disk = Point(unit.position).buffer(radius)
        self.planned.append(_Copper(unit.net, None, disk, OTHER_NET_CLEARANCE))
        self.planned_vias.append(_PlannedVia(unit.position, radius, unit.net))
        hole = [*unit.position, unit.size.drill / 2]
        self.drills = np.vstack([self.drills, hole])
        if unit.stub is not None:
            s = unit.stub
            line = LineString([s.start, s.end]).buffer(s.width / 2)
            self.planned.append(_Copper(unit.net, s.layer, line, OTHER_NET_CLEARANCE))

    def _nearby(self, geom: BaseGeometry, reach: float) -> Iterator[_Copper]:
        for i in self.tree.query(geom, predicate="dwithin", distance=reach):
            yield self.copper[int(i)]
        yield from self.planned

    def nearby_copper(
        self, geom: BaseGeometry, reach: float, net: str, layer: str | None
    ) -> list[BaseGeometry]:
        """`net`'s copper on exactly `layer`, planned units included.

        `layer` None selects through copper (vias). Items farther than
        `reach` from `geom` may be left out but are not guaranteed to be.
        """
        return [
            c.geometry
            for c in self._nearby(geom, reach)
            if c.net == net and c.layer == layer
        ]

    def in_bounds(self, position: Point2, size: ViaSize) -> bool:
        """Board edge, via keepout and forbidden-position rules."""
        pt = Point(position)
        r = size.diameter / 2
        if self.snapshot.board_outline.is_empty:
            return False
        if not self._placeable.contains(pt) or pt.distance(self._edge) < EDGE_GAP + r:
            return False
        if self.snapshot.in_via_keepout(position, r):
            return False
        return all(math.dist(position, f) >= FORBIDDEN_RADIUS for f in self.forbidden)

    def clashes_with_other_nets(
        self, geom: BaseGeometry, half_width: float, net: str, layer: str | None
    ) -> bool:
        """Other-net copper within clearance of `geom` buffered by `half_width`.

        `layer` None means the copper spans every layer, like a via.
        """
        reach = half_width + self.max_clearance
        for c in self._nearby(geom, reach):
            if c.net == net:
                continue
            if layer is not None and c.layer is not None and c.layer != layer:
                continue
            if geom.distance(c.geometry) < half_width + c.clearance:
                return True
        return False

    def _clashes_with_planned_vias(
        self, position: Point2, radius: float, net: str
    ) -> bool:
        return any(
            v.net != net and v.gap(position, radius) < OTHER_NET_CLEARANCE
            for v in self.planned_vias
        )

    def via_is_clear(
        self,
        position: Point2,
        net: str,
        size: ViaSize,
        *,
        polygon_disk: bool,
        exempt_pad: str | None = None,
    ) -> bool:
        """Every clearance rule a full-clearance via must pass.

        NPTH keep-off, other-net copper on every layer, the same-net pad gap
        (sparing `exempt_pad`) and hole-to-hole. `polygon_disk` measures
        copper gaps from the via's polygon disk, as returnvia does, instead
        of from its centre point less its radius, as padvia does; see the
        class docstring.
        """
        centre = Point(position)
        radius = size.diameter / 2
        geom, half_width = (
            (centre.buffer(radius), 0.0) if polygon_disk else (centre, radius)
        )
        return not (
            self.clashes_with_unplated(centre, radius)
            or self._clashes_with_planned_vias(position, radius, net)
            or self.clashes_with_other_nets(geom, half_width, net, None)
            or self.clashes_with_same_net_pads(geom, half_width, net, exempt_pad)
            or self.clashes_with_holes(position, size.drill / 2)
        )

    def clashes_with_unplated(self, geom: BaseGeometry, half_width: float) -> bool:
        """Inside an NPTH keep-off; these bind copper of every net and layer."""
        return any(
            geom.distance(h.centre) < h.radius + half_width + h.copper_gap
            for h in self.unplated
        )

    def clashes_with_holes(self, position: Point2, drill_radius: float) -> bool:
        """A drill at `position` too close to any existing or planned hole."""
        gaps = np.hypot(
            self.drills[:, 0] - position[0], self.drills[:, 1] - position[1]
        )
        gaps -= self.drills[:, 2] + drill_radius
        return bool((gaps < HOLE_TO_HOLE_GAP).any())

    def clashes_with_same_net_pads(
        self,
        geom: BaseGeometry,
        half_width: float,
        net: str,
        exempt_pad: str | None = None,
    ) -> bool:
        """Same-net pad copper within the via-in-pad gap.

        `exempt_pad` names a pad the via may approach, such as the one its
        stub connects to.
        """
        reach = half_width + SAME_NET_PAD_GAP
        return any(
            c.pad_id not in (None, exempt_pad)
            and c.net == net
            and geom.distance(c.geometry) < reach
            for c in self._nearby(geom, reach)
        )

    def reaches_zones(
        self,
        position: Point2,
        net: str,
        size: ViaSize,
        *,
        filled: bool,
        exempt_layer: str | None = None,
    ) -> bool:
        """The zone-layer rule: inside `net`'s zones on at least two layers.

        With `filled`, the via must also sit inside the fill on each of those
        layers; the forced pass of returnvia only asks for the zone outlines.
        `exempt_layer` is left out of the fill check: padvia exempts the
        pad's own layer, where the stub connects the via and the fill around
        the pad is cut by its thermal gap.
        """
        layers = self.snapshot.required_layers(net, position)
        if len(layers) < MIN_STITCH_LAYERS:
            return False
        if not filled:
            return True
        disk = Point(position).buffer(size.diameter / 2)
        return all(
            self._fill(net, layer).contains(disk)
            for layer in layers
            if layer != exempt_layer
        )

    def _fill(self, net: str, layer: str) -> PreparedGeometry:
        key = (net, layer)
        if key not in self._fills:
            self._fills[key] = prep(self.snapshot.fill(net, layer))
        return self._fills[key]

    def keeps_unrelaxed_rules(self, position: Point2, net: str, size: ViaSize) -> bool:
        """Rules a forced via keeps: hole-to-hole and the same-net pad gap.

        Only other-net copper clearance is relaxed for a forced via.
        """
        if self.clashes_with_holes(position, size.drill / 2):
            return False
        disk = Point(position).buffer(size.diameter / 2)
        return not self.clashes_with_same_net_pads(disk, 0.0, net)

    def clearance_to_other_nets(
        self, position: Point2, net: str, size: ViaSize
    ) -> float:
        """Smallest gap from a via to other-net copper, capped at OTHER_NET_CLEARANCE.

        An NPTH keep-off counts as copper; zero or less overlaps.
        """
        radius = size.diameter / 2
        disk = Point(position).buffer(radius)
        centre = Point(position)
        gaps = [
            disk.distance(c.geometry)
            for c in self._nearby(disk, OTHER_NET_CLEARANCE)
            if c.net != net
        ]
        gaps += [v.gap(position, radius) for v in self.planned_vias if v.net != net]
        gaps += [
            centre.distance(h.centre) - h.radius - radius - h.copper_gap
            for h in self.unplated
        ]
        return min([OTHER_NET_CLEARANCE, *gaps])
