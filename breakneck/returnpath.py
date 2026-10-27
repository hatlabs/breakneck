"""Return distance through the copper of each signal via's reference planes.

On a plane, the distance from a signal via to a return is the shortest path
from the via's centre to the return's centre through the plane's ground fill,
plus each return's passable disc and a straight hop across the via's own
void. This module needs the `place` extra.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import ndimage
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from breakneck.plan import QUANTISATION, THERMAL_REACH
from breakneck.raster import BoolMap, FloatMap, Grid, Pixel, geodesic, touch
from breakneck.returnvia import (
    OUTSIDE_GROUND,
    ReferencePlane,
    ViaReferences,
    via_references,
)
from breakneck.snapshot import PadKind, Point2, Snapshot

# The void floods through edge-sharing pixels only, so it never leaks through
# two fill pixels that meet at a corner.
VOID_CONNECTIVITY = ndimage.generate_binary_structure(2, 1)
ADJACENT = np.ones((3, 3), bool)


class NoBoardOutlineError(ValueError):
    pass


class Outcome(Enum):
    MEASURED = "measured"
    NO_GROUND_FILL = "no ground fill"
    OUTSIDE_GROUND = "outside ground"


@dataclass(frozen=True)
class PlaneReturn:
    """Return distance of one signal via on one of its reference planes.

    `layers` are the connected layers that reference the plane. `distance`
    is inf when no copper path reaches a return, and for the outcomes other
    than MEASURED. `plane` is None outside every ground zone.
    """

    position: Point2
    net: str
    layers: tuple[str, ...]
    plane: ReferencePlane | None
    distance: float
    served: bool
    outcome: Outcome

    @property
    def reason(self) -> str | None:
        """Why the via can't be served on this plane, if no return via can fix it."""
        match self.outcome:
            case Outcome.NO_GROUND_FILL:
                assert self.plane is not None
                return f"no {self.plane.ground} fill at the via on {self.plane.layer}"
            case Outcome.OUTSIDE_GROUND:
                return f"{OUTSIDE_GROUND} for {self.layers[0]}"
        return None


class ReturnFields:
    """Per (plane, ground net): the fill and the distance field to its returns."""

    def __init__(self, snapshot: Snapshot) -> None:
        if snapshot.board_outline.is_empty:
            raise NoBoardOutlineError("Board has no closed Edge.Cuts outline")
        self.snapshot = snapshot
        self.grid = Grid.covering(snapshot.board_outline.bounds)
        self._fills: dict[ReferencePlane, BoolMap] = {}
        self._fields: dict[ReferencePlane, FloatMap] = {}

    def fill(self, plane: ReferencePlane) -> BoolMap:
        if plane not in self._fills:
            geom = self.snapshot.fill(plane.ground, plane.layer)
            self._fills[plane] = self.grid.mask(geom)
        return self._fills[plane]

    def field(self, plane: ReferencePlane) -> FloatMap:
        """Distance (mm) through the fill to the nearest return's centre."""
        if plane not in self._fields:
            mask = self.fill(plane).copy()
            seeds: list[Pixel] = []
            for centre, disc in self._returns(plane):
                win = self.grid.bounds_window(disc.bounds)
                inside = self.grid.mask(disc, win)
                mask[win] |= inside
                seeds += self._seeds(centre, win, inside)
            self._fields[plane] = geodesic(self.grid, mask, seeds)
        return self._fields[plane]

    def _seeds(
        self, centre: Point2, win: tuple[slice, slice], inside: BoolMap
    ) -> list[Pixel]:
        """The return's centre pixel, or the in-grid pixels of its disc when
        the centre is off the grid, as on or past the board outline."""
        px = self.grid.index(centre)
        if self.grid.contains(px):
            return [px]
        rr, cc = np.nonzero(inside)
        return list(zip((rr + win[0].start).tolist(), (cc + win[1].start).tolist()))

    def _returns(self, plane: ReferencePlane) -> list[tuple[Point2, BaseGeometry]]:
        """Centres and passable discs of the ground's vias and PTH pads.

        As for pourstitch stitches, one counts as a return only where fill
        lies within its thermal reach.
        """
        snap = self.snapshot
        candidates = [
            (v.position, Point(v.position).buffer(v.diameter / 2))
            for v in snap.vias
            if v.net == plane.ground
        ] + [
            (p.position, p.shapes[plane.layer])
            for p in snap.pads
            if p.net == plane.ground
            and p.kind is PadKind.PTH
            and p.drill
            and plane.layer in p.shapes
        ]
        fill = self.fill(plane)
        reached = [(c, shape.buffer(THERMAL_REACH)) for c, shape in candidates]
        return [(c, disc) for c, disc in reached if touch(self.grid, fill, disc)]

    def distance_at(
        self, position: Point2, plane: ReferencePlane, distance: float
    ) -> float | None:
        """Return distance of a via at `position`; None with no fill near it.

        The via's void is the non-fill region around its centre, within
        `distance`; every fill pixel next to it is a way in, reached by a
        straight hop from the centre.
        """
        g = self.grid
        centre = g.index(position)
        if not g.contains(centre):
            return None
        win, near = g.disc(centre, distance)
        fill = self.fill(plane)[win]
        if not (fill & near).any():
            return None
        local = (centre[0] - win[0].start, centre[1] - win[1].start)
        edge = _void_edge(fill, near, local)
        rr, cc = np.nonzero(edge)
        if not len(rr):
            return math.inf
        hops = np.hypot(g.xs[win[1]][cc] - position[0], g.ys[win[0]][rr] - position[1])
        return float(np.min(self.field(plane)[win][rr, cc] + hops))

    def void(
        self, position: Point2, plane: ReferencePlane, distance: float
    ) -> tuple[tuple[slice, slice], BoolMap] | None:
        """The window around `position` and the void in `plane`'s fill that
        contains it, within `distance`: the antipad `distance_at` hops
        across. None off the grid."""
        g = self.grid
        centre = g.index(position)
        if not g.contains(centre):
            return None
        win, near = g.disc(centre, distance)
        local = (centre[0] - win[0].start, centre[1] - win[1].start)
        return win, _void(self.fill(plane)[win], near, local)

    def measure(
        self, references: Iterable[ViaReferences], distance: float
    ) -> list[PlaneReturn]:
        """One result per (non-exempt via, plane) and per layer with no plane."""
        out = []
        for refs in references:
            if refs.exempt:
                continue
            v = refs.via
            for plane in refs.planes:
                layers = tuple(c.layer for c in refs.layers if c.plane == plane)
                found = self.distance_at(v.position, plane, distance)
                outcome = (
                    Outcome.MEASURED if found is not None else Outcome.NO_GROUND_FILL
                )
                d = math.inf if found is None else found
                out.append(
                    PlaneReturn(
                        v.position,
                        v.net,
                        layers,
                        plane,
                        d,
                        d <= distance + QUANTISATION,
                        outcome,
                    )
                )
            out += [
                PlaneReturn(
                    v.position,
                    v.net,
                    (c.layer,),
                    None,
                    math.inf,
                    False,
                    Outcome.OUTSIDE_GROUND,
                )
                for c in refs.layers
                if c.plane is None
            ]
        return out


def _void(fill: BoolMap, near: BoolMap, centre: Pixel) -> BoolMap:
    """The non-fill region containing `centre`, within `near`; empty when
    `centre` is on fill."""
    if fill[centre]:
        return np.zeros_like(fill)
    labels, _ = ndimage.label(~fill & near, VOID_CONNECTIVITY)
    return labels == labels[centre]


def _void_edge(fill: BoolMap, near: BoolMap, centre: Pixel) -> BoolMap:
    """Fill pixels next to the void containing `centre`, within `near`."""
    if fill[centre]:
        edge = np.zeros_like(fill)
        edge[centre] = True
        return edge
    void = _void(fill, near, centre)
    return ndimage.binary_dilation(void, ADJACENT) & fill & near


def measure_returns(
    snapshot: Snapshot, ground_nets: Sequence[str] | None, distance: float
) -> list[PlaneReturn]:
    """Return distance of every non-exempt signal via on each reference plane."""
    fields = ReturnFields(snapshot)
    return fields.measure(via_references(snapshot, ground_nets), distance)
