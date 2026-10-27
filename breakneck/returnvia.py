"""Return vias: a ground via near every via on a non-ground net.

Each via's reference planes are resolved here. Measuring the distance
through them (`returnpath`) and planning by it (`returnvia_planner`) need
the `place` extra.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from loguru import logger
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from breakneck.snapshot import Point2, Snapshot, Via, ViaSize

DEFAULT_DISTANCE = 1.5
FALLBACK_GROUND = "GND"
DISTANCE_EPSILON = 1e-9

OUTSIDE_GROUND = "outside every ground zone"


@dataclass(frozen=True)
class ReturnViaOptions:
    distance: float = DEFAULT_DISTANCE
    ground_nets: tuple[str, ...] | None = None
    via: ViaSize | None = None
    force: bool = False


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
    """The first reference outline containing `position`; overlapping
    outlines on a layer resolve by zone priority."""
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
