"""KiCad-independent copper geometry snapshot for the via planners.

Coordinates are KiCad-native millimetres with y pointing down, the same frame
as kicad-cli DRC reports and board files. This deliberately differs from the
y-flipped helpers in `breakneck.conversions`.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property, reduce

import kipy.board
import kipy.board_types
import shapely
from google.protobuf.message import Message
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.geometry import PolygonWithHoles, PolyLine, Vector2
from kipy.proto.board.board_types_pb2 import BoardLayer, PadType, ZoneType
from kipy.util.board_layer import canonical_name
from loguru import logger
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

Point2 = tuple[float, float]

NM_PER_MM = 1_000_000
FRONT_COPPER = "F.Cu"
BACK_COPPER = "B.Cu"
OUTER_COPPER = frozenset({FRONT_COPPER, BACK_COPPER})
EDGE_CUTS = "Edge.Cuts"

# Edge.Cuts arc endpoints don't meet exactly in floating point; snapping to
# 1 µm lets polygonize close the outline.
EDGE_DECIMALS = 3
ARC_SEGMENT_ANGLE = math.radians(5)
MIN_ARC_SEGMENTS = 4
COLLINEAR_EPSILON = 1e-12
MIN_STITCH_LAYERS = 2
MIN_GROUND_INNER_LAYERS = 2


class PadKind(enum.Enum):
    SMD = "smd"
    PTH = "pth"
    NPTH = "npth"
    OTHER = "other"


@dataclass(frozen=True)
class ViaSize:
    diameter: float
    drill: float


DEFAULT_VIA_SIZE = ViaSize(0.6, 0.3)


@dataclass(frozen=True, eq=False)
class Footprint:
    id: str
    ref: str
    position: Point2
    layer: str


@dataclass(frozen=True, eq=False)
class Pad:
    id: str
    footprint_id: str | None
    net: str
    kind: PadKind
    number: str
    position: Point2
    drill: float
    shapes: Mapping[str, BaseGeometry]
    clearance_override: float | None = None


@dataclass(frozen=True, eq=False)
class Track:
    id: str
    net: str
    layer: str
    width: float
    centreline: LineString

    @cached_property
    def geometry(self) -> BaseGeometry:
        return self.centreline.buffer(self.width / 2)


@dataclass(frozen=True, eq=False)
class Via:
    id: str
    net: str
    position: Point2
    diameter: float
    drill: float
    locked: bool = False


@dataclass(frozen=True, eq=False)
class Zone:
    net: str | None
    layers: tuple[str, ...]
    priority: int
    outline: BaseGeometry
    fills: Mapping[str, BaseGeometry] = field(default_factory=dict)
    is_rule_area: bool = False
    keepout_vias: bool = False


@dataclass(eq=False)
class Snapshot:
    """Board copper and outline geometry, in mm with y down."""

    copper_layers: tuple[str, ...]
    pads: Sequence[Pad] = ()
    tracks: Sequence[Track] = ()
    vias: Sequence[Via] = ()
    zones: Sequence[Zone] = ()
    footprints: Sequence[Footprint] = ()
    edge_paths: Sequence[Sequence[Point2]] = ()
    via_sizes: Mapping[str, ViaSize] = field(default_factory=dict)
    # Canonical copper layer name to the board's own name, which DRC uses.
    layer_names: Mapping[str, str] = field(default_factory=dict)

    @cached_property
    def inner_layers(self) -> tuple[str, ...]:
        return tuple(L for L in self.copper_layers if L not in OUTER_COPPER)

    @cached_property
    def board_outline(self) -> BaseGeometry:
        return outline_from_edge_paths(self.edge_paths)

    @cached_property
    def via_keepouts(self) -> BaseGeometry:
        """Area of the rule areas that keep out vias."""
        return unary_union(
            [z.outline for z in self.zones if z.is_rule_area and z.keepout_vias]
        )

    def in_via_keepout(self, position: Point2, radius: float) -> bool:
        """Whether a via of `radius` at `position` touches a via keepout."""
        keepouts = self.via_keepouts
        return not keepouts.is_empty and Point(position).distance(keepouts) <= radius

    @cached_property
    def copper_zones(self) -> tuple[Zone, ...]:
        return tuple(z for z in self.zones if not z.is_rule_area and z.net)

    def zone_layers(self, net: str) -> frozenset[str]:
        """Copper layers carrying a zone of `net` anywhere on the board."""
        return frozenset(L for z in self.copper_zones if z.net == net for L in z.layers)

    def is_stitchable(self, net: str) -> bool:
        return len(self.zone_layers(net)) >= MIN_STITCH_LAYERS

    def split_stitchable(self, nets: Iterable[str]) -> tuple[list[str], list[str]]:
        """Partition nets into stitchable and not; log the ones skipped."""
        ok, skipped = [], []
        for net in nets:
            (ok if self.is_stitchable(net) else skipped).append(net)
        for net in skipped:
            layers = sorted(self.zone_layers(net)) or ["none"]
            logger.warning(
                f"Net {net} has zones on fewer than {MIN_STITCH_LAYERS} layers "
                f"({', '.join(layers)}); skipping"
            )
        return ok, skipped

    def required_layers(self, net: str, position: Point2) -> frozenset[str]:
        """Layers where a via for `net` at `position` must sit inside the fill.

        These are the layers on which an outline of one of the net's zones
        contains the point. A via is only useful where this has at least two
        layers.
        """
        pt = Point(position)
        return frozenset(
            L
            for z in self.copper_zones
            if z.net == net and z.outline.contains(pt)
            for L in z.layers
        )

    def default_ground_nets(self) -> list[str]:
        """Nets with zones on at least two inner layers."""
        nets = {z.net for z in self.copper_zones if z.net is not None}
        return sorted(
            n
            for n in nets
            if len(self.zone_layers(n) & set(self.inner_layers))
            >= MIN_GROUND_INNER_LAYERS
        )

    def via_size(self, net: str) -> ViaSize:
        return self.via_sizes.get(net, DEFAULT_VIA_SIZE)

    def fill(self, net: str, layer: str) -> BaseGeometry:
        return self._fills.get((net, layer), Polygon())

    @cached_property
    def _fills(self) -> dict[tuple[str, str], BaseGeometry]:
        parts: dict[tuple[str, str], list[BaseGeometry]] = {}
        for z in self.copper_zones:
            assert z.net is not None
            for layer, geom in z.fills.items():
                parts.setdefault((z.net, layer), []).append(geom)
        return {k: unary_union(v) for k, v in parts.items()}

    @classmethod
    def from_board(cls, board: kipy.board.Board) -> Snapshot:
        """Read the board's committed copper state over the IPC API."""
        layer_ids = {
            canonical_name(L): L
            for L in board.get_enabled_layers()
            if canonical_name(L).endswith(".Cu")
        }
        copper_layers = tuple(
            L
            for L in _copper_stack_order(board.get_copper_layer_count())
            if L in layer_ids
        )
        kipy_pads = board.get_pads()
        nets = board.get_nets()
        snap = cls(
            copper_layers=copper_layers,
            pads=_read_pads(board, kipy_pads, copper_layers, layer_ids),
            tracks=[_read_track(t) for t in board.get_tracks()],
            vias=[_read_via(v) for v in board.get_vias()],
            zones=[
                _read_zone(z)
                for z in board.get_zones()
                if z.type in (ZoneType.ZT_COPPER, ZoneType.ZT_RULE_AREA)
            ],
            footprints=[_read_footprint(f) for f in board.get_footprints()],
            edge_paths=read_edge_paths(board.get_shapes()),
            via_sizes=_read_via_sizes(board, nets),
            layer_names={L: _layer_name(board, L, layer_ids[L]) for L in copper_layers},
        )
        unfilled = [z for z in snap.copper_zones if not z.fills]
        logger.info(
            f"Snapshot: {len(snap.pads)} pads, {len(snap.tracks)} tracks, "
            f"{len(snap.vias)} vias, {len(snap.copper_zones)} copper zones "
            f"({len(unfilled)} without fill)"
        )
        return snap


def _snap(p: Point2) -> Point2:
    return (round(p[0], EDGE_DECIMALS), round(p[1], EDGE_DECIMALS))


def arc_points(start: Point2, mid: Point2, end: Point2) -> list[Point2]:
    """Tessellate the circular arc through start, mid and end."""
    (x1, y1), (x2, y2), (x3, y3) = start, mid, end
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < COLLINEAR_EPSILON:
        return [start, end]
    s1, s2, s3 = x1**2 + y1**2, x2**2 + y2**2, x3**2 + y3**2
    ux = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    uy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    r = math.hypot(x1 - ux, y1 - uy)
    a1, a2, a3 = (math.atan2(p[1] - uy, p[0] - ux) for p in (start, mid, end))
    sweep = (a3 - a1) % (2 * math.pi)
    if (a2 - a1) % (2 * math.pi) > sweep:
        sweep -= 2 * math.pi
    n = max(MIN_ARC_SEGMENTS, math.ceil(abs(sweep) / ARC_SEGMENT_ANGLE))
    pts = [
        (ux + r * math.cos(a1 + sweep * i / n), uy + r * math.sin(a1 + sweep * i / n))
        for i in range(1, n)
    ]
    # Keep the exact endpoints so neighbouring segments still meet.
    return [start, *pts, end]


def circle_points(centre: Point2, radius: float) -> list[Point2]:
    n = max(MIN_ARC_SEGMENTS, math.ceil(2 * math.pi / ARC_SEGMENT_ANGLE))
    cx, cy = centre
    pts = [
        (
            cx + radius * math.cos(2 * math.pi * i / n),
            cy + radius * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]
    return [*pts, pts[0]]


def valid_polygon(
    shell: Sequence[Point2], holes: Sequence[Sequence[Point2]] = ()
) -> BaseGeometry:
    poly = Polygon(shell, list(holes))
    return poly if poly.is_valid else poly.buffer(0)


# ---- kipy conversion ---------------------------------------------------------


def _mm(v: Vector2) -> Point2:
    return (v.x / NM_PER_MM, v.y / NM_PER_MM)


def _copper_stack_order(count: int) -> list[str]:
    return [FRONT_COPPER, *(f"In{i}.Cu" for i in range(1, count - 1)), BACK_COPPER]


def _ring(polyline: PolyLine) -> list[Point2]:
    pts: list[Point2] = []
    for node in polyline.nodes:
        if node.has_point:
            pts.append(_mm(node.point))
        else:
            a = node.arc
            pts.extend(arc_points(_mm(a.start), _mm(a.mid), _mm(a.end)))
    return pts


def _polygon(pwh: PolygonWithHoles) -> BaseGeometry:
    return valid_polygon(_ring(pwh.outline), [_ring(h) for h in pwh.holes])


def _pad_kind(pad_type: PadType.ValueType) -> PadKind:
    return {
        PadType.PT_SMD: PadKind.SMD,
        PadType.PT_PTH: PadKind.PTH,
        PadType.PT_NPTH: PadKind.NPTH,
    }.get(pad_type, PadKind.OTHER)


def _pad_shapes(
    board: kipy.board.Board,
    pads: Sequence[kipy.board_types.Pad],
    copper_layers: Sequence[str],
    layer_ids: Mapping[str, BoardLayer.ValueType],
) -> dict[str, dict[str, BaseGeometry]]:
    """Pad copper per layer.

    `get_pad_shapes_as_polygons` silently drops pads with no shape on the
    layer, so the result can't be zipped with the request. Ask only for pads
    whose padstack lists the layer and fall back to per-pad queries if the
    lengths still differ.
    """
    shapes: dict[str, dict[str, BaseGeometry]] = {p.id.value: {} for p in pads}
    for layer in copper_layers:
        lid = layer_ids[layer]
        on = [p for p in pads if lid in set(p.padstack.layers)]
        if not on:
            continue
        result: list[PolygonWithHoles | None] = list(
            board.get_pad_shapes_as_polygons(on, lid)
        )
        if len(result) != len(on):
            logger.debug(
                f"Pad shape query on {layer} dropped pads; querying one by one"
            )
            result = [board.get_pad_shapes_as_polygons(p, lid) for p in on]
        for pad, pwh in zip(on, result):
            if pwh is not None:
                shapes[pad.id.value][layer] = _polygon(pwh)
    return shapes


def _clearance_override(proto_owner: Message, field_name: str) -> float | None:
    if not proto_owner.HasField(field_name):
        return None
    return getattr(proto_owner, field_name).value_nm / NM_PER_MM


def _read_pads(
    board: kipy.board.Board,
    pads: Sequence[kipy.board_types.Pad],
    copper_layers: Sequence[str],
    layer_ids: Mapping[str, BoardLayer.ValueType],
) -> list[Pad]:
    shapes = _pad_shapes(board, pads, copper_layers, layer_ids)
    fp_overrides = {
        f.id.value: _clearance_override(f.proto.overrides, "copper_clearance")
        for f in board.get_footprints()
    }
    out = []
    for p in pads:
        drill = p.padstack.drill.diameter
        parent = p.parent.value if p.parent is not None else None
        override = _clearance_override(p.proto, "copper_clearance_override")
        if override is None and parent is not None:
            override = fp_overrides.get(parent)
        out.append(
            Pad(
                id=p.id.value,
                footprint_id=parent,
                net=p.net.name,
                kind=_pad_kind(p.pad_type),
                number=p.number,
                position=_mm(p.position),
                # Oval drills: the larger axis is the conservative clearance.
                drill=max(drill.x, drill.y) / NM_PER_MM,
                shapes=shapes[p.id.value],
                clearance_override=override,
            )
        )
    return out


def _read_track(t: kipy.board_types.Track | kipy.board_types.ArcTrack) -> Track:
    if isinstance(t, kipy.board_types.ArcTrack):
        line = LineString(arc_points(_mm(t.start), _mm(t.mid), _mm(t.end)))
    else:
        line = LineString([_mm(t.start), _mm(t.end)])
    return Track(
        id=t.id.value,
        net=t.net.name,
        layer=canonical_name(t.layer),
        width=t.width / NM_PER_MM,
        centreline=line,
    )


def _read_via(v: kipy.board_types.Via) -> Via:
    return Via(
        id=v.id.value,
        net=v.net.name,
        position=_mm(v.position),
        diameter=v.diameter / NM_PER_MM,
        drill=v.drill_diameter / NM_PER_MM,
        locked=v.locked,
    )


def _read_zone(z: kipy.board_types.Zone) -> Zone:
    rule = z.is_rule_area()
    fills = (
        {}
        if rule
        else {
            canonical_name(L): unary_union([_polygon(p) for p in polys])
            for L, polys in z.filled_polygons.items()
        }
    )
    net = z.net
    return Zone(
        net=net.name if net is not None and net.name else None,
        layers=tuple(canonical_name(L) for L in z.layers),
        priority=z.priority,
        outline=_polygon(z.outline),
        fills=fills,
        is_rule_area=rule,
        keepout_vias=rule and z.proto.rule_area_settings.keepout_vias,
    )


def _read_footprint(f: kipy.board_types.FootprintInstance) -> Footprint:
    return Footprint(
        id=f.id.value,
        ref=f.reference_field.text.value,
        position=_mm(f.position),
        layer=canonical_name(f.layer),
    )


def outline_from_edge_paths(paths: Iterable[Sequence[Point2]]) -> BaseGeometry:
    """Board area polygonised from Edge.Cuts paths.

    Closed Edge.Cuts loops nest even-odd: a loop inside the outline is a
    cutout, a loop inside a cutout is board again.
    """
    lines = [LineString([_snap(p) for p in path]) for path in paths if len(path) >= 2]
    if not lines:
        return Polygon()
    shells = [Polygon(f.exterior) for f in polygonize(unary_union(lines))]
    return reduce(shapely.symmetric_difference, shells, Polygon())


def read_edge_paths(
    shapes: Iterable[kipy.board_types.BoardShape],
) -> list[list[Point2]]:
    """The Edge.Cuts paths among `shapes`."""
    return [
        path
        for s in shapes
        if canonical_name(s.layer) == EDGE_CUTS
        for path in _edge_paths(s)
    ]


def _edge_paths(shape: kipy.board_types.BoardShape) -> list[list[Point2]]:
    match shape:
        case kipy.board_types.BoardSegment():
            return [[_mm(shape.start), _mm(shape.end)]]
        case kipy.board_types.BoardArc():
            return [arc_points(_mm(shape.start), _mm(shape.mid), _mm(shape.end))]
        case kipy.board_types.BoardCircle():
            return [circle_points(_mm(shape.center), shape.radius() / NM_PER_MM)]
        case kipy.board_types.BoardRectangle():
            (x0, y0), (x1, y1) = _mm(shape.top_left), _mm(shape.bottom_right)
            return [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]
        case kipy.board_types.BoardPolygon():
            return [
                [*ring, ring[0]]
                for pwh in shape.polygons
                for ring in (_ring(pwh.outline), *(_ring(h) for h in pwh.holes))
            ]
    logger.warning(f"Ignoring unsupported Edge.Cuts shape {type(shape).__name__}")
    return []


def _layer_name(
    board: kipy.board.Board, canonical: str, layer_id: BoardLayer.ValueType
) -> str:
    try:
        return board.get_layer_name(layer_id) or canonical
    except (ApiError, KiCadConnectionError) as e:
        logger.debug(f"KiCad did not report the name of {canonical}: {e}")
        return canonical


def _read_via_sizes(
    board: kipy.board.Board, nets: Sequence[kipy.board_types.Net]
) -> dict[str, ViaSize]:
    sizes = {}
    for name, netclass in board.get_netclass_for_nets(nets).items():
        dia, drill = netclass.via_diameter, netclass.via_drill
        if dia and drill:
            sizes[name] = ViaSize(dia / NM_PER_MM, drill / NM_PER_MM)
    return sizes
