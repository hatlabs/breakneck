import math
from typing import cast

import kipy.board
import kipy.board_types as kbt
import pytest
from kipy.geometry import PolygonWithHoles, PolyLineNode, Vector2
from kipy.project_types import NetClass
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from kipy.proto.board.board_types_pb2 import PadType, ZoneType
from kipy.proto.common.types import project_settings_pb2
from kipy.util.board_layer import canonical_name
from shapely.geometry import Point, box

from breakneck.snapshot import (
    DEFAULT_VIA_SIZE,
    NM_PER_MM,
    Snapshot,
    ViaSize,
    Zone,
    arc_points,
)

FOUR_LAYERS = ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
NUDGE = 1e-9


def zone(net, layers, outline, priority=0, filled=True):
    fills = {L: outline for L in layers} if filled else {}
    return Zone(
        net=net, layers=tuple(layers), priority=priority, outline=outline, fills=fills
    )


def rounded_rect_paths(w, h, r, nudge=NUDGE):
    """Edge.Cuts paths of a rounded rectangle; arc ends are nudged off the lines."""

    def arc(cx, cy, a0):
        a1 = a0 + math.pi / 2
        am = (a0 + a1) / 2
        p = lambda a, n=0.0: (cx + r * math.cos(a) + n, cy + r * math.sin(a) - n)
        return arc_points(p(a0, nudge), p(am), p(a1, nudge))

    return [
        [(r, 0), (w - r, 0)],
        arc(w - r, r, -math.pi / 2),
        [(w, r), (w, h - r)],
        arc(w - r, h - r, 0),
        [(w - r, h), (r, h)],
        arc(r, h - r, math.pi / 2),
        [(0, h - r), (0, r)],
        arc(r, r, math.pi),
    ]


def test_edge_segments_and_arcs_with_float_noise_form_one_outline():
    snap = Snapshot(copper_layers=FOUR_LAYERS, edge_paths=rounded_rect_paths(20, 10, 2))

    outline = snap.board_outline

    assert outline.geom_type == "Polygon"
    expected = 20 * 10 - (4 - math.pi) * 2**2
    assert outline.area == pytest.approx(expected, rel=1e-3)


def test_open_edge_outline_gives_empty_board():
    paths = rounded_rect_paths(20, 10, 2)[:-1]
    snap = Snapshot(copper_layers=FOUR_LAYERS, edge_paths=paths)

    assert snap.board_outline.is_empty


def square_path(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def test_internal_cutout_is_a_hole_in_the_outline():
    paths = [square_path(0, 0, 10, 10), square_path(4, 4, 6, 6)]
    snap = Snapshot(copper_layers=FOUR_LAYERS, edge_paths=paths)

    outline = snap.board_outline

    assert outline.area == pytest.approx(96)
    assert not outline.contains(Point(5, 5))
    assert outline.boundary.distance(Point(5, 4.5)) == pytest.approx(0.5)


def test_island_inside_a_cutout_is_board_again():
    paths = [
        square_path(0, 0, 10, 10),
        square_path(2, 2, 8, 8),
        square_path(4, 4, 6, 6),
    ]
    snap = Snapshot(copper_layers=FOUR_LAYERS, edge_paths=paths)

    outline = snap.board_outline

    assert outline.area == pytest.approx(100 - 36 + 4)
    assert outline.contains(Point(5, 5))
    assert not outline.contains(Point(3, 3))


def test_arc_points_keep_exact_endpoints_and_pass_through_mid():
    start, mid, end = (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)

    pts = arc_points(start, mid, end)

    assert pts[0] == start and pts[-1] == end
    assert all(math.hypot(x, y) == pytest.approx(1.0) for x, y in pts)
    assert all(y >= -1e-12 for _, y in pts)  # went via the mid side, not the other way


def test_arc_points_clockwise_arc_goes_through_mid():
    start, mid, end = (-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)

    pts = arc_points(start, mid, end)

    assert all(y >= -1e-12 for _, y in pts)


def test_zone_layer_rule_in_two_layer_pour_is_front_and_back():
    pour = box(0, 0, 10, 10)
    snap = Snapshot(
        copper_layers=FOUR_LAYERS, zones=[zone("+3.3V", ["F.Cu", "B.Cu"], pour)]
    )

    assert snap.required_layers("+3.3V", (5, 5)) == {"F.Cu", "B.Cu"}
    assert snap.required_layers("+3.3V", (15, 5)) == set()


def test_zone_layer_rule_adds_outer_pour_over_inner_planes():
    snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        zones=[
            zone("GND", ["In1.Cu", "In2.Cu"], box(0, 0, 20, 20)),
            zone("GND", ["F.Cu"], box(0, 0, 10, 20)),
        ],
    )

    assert snap.required_layers("GND", (5, 5)) == {"In1.Cu", "In2.Cu", "F.Cu"}
    assert snap.required_layers("GND", (15, 5)) == {"In1.Cu", "In2.Cu"}


def test_rule_areas_do_not_count_for_the_zone_layer_rule():
    keepout = Zone(
        net=None,
        layers=FOUR_LAYERS,
        priority=0,
        outline=box(0, 0, 10, 10),
        is_rule_area=True,
        keepout_vias=True,
    )
    snap = Snapshot(copper_layers=FOUR_LAYERS, zones=[keepout])

    assert snap.copper_zones == ()
    assert snap.default_ground_nets() == []


def test_default_ground_nets_are_nets_with_two_inner_zone_layers():
    snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        zones=[
            zone("GND", ["In1.Cu", "In2.Cu"], box(0, 0, 10, 10)),
            zone("GND_ISO", ["In1.Cu"], box(10, 0, 20, 10)),
            zone("GND_ISO", ["In2.Cu"], box(10, 0, 20, 10)),
            zone("+3.3V", ["F.Cu", "B.Cu"], box(0, 0, 20, 10)),
            zone("VBUS", ["In1.Cu", "F.Cu"], box(0, 0, 5, 5)),
        ],
    )

    assert snap.default_ground_nets() == ["GND", "GND_ISO"]


def test_overlapping_ground_outlines_resolve_to_higher_priority():
    snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        zones=[
            zone("GND", ["In1.Cu", "In2.Cu"], box(0, 0, 20, 10), priority=0),
            zone("GND_ISO", ["In1.Cu", "In2.Cu"], box(10, 0, 20, 10), priority=2),
        ],
    )
    grounds = ["GND", "GND_ISO"]

    assert snap.ground_net_at((5, 5), grounds) == "GND"
    assert snap.ground_net_at((15, 5), grounds) == "GND_ISO"


def test_ground_falls_back_to_outer_zones_then_none():
    snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        zones=[
            zone("GND", ["In1.Cu", "In2.Cu"], box(0, 0, 10, 10)),
            zone("GND_ISO", ["F.Cu", "B.Cu"], box(20, 0, 30, 10)),
        ],
    )
    grounds = ["GND", "GND_ISO"]

    assert snap.ground_net_at((25, 5), grounds) == "GND_ISO"
    assert snap.ground_net_at((15, 5), grounds) is None


def test_inner_ground_outline_wins_over_higher_priority_outer_zone():
    snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        zones=[
            zone("GND", ["In1.Cu", "In2.Cu"], box(0, 0, 10, 10), priority=0),
            zone("GND_ISO", ["F.Cu"], box(0, 0, 10, 10), priority=5),
        ],
    )

    assert snap.ground_net_at((5, 5), ["GND", "GND_ISO"]) == "GND"


def test_net_with_zone_on_one_layer_is_not_stitchable():
    snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        zones=[
            zone("SW", ["F.Cu"], box(0, 0, 5, 5)),
            zone("+3.3V", ["F.Cu", "B.Cu"], box(0, 0, 5, 5)),
        ],
    )

    ok, skipped = snap.split_stitchable(["SW", "+3.3V", "NOZONE"])

    assert ok == ["+3.3V"]
    assert skipped == ["SW", "NOZONE"]
    assert not snap.is_stitchable("SW")


def test_net_whose_layers_come_from_separate_zones_is_stitchable():
    snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        zones=[
            zone("+5V", ["F.Cu"], box(0, 0, 5, 5)),
            zone("+5V", ["B.Cu"], box(0, 0, 5, 5)),
        ],
    )

    assert snap.is_stitchable("+5V")


def test_via_size_comes_from_netclass_with_default_fallback():
    snap = Snapshot(copper_layers=FOUR_LAYERS, via_sizes={"GND": ViaSize(0.8, 0.4)})

    assert snap.via_size("GND") == ViaSize(0.8, 0.4)
    assert snap.via_size("OTHER") == DEFAULT_VIA_SIZE


def test_fill_unions_zones_of_the_same_net_and_layer():
    snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        zones=[
            zone("GND", ["F.Cu"], box(0, 0, 5, 5)),
            zone("GND", ["F.Cu"], box(5, 0, 10, 5)),
        ],
    )

    assert snap.fill("GND", "F.Cu").area == pytest.approx(50)
    assert snap.fill("GND", "B.Cu").is_empty
    assert snap.fill("GND", "F.Cu").contains(Point(5, 2.5))


# ---- reading a board over the API ---------------------------------------------


def nm(v):
    return round(v * NM_PER_MM)


def kipy_rect(x0, y0, x1, y1):
    pwh = PolygonWithHoles()
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        pwh.outline.append(PolyLineNode.from_xy(nm(x), nm(y)))
    pwh.outline.closed = True
    return pwh


def kipy_pad(pad_id, net, x, y, layers=(BL.BL_F_Cu,), override=None, parent=None):
    p = kbt.Pad()
    p.proto.id.value = pad_id
    p.proto.net.name = net
    p.proto.type = PadType.PT_SMD
    p.position = Vector2.from_xy(nm(x), nm(y))
    p.proto.pad_stack.layers.extend(layers)
    if override is not None:
        p.proto.copper_clearance_override.value_nm = nm(override)
    if parent is not None:
        p.proto.parent.value = parent
    return p


def kipy_footprint(fp_id, override=None):
    f = kbt.FootprintInstance()
    f.proto.id.value = fp_id
    if override is not None:
        f.proto.overrides.copper_clearance.value_nm = nm(override)
    return f


def kipy_zone(outline, layers, net=None, rule_area=False, keepout_vias=False):
    z = kbt.Zone()
    z.proto.type = ZoneType.ZT_RULE_AREA if rule_area else ZoneType.ZT_COPPER
    z.proto.layers.extend(layers)
    z.proto.outline.polygons.add().CopyFrom(outline.proto)
    if rule_area:
        z.proto.rule_area_settings.keepout_vias = keepout_vias
    else:
        z.proto.copper_settings.net.name = net
        for layer in layers:
            filled = z.proto.filled_polygons.add()
            filled.layer = layer
            filled.shapes.polygons.add().CopyFrom(outline.proto)
    return z


def netclass(name, via=None):
    # NetClass() without a proto shares one default proto between instances.
    nc = NetClass(project_settings_pb2.NetClass(name=name))
    if via is not None:
        stack = nc.proto.board.via_stack
        stack.copper_layers.add().size.x_nm = nm(via.diameter)
        stack.drill.diameter.x_nm = nm(via.drill)
    return nc


class FakeBoard:
    """Answers the reads `Snapshot.from_board` makes."""

    def __init__(
        self,
        pads=(),
        zones=(),
        footprints=(),
        layers=(BL.BL_F_Cu, BL.BL_B_Cu),
        netclasses=None,
        drops_pads=False,
    ):
        self.pads = list(pads)
        self.zones = list(zones)
        self.footprints = list(footprints)
        self.layers = list(layers)
        self.netclasses = netclasses or {}
        self.drops_pads = drops_pads

    def get_enabled_layers(self):
        return self.layers

    def get_copper_layer_count(self):
        return sum(1 for L in self.layers if canonical_name(L).endswith(".Cu"))

    def get_pads(self):
        return self.pads

    def get_pad_shapes_as_polygons(self, pads, layer):
        def shape(p):
            x, y = p.position.x / NM_PER_MM, p.position.y / NM_PER_MM
            return kipy_rect(x - 0.25, y - 0.25, x + 0.25, y + 0.25)

        if isinstance(pads, kbt.Pad):
            return shape(pads)
        # KiCad silently leaves out pads it has no shape for.
        return [shape(p) for p in pads[: -1 if self.drops_pads else None]]

    def get_nets(self):
        return []

    def get_tracks(self):
        return []

    def get_vias(self):
        return []

    def get_zones(self):
        return self.zones

    def get_footprints(self):
        return self.footprints

    def get_shapes(self):
        return []

    def get_netclass_for_nets(self, nets):
        return self.netclasses


def read(board):
    return Snapshot.from_board(cast(kipy.board.Board, board))


def test_pad_shape_fallback_fills_every_pad_when_the_batch_drops_one():
    pads = [kipy_pad(f"p{i}", "GND", i, 0) for i in range(3)]

    snap = read(FakeBoard(pads=pads, drops_pads=True))

    assert [sorted(p.shapes) for p in snap.pads] == [["F.Cu"]] * 3
    assert snap.pads[2].shapes["F.Cu"].centroid.x == pytest.approx(2)


def test_pad_inherits_footprint_clearance_override_unless_it_has_its_own():
    pads = [
        kipy_pad("own", "GND", 0, 0, override=0.2, parent="fp"),
        kipy_pad("inherits", "GND", 1, 0, parent="fp"),
        kipy_pad("free", "GND", 2, 0),
    ]

    snap = read(FakeBoard(pads=pads, footprints=[kipy_footprint("fp", 1.6)]))

    assert [p.clearance_override for p in snap.pads] == [0.2, 1.6, None]


def test_rule_area_keeps_its_via_keepout_and_has_no_fills():
    zones = [
        kipy_zone(
            kipy_rect(0, 0, 5, 5), [BL.BL_F_Cu], rule_area=True, keepout_vias=True
        ),
        kipy_zone(kipy_rect(0, 0, 9, 9), [BL.BL_F_Cu, BL.BL_B_Cu], net="GND"),
    ]

    rule, copper = read(FakeBoard(zones=zones)).zones

    assert rule.is_rule_area and rule.keepout_vias and rule.net is None
    assert rule.fills == {}
    assert rule.outline.area == pytest.approx(25)
    assert copper.net == "GND" and not copper.keepout_vias
    assert sorted(copper.fills) == ["B.Cu", "F.Cu"]


def test_copper_layers_are_in_stack_order():
    layers = [BL.BL_B_Cu, BL.BL_In2_Cu, BL.BL_Edge_Cuts, BL.BL_F_Cu, BL.BL_In1_Cu]

    snap = read(FakeBoard(layers=layers))

    assert snap.copper_layers == FOUR_LAYERS


def test_via_sizes_come_from_netclasses_that_set_them():
    netclasses = {"GND": netclass("Power", ViaSize(0.8, 0.4)), "SIG": netclass("Bare")}

    snap = read(FakeBoard(netclasses=netclasses))

    assert snap.via_sizes == {"GND": ViaSize(0.8, 0.4)}
    assert snap.via_size("SIG") == DEFAULT_VIA_SIZE
