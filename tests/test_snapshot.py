import math

import pytest
from shapely.geometry import Point, box

from breakneck.snapshot import (
    DEFAULT_VIA_SIZE,
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
