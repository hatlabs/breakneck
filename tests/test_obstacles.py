import math

import pytest
from shapely.geometry import LineString, Point, box

from breakneck.obstacles import Obstacles
from breakneck.plan import (
    EDGE_GAP,
    FORBIDDEN_RADIUS,
    HOLE_TO_HOLE_GAP,
    OTHER_NET_CLEARANCE,
    OVERRIDE_MARGIN,
    SAME_NET_PAD_GAP,
    PlannedUnit,
    Stub,
)
from breakneck.snapshot import Pad, PadKind, Snapshot, Track, Via, ViaSize, Zone

LAYERS = ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
BOARD = box(0, 0, 30, 20)
VIA = ViaSize(0.6, 0.3)
R = VIA.diameter / 2
# Keeps probes clear of exact rule boundaries, where float rounding decides.
SLACK = 0.01
MOUNTING_HOLE_OVERRIDE = 1.6


def zone(net, layers, outline=BOARD, fills=None):
    return Zone(
        net=net,
        layers=tuple(layers),
        priority=0,
        outline=outline,
        fills=fills if fills is not None else {L: outline for L in layers},
    )


def pad(net, centre, layer="F.Cu", w=1.0, h=1.0, pad_id=None, override=None):
    x, y = centre
    return Pad(
        id=pad_id or f"{net}@{centre}",
        footprint_id=None,
        net=net,
        kind=PadKind.SMD,
        number="1",
        position=centre,
        drill=0.0,
        shapes={layer: box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)},
        clearance_override=override,
    )


def npth(centre, drill, override=None):
    return Pad(
        id="hole",
        footprint_id=None,
        net="",
        kind=PadKind.NPTH,
        number="",
        position=centre,
        drill=drill,
        shapes={},
        clearance_override=override,
    )


def track(net, start, end, layer="F.Cu", width=0.2):
    return Track(
        id=f"{net}:{start}-{end}",
        net=net,
        layer=layer,
        width=width,
        centreline=LineString([start, end]),
    )


def via(net, centre, size=VIA):
    return Via(
        id=f"{net}@{centre}",
        net=net,
        position=centre,
        diameter=size.diameter,
        drill=size.drill,
    )


def snapshot(pads=(), tracks=(), vias=(), zones=None, outline=BOARD):
    edge = [[(x, y) for x, y in outline.exterior.coords]] if outline else []
    return Snapshot(
        copper_layers=LAYERS,
        pads=list(pads),
        tracks=list(tracks),
        vias=list(vias),
        zones=zones if zones is not None else [zone("GND", ("F.Cu", "B.Cu"))],
        edge_paths=edge,
    )


def obstacles(forbidden=(), **kwargs):
    return Obstacles(snapshot(**kwargs), list(forbidden))


def via_clashes(obs, pos, net="GND"):
    return obs.clashes_with_other_nets(Point(pos), R, net, None)


# ---- other-net copper --------------------------------------------------------


def test_via_keeps_other_net_clearance_from_copper_on_any_layer():
    # Track edge at x = 10.1; a via's edge must stay OTHER_NET_CLEARANCE off it.
    edge = 10.1
    t = track("SIG", (10, 0), (10, 20), layer="In2.Cu")
    obs = obstacles(tracks=[t])

    assert via_clashes(obs, (edge + R + OTHER_NET_CLEARANCE - SLACK, 10))
    assert not via_clashes(obs, (edge + R + OTHER_NET_CLEARANCE + SLACK, 10))


def test_same_net_copper_is_not_an_other_net_obstacle():
    obs = obstacles(tracks=[track("GND", (10, 0), (10, 20))])

    assert not via_clashes(obs, (10.5, 10))


def test_stub_only_clashes_with_copper_on_its_own_layer():
    obs = obstacles(tracks=[track("SIG", (10, 0), (10, 20), layer="B.Cu")])
    stub = LineString([(10.3, 5), (10.3, 15)])

    assert obs.clashes_with_other_nets(stub, 0.1, "GND", "B.Cu")
    assert not obs.clashes_with_other_nets(stub, 0.1, "GND", "F.Cu")


def test_existing_via_blocks_a_stub_on_every_layer():
    obs = obstacles(vias=[via("SIG", (10, 10))])
    stub = LineString([(10.5, 5), (10.5, 15)])

    for layer in LAYERS:
        assert obs.clashes_with_other_nets(stub, 0.1, "GND", layer)


def test_pad_clearance_override_widens_the_gap_around_that_pad():
    plain = obstacles(pads=[pad("SIG", (10, 10))])
    overridden = obstacles(pads=[pad("SIG", (10, 10), override=MOUNTING_HOLE_OVERRIDE)])
    pad_edge = 10.5
    spot = (pad_edge + R + OTHER_NET_CLEARANCE + SLACK, 10)
    gap = MOUNTING_HOLE_OVERRIDE + OVERRIDE_MARGIN

    assert not via_clashes(plain, spot)
    assert via_clashes(overridden, spot)
    assert via_clashes(overridden, (pad_edge + R + gap - SLACK, 10))
    assert not via_clashes(overridden, (pad_edge + R + gap + SLACK, 10))


# ---- NPTH keep-offs ----------------------------------------------------------


def test_npth_keep_off_binds_a_via_of_any_net():
    drill = 3.0
    obs = obstacles(pads=[npth((10, 10), drill, MOUNTING_HOLE_OVERRIDE)])
    reach = drill / 2 + R + MOUNTING_HOLE_OVERRIDE + OVERRIDE_MARGIN

    assert obs.clashes_with_unplated(Point(10 + reach - SLACK, 10), R)
    assert not obs.clashes_with_unplated(Point(10 + reach + SLACK, 10), R)


def test_npth_keep_off_binds_a_stub():
    drill = 2.0
    obs = obstacles(pads=[npth((10, 10), drill)])
    half = 0.1
    reach = drill / 2 + half + OTHER_NET_CLEARANCE
    near = LineString([(10 + reach - SLACK, 5), (10 + reach - SLACK, 15)])
    far = LineString([(10 + reach + SLACK, 5), (10 + reach + SLACK, 15)])

    assert obs.clashes_with_unplated(near, half)
    assert not obs.clashes_with_unplated(far, half)


# ---- hole-to-hole ------------------------------------------------------------


def test_drill_keeps_hole_to_hole_gap_from_existing_and_planned_vias():
    obs = obstacles(vias=[via("SIG", (10, 10))])
    reach = VIA.drill + HOLE_TO_HOLE_GAP

    assert obs.clashes_with_holes((10 + reach - SLACK, 10), VIA.drill / 2)
    assert not obs.clashes_with_holes((10 + reach + SLACK, 10), VIA.drill / 2)
    assert not obs.clashes_with_holes((20, 10), VIA.drill / 2)

    obs.add_unit(PlannedUnit((20 + reach - 2 * SLACK, 10), "GND", VIA))

    assert obs.clashes_with_holes((20, 10), VIA.drill / 2)


def test_plated_pad_drill_counts_as_a_hole():
    pth = Pad(
        id="pth",
        footprint_id=None,
        net="SIG",
        kind=PadKind.PTH,
        number="1",
        position=(10, 10),
        drill=1.0,
        shapes={},
    )
    obs = obstacles(pads=[pth])

    assert obs.clashes_with_holes((10.9, 10), VIA.drill / 2)
    assert not obs.clashes_with_unplated(Point(10.9, 10), R)


# ---- bounds ------------------------------------------------------------------


def test_via_keeps_the_edge_gap_inside_the_outline():
    obs = obstacles()
    inset = EDGE_GAP + R

    assert not obs.in_bounds((inset - SLACK, 10), VIA)
    assert obs.in_bounds((inset + SLACK, 10), VIA)
    assert not obs.in_bounds((-5, 10), VIA)


def test_nothing_is_in_bounds_without_an_outline():
    obs = obstacles(outline=None)

    assert not obs.in_bounds((15, 10), VIA)


def test_via_keepout_rule_area_is_out_of_bounds():
    keepout = Zone(
        None, LAYERS, 0, box(10, 5, 14, 9), is_rule_area=True, keepout_vias=True
    )
    obs = obstacles(zones=[zone("GND", ("F.Cu", "B.Cu")), keepout])

    assert not obs.in_bounds((10 - R + SLACK, 7), VIA)
    assert obs.in_bounds((10 - R - SLACK, 7), VIA)


def test_forbidden_positions_keep_their_radius():
    obs = obstacles(forbidden=[(15, 10)])

    assert not obs.in_bounds((15 + FORBIDDEN_RADIUS - SLACK, 10), VIA)
    assert obs.in_bounds((15 + FORBIDDEN_RADIUS + SLACK, 10), VIA)


# ---- same-net pads -----------------------------------------------------------


def test_via_keeps_out_of_same_net_pads_unless_exempt():
    own = pad("GND", (10, 10), pad_id="own")
    obs = obstacles(pads=[own])
    pad_edge = 10.5
    near = Point(pad_edge + R + SAME_NET_PAD_GAP - SLACK, 10)
    far = Point(pad_edge + R + SAME_NET_PAD_GAP + SLACK, 10)

    assert obs.clashes_with_same_net_pads(near, R, "GND")
    assert not obs.clashes_with_same_net_pads(far, R, "GND")
    assert not obs.clashes_with_same_net_pads(near, R, "GND", exempt_pad="own")
    assert not obs.clashes_with_same_net_pads(near, R, "SIG")


def test_same_net_tracks_and_vias_do_not_count_as_pads():
    obs = obstacles(tracks=[track("GND", (10, 0), (10, 20))], vias=[via("GND", (5, 5))])

    assert not obs.clashes_with_same_net_pads(Point(10.3, 10), R, "GND")
    assert not obs.clashes_with_same_net_pads(Point(5.5, 5), R, "GND")


# ---- planned units -----------------------------------------------------------


def test_planned_via_and_stub_block_other_nets():
    obs = obstacles()
    stub = Stub(start=(5, 10), end=(10, 10), width=0.3, layer="F.Cu")
    obs.add_unit(PlannedUnit((10, 10), "GND", VIA, stub=stub))
    stub_probe = LineString([(7, 9.6), (8, 9.6)])

    assert via_clashes(obs, (10.8, 10), net="SIG")
    assert not via_clashes(obs, (10.8, 10), net="GND")
    assert obs.clashes_with_other_nets(stub_probe, 0.1, "SIG", "F.Cu")
    assert not obs.clashes_with_other_nets(stub_probe, 0.1, "SIG", "B.Cu")


def test_nearby_copper_selects_net_and_exact_layer_including_planned():
    gnd_track = track("GND", (10, 0), (10, 20))
    obs = obstacles(
        tracks=[gnd_track, track("SIG", (11, 0), (11, 20))],
        vias=[via("GND", (12, 10))],
    )
    obs.add_unit(PlannedUnit((13, 10), "GND", VIA))
    area = box(9, 9, 14, 11)

    on_front = obs.nearby_copper(area, 0.0, "GND", "F.Cu")
    through = obs.nearby_copper(area, 0.0, "GND", None)

    assert [g.equals(gnd_track.geometry) for g in on_front] == [True]
    assert sorted(round(g.centroid.x, 3) for g in through) == [12, 13]


# ---- zone-layer rule ---------------------------------------------------------


def test_zone_rule_needs_two_layers_and_the_fill_when_filled():
    hole = box(14, 9, 16, 11)
    fills = {"F.Cu": BOARD, "B.Cu": BOARD.difference(hole)}
    obs = obstacles(zones=[zone("GND", ("F.Cu", "B.Cu"), fills=fills)])

    assert obs.reaches_zones((5, 5), "GND", VIA, filled=True)
    assert not obs.reaches_zones((15, 10), "GND", VIA, filled=True)
    assert obs.reaches_zones((15, 10), "GND", VIA, filled=False)
    assert obs.reaches_zones((15, 10), "GND", VIA, filled=True, exempt_layer="B.Cu")
    assert not obs.reaches_zones((15, 10), "GND", VIA, filled=True, exempt_layer="F.Cu")


def test_zone_rule_fails_with_a_zone_on_one_layer():
    obs = obstacles(zones=[zone("GND", ("F.Cu",))])

    assert not obs.reaches_zones((5, 5), "GND", VIA, filled=False)


# ---- forced-pass helpers -----------------------------------------------------


def test_clearance_to_other_nets_is_capped_at_the_rule():
    obs = obstacles()

    assert obs.clearance_to_other_nets((15, 10), "GND", VIA) == OTHER_NET_CLEARANCE


def test_clearance_to_other_nets_measures_copper_planned_vias_and_keep_offs():
    t = track("SIG", (10, 0), (10, 20))
    obs = obstacles(tracks=[t], pads=[npth((20, 10), 2.0)])
    obs.add_unit(PlannedUnit((15, 15), "SIG", VIA))
    gap = 0.1
    abs_tol = 1e-3

    by_track = obs.clearance_to_other_nets((10.1 + R + gap, 10), "GND", VIA)
    by_planned = obs.clearance_to_other_nets((15 + 2 * R + gap, 15), "GND", VIA)
    by_hole = obs.clearance_to_other_nets(
        (20 + 1.0 + R + OTHER_NET_CLEARANCE + gap, 10), "GND", VIA
    )
    same_net = obs.clearance_to_other_nets((10.1 + R + gap, 10), "SIG", VIA)

    assert by_track == pytest.approx(gap, abs=abs_tol)
    assert by_planned == pytest.approx(gap, abs=abs_tol)
    assert by_hole == pytest.approx(gap, abs=abs_tol)
    assert same_net == OTHER_NET_CLEARANCE


@pytest.mark.parametrize("polygon_disk", [False, True])
def test_planned_via_of_another_net_is_an_exact_circle(polygon_disk):
    # Along this bearing both 64-segment disks are flat sides, which would
    # add 0.7 µm to the gap and accept the spot.
    bearing = math.pi / 64
    gap = OTHER_NET_CLEARANCE - 0.0006
    obs = obstacles()
    obs.add_unit(PlannedUnit((15, 10), "GND_ISO", VIA))
    d = 2 * R + gap
    spot = (15 + d * math.cos(bearing), 10 + d * math.sin(bearing))

    assert not obs.via_is_clear(spot, "GND", VIA, polygon_disk=polygon_disk)
    assert obs.clearance_to_other_nets(spot, "GND", VIA) == pytest.approx(gap)


def test_clearance_to_other_nets_is_negative_on_overlap():
    obs = obstacles(tracks=[track("SIG", (10, 0), (10, 20))])

    assert obs.clearance_to_other_nets((10, 10), "GND", VIA) <= 0


def test_unrelaxed_rules_ignore_other_nets_but_keep_holes_and_pads():
    gnd_pad = pad("GND", (20, 10), pad_id="own")
    obs = obstacles(
        tracks=[track("SIG", (10, 0), (10, 20))],
        vias=[via("SIG", (5, 5))],
        pads=[gnd_pad],
    )

    assert obs.keeps_unrelaxed_rules((10.3, 10), "GND", VIA)
    assert not obs.keeps_unrelaxed_rules((5.4, 5), "GND", VIA)
    assert not obs.keeps_unrelaxed_rules((20.8, 10), "GND", VIA)
