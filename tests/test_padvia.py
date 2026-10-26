import math
from dataclasses import replace

import pytest
from shapely.geometry import LineString, Point, box

from breakneck import padvia
from breakneck.obstacles import Obstacles
from breakneck.padvia import PadviaOptions, plan_padvia
from breakneck.plan import FORBIDDEN_RADIUS, HOLE_TO_HOLE_GAP
from breakneck.snapshot import (
    Footprint,
    Pad,
    PadKind,
    Snapshot,
    Track,
    Via,
    ViaSize,
    Zone,
)

FOUR_LAYERS = ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
BOARD = box(0, 0, 30, 20)
CENTRE = (10.0, 10.0)
VIA = ViaSize(0.6, 0.3)
GND = PadviaOptions(nets=("GND",))
MOUNTING_HOLE_OVERRIDE = 1.6


def edge_paths(outline=BOARD):
    return [[(x, y) for x, y in outline.exterior.coords]]


def zone(net, layers, outline=BOARD, fills=None):
    return Zone(
        net=net,
        layers=tuple(layers),
        priority=0,
        outline=outline,
        fills=fills if fills is not None else {L: outline for L in layers},
    )


def pad(
    net,
    centre,
    w=0.5,
    h=0.5,
    layer="F.Cu",
    footprint_id=None,
    pad_id=None,
    kind=PadKind.SMD,
):
    x, y = centre
    return Pad(
        id=pad_id or f"{net}@{x},{y}",
        footprint_id=footprint_id,
        net=net,
        kind=kind,
        number="1",
        position=centre,
        drill=0.0,
        shapes={layer: box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)},
    )


def blocker(layer, corridor, centre=CENTRE, inner=0.45, outer=3.0):
    """No-net copper ringing `centre` on `layer`, open along `corridor`."""
    ring = Point(centre).buffer(outer).difference(Point(centre).buffer(inner))
    return Pad(
        id=f"blocker-{layer}",
        footprint_id=None,
        net="",
        kind=PadKind.SMD,
        number="",
        position=centre,
        drill=0.0,
        shapes={layer: ring.difference(corridor)},
    )


EAST_CORRIDOR = box(10.3, 9.0, 13.5, 11.0)


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


def snapshot(pads=(), tracks=(), vias=(), zones=None, footprints=()):
    return Snapshot(
        copper_layers=FOUR_LAYERS,
        pads=list(pads),
        tracks=list(tracks),
        vias=list(vias),
        zones=zones if zones is not None else [zone("GND", ("F.Cu", "B.Cu"))],
        footprints=list(footprints),
        edge_paths=edge_paths(),
        via_sizes={"GND": VIA},
    )


def track(net, start, end, layer="F.Cu", width=0.2, track_id=None):
    return Track(
        id=track_id or f"{net}:{start}-{end}",
        net=net,
        layer=layer,
        width=width,
        centreline=LineString([start, end]),
    )


def with_plan_applied(snap, plan):
    """The snapshot a second run would see after committing `plan`."""
    vias = [
        Via(f"new{i}", u.net, u.position, u.size.diameter, u.size.drill)
        for i, u in enumerate(plan.units)
    ]
    tracks = [
        track(u.net, u.stub.start, u.stub.end, u.stub.layer, u.stub.width, f"stub{i}")
        for i, u in enumerate(plan.units)
        if u.stub is not None
    ]
    return snapshot(
        pads=snap.pads,
        tracks=[*snap.tracks, *tracks],
        vias=[*snap.vias, *vias],
        zones=list(snap.zones),
        footprints=snap.footprints,
    )


def r0402(net_a="GND", net_b="SIG"):
    """An 0402 at CENTRE with its GND pad on the -x side.

    The planner scans rays counter-clockwise from +x, so without the inward
    penalty the free +y ray would win over the outward -x one.
    """
    fp = Footprint(id="R1", ref="R1", position=CENTRE, layer="F.Cu")
    pads = [
        pad(net_a, (9.52, 10.0), 0.56, 0.62, footprint_id="R1", pad_id="R1.1"),
        pad(net_b, (10.48, 10.0), 0.56, 0.62, footprint_id="R1", pad_id="R1.2"),
    ]
    return fp, pads


# ---- placement -------------------------------------------------------------------


def test_0402_gnd_pad_gets_via_away_from_footprint_centre():
    fp, pads = r0402()

    plan = plan_padvia(snapshot(pads, footprints=[fp]), GND, [])

    assert len(plan.units) == 1
    u = plan.units[0]
    assert u.net == "GND"
    assert u.size == VIA
    assert u.position[0] < 9.52 - 0.28
    assert u.position[1] == pytest.approx(10.0, abs=1e-3)
    assert u.stub is not None
    assert u.stub.layer == "F.Cu"
    assert u.stub.start == pytest.approx((9.52, 10.0))
    assert u.stub.end == u.position
    assert not plan.unfixable


def test_via_option_overrides_netclass_size():
    fp, pads = r0402()
    big = ViaSize(0.8, 0.4)

    plan = plan_padvia(
        snapshot(pads, footprints=[fp]), PadviaOptions(("GND",), via=big), []
    )

    assert plan.units[0].size == big


def test_only_pads_of_requested_nets_and_single_layer_smd_are_candidates():
    pads = [
        pad("GND", (5, 5)),
        pad("SIG", (15, 5)),
        pad("GND", (20, 5), kind=PadKind.PTH),
        Pad(
            id="two-layer",
            footprint_id=None,
            net="GND",
            kind=PadKind.SMD,
            number="1",
            position=(25, 5),
            drill=0.0,
            shapes={
                "F.Cu": box(24.75, 4.75, 25.25, 5.25),
                "B.Cu": box(24.75, 4.75, 25.25, 5.25),
            },
        ),
    ]

    plan = plan_padvia(snapshot(pads), GND, [])

    assert len(plan.units) == 1
    assert Point(plan.units[0].position).distance(Point(5, 5)) < 2


@pytest.mark.parametrize(
    ("w", "h", "expected"),
    [(0.25, 0.8, 0.25), (1.0, 1.0, 0.4), (0.15, 0.6, 0.2)],
)
def test_stub_width_is_pad_minor_dimension_within_limits(w, h, expected):
    plan = plan_padvia(snapshot([pad("GND", CENTRE, w, h)]), GND, [])

    stub = plan.units[0].stub
    assert stub is not None
    assert stub.width == pytest.approx(expected)


# ---- served pads -----------------------------------------------------------------


def test_pad_joined_to_nearby_via_by_track_is_served():
    p = pad("GND", CENTRE)
    via = Via("v", "GND", (11.0, 10.0), 0.6, 0.3)
    joined = track("GND", CENTRE, (11.0, 10.0))

    assert plan_padvia(snapshot([p], tracks=[joined], vias=[via]), GND, []).units == ()


def test_nearby_via_without_connecting_copper_does_not_serve():
    p = pad("GND", CENTRE)
    via = Via("v", "GND", (11.0, 10.0), 0.6, 0.3)
    pour_elsewhere = box(20, 0, 30, 20)
    zones = [
        zone("GND", ("F.Cu", "B.Cu"), fills={"F.Cu": pour_elsewhere, "B.Cu": BOARD})
    ]

    plan = plan_padvia(snapshot([p], vias=[via], zones=zones), GND, [])

    assert len(plan.units) == 1


def test_pad_and_nearby_via_in_the_same_pour_on_the_pad_layer_is_served():
    p = pad("GND", CENTRE)
    via = Via("v", "GND", (10.8, 10.0), 0.6, 0.3)

    assert plan_padvia(snapshot([p], vias=[via]), GND, []).units == ()


def test_via_in_a_separate_island_of_the_pad_layer_pour_does_not_serve():
    p = pad("GND", CENTRE)
    via = Via("v", "GND", (10.8, 10.0), 0.6, 0.3)
    split = BOARD.difference(box(10.4, 0, 10.45, 20))
    zones = [zone("GND", ("F.Cu", "B.Cu"), fills={"F.Cu": split, "B.Cu": BOARD})]

    plan = plan_padvia(snapshot([p], vias=[via], zones=zones), GND, [])

    assert len(plan.units) == 1


def test_connected_via_beyond_served_distance_does_not_serve():
    p = pad("GND", CENTRE)
    via = Via("v", "GND", (13.0, 10.0), 0.6, 0.3)
    joined = track("GND", CENTRE, (13.0, 10.0))

    plan = plan_padvia(snapshot([p], tracks=[joined], vias=[via]), GND, [])

    assert len(plan.units) == 1


# ---- obstacles -------------------------------------------------------------------


def test_free_spot_inside_mounting_hole_clearance_is_unfixable():
    hole = npth((13.5, 10.0), 3.0, override=MOUNTING_HOLE_OVERRIDE)
    pads = [pad("GND", CENTRE), blocker("B.Cu", EAST_CORRIDOR), hole]

    plan = plan_padvia(snapshot(pads), GND, [])

    assert plan.units == ()
    assert len(plan.unfixable) == 1
    spot = plan.unfixable[0]
    assert spot.net == "GND"
    assert spot.layer == "F.Cu"
    assert spot.position == pytest.approx(CENTRE)


def test_via_keeps_edge_gap_from_a_board_cutout():
    pads = [pad("GND", CENTRE), blocker("B.Cu", EAST_CORRIDOR)]
    slot = box(10.8, 8.5, 14, 11.5)
    snap = snapshot(pads)
    assert plan_padvia(snap, GND, []).units  # the corridor alone has room

    cut = replace(snap, edge_paths=[*snap.edge_paths, list(slot.exterior.coords)])
    plan = plan_padvia(cut, GND, [])

    assert plan.units == ()
    assert [s.position for s in plan.unfixable] == [pytest.approx(CENTRE)]


def test_hole_without_clearance_override_leaves_the_spot_free():
    hole = npth((13.5, 10.0), 3.0)
    pads = [pad("GND", CENTRE), blocker("B.Cu", EAST_CORRIDOR), hole]

    plan = plan_padvia(snapshot(pads), GND, [])

    assert len(plan.units) == 1
    assert plan.units[0].position[0] > 10.3


def test_other_net_pad_clearance_override_is_honoured():
    near = pad("SIG", (11.9, 10.0), 0.4, 0.4)
    padded = replace(near, clearance_override=1.0)
    pads = [pad("GND", CENTRE), blocker("B.Cu", EAST_CORRIDOR)]

    plain = plan_padvia(snapshot([*pads, near]), GND, [])
    overridden = plan_padvia(snapshot([*pads, padded]), GND, [])

    assert len(plain.units) == 1
    assert overridden.units == ()


def test_stub_crossing_other_net_track_on_pad_layer_is_rejected():
    crossing = track("SIG", (11.0, 8.5), (11.0, 11.5), layer="F.Cu")
    pads = [pad("GND", CENTRE), blocker("B.Cu", EAST_CORRIDOR)]

    plan = plan_padvia(snapshot(pads, tracks=[crossing]), GND, [])

    assert plan.units == ()
    assert len(plan.unfixable) == 1


def test_via_may_hop_over_other_net_track_on_another_layer():
    inner = track("SIG", (11.0, 8.5), (11.0, 11.5), layer="In1.Cu")
    pads = [pad("GND", CENTRE), blocker("B.Cu", EAST_CORRIDOR)]

    plan = plan_padvia(snapshot(pads, tracks=[inner]), GND, [])

    assert len(plan.units) == 1
    x, _ = plan.units[0].position
    assert x - 11.0 >= 0.1 + 0.3 + 0.3 - 1e-9


def test_via_clears_other_net_copper_on_every_layer():
    pads = [pad("GND", CENTRE), blocker("In2.Cu", EAST_CORRIDOR)]

    plan = plan_padvia(snapshot(pads), GND, [])

    assert len(plan.units) == 1
    assert plan.units[0].position[0] > 10.3


# ---- zone-layer rule ---------------------------------------------------------------


def test_net_with_zone_on_one_layer_only_is_skipped():
    zones = [zone("GND", ("F.Cu", "B.Cu")), zone("+5V", ("F.Cu",))]
    pads = [pad("+5V", CENTRE)]

    plan = plan_padvia(snapshot(pads, zones=zones), PadviaOptions(("+5V",)), [])

    assert plan.units == ()
    assert plan.unfixable == ()


def test_via_must_sit_inside_fill_on_required_layers_other_than_pad_layer():
    back_fill = BOARD.difference(Point(CENTRE).buffer(3.0))
    zones = [
        Zone(
            net="GND",
            layers=("F.Cu", "B.Cu"),
            priority=0,
            outline=BOARD,
            fills={"F.Cu": BOARD, "B.Cu": back_fill},
        )
    ]

    plan = plan_padvia(snapshot([pad("GND", CENTRE)], zones=zones), GND, [])

    assert plan.units == ()
    assert len(plan.unfixable) == 1


def test_via_outside_second_zone_outline_is_unfixable():
    zones = [
        zone("GND", ("F.Cu",)),
        zone("GND", ("B.Cu",), outline=box(20, 0, 30, 20)),
    ]

    plan = plan_padvia(snapshot([pad("GND", CENTRE)], zones=zones), GND, [])

    assert plan.units == ()
    assert len(plan.unfixable) == 1


# ---- prune loop and repeat runs ------------------------------------------------------


def test_via_avoids_a_via_keepout_rule_area():
    fp, pads = r0402()
    nearest = plan_padvia(snapshot(pads, footprints=[fp]), GND, []).units[0]
    keepout = Point(nearest.position).buffer(0.4)
    rule_area = Zone(
        net=None,
        layers=FOUR_LAYERS,
        priority=0,
        outline=keepout,
        is_rule_area=True,
        keepout_vias=True,
    )
    zones = [zone("GND", ("F.Cu", "B.Cu")), rule_area]

    plan = plan_padvia(snapshot(pads, zones=zones, footprints=[fp]), GND, [])

    (u,) = plan.units
    assert not Point(u.position).buffer(VIA.diameter / 2).intersects(keepout)


def test_forbidden_position_is_not_reused():
    fp, pads = r0402()
    snap = snapshot(pads, footprints=[fp])
    first = plan_padvia(snap, GND, []).units[0].position

    plan = plan_padvia(snap, GND, [first])

    assert len(plan.units) == 1
    assert math.dist(plan.units[0].position, first) >= FORBIDDEN_RADIUS


def test_second_run_on_output_places_nothing():
    fp, pads = r0402()
    other_fp = Footprint(id="C1", ref="C1", position=(20.0, 10.0), layer="F.Cu")
    more = [
        pad("GND", (20.48, 10.0), 0.56, 0.62, footprint_id="C1", pad_id="C1.1"),
        pad("+3V3", (19.52, 10.0), 0.56, 0.62, footprint_id="C1", pad_id="C1.2"),
    ]
    snap = snapshot([*pads, *more], footprints=[fp, other_fp])
    first = plan_padvia(snap, GND, [])
    assert len(first.units) == 2

    second = plan_padvia(with_plan_applied(snap, first), GND, [])

    assert second.units == ()
    assert second.unfixable == ()


def test_planned_vias_respect_each_other():
    pads = [pad("GND", (10.0, 10.0)), pad("SIG2", (10.0, 11.4))]
    sig_zone = zone("SIG2", ("F.Cu", "B.Cu"))
    snap = snapshot(pads, zones=[zone("GND", ("F.Cu", "B.Cu")), sig_zone])

    plan = plan_padvia(snap, PadviaOptions(("GND", "SIG2")), [])

    assert len(plan.units) == 2
    a, b = (Point(u.position) for u in plan.units)
    assert a.distance(b) >= 0.6 + 0.3 - 1e-9


def test_a_later_via_keeps_hole_to_hole_from_an_earlier_planned_one():
    # Both pads' first free spot is due east; their holes would be 0.25 mm
    # apart. No pour on the pad layer, so the first via does not serve the
    # second pad.
    pads = [pad("GND", (10.0, 10.0)), pad("GND", (10.0, 10.55))]
    snap = snapshot(pads, zones=[zone("GND", ("In1.Cu", "B.Cu"))])

    plan = plan_padvia(snap, GND, [])

    a, b = plan.units
    assert math.dist(a.position, b.position) - VIA.drill >= HOLE_TO_HOLE_GAP


def test_stub_through_a_mounting_hole_keep_off_is_rejected():
    # The corridor opens only beyond the hole, so every stub passes it.
    corridor = box(11.2, 9.0, 13.5, 11.0)
    pads = [pad("GND", CENTRE), blocker("B.Cu", corridor)]
    (unit,) = plan_padvia(snapshot(pads), GND, []).units
    snap = snapshot([*pads, npth((10.9, 10.0), 0.4)])

    plan = plan_padvia(snap, GND, [])

    assert plan.units == ()
    assert len(plan.unfixable) == 1
    # The via spot itself keeps clear of the hole; only the stub does not.
    own = pads[0]
    assert padvia._via_ok(Obstacles(snap, []), unit.position, "GND", VIA, own)


def test_via_may_sit_within_the_via_in_pad_gap_of_its_own_pad():
    # Rounding the first clear spot to 1 µm pulls it 0.4 µm inside the gap.
    plan = plan_padvia(snapshot([pad("GND", (10.0004, 10.0))]), GND, [])

    assert plan.units[0].position == (10.7, 10.0)
