import math

import pytest
from shapely.geometry import LineString, Point, box

from breakneck.plan import FORBIDDEN_RADIUS
from breakneck.returnvia import (
    CLEARANCE,
    ReturnViaOptions,
    plan_return_vias,
    return_lines,
    unserved_vias,
)
from breakneck.snapshot import (
    Pad,
    PadKind,
    Snapshot,
    Track,
    Via,
    ViaSize,
    Zone,
)

FOUR_LAYERS = ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
INNER = ("In1.Cu", "In2.Cu")
VIA_RADIUS = 0.3
EPS = 1e-6
BOARD_W, BOARD_H = 40.0, 20.0
GND_AREA = box(0, 0, 20, 20)
ISO_AREA = box(20, 0, 40, 20)


def zone(net, layers, outline, fills=None, priority=0):
    return Zone(
        net=net,
        layers=tuple(layers),
        priority=priority,
        outline=outline,
        fills=fills if fills is not None else {L: outline for L in layers},
    )


def via(net, x, y, dia=0.6, drill=0.3):
    return Via(id=f"{net}@{x},{y}", net=net, position=(x, y), diameter=dia, drill=drill)


def track(net, layer, start, end, width=0.2):
    return Track(
        id=f"{net}:{start}-{end}",
        net=net,
        layer=layer,
        width=width,
        centreline=LineString([start, end]),
    )


def pad(net, pos, shapes, kind=PadKind.SMD, drill=0.0, override=None):
    return Pad(
        id=f"{net}@{pos}",
        footprint_id=None,
        net=net,
        kind=kind,
        number="1",
        position=pos,
        drill=drill,
        shapes=shapes,
        clearance_override=override,
    )


def snapshot(vias=(), tracks=(), pads=(), zones=None, via_sizes=None):
    edge = [[(0, 0), (BOARD_W, 0), (BOARD_W, BOARD_H), (0, BOARD_H), (0, 0)]]
    if zones is None:
        zones = [zone("GND", INNER, GND_AREA), zone("GND_ISO", INNER, ISO_AREA)]
    return Snapshot(
        copper_layers=FOUR_LAYERS,
        pads=list(pads),
        tracks=list(tracks),
        vias=list(vias),
        zones=list(zones),
        edge_paths=edge,
        via_sizes=via_sizes or {},
    )


def plan(snap, forbidden=(), **kwargs):
    return plan_return_vias(snap, ReturnViaOptions(**kwargs), list(forbidden))


def disk(position, radius=VIA_RADIUS):
    return Point(position).buffer(radius)


def test_signal_via_gets_return_within_distance_clear_of_other_net_track():
    sig = via("SIG", 10, 10)
    other = track("X", "F.Cu", (10.9, 5), (10.9, 15))
    result = plan(snapshot(vias=[sig], tracks=[other]))

    assert len(result.units) == 1
    (u,) = result.units
    assert u.net == "GND"
    assert not u.forced
    assert math.dist(u.position, sig.position) <= 1.5 + EPS
    assert disk(u.position).distance(other.geometry) >= CLEARANCE - EPS
    assert disk(u.position).distance(disk(sig.position)) >= CLEARANCE - EPS
    assert not result.unfixable


def test_default_distance_is_one_and_a_half_mm():
    assert ReturnViaOptions().distance == 1.5


def test_ground_via_one_mm_away_serves_the_signal_via():
    result = plan(snapshot(vias=[via("SIG", 10, 10), via("GND", 11, 10)]))
    assert not result.units and not result.unfixable


def test_ground_pth_pad_counts_as_a_return():
    gnd_pad = pad(
        "GND",
        (11, 10),
        {L: disk((11, 10), 0.5) for L in FOUR_LAYERS},
        kind=PadKind.PTH,
        drill=0.4,
    )
    result = plan(snapshot(vias=[via("SIG", 10, 10)], pads=[gnd_pad]))
    assert not result.units and not result.unfixable


def test_ground_via_of_another_ground_does_not_serve():
    result = plan(snapshot(vias=[via("SIG", 30, 10), via("GND", 30.5, 10)]))
    assert [u.net for u in result.units] == ["GND_ISO"]


def test_via_in_isolated_ground_area_gets_isolated_ground_return():
    result = plan(snapshot(vias=[via("SIG_ISO", 30, 10)]))
    assert [u.net for u in result.units] == ["GND_ISO"]
    assert ISO_AREA.contains(disk(result.units[0].position))


def test_ground_via_never_needs_a_return():
    result = plan(snapshot(vias=[via("GND_ISO", 30, 10), via("GND", 5, 5)]))
    assert not result.units and not result.unfixable


def test_via_outside_every_ground_outline_is_unfixable_and_never_forced():
    zones = [zone("GND", INNER, GND_AREA)]
    result = plan(snapshot(vias=[via("SIG", 35, 10)], zones=zones), force=True)
    assert not result.units
    assert [(s.position, s.net) for s in result.unfixable] == [((35, 10), "SIG")]


def test_outer_ground_zone_serves_where_no_inner_outline_does():
    zones = [
        zone("GND", INNER, GND_AREA),
        zone("GND_ISO", ("F.Cu", "B.Cu"), ISO_AREA),
        zone("GND_ISO", ("In1.Cu", "In2.Cu"), box(0, 0, 1, 1)),
    ]
    result = plan(snapshot(vias=[via("SIG", 30, 10)], zones=zones))
    assert [u.net for u in result.units] == ["GND_ISO"]


def test_return_via_sits_inside_the_fill_on_every_required_layer():
    # An outer GND pour adds F.Cu to the required layers, filled only left of x=10.
    left = box(0, 0, 10, 20)
    zones = [
        zone("GND", INNER, GND_AREA),
        zone("GND", ("F.Cu",), GND_AREA, fills={"F.Cu": left}),
        zone("GND_ISO", INNER, ISO_AREA),
    ]
    result = plan(snapshot(vias=[via("SIG", 10, 10)], zones=zones))
    (u,) = result.units
    assert left.contains(disk(u.position))


def test_netclass_via_size_with_override():
    snap = snapshot(vias=[via("SIG", 10, 10)], via_sizes={"GND": ViaSize(0.7, 0.35)})
    assert plan(snap).units[0].size == ViaSize(0.7, 0.35)
    assert plan(snap, via=ViaSize(0.5, 0.25)).units[0].size == ViaSize(0.5, 0.25)


def test_one_return_via_serves_several_nearby_signal_vias():
    result = plan(snapshot(vias=[via("A", 10, 10), via("B", 10, 11.2)]))
    assert len(result.units) == 1


def test_forbidden_position_is_not_reused():
    snap = snapshot(vias=[via("SIG", 10, 10)])
    (first,) = plan(snap).units
    (second,) = plan(snap, forbidden=[first.position]).units
    assert math.dist(first.position, second.position) >= FORBIDDEN_RADIUS
    assert math.dist(second.position, (10, 10)) <= 1.5 + EPS


@pytest.mark.parametrize("layer", FOUR_LAYERS)
def test_other_net_pad_clearance_override_is_enforced_on_every_layer(layer):
    override = 1.0
    other = pad("X", (11.2, 10), {layer: box(11, 9, 11.4, 11)}, override=override)
    result = plan(snapshot(vias=[via("SIG", 10, 10)], pads=[other]))
    (u,) = result.units
    assert disk(u.position).distance(other.shapes[layer]) >= override - EPS


def test_mounting_hole_clearance_override_is_enforced():
    hole = pad("", (13, 10), {}, kind=PadKind.NPTH, drill=3.2, override=1.6)
    result = plan(snapshot(vias=[via("SIG", 10, 10)], pads=[hole]))
    (u,) = result.units
    assert disk(u.position).distance(disk(hole.position, 1.6)) >= 1.6 - EPS


def test_return_via_keeps_clear_of_board_edge():
    # The only room left of the track is too close to the board edge.
    wall = track("X", "F.Cu", (2.1, 5), (2.1, 15), width=1.8)
    result = plan(snapshot(vias=[via("SIG", 0.6, 10)], tracks=[wall]))
    assert not result.units
    assert [s.position for s in result.unfixable] == [(0.6, 10)]


def _crowded(signal_y=10.0, pitch=1.0, width=0.2):
    """Other-net tracks on F.Cu every `pitch` mm, leaving a gap around y=10."""
    rows = [signal_y + 0.5 + k * pitch for k in range(-4, 4)]
    return [track("X", "F.Cu", (5, y), (15, y), width) for y in rows]


def test_crowded_via_without_force_is_unfixable():
    result = plan(snapshot(vias=[via("SIG", 10, 10)], tracks=_crowded()))
    assert not result.units
    assert [s.position for s in result.unfixable] == [(10, 10)]


def test_force_places_crowded_via_at_maximum_clearance():
    tracks = _crowded()
    result = plan(snapshot(vias=[via("SIG", 10, 10)], tracks=tracks), force=True)

    (u,) = result.units
    assert u.forced and u.net == "GND"
    assert math.dist(u.position, (10, 10)) <= 1.5 + EPS
    # Tracks 1 mm apart leave 0.8 mm gaps: a 0.6 mm via gets 0.1 mm either side.
    gap = min(disk(u.position).distance(t.geometry) for t in tracks)
    assert gap == pytest.approx(0.1, abs=0.01)
    assert not result.unfixable


def test_force_skips_and_reports_spot_that_would_overlap_other_net_copper():
    blanket = track("X", "F.Cu", (5, 10), (15, 10), width=5)
    result = plan(snapshot(vias=[via("SIG", 10, 10)], tracks=[blanket]), force=True)
    assert not result.units
    assert [s.position for s in result.unfixable] == [(10, 10)]


def test_force_never_enters_a_mounting_hole_keep_off():
    # The keep-off (hole radius 0.5 + 2.5 override) covers every reachable spot.
    hole = pad("", (10, 11), {}, kind=PadKind.NPTH, drill=1.0, override=2.5)
    result = plan(snapshot(vias=[via("SIG", 10, 10)], pads=[hole]), force=True)
    assert not result.units
    assert [s.position for s in result.unfixable] == [(10, 10)]


def test_one_forced_via_serves_a_nearby_crowded_via_too():
    vias = [via("A", 10, 10), via("B", 10, 11)]
    result = plan(snapshot(vias=vias, tracks=_crowded()), force=True)
    assert len(result.units) == 1 and result.units[0].forced
    assert not result.unfixable


def test_planned_returns_satisfy_the_gndvia_check():
    # A keep-off around the via leaves room only on the outermost ring, where
    # rounding to 1 µm would push this off-grid position past the distance.
    centre = (10.0006, 10)
    hole = pad("", centre, {}, kind=PadKind.NPTH, drill=0.4, override=0.98)
    snap = snapshot(vias=[via("SIG", *centre)], pads=[hole])
    result = plan(snap)

    placed = [via(u.net, *u.position) for u in result.units]
    after = snapshot(vias=[*snap.vias, *placed], pads=[hole])
    assert {u.position for u in unserved_vias(after, None, 1.5)} == {
        s.position for s in result.unfixable
    }


def test_gndvia_reports_exactly_the_vias_returnvia_serves():
    zones = [zone("GND", INNER, GND_AREA), zone("GND_ISO", INNER, box(20, 0, 30, 20))]
    served = via("SERVED", 5, 5)
    needs = via("NEEDS", 10, 10)
    iso = via("ISO", 25, 10)
    outside = via("OUTSIDE", 35, 10)
    snap = snapshot(
        vias=[served, via("GND", 5.8, 5), needs, iso, outside, via("GND_ISO", 28, 10)],
        zones=zones,
    )
    options = ReturnViaOptions()

    reported = {u.net for u in unserved_vias(snap, ["GND", "GND_ISO"], 1.5)}
    result = plan_return_vias(snap, options, [])
    acted_on = {s.net for s in result.unfixable} | {
        v.net
        for v in (needs, iso)
        if any(math.dist(u.position, v.position) <= 1.5 + EPS for u in result.units)
    }

    assert reported == {"NEEDS", "ISO", "OUTSIDE"}
    assert reported == acted_on


def test_gndvia_lines_go_to_the_nearest_return_of_the_local_ground():
    zones = [zone("GND", INNER, GND_AREA), zone("GND_ISO", INNER, ISO_AREA)]
    snap = snapshot(
        vias=[
            via("ISO", 25, 10),
            via("GND", 19, 10),  # nearer, but the wrong ground
            via("GND_ISO", 28, 10),
            via("GND_ISO", 35, 10),
        ],
        zones=zones,
    )
    lines, markers = return_lines(snap, None, 1.5)
    assert lines == [((25, 10), (28, 10))]
    assert markers == []


def test_gndvia_marks_vias_without_a_ground_or_a_return():
    zones = [zone("GND", INNER, GND_AREA)]
    snap = snapshot(vias=[via("A", 10, 10), via("B", 35, 10)], zones=zones)
    lines, markers = return_lines(snap, None, 1.5)
    assert lines == []
    assert sorted(markers) == [(10, 10), (35, 10)]

