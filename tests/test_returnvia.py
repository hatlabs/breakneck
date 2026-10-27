import math
import sys
from dataclasses import replace

import pytest
from shapely.geometry import LineString, Point, box

from breakneck import cli, returnvia_command
from breakneck.plan import (
    EDGE_GAP,
    FORBIDDEN_RADIUS,
    HOLE_TO_HOLE_GAP,
    OTHER_NET_CLEARANCE,
    SAME_NET_PAD_GAP,
)
from breakneck.returnpath import measure_returns
from breakneck.returnvia import (
    NoGroundNetsError,
    ReferencePlane,
    ReturnViaOptions,
    resolve_ground_nets,
    via_references,
)
from breakneck.returnvia_planner import ReturnViaPlanner
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


SIGNAL_ID = "signal:"


def signal(net, x, y):
    """A via that `snapshot` ties to F.Cu and B.Cu, so it changes layers."""
    return replace(via(net, x, y), id=f"{SIGNAL_ID}{net}@{x},{y}")


def ties(vias):
    """F.Cu and B.Cu stubs within each signal via's own copper."""
    return [
        track(v.net, L, v.position, (v.position[0] - 0.1, v.position[1]))
        for v in vias
        if v.id.startswith(SIGNAL_ID)
        for L in ("F.Cu", "B.Cu")
    ]


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
        tracks=[*ties(vias), *tracks],
        vias=list(vias),
        zones=list(zones),
        edge_paths=edge,
        via_sizes=via_sizes or {},
    )


def plan(snap, forbidden=(), **kwargs):
    return ReturnViaPlanner()(snap, ReturnViaOptions(**kwargs), list(forbidden))


def unserved(snap, ground_nets=None, distance=1.5):
    return [r for r in measure_returns(snap, ground_nets, distance) if not r.served]


def disk(position, radius=VIA_RADIUS):
    return Point(position).buffer(radius)


def test_signal_via_gets_return_within_distance_clear_of_other_net_track():
    sig = signal("SIG", 10, 10)
    other = track("X", "F.Cu", (10.9, 5), (10.9, 15))
    result = plan(snapshot(vias=[sig], tracks=[other]))

    assert len(result.units) == 1
    (u,) = result.units
    assert u.net == "GND"
    assert not u.forced
    assert math.dist(u.position, sig.position) <= 1.5 + EPS
    assert disk(u.position).distance(other.geometry) >= OTHER_NET_CLEARANCE - EPS
    assert disk(u.position).distance(disk(sig.position)) >= OTHER_NET_CLEARANCE - EPS
    assert not result.unfixable


def test_default_distance_is_one_and_a_half_mm():
    assert ReturnViaOptions().distance == 1.5


def test_ground_via_one_mm_away_serves_the_signal_via():
    result = plan(snapshot(vias=[signal("SIG", 10, 10), via("GND", 11, 10)]))
    assert not result.units and not result.unfixable


def test_ground_pth_pad_counts_as_a_return():
    gnd_pad = pad(
        "GND",
        (11, 10),
        {L: disk((11, 10), 0.5) for L in FOUR_LAYERS},
        kind=PadKind.PTH,
        drill=0.4,
    )
    result = plan(snapshot(vias=[signal("SIG", 10, 10)], pads=[gnd_pad]))
    assert not result.units and not result.unfixable


def test_ground_via_of_another_ground_does_not_serve():
    result = plan(snapshot(vias=[signal("SIG", 30, 10), via("GND", 30.5, 10)]))
    assert [u.net for u in result.units] == ["GND_ISO"]


def test_via_in_isolated_ground_area_gets_isolated_ground_return():
    result = plan(snapshot(vias=[signal("SIG_ISO", 30, 10)]))
    assert [u.net for u in result.units] == ["GND_ISO"]
    assert ISO_AREA.contains(disk(result.units[0].position))


def test_ground_via_never_needs_a_return():
    result = plan(snapshot(vias=[via("GND_ISO", 30, 10), via("GND", 5, 5)]))
    assert not result.units and not result.unfixable


def test_via_outside_every_ground_outline_is_unfixable_and_never_forced():
    zones = [zone("GND", INNER, GND_AREA)]
    result = plan(snapshot(vias=[signal("SIG", 35, 10)], zones=zones), force=True)
    assert not result.units
    assert [(s.position, s.net, s.reason) for s in result.unfixable] == [
        ((35, 10), "SIG", "outside every ground zone for F.Cu"),
        ((35, 10), "SIG", "outside every ground zone for B.Cu"),
    ]


def test_outer_ground_zone_serves_where_no_inner_outline_does():
    zones = [
        zone("GND", INNER, GND_AREA),
        zone("GND_ISO", ("F.Cu", "B.Cu"), ISO_AREA),
        zone("GND_ISO", ("In1.Cu", "In2.Cu"), box(0, 0, 1, 1)),
    ]
    result = plan(snapshot(vias=[signal("SIG", 30, 10)], zones=zones))
    assert [u.net for u in result.units] == ["GND_ISO"]


def test_return_via_sits_inside_the_fill_on_every_required_layer():
    # An outer GND pour adds F.Cu to the required layers, filled only left of x=10.
    left = box(0, 0, 10, 20)
    zones = [
        zone("GND", INNER, GND_AREA),
        zone("GND", ("F.Cu",), GND_AREA, fills={"F.Cu": left}),
        zone("GND_ISO", INNER, ISO_AREA),
    ]
    result = plan(snapshot(vias=[signal("SIG", 10, 10)], zones=zones))
    (u,) = result.units
    assert left.contains(disk(u.position))


def test_netclass_via_size_with_override():
    snap = snapshot(vias=[signal("SIG", 10, 10)], via_sizes={"GND": ViaSize(0.7, 0.35)})
    assert plan(snap).units[0].size == ViaSize(0.7, 0.35)
    assert plan(snap, via=ViaSize(0.5, 0.25)).units[0].size == ViaSize(0.5, 0.25)


def test_one_return_via_serves_several_nearby_signal_vias():
    result = plan(snapshot(vias=[signal("A", 10, 10), signal("B", 10, 11.2)]))
    assert len(result.units) == 1


def via_keepout(outline):
    return Zone(
        net=None,
        layers=FOUR_LAYERS,
        priority=0,
        outline=outline,
        is_rule_area=True,
        keepout_vias=True,
    )


def test_return_via_avoids_a_via_keepout_rule_area():
    grounds = [zone("GND", INNER, GND_AREA), zone("GND_ISO", INNER, ISO_AREA)]
    (nearest,) = plan(snapshot(vias=[signal("SIG", 10, 10)], zones=grounds)).units
    keepout = Point(nearest.position).buffer(0.4)

    snap = snapshot(
        vias=[signal("SIG", 10, 10)], zones=[*grounds, via_keepout(keepout)]
    )
    (u,) = plan(snap).units

    assert not disk(u.position).intersects(keepout)


def test_return_vias_of_different_grounds_keep_clearance_from_each_other():
    # Signal vias either side of the boundary between the two grounds, whose
    # fills are strips along it, so both returns land near the boundary.
    strips = {"GND": box(19.1, 0, 20, 20), "GND_ISO": box(20, 0, 20.9, 20)}
    grounds = [
        zone(net, INNER, area, fills=dict.fromkeys(INNER, strips[net]))
        for net, area in (("GND", GND_AREA), ("GND_ISO", ISO_AREA))
    ]
    vias = [signal("A", 19.2, 10), signal("B", 20.8, 10)]
    result = plan(snapshot(vias=vias, zones=grounds))

    gnd, iso = sorted(result.units, key=lambda u: u.net)
    assert (gnd.net, iso.net) == ("GND", "GND_ISO")
    gap = disk(gnd.position).distance(disk(iso.position))
    assert gap >= OTHER_NET_CLEARANCE - EPS


def test_forbidden_position_is_not_reused():
    snap = snapshot(vias=[signal("SIG", 10, 10)])
    (first,) = plan(snap).units
    (second,) = plan(snap, forbidden=[first.position]).units
    assert math.dist(first.position, second.position) >= FORBIDDEN_RADIUS
    assert math.dist(second.position, (10, 10)) <= 1.5 + EPS


@pytest.mark.parametrize("layer", FOUR_LAYERS)
def test_other_net_pad_clearance_override_is_enforced_on_every_layer(layer):
    override = 1.0
    other = pad("X", (11.2, 10), {layer: box(11, 9, 11.4, 11)}, override=override)
    result = plan(snapshot(vias=[signal("SIG", 10, 10)], pads=[other]))
    (u,) = result.units
    assert disk(u.position).distance(other.shapes[layer]) >= override - EPS


def test_mounting_hole_clearance_override_is_enforced():
    hole = pad("", (13, 10), {}, kind=PadKind.NPTH, drill=3.2, override=1.6)
    result = plan(snapshot(vias=[signal("SIG", 10, 10)], pads=[hole]))
    (u,) = result.units
    assert disk(u.position).distance(disk(hole.position, 1.6)) >= 1.6 - EPS


def test_return_via_keeps_clear_of_board_edge():
    # The only room left of the track is too close to the board edge.
    wall = track("X", "F.Cu", (2.1, 5), (2.1, 15), width=1.8)
    result = plan(snapshot(vias=[signal("SIG", 0.6, 10)], tracks=[wall]))
    assert not result.units
    assert [s.position for s in result.unfixable] == [(0.6, 10)]


def test_force_never_places_a_via_in_or_near_a_board_cutout():
    # Other-net copper everywhere within reach except a slot cut in the board:
    # the slot is the only spot with clearance.
    slot = box(10.6, 9.4, 12, 10.6)
    blanket = pad("X", (10, 10), {"F.Cu": box(5, 5, 15, 15).difference(slot)})
    snap = snapshot(vias=[signal("SIG", 10, 10)], pads=[blanket])
    assert plan(snap, force=True).units  # without the cutout the slot is used

    slot_edge = list(slot.exterior.coords)
    cut = replace(snap, edge_paths=[*snap.edge_paths, slot_edge])
    result = plan(cut, force=True)

    assert all(
        Point(u.position).distance(slot) >= EDGE_GAP + VIA_RADIUS - EPS
        for u in result.units
    )
    assert [s.position for s in result.unfixable] == [(10, 10)]


def _crowded(signal_y=10.0, pitch=1.0, width=0.2):
    """Other-net tracks on F.Cu every `pitch` mm, leaving a gap around y=10."""
    rows = [signal_y + 0.5 + k * pitch for k in range(-4, 4)]
    return [track("X", "F.Cu", (5, y), (15, y), width) for y in rows]


def test_crowded_via_without_force_is_unfixable():
    result = plan(snapshot(vias=[signal("SIG", 10, 10)], tracks=_crowded()))
    assert not result.units
    assert [s.position for s in result.unfixable] == [(10, 10)]


def test_force_places_crowded_via_at_maximum_clearance():
    tracks = _crowded()
    result = plan(snapshot(vias=[signal("SIG", 10, 10)], tracks=tracks), force=True)

    (u,) = result.units
    assert u.forced and u.net == "GND"
    assert math.dist(u.position, (10, 10)) <= 1.5 + EPS
    # Tracks 1 mm apart leave 0.8 mm gaps: a 0.6 mm via gets 0.1 mm either side.
    gap = min(disk(u.position).distance(t.geometry) for t in tracks)
    assert gap == pytest.approx(0.1, abs=0.01)
    assert not result.unfixable


def test_force_skips_and_reports_spot_that_would_overlap_other_net_copper():
    blanket = track("X", "F.Cu", (5, 10), (15, 10), width=5)
    result = plan(snapshot(vias=[signal("SIG", 10, 10)], tracks=[blanket]), force=True)
    assert not result.units
    assert [s.position for s in result.unfixable] == [(10, 10)]


def test_force_never_enters_a_mounting_hole_keep_off():
    # The keep-off (hole radius 0.5 + 2.5 override) covers every reachable spot.
    hole = pad("", (10, 11), {}, kind=PadKind.NPTH, drill=1.0, override=2.5)
    result = plan(snapshot(vias=[signal("SIG", 10, 10)], pads=[hole]), force=True)
    assert not result.units
    assert [s.position for s in result.unfixable] == [(10, 10)]


POCKET_CENTRE = (11, 10)


def _pocket(centre=POCKET_CENTRE):
    return box(5, 5, 15, 15).difference(Point(centre).buffer(0.4))


def _blanket(*pockets):
    """Other-net copper around the signal vias except pockets of room.

    Each pocket is the only spot with positive clearance near it, so an
    unrestricted forced pass puts a via at its centre.
    """
    copper = pockets[0]
    for p in pockets[1:]:
        copper = copper.intersection(p)
    return pad("X", (7, 7), {"F.Cu": copper})


def test_force_keeps_hole_to_hole_from_an_existing_via():
    # A ground via 1.75 mm from the signal via doesn't serve it, but its hole
    # is 0.55 mm from the pocket centre.
    gnd = via("GND", 11.75, 10)
    pocket = _pocket((11.2, 10))
    snap = snapshot(vias=[signal("SIG", 10, 10), gnd], pads=[_blanket(pocket)])

    result = plan(snap, force=True)

    for u in result.units:
        hole_gap = math.dist(u.position, gnd.position) - u.size.drill / 2 - 0.15
        assert hole_gap >= HOLE_TO_HOLE_GAP - EPS
    assert result.units or result.unfixable


def test_force_keeps_hole_to_hole_from_an_earlier_forced_via():
    # Signal vias 2.55 mm apart, each with a pocket 1 mm away; the pockets are
    # 0.55 mm apart, so the second forced via lands next to the first.
    pockets = _blanket(_pocket((11, 10)), _pocket((11.55, 10)))
    snap = snapshot(vias=[signal("A", 10, 10), signal("B", 12.55, 10)], pads=[pockets])

    result = plan(snap, force=True)

    holes = [u.position for u in result.units]
    for i, a in enumerate(holes):
        for b in holes[i + 1 :]:
            assert math.dist(a, b) - 0.3 >= HOLE_TO_HOLE_GAP - EPS
    assert len(result.units) + len(result.unfixable) == 2


def test_force_keeps_out_of_same_net_pads():
    gnd_pad = pad("GND", (11.25, 10), {"F.Cu": box(11.15, 9.95, 11.35, 10.05)})
    snap = snapshot(vias=[signal("SIG", 10, 10)], pads=[_blanket(_pocket()), gnd_pad])

    result = plan(snap, force=True)

    for u in result.units:
        gap = disk(u.position).distance(gnd_pad.shapes["F.Cu"])
        assert gap >= SAME_NET_PAD_GAP - EPS
    assert result.units or result.unfixable


def test_force_uses_a_spot_inside_the_zone_outline_but_outside_the_fill():
    # The fill has a hole around the signal via that leaves no full-clearance
    # spot within reach, though fill lies within the distance.
    fill = GND_AREA.difference(Point(10, 10).buffer(1.3))
    grounds = [zone("GND", INNER, GND_AREA, fills=dict.fromkeys(INNER, fill))]
    snap = snapshot(vias=[signal("SIG", 10, 10)], zones=grounds)
    assert not plan(snap).units

    (u,) = plan(snap, force=True).units

    assert u.forced
    assert GND_AREA.contains(disk(u.position))
    assert not fill.intersects(disk(u.position))


def test_one_forced_via_stands_in_for_a_nearby_crowded_via_too():
    vias = [signal("A", 10, 10), signal("B", 10, 11)]
    result = plan(snapshot(vias=vias, tracks=_crowded()), force=True)
    (u,) = result.units
    assert u.forced
    assert [(s.net, s.reason) for s in result.unfixable] == [
        ("B", f"relies on forced via at ({u.position[0]:.3f}, {u.position[1]:.3f})")
    ]


def test_planned_returns_satisfy_the_check():
    # A keep-off around the via leaves room only on the outermost ring, where
    # rounding to 1 µm would push this off-grid position past the distance.
    centre = (10.0006, 10)
    hole = pad("", centre, {}, kind=PadKind.NPTH, drill=0.4, override=0.98)
    snap = snapshot(vias=[signal("SIG", *centre)], pads=[hole])
    result = plan(snap)

    after = returnvia_command.with_returns(snap, result.units)
    assert {r.position for r in unserved(after)} == {
        s.position for s in result.unfixable
    }


def test_planner_acts_on_exactly_the_unserved_vias():
    zones = [zone("GND", INNER, GND_AREA), zone("GND_ISO", INNER, box(20, 0, 30, 20))]
    served = signal("SERVED", 5, 5)
    needs = signal("NEEDS", 10, 10)
    iso = signal("ISO", 25, 10)
    outside = signal("OUTSIDE", 35, 10)
    snap = snapshot(
        vias=[served, via("GND", 5.8, 5), needs, iso, outside, via("GND_ISO", 28, 10)],
        zones=zones,
    )
    options = ReturnViaOptions()

    reported = {r.net for r in unserved(snap, ["GND", "GND_ISO"])}
    result = ReturnViaPlanner()(snap, options, [])
    acted_on = {s.net for s in result.unfixable} | {
        v.net
        for v in (needs, iso)
        if any(math.dist(u.position, v.position) <= 1.5 + EPS for u in result.units)
    }

    assert reported == {"NEEDS", "ISO", "OUTSIDE"}
    assert reported == acted_on


TWO_LAYERS = ("F.Cu", "B.Cu")


def two_layer_snapshot(vias=(), zones=()):
    edge = [[(0, 0), (BOARD_W, 0), (BOARD_W, BOARD_H), (0, BOARD_H), (0, 0)]]
    return Snapshot(
        copper_layers=TWO_LAYERS,
        vias=list(vias),
        tracks=ties(vias),
        zones=list(zones),
        edge_paths=edge,
    )


def test_two_layer_board_falls_back_to_the_net_named_gnd():
    snap = two_layer_snapshot(
        vias=[via("GND", 5, 5), signal("SIG", 5.8, 5), signal("FAR", 15, 15)],
        zones=[zone("GND", TWO_LAYERS, GND_AREA)],
    )

    assert resolve_ground_nets(snap, None) == ["GND"]
    assert {r.net for r in unserved(snap)} == {"FAR"}
    (u,) = plan(snap).units
    assert u.net == "GND"


def test_gnd_with_only_vias_still_counts_as_the_ground():
    snap = two_layer_snapshot(vias=[via("GND", 5, 5), signal("SIG", 5.8, 5)])

    assert resolve_ground_nets(snap, None) == ["GND"]


@pytest.mark.parametrize("ground_nets", [None, ["GND"]])
def test_ground_vias_are_never_unserved(ground_nets):
    snap = snapshot(vias=[via("GND", 10, 10), via("GND", 30, 10)])

    assert unserved(snap, ground_nets) == []


def test_no_ground_net_found_is_an_error():
    snap = two_layer_snapshot(
        vias=[via("SIG", 5, 5)], zones=[zone("PWR", TWO_LAYERS, GND_AREA)]
    )

    with pytest.raises(NoGroundNetsError, match="--ground-nets"):
        resolve_ground_nets(snap, None)
    with pytest.raises(NoGroundNetsError):
        plan(snap)


def test_returnvia_without_a_ground_exits_with_an_error(monkeypatch):
    snap = two_layer_snapshot(vias=[via("SIG", 5, 5)])
    monkeypatch.setattr(
        returnvia_command,
        "run_placement",
        lambda planner, options, settings, kicad: planner(snap, options, []),
    )
    monkeypatch.setattr(returnvia_command, "check_outline", lambda board: None)
    monkeypatch.setattr(
        returnvia_command.kipy,
        "KiCad",
        lambda: type("K", (), {"get_board": lambda self: None})(),
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.execute_returnvia(_parse(monkeypatch, "returnvia"))

    assert exit_info.value.code == 1


def _parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["breakneck", *argv])
    return cli.parse_args()


def test_returnvia_default_distance(monkeypatch):
    assert _parse(monkeypatch, "returnvia").distance == 1.5


def test_gndvia_is_an_unknown_command(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "gndvia")


def test_returnvia_arguments(monkeypatch):
    args = _parse(
        monkeypatch,
        "returnvia",
        "--via",
        "0.5/0.25",
        "--ground-nets",
        "GND",
        "GND_ISO",
        "--force",
    )
    assert args.via == ViaSize(0.5, 0.25)
    assert args.ground_nets == ["GND", "GND_ISO"]
    assert args.force


@pytest.mark.parametrize("bad", ["0.6", "0.3/0.6", "a/b"])
def test_returnvia_rejects_malformed_via_size(monkeypatch, bad):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "returnvia", "--via", bad)


# ---- reference planes ----------------------------------------------------------

GROUNDS = ["GND", "GND_ISO"]
SIG_AT = (10, 10)


def references(snap, ground_nets=GROUNDS):
    return via_references(snap, ground_nets)


def front_track(start=(5, 10), end=SIG_AT, net="SIG"):
    return track(net, "F.Cu", start, end)


def back_track(start=(15, 10), end=SIG_AT, net="SIG"):
    return track(net, "B.Cu", start, end)


def connected(snap):
    (r,) = references(snap)
    return [c.layer for c in r.layers]


def test_via_with_a_same_net_pad_on_the_back_connects_on_both_layers():
    back_pad = pad("SIG", (10.5, 10), {"B.Cu": box(10.2, 9.7, 10.8, 10.3)})
    snap = snapshot(vias=[via("SIG", *SIG_AT)], tracks=[front_track()], pads=[back_pad])

    assert connected(snap) == ["F.Cu", "B.Cu"]


def test_via_in_its_own_nets_pour_connects_on_the_pour_layer():
    pour = zone("SIG", ["In2.Cu"], box(8, 8, 12, 12))
    snap = snapshot(
        vias=[via("SIG", *SIG_AT)],
        tracks=[front_track()],
        zones=[zone("GND", INNER, GND_AREA), pour],
    )

    assert connected(snap) == ["F.Cu", "In2.Cu"]


def test_track_touching_the_via_edge_on_connects():
    edge_on = 10 + VIA_RADIUS + 0.1  # track half-width 0.1
    snap = snapshot(
        vias=[via("SIG", *SIG_AT)],
        tracks=[front_track(), back_track(start=(15, 10), end=(edge_on, 10))],
    )

    assert connected(snap) == ["F.Cu", "B.Cu"]


# Copper that must not add a layer to the via's F.Cu track.
MISSES = {
    "track-short-of-the-disc": {
        "tracks": [track("SIG", "B.Cu", (15, 10), (10.45, 10))]
    },
    "other-net-track": {"tracks": [track("X", "B.Cu", (15, 10), SIG_AT)]},
    "pad-on-another-layer": {"pads": [pad("SIG", SIG_AT, {"F.Cu": disk(SIG_AT, 0.5)})]},
    "pour-fill-short-of-the-disc": {
        "zones": [
            zone("GND", INNER, GND_AREA),
            zone(
                "SIG",
                ["In2.Cu"],
                box(8, 8, 12, 12),
                fills={"In2.Cu": box(10.35, 8, 12, 12)},
            ),
        ]
    },
}


@pytest.mark.parametrize("miss", MISSES.values(), ids=MISSES.keys())
def test_copper_that_misses_the_via_does_not_connect(miss):
    extra = dict(miss)
    tracks = [front_track(), *extra.pop("tracks", [])]
    snap = snapshot(vias=[via("SIG", *SIG_AT)], tracks=tracks, **extra)

    (r,) = references(snap)
    assert [c.layer for c in r.layers] == ["F.Cu"]
    assert r.exempt


def test_floating_via_is_exempt():
    (r,) = references(snapshot(vias=[via("SIG", *SIG_AT)]))

    assert r.layers == ()
    assert r.exempt
    assert r.planes == ()


@pytest.mark.parametrize("net", ["GND", "GND_ISO", ""])
def test_ground_and_netless_vias_are_not_considered(net):
    snap = snapshot(
        vias=[via(net, *SIG_AT)],
        tracks=[front_track(net=net), back_track(net=net)],
    )

    assert references(snap) == []


def through_via(zones=None, ground_nets=GROUNDS):
    """The plane per connected layer of a via with tracks on F.Cu and B.Cu."""
    snap = snapshot(
        vias=[via("SIG", *SIG_AT)], tracks=[front_track(), back_track()], zones=zones
    )
    (r,) = via_references(snap, ground_nets)
    return r


def planes_by_layer(r):
    return {c.layer: c.plane for c in r.layers}


def test_each_end_of_a_through_via_references_the_nearest_ground_plane():
    r = through_via(ground_nets=None)

    assert planes_by_layer(r) == {
        "F.Cu": ReferencePlane("In1.Cu", "GND"),
        "B.Cu": ReferencePlane("In2.Cu", "GND"),
    }
    assert not r.exempt
    assert len(r.planes) == 2


def test_layer_whose_ground_outline_misses_the_via_is_skipped():
    r = through_via(
        zones=[
            zone("GND", ["In1.Cu"], box(30, 0, 40, 20)),
            zone("GND", ["In2.Cu"], GND_AREA),
        ]
    )

    assert planes_by_layer(r)["F.Cu"] == ReferencePlane("In2.Cu", "GND")


def test_plane_takes_the_ground_net_of_its_own_zone():
    r = through_via(
        zones=[
            zone("GND_ISO", ["In1.Cu"], box(5, 5, 15, 15)),
            zone("GND", ["In2.Cu"], GND_AREA),
        ]
    )

    assert planes_by_layer(r) == {
        "F.Cu": ReferencePlane("In1.Cu", "GND_ISO"),
        "B.Cu": ReferencePlane("In2.Cu", "GND"),
    }


def test_overlapping_ground_outlines_resolve_to_the_highest_priority():
    r = through_via(
        zones=[
            zone("GND", INNER, GND_AREA),
            zone("GND_ISO", ["In1.Cu"], box(5, 5, 15, 15), priority=1),
        ]
    )

    assert planes_by_layer(r)["F.Cu"] == ReferencePlane("In1.Cu", "GND_ISO")


def test_two_ends_on_the_same_plane_are_one_measurement():
    r = through_via(
        zones=[
            zone("GND", ["In1.Cu"], GND_AREA),
            zone("GND", ["In2.Cu"], box(30, 0, 40, 20)),
        ]
    )

    assert planes_by_layer(r) == {
        "F.Cu": ReferencePlane("In1.Cu", "GND"),
        "B.Cu": ReferencePlane("In1.Cu", "GND"),
    }
    assert r.planes == (ReferencePlane("In1.Cu", "GND"),)


def test_end_with_no_ground_on_any_other_layer_is_outside_ground():
    r = through_via(
        zones=[zone("GND", ["F.Cu"], GND_AREA)],
        ground_nets=["GND"],
    )

    assert planes_by_layer(r) == {
        "F.Cu": None,
        "B.Cu": ReferencePlane("F.Cu", "GND"),
    }
    assert r.planes == (ReferencePlane("F.Cu", "GND"),)


def test_inner_layer_equidistant_from_two_planes_takes_the_upper_one():
    snap = snapshot(
        vias=[via("SIG", *SIG_AT)],
        tracks=[track("SIG", "In1.Cu", (5, 10), SIG_AT), back_track()],
        zones=[zone("GND", ["F.Cu", "In2.Cu"], GND_AREA)],
    )

    (r,) = via_references(snap, ["GND"])
    assert planes_by_layer(r)["In1.Cu"] == ReferencePlane("F.Cu", "GND")
