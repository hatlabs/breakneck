import math

from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from breakneck import drc
from breakneck.placement import MarkerKind, markers_for
from breakneck.returnpath import Outcome, measure_returns
from breakneck.returnvia import ReturnViaOptions
from breakneck.returnvia_command import with_returns
from breakneck.returnvia_planner import ReturnViaPlanner
from breakneck.snapshot import Pad, PadKind, Snapshot, Track, Via, ViaSize, Zone

FOUR_LAYERS = ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
INNER = ("In1.Cu", "In2.Cu")
BOARD = box(0, 0, 40, 20)
DISTANCE = 1.5
SIG = (10.0, 10.0)
VIA_RADIUS = 0.3
ANTIPAD = VIA_RADIUS + 0.2
# Thicker than a return's passable disc reaches past its own copper, so no
# return can bridge it.
WALL = 0.5
TWO_GROUNDS = ("GND", "GND_ISO")


def via(net, pos):
    return Via(id=f"{net}@{pos}", net=net, position=pos, diameter=0.6, drill=0.3)


def signal_via(pos, net="SIG"):
    """A via tied to F.Cu and B.Cu by stubs within its own copper."""
    x, y = pos
    stub = LineString([pos, (x - 0.1, y)])
    tracks = [Track(f"{net}:{L}@{pos}", net, L, 0.2, stub) for L in ("F.Cu", "B.Cu")]
    return via(net, pos), tracks


def holes(*discs):
    return unary_union([Point(p).buffer(r) for p, r in discs])


def flush(*signals):
    return BOARD.difference(holes(*((s, ANTIPAD) for s in signals)))


def gnd_zone(fills, net="GND", outline=BOARD):
    return Zone(net=net, layers=tuple(fills), priority=0, outline=outline, fills=fills)


def board(
    signals, returns=(), fills=None, zones=None, pads=(), tracks=(), via_sizes=None
):
    """Signal vias over GND planes on In1 and In2, by default the board
    minus each signal via's antipad."""
    vias, ties = [], []
    for s in signals:
        v, t = signal_via(s)
        vias.append(v)
        ties += t
    if zones is None:
        zones = [gnd_zone(fills or dict.fromkeys(INNER, flush(*signals)))]
    return Snapshot(
        copper_layers=FOUR_LAYERS,
        vias=vias + [via(n, p) for n, p in returns],
        tracks=ties + list(tracks),
        zones=zones,
        pads=list(pads),
        edge_paths=[list(BOARD.exterior.coords)],
        via_sizes=via_sizes or {},
    )


def plan(snap, forbidden=(), **kwargs):
    return ReturnViaPlanner()(snap, ReturnViaOptions(**kwargs), list(forbidden))


def check(snap, result, ground_nets=None):
    """The `--check` measurement of the board with the planned vias."""
    return measure_returns(with_returns(snap, result.units), ground_nets, DISTANCE)


def unserved(results):
    return {r.position for r in results if not r.served}


def other_net_copper(area):
    return Pad("X@", None, "X", PadKind.SMD, "1", (0, 0), 0.0, {"F.Cu": area})


def slotted_in1(returns=(("GND", (SIG[0] + 1.4, SIG[1])),), pads=()):
    """A thin slot in In1 between SIG and the east; In2 is whole."""
    slot = box(SIG[0] + 0.6, SIG[1] - 2, SIG[0] + 0.7, SIG[1] + 2)
    fills = {"In1.Cu": flush(SIG).difference(slot), "In2.Cu": flush(SIG)}
    return board([SIG], returns, fills=fills, pads=pads), slot


def test_slot_in_one_plane_gets_a_return_on_its_near_side():
    snap, slot = slotted_in1()
    before = {r.plane.layer: r.served for r in measure_returns(snap, None, DISTANCE)}
    assert before == {"In1.Cu": False, "In2.Cu": True}

    result = plan(snap)

    (u,) = result.units
    assert Point(u.position).buffer(VIA_RADIUS).bounds[2] <= slot.bounds[0]
    assert not unserved(check(snap, result))


def test_spot_across_a_slot_is_rejected_even_with_clearance():
    # Other-net copper leaves room only across the slot, where In1 has no
    # short path back to the via.
    near_side = box(5, 5, SIG[0] + 0.8, 15).difference(holes((SIG, VIA_RADIUS)))
    snap, _ = slotted_in1(returns=(), pads=[other_net_copper(near_side)])

    result = plan(snap)

    assert not result.units
    assert [s.position for s in result.unfixable] == [SIG]


def test_ends_failing_on_two_ground_nets_get_one_via_each():
    # GND under F.Cu, GND_ISO under B.Cu; each also pours an outer layer so
    # a return via of it reaches two layers.
    plain = flush(SIG)
    zones = [
        gnd_zone(dict.fromkeys(("F.Cu", "In1.Cu"), plain)),
        gnd_zone(dict.fromkeys(("In2.Cu", "B.Cu"), plain), net="GND_ISO"),
    ]
    snap = board([SIG], zones=zones)

    result = plan(snap, ground_nets=TWO_GROUNDS)

    assert sorted(u.net for u in result.units) == ["GND", "GND_ISO"]
    assert not unserved(check(snap, result, TWO_GROUNDS))


def test_spot_on_the_outermost_ring_passes_the_check():
    # A keep-off leaves room only on the last ring, 1.55 mm out.
    hole = Pad("h", None, "", PadKind.NPTH, "", SIG, 0.4, {}, clearance_override=0.92)
    snap = board([SIG], pads=[hole])

    result = plan(snap)

    (u,) = result.units
    assert math.dist(u.position, SIG) > DISTANCE
    assert not unserved(check(snap, result))


def test_via_served_by_a_via_planned_earlier_gets_none_of_its_own():
    later = (SIG[0] + 0.8, SIG[1] + 0.9)
    snap = board([SIG, later])

    result = plan(snap)

    assert len(result.units) == 1
    assert not unserved(check(snap, result))


def test_search_covers_only_the_planes_earlier_vias_leave_unserved():
    # A's via, planned first, serves B on In1 through the open west side of
    # a C-shaped wall around B, but not on In2, where a wall west of B
    # blocks it. Every spot left for B lies outside the C, so it serves B on
    # In2 only: a search over both planes would find none.
    b = (20.0, 10.0)
    a = (b[0] - 2.35, b[1])
    inner = box(b[0] - 0.6, b[1] - 0.6, b[0] + 0.6, b[1] + 0.6)
    outer = inner.buffer(WALL, join_style="mitre")
    c_wall = outer.difference(inner).difference(box(0, 0, b[0] - 0.6, 20))
    west_wall = box(b[0] - 0.6 - WALL, b[1] - 2, b[0] - 0.6, b[1] + 2)
    fills = {
        "In1.Cu": flush(a, b).difference(c_wall),
        "In2.Cu": flush(a, b).difference(west_wall),
    }
    snap = board([a, b], fills=fills)

    result = plan(snap)

    first, second = result.units
    assert math.dist(first.position, a) < math.dist(first.position, b)
    assert not outer.contains(Point(second.position))
    assert not unserved(check(snap, result))
    alone = check(snap, type(result)(units=(second,)))
    assert {r.plane.layer for r in alone if r.position == b and not r.served} == {
        "In1.Cu"
    }


def test_prune_rounds_plan_as_a_fresh_planner_given_the_forbidden_spots():
    snap = board([SIG])
    options = ReturnViaOptions()
    planner = ReturnViaPlanner()

    (first,) = planner(snap, options, []).units
    second = planner(snap, options, [first.position])

    assert second.units and second.units[0].position != first.position
    assert second == ReturnViaPlanner()(snap, options, [first.position])


def crowded():
    """Other-net F.Cu tracks every 1 mm, leaving SIG only 0.8 mm gaps."""
    rows = [SIG[1] + 0.5 + k for k in range(-4, 4)]
    return [
        Track(f"X{y}", "X", "F.Cu", 0.2, LineString([(5, y), (15, y)])) for y in rows
    ]


def test_forced_via_leaves_its_via_unserved_and_marked():
    snap = board([SIG], tracks=crowded())

    result = plan(snap, force=True)

    (u,) = result.units
    assert u.forced
    assert unserved(check(snap, result)) == {SIG}
    prune = drc.PruneResult(result, (), (), (), 1)
    assert (u.position, MarkerKind.FORCED) in {
        (m.position, m.kind) for m in markers_for(prune, True)
    }


def test_no_ground_fill_end_is_unfixable_and_never_planned():
    # In1 has a GND outline around SIG but no fill; GND pours In2 and B.Cu.
    around = box(SIG[0] - 0.8, SIG[1] - 0.8, SIG[0] + 0.8, SIG[1] + 0.8)
    zones = [
        gnd_zone({"In1.Cu": Polygon()}, outline=around),
        gnd_zone(dict.fromkeys(("In2.Cu", "B.Cu"), flush(SIG))),
    ]
    snap = board([SIG], zones=zones)

    result = plan(snap, force=True)

    (u,) = result.units
    assert not u.forced
    assert [(s.position, s.reason) for s in result.unfixable] == [
        (SIG, "no GND fill at the via on In1.Cu")
    ]
    after = {r.plane.layer: r for r in check(snap, result)}
    assert after["In1.Cu"].outcome is Outcome.NO_GROUND_FILL
    assert after["In2.Cu"].served


def test_end_outside_every_ground_zone_is_unfixable_and_the_other_end_planned():
    # GND pours F.Cu, and In1 everywhere but around SIG: the F.Cu end finds
    # no plane, the B.Cu end references the F.Cu pour.
    around = box(SIG[0] - 0.6, SIG[1] - 0.6, SIG[0] + 0.6, SIG[1] + 0.6)
    zones = [
        gnd_zone({"F.Cu": flush(SIG)}),
        gnd_zone({"In1.Cu": flush(SIG)}, outline=BOARD.difference(around)),
    ]
    snap = board([SIG], zones=zones)

    result = plan(snap, ground_nets=["GND"], force=True)

    (u,) = result.units
    assert not u.forced
    assert [(s.position, s.reason) for s in result.unfixable] == [
        (SIG, "outside every ground zone for F.Cu")
    ]
    after = check(snap, result, ["GND"])
    assert [(r.layers, r.served) for r in after] == [
        (("B.Cu",), True),
        (("F.Cu",), False),
    ]


def test_planned_run_leaves_only_the_unfixable_vias_unserved():
    crowded_via = (SIG[0], SIG[1])
    slotted = (25.0, 10.0)
    plain = (30.0, 5.0)
    slot = box(slotted[0] + 0.6, slotted[1] - 2, slotted[0] + 0.7, slotted[1] + 2)
    fill = flush(crowded_via, slotted, plain)
    snap = board(
        [crowded_via, slotted, plain],
        returns=[("GND", (slotted[0] + 1.4, slotted[1]))],
        fills={"In1.Cu": fill.difference(slot), "In2.Cu": fill},
        tracks=crowded(),
    )

    result = plan(snap)

    assert {s.position for s in result.unfixable} == {crowded_via}
    assert len(result.units) == 2
    assert unserved(check(snap, result)) == {crowded_via}


def pocketed(*pockets, signals=(SIG,)):
    """Other-net F.Cu copper around `signals` except the given pockets."""
    vias = tuple((p, VIA_RADIUS) for p in signals)
    return other_net_copper(box(4, 4, 18, 16).difference(holes(*vias, *pockets)))


def test_pending_via_served_by_a_via_planned_later_is_not_unfixable():
    # A has no spot of its own; B's via, planned later, serves it too.
    a = SIG
    b_via = (
        round(a[0] + 1.225 * math.cos(math.radians(5)), 3),
        round(a[1] + 1.225 * math.sin(math.radians(5)), 3),
    )
    b = (round(b_via[0] + 1.0, 3), b_via[1])
    snap = board([a, b], pads=[pocketed((b_via, 0.63), signals=(a, b))])

    result = plan(snap)

    assert [u.position for u in result.units] == [b_via]
    assert not result.unfixable
    assert not unserved(check(snap, result))


def slot_between(a, b, pockets):
    """A wide slot in both planes east of `a`; room only in `pockets`."""
    slot = box(a[0] + 0.6, a[1] - 4, a[0] + 1.1, a[1] + 4)
    fills = dict.fromkeys(INNER, flush(a, b).difference(slot))
    return board([a, b], fills=fills, pads=[pocketed(*pockets, signals=(a, b))])


def test_full_clearance_via_never_stands_in_for_another():
    # B's via sits across the slot from A, within a straight 1.5 mm of it.
    a, b, b_via = SIG, (SIG[0] + 2.5, SIG[1]), (SIG[0] + 1.5, SIG[1])
    snap = slot_between(a, b, [(b_via, 0.65)])

    result = plan(snap)

    assert [u.position for u in result.units] == [b_via]
    assert [(s.position, s.reason) for s in result.unfixable] == [
        (a, "no clear spot reaches the plane within 1.5 mm")
    ]


def test_forced_via_of_another_ground_never_stands_in():
    a, b = SIG, (SIG[0] + 1.2, SIG[1])
    left, right = box(0, 0, 10.6, 20), box(10.6, 0, 40, 20)
    plain = flush(a, b)
    zones = [
        gnd_zone({L: plain.intersection(left) for L in INNER}, outline=left),
        gnd_zone(
            {L: plain.intersection(right) for L in INNER}, "GND_ISO", outline=right
        ),
    ]
    pockets = pocketed(((10.2, 10.9), 0.45), ((12.2, 10.0), 0.45), signals=(a, b))
    snap = board([b, a], zones=zones, pads=[pockets])

    result = plan(snap, ground_nets=TWO_GROUNDS, force=True)

    assert sorted((u.net, u.forced) for u in result.units) == [
        ("GND", True),
        ("GND_ISO", True),
    ]
    assert not result.unfixable


def test_via_relying_on_a_forced_via_is_unfixable_and_names_it():
    # A's forced via lands across the slot from A, beside B, which then has
    # no spot of its own.
    a, b = SIG, (SIG[0] + 2.5, SIG[1])
    snap = slot_between(a, b, [((11.5, 10.0), 0.45), ((13.5, 10.0), 0.45)])

    result = plan(snap, force=True)

    (u,) = result.units
    assert u.forced
    x, y = u.position
    assert [(s.position, s.reason) for s in result.unfixable] == [
        (b, f"relies on forced via at ({x:.3f}, {y:.3f})")
    ]
    prune = drc.PruneResult(result, (), (), (), 1)
    assert b in {m.position for m in markers_for(prune, True)}


def test_no_clear_spot_is_no_room():
    snap = board([SIG], pads=[pocketed()])

    result = plan(snap)

    assert [(s.reason, s.layer) for s in result.unfixable] == [
        ("no room for a return via", "In1.Cu, In2.Cu")
    ]


def thermal_only(centre, hole):
    """GND pours In2 and B.Cu. In1's GND fill and outline stop `hole` short
    of `centre`, so a via there reaches In1 only through its thermal reach.
    Other-net copper leaves room for a via only at `centre`."""
    gap = Point(centre).buffer(hole)
    zones = [
        gnd_zone({"In1.Cu": flush(SIG).difference(gap)}, outline=BOARD.difference(gap)),
        gnd_zone(dict.fromkeys(("In2.Cu", "B.Cu"), flush(SIG))),
    ]
    return board([SIG], zones=zones, pads=[pocketed((centre, VIA_RADIUS + 0.35))])


def test_return_whose_centre_is_off_the_fill_serves_through_its_thermal_reach():
    centre = (SIG[0] + 1.0, SIG[1])
    snap = thermal_only(centre, VIA_RADIUS + 0.1)

    result = plan(snap)

    assert [u.position for u in result.units] == [centre]
    assert not unserved(check(snap, result))


def test_return_reach_follows_its_own_ground_nets_via_size():
    # GND_ISO's larger via reaches In1's fill past the gap; GND's would not.
    centre = (30.0 + 1.2, SIG[1])
    sig = (30.0, SIG[1])
    sizes = {"GND": ViaSize(0.6, 0.3), "GND_ISO": ViaSize(1.0, 0.5)}
    gap = Point(centre).buffer(0.75)
    left, right = box(0, 0, 20, 20), box(20, 0, 40, 20)
    plain = flush(sig)
    zones = [
        gnd_zone(
            dict.fromkeys(("In1.Cu", "In2.Cu", "B.Cu"), plain.intersection(left)),
            outline=left,
        ),
        gnd_zone(
            {"In1.Cu": plain.intersection(right).difference(gap)},
            "GND_ISO",
            right.difference(gap),
        ),
        gnd_zone(
            dict.fromkeys(("In2.Cu", "B.Cu"), plain.intersection(right)),
            "GND_ISO",
            right,
        ),
    ]
    room = holes((sig, VIA_RADIUS), (centre, 0.85))
    pocket = other_net_copper(box(24, 4, 38, 16).difference(room))
    snap = board([sig], zones=zones, pads=[pocket], via_sizes=sizes)

    result = plan(snap, ground_nets=TWO_GROUNDS)

    (u,) = result.units
    assert (u.net, u.position, u.size) == ("GND_ISO", centre, sizes["GND_ISO"])
    assert not unserved(check(snap, result, TWO_GROUNDS))
