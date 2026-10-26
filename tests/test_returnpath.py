import math

import pytest
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from breakneck.plan import QUANTISATION
from breakneck.returnpath import (
    NoBoardOutlineError,
    Outcome,
    PlaneReturn,
    measure_returns,
)
from breakneck.returnvia import ReferencePlane
from breakneck.snapshot import Pad, PadKind, Snapshot, Track, Via, Zone

FOUR_LAYERS = ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
INNER = ("In1.Cu", "In2.Cu")
BOARD = box(0, 0, 40, 20)
DISTANCE = 1.5
SIG = (10.0, 10.0)
VIA_RADIUS = 0.3
ANTIPAD = VIA_RADIUS + 0.2
# 8-connected steps overstate a straight run by up to about 8 %.
STEP_ERROR = 0.08


def via(net, pos, dia=2 * VIA_RADIUS):
    return Via(id=f"{net}@{pos}", net=net, position=pos, diameter=dia, drill=0.3)


def signal_via(pos=SIG, net="SIG"):
    """A via joining F.Cu and B.Cu tracks: its ends reference In1 and In2."""
    x, y = pos
    tracks = [
        Track(f"{net}:{L}@{pos}", net, L, 0.2, LineString([pos, (x - 3, y)]))
        for L in ("F.Cu", "B.Cu")
    ]
    return via(net, pos), tracks


def holes(*discs):
    return unary_union([Point(p).buffer(r) for p, r in discs])


def gnd_zone(fills, net="GND", outline=BOARD, layers=INNER):
    return Zone(net=net, layers=tuple(layers), priority=0, outline=outline, fills=fills)


def board(signals, returns=(), fills=None, zones=None, pads=(), edge=True):
    """Signal vias with F.Cu/B.Cu tracks over GND planes on In1 and In2.

    By default both planes are the board minus each signal via's antipad.
    """
    vias, tracks = [], []
    for s in signals:
        v, t = signal_via(s)
        vias.append(v)
        tracks += t
    if zones is None:
        if fills is None:
            plain = BOARD.difference(holes(*((s, ANTIPAD) for s in signals)))
            fills = {L: plain for L in INNER}
        zones = [gnd_zone(fills)]
    outline = [[(0, 0), (40, 0), (40, 20), (0, 20), (0, 0)]] if edge else []
    return Snapshot(
        copper_layers=FOUR_LAYERS,
        vias=vias + [via(n, p) for n, p in returns],
        tracks=tracks,
        zones=zones,
        pads=list(pads),
        edge_paths=outline,
    )


def measure(snap, distance=DISTANCE) -> dict[str, PlaneReturn]:
    """Results of the via at SIG by plane layer."""
    return {
        r.plane.layer: r
        for r in measure_returns(snap, None, distance)
        if r.position == SIG and r.plane
    }


def polar(r, degrees, centre=SIG):
    a = math.radians(degrees)
    return (centre[0] + r * math.cos(a), centre[1] + r * math.sin(a))


def assert_close(measured, straight):
    assert abs(measured - straight) <= QUANTISATION + STEP_ERROR * straight


@pytest.mark.parametrize("degrees", [0, 22.5, 45])
@pytest.mark.parametrize("r, served", [(1.4, True), (1.7, False)])
def test_unobstructed_plane_measures_the_straight_distance(degrees, r, served):
    results = measure(board([SIG], [("GND", polar(r, degrees))]))

    assert set(results) == set(INNER)
    for res in results.values():
        assert res.outcome is Outcome.MEASURED
        assert res.served is served
        assert_close(res.distance, r)


def test_ground_via_at_exactly_the_distance_on_an_axis_is_served():
    results = measure(board([SIG], [("GND", (SIG[0] + DISTANCE, SIG[1]))]))
    assert all(r.served for r in results.values())


def test_wide_zone_clearance_still_measures_from_the_void_edge():
    wide = BOARD.difference(holes((SIG, VIA_RADIUS + 0.5)))
    snap = board([SIG], [("GND", polar(1.4, 0))], fills={L: wide for L in INNER})

    for res in measure(snap).values():
        assert res.served
        assert_close(res.distance, 1.4)


def test_row_of_vias_with_merged_antipads_measures_across_the_void():
    row = [(SIG[0] + 0.8 * k, SIG[1]) for k in range(-2, 3)]
    ret = (SIG[0], SIG[1] + 1.0)
    snap = board(row, [("GND", ret)])

    results = [r for r in measure_returns(snap, None, DISTANCE) if r.plane]
    assert len(results) == 2 * len(row)
    for res in results:
        straight = math.dist(res.position, ret)
        assert res.outcome is Outcome.MEASURED
        assert_close(res.distance, straight)
        assert res.served is (straight <= DISTANCE)


def test_return_beyond_a_large_pad_antipad_is_measured_across_the_void():
    pad_antipad = box(SIG[0] + 0.3, SIG[1] - 1.5, SIG[0] + 1.3, SIG[1] + 1.5)
    fill = BOARD.difference(holes((SIG, ANTIPAD)).union(pad_antipad))
    snap = board([SIG], [("GND", polar(1.4, 0))], fills={L: fill for L in INNER})

    for res in measure(snap).values():
        assert res.served
        assert_close(res.distance, 1.4)


def test_island_nearer_than_the_main_pour_does_not_hide_the_pour():
    island = box(SIG[0] + 0.55, SIG[1] - 0.2, SIG[0] + 0.75, SIG[1] + 0.2)
    fill = BOARD.difference(holes((SIG, 1.0))).union(island)
    snap = board([SIG], [("GND", polar(1.4, 90))], fills={L: fill for L in INNER})

    for res in measure(snap).values():
        assert res.served
        assert_close(res.distance, 1.4)


def test_thermal_relief_return_is_measured_from_its_centre():
    ret = polar(1.4, 0)
    fill = BOARD.difference(holes((SIG, ANTIPAD), (ret, VIA_RADIUS + 0.3)))
    snap = board([SIG], [("GND", ret)], fills={L: fill for L in INNER})

    for res in measure(snap).values():
        assert res.served
        assert_close(res.distance, 1.4)


def test_ground_via_without_fill_within_thermal_reach_is_no_return():
    ret = polar(1.2, 0)
    gap = BOARD.difference(holes((SIG, ANTIPAD), (ret, VIA_RADIUS + 0.5)))
    flush = BOARD.difference(holes((SIG, ANTIPAD)))
    snap = board([SIG], [("GND", ret)], fills={"In1.Cu": gap, "In2.Cu": flush})

    results = measure(snap)
    assert results["In1.Cu"].outcome is Outcome.MEASURED
    assert results["In1.Cu"].distance == math.inf
    assert not results["In1.Cu"].served
    assert results["In2.Cu"].served


def test_no_fill_near_the_via_is_its_own_outcome():
    far = box(15, 0, 20, 20)
    flush = BOARD.difference(holes((SIG, ANTIPAD)))
    snap = board(
        [SIG], [("GND", polar(1.4, 0))], fills={"In1.Cu": far, "In2.Cu": flush}
    )

    results = measure(snap)
    assert results["In1.Cu"].outcome is Outcome.NO_GROUND_FILL
    assert not results["In1.Cu"].served
    assert results["In1.Cu"].reason == "no GND fill at the via on In1.Cu"
    assert results["In2.Cu"].served


def test_slot_in_one_plane_fails_only_the_end_that_references_it():
    # The return's passable disc (radius + THERMAL_REACH) must not bridge
    # the slot, so the return sits 1.4 mm away rather than closer.
    ret = polar(1.4, 0)
    slot = box(SIG[0] + 0.6, SIG[1] - 2, SIG[0] + 0.7, SIG[1] + 2)
    flush = BOARD.difference(holes((SIG, ANTIPAD)))
    snap = board(
        [SIG], [("GND", ret)], fills={"In1.Cu": flush.difference(slot), "In2.Cu": flush}
    )

    results = measure(snap)
    assert not results["In1.Cu"].served
    assert results["In1.Cu"].distance > 2 * 2
    assert results["In2.Cu"].served
    assert_close(results["In2.Cu"].distance, 1.4)


# Midpoint and inward direction of each board outline edge.
EDGES = {
    "left": ((0.0, 10.0), (1, 0)),
    "right": ((40.0, 10.0), (-1, 0)),
    "top": ((20.0, 0.0), (0, 1)),
    "bottom": ((20.0, 20.0), (0, -1)),
}
OPPOSITE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
# Less than the return's passable disc radius, so the disc still reaches fill.
PAST_EDGE = 0.5
INSET = 1.0


def inset(edge, d):
    (x, y), (dx, dy) = EDGES[edge]
    return (x + d * dx, y + d * dy)


def results_at(snap, pos):
    return [r for r in measure_returns(snap, None, DISTANCE) if r.position == pos]


@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("offset", [0.0, PAST_EDGE])
def test_return_on_or_past_the_outline_serves_a_via_inside(edge, offset):
    sig = inset(edge, INSET)
    snap = board([sig], [("GND", inset(edge, -offset))])

    results = results_at(snap, sig)
    assert len(results) == len(INNER)
    assert all(r.served for r in results)


@pytest.mark.parametrize("edge", EDGES)
def test_return_past_the_outline_does_not_serve_the_far_side(edge):
    sig = inset(OPPOSITE[edge], INSET)
    snap = board([sig], [("GND", inset(edge, -PAST_EDGE))])

    results = results_at(snap, sig)
    assert len(results) == len(INNER)
    assert not any(r.served for r in results)


def test_ground_via_of_another_net_does_not_serve_the_plane():
    iso = box(5, 5, 15, 15)
    antipads = holes((SIG, ANTIPAD), (polar(1.0, 0), ANTIPAD))
    zones = [
        gnd_zone({"In1.Cu": iso.difference(antipads)}, "GND_ISO", iso, ["In1.Cu"]),
        gnd_zone({"In2.Cu": BOARD.difference(holes((SIG, ANTIPAD)))}, "GND"),
    ]
    snap = board([SIG], [("GND", polar(1.0, 0))], zones=zones)

    results = measure_returns(snap, ["GND", "GND_ISO"], DISTANCE)
    by_layer = {r.plane.layer: r for r in results if r.plane}
    assert by_layer["In1.Cu"].plane == ReferencePlane("In1.Cu", "GND_ISO")
    assert not by_layer["In1.Cu"].served
    assert by_layer["In2.Cu"].plane == ReferencePlane("In2.Cu", "GND")
    assert by_layer["In2.Cu"].served


def test_ground_pth_pad_is_a_return_but_an_smd_pad_is_not():
    pos = polar(1.4, 0)
    shapes = {L: Point(pos).buffer(0.5) for L in FOUR_LAYERS}

    def pad(kind, drill):
        return Pad("p", None, "GND", kind, "1", pos, drill, shapes)

    pth = measure(board([SIG], pads=[pad(PadKind.PTH, 0.4)]))
    smd = measure(board([SIG], pads=[pad(PadKind.SMD, 0.0)]))

    assert all(r.served for r in pth.values())
    assert not any(r.served for r in smd.values())


def test_two_ends_on_one_plane_are_one_measurement():
    zones = [
        gnd_zone({"In1.Cu": BOARD.difference(holes((SIG, ANTIPAD)))}, layers=["In1.Cu"])
    ]
    snap = board([SIG], [("GND", polar(1.0, 0))], zones=zones)

    (res,) = measure_returns(snap, ["GND"], DISTANCE)
    assert res.layers == ("F.Cu", "B.Cu")
    assert res.plane == ReferencePlane("In1.Cu", "GND")
    assert res.served


def test_layer_without_reference_plane_is_unserved_outside_ground():
    zones = [
        gnd_zone({L: box(20, 0, 40, 20) for L in INNER}, outline=box(20, 0, 40, 20))
    ]
    snap = board([SIG], zones=zones)

    results = measure_returns(snap, ["GND"], DISTANCE)
    assert [(r.layers, r.outcome, r.served) for r in results] == [
        (("F.Cu",), Outcome.OUTSIDE_GROUND, False),
        (("B.Cu",), Outcome.OUTSIDE_GROUND, False),
    ]
    assert results[0].plane is None
    assert results[0].reason == "outside every ground zone for F.Cu"


def test_exempt_via_is_not_measured():
    lone = via("SIG", SIG)
    snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        vias=[lone],
        zones=[gnd_zone({L: BOARD for L in INNER})],
        edge_paths=[[(0, 0), (40, 0), (40, 20), (0, 20), (0, 0)]],
    )
    assert measure_returns(snap, None, DISTANCE) == []


def test_board_without_outline_raises_a_typed_error():
    with pytest.raises(NoBoardOutlineError):
        measure_returns(board([SIG], edge=False), None, DISTANCE)
