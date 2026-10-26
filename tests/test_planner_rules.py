"""padvia and returnvia must accept and reject the same via spots."""

import pytest
from shapely.geometry import box

from breakneck import padvia, returnvia
from breakneck.plan import OTHER_NET_CLEARANCE, OVERRIDE_MARGIN, pad_clearance
from breakneck.snapshot import Pad, PadKind, Snapshot, ViaSize, Zone

LAYERS = ("F.Cu", "B.Cu")
BOARD = box(0, 0, 30, 20)
VIA = ViaSize(0.6, 0.3)
HOLE_CENTRE = (10.0, 10.0)
HOLE_DRILL = 3.0
MOUNTING_HOLE_OVERRIDE = 1.6
KEEPOUT = box(18, 4, 22, 8)
STEP = 0.01


def gnd_zone():
    return Zone("GND", LAYERS, 0, BOARD, {L: BOARD for L in LAYERS})


def own_pad():
    """A far-away GND SMD pad; padvia judges spots for some pad."""
    return Pad(
        id="own",
        footprint_id=None,
        net="GND",
        kind=PadKind.SMD,
        number="1",
        position=(27, 17),
        drill=0.0,
        shapes={"F.Cu": box(26.75, 16.75, 27.25, 17.25)},
    )


def snapshot(pads=(), zones=()):
    return Snapshot(
        copper_layers=LAYERS,
        pads=[own_pad(), *pads],
        zones=[gnd_zone(), *zones],
        edge_paths=[[(x, y) for x, y in BOARD.exterior.coords]],
    )


def returnvia_accepts(snap, pos):
    board = returnvia._Board(snap, [])
    return (
        board.in_bounds(pos, VIA)
        and board.connects(pos, "GND", VIA, filled=True)
        and board.clear(pos, "GND", VIA)
    )


def padvia_accepts(snap, pos):
    return padvia._Board(snap, []).via_ok(pos, "GND", VIA, own_pad())


def probes(x0, x1, y):
    """Spots STEP apart, offset half a step so none sits exactly on a rule
    boundary, where float rounding alone would decide."""
    n = round((x1 - x0) / STEP)
    return [(round(x0 + (i + 0.5) * STEP, 4), y) for i in range(n)]


def test_pad_clearance_adds_the_margin_to_an_override():
    assert pad_clearance(None) == OTHER_NET_CLEARANCE
    assert pad_clearance(0.1) == OTHER_NET_CLEARANCE
    assert pad_clearance(1.6) == pytest.approx(1.6 + OVERRIDE_MARGIN)


def test_planners_agree_near_an_overridden_mounting_hole():
    hole = Pad(
        id="hole",
        footprint_id=None,
        net="",
        kind=PadKind.NPTH,
        number="",
        position=HOLE_CENTRE,
        drill=HOLE_DRILL,
        shapes={},
        clearance_override=MOUNTING_HOLE_OVERRIDE,
    )
    snap = snapshot(pads=[hole])
    edge = HOLE_CENTRE[0] + HOLE_DRILL / 2 + VIA.diameter / 2
    spots = probes(edge + 1.4, edge + 2.0, HOLE_CENTRE[1])

    accepted_r = [p for p in spots if returnvia_accepts(snap, p)]
    accepted_p = [p for p in spots if padvia_accepts(snap, p)]

    assert accepted_r == accepted_p
    gap = min(p[0] for p in accepted_r) - edge
    assert gap == pytest.approx(MOUNTING_HOLE_OVERRIDE + OVERRIDE_MARGIN, abs=STEP)


def test_planners_agree_at_a_via_keepout_rule_area():
    keepout = Zone(
        net=None,
        layers=LAYERS,
        priority=0,
        outline=KEEPOUT,
        is_rule_area=True,
        keepout_vias=True,
    )
    snap = snapshot(zones=[keepout])
    spots = probes(14, 20, 6)

    accepted_r = [p for p in spots if returnvia_accepts(snap, p)]
    accepted_p = [p for p in spots if padvia_accepts(snap, p)]

    assert accepted_r == accepted_p
    assert max(p[0] for p in accepted_r) < 18 - VIA.diameter / 2


def test_via_keepouts_are_rule_areas_that_keep_out_vias():
    rule = Zone(None, LAYERS, 0, KEEPOUT, is_rule_area=True, keepout_vias=True)
    tracks_only = Zone(None, LAYERS, 0, box(0, 0, 1, 1), is_rule_area=True)

    snap = snapshot(zones=[rule, tracks_only])

    assert snap.via_keepouts.equals(KEEPOUT)
