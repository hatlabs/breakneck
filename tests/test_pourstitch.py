import math

import numpy as np
import pytest
from shapely.geometry import Point, box
from shapely.ops import unary_union

from breakneck.plan import FORBIDDEN_RADIUS, PlannedUnit
from breakneck.pourstitch import (
    CLEARANCE,
    QUANTISATION,
    Options,
    PourStitcher,
    lattice_sites,
)
from breakneck.snapshot import Pad, PadKind, Snapshot, Via, ViaSize, Zone

TWO_LAYERS = ("F.Cu", "B.Cu")
D = 2.5
NET = "GND"
VIA = ViaSize(0.6, 0.3)


def outline_paths(geom) -> list[list[tuple[float, float]]]:
    return [[(x, y) for x, y in geom.exterior.coords]]


def pour_snapshot(pour, board=None, vias=(), pads=(), net=NET):
    board = board or pour.envelope
    zone = Zone(
        net=net,
        layers=TWO_LAYERS,
        priority=0,
        outline=pour,
        fills={L: pour for L in TWO_LAYERS},
    )
    return Snapshot(
        copper_layers=TWO_LAYERS,
        zones=[zone],
        vias=list(vias),
        pads=list(pads),
        edge_paths=outline_paths(board),
        via_sizes={net: VIA},
    )


def options(**kw):
    return Options(nets=(NET,), distance=D, **kw)


def max_uncovered(stitcher, snap, opts, units):
    return max(
        float(np.nanmax(m.distance)) for m in stitcher.coverage(snap, opts, units)
    )


def test_square_pour_ends_fully_covered():
    snap = pour_snapshot(box(0, 0, 20, 20))
    stitcher = PourStitcher()

    plan = stitcher(snap, options(), [])

    assert plan.units
    assert not plan.unfixable
    assert max_uncovered(stitcher, snap, options(), plan.units) <= D


def test_lattice_pitch_is_distance_times_root_two():
    sites = lattice_sites((0, 0, 20, 20), options().pitch, 0)
    xs = sorted({round(x, 6) for x, _ in sites})

    assert np.diff(xs) == pytest.approx(D * math.sqrt(2))


def test_open_pour_lattice_vias_sit_on_lattice_sites():
    snap = pour_snapshot(box(0, 0, 30, 30))
    plan = PourStitcher()(snap, options(), [])
    sites = lattice_sites((0, 0, 30, 30), options().pitch, 0)

    on_site = [
        u for u in plan.units if min(math.dist(u.position, s) for s in sites) <= 0.1
    ]

    interior = [s for s in sites if 2 < s[0] < 28 and 2 < s[1] < 28]
    assert len(on_site) >= len(interior)


def u_pour():
    """Two 1.2 mm arms 1 mm apart, joined 10 mm below their tips."""
    left = box(0, 0, 1.2, 12)
    right = box(2.2, 0, 3.4, 12)
    base = box(0, 10, 3.4, 12)
    return unary_union([left, right, base])


def test_u_pour_arm_tip_not_covered_from_the_other_arm():
    pour = u_pour()
    via = Via("v", NET, (0.6, 1.0), 0.6, 0.3)
    snap = pour_snapshot(pour, board=box(-1, -1, 4.4, 13), vias=[via])
    stitcher = PourStitcher()

    maps = stitcher.coverage(snap, options(), [])
    grid = stitcher.grid(snap, options())
    tip = grid.index((2.8, 0.5))

    assert math.dist((0.6, 1.0), (2.8, 0.5)) < D
    assert all(m.distance[tip] > D for m in maps)

    plan = stitcher(snap, options(), [])
    assert any(u.position[0] > 2.2 and u.position[1] < 1 + D for u in plan.units)


def finger_snapshot(width):
    """A pour with a long finger of `width` sticking out to the right."""
    body = box(0, 0, 6, 6)
    finger = box(6, 2.5, 16, 2.5 + width)
    return pour_snapshot(unary_union([body, finger]), board=box(-1, -1, 17, 7))


def test_narrow_finger_gets_fallback_via():
    snap = finger_snapshot(0.8)
    plan = PourStitcher()(snap, options(), [])

    in_finger = [u for u in plan.units if u.position[0] > 7]
    assert in_finger
    assert all(u.size == options().fallback_via for u in in_finger)
    assert not plan.unfixable


def test_too_narrow_finger_is_reported_unfixable():
    snap = finger_snapshot(0.4)
    plan = PourStitcher()(snap, options(), [])

    assert not [u for u in plan.units if u.position[0] > 7]
    assert plan.unfixable
    assert all(s.position[0] > 6 for s in plan.unfixable)


def test_forbidden_position_is_not_reused():
    snap = pour_snapshot(box(0, 0, 20, 20))
    stitcher = PourStitcher()
    first = stitcher(snap, options(), [])
    forbidden = first.units[len(first.units) // 2].position

    plan = stitcher(snap, options(), [forbidden])

    assert all(math.dist(u.position, forbidden) > FORBIDDEN_RADIUS for u in plan.units)
    covered = max_uncovered(stitcher, snap, options(), plan.units) <= D
    assert covered or plan.unfixable


def test_second_run_on_output_places_nothing():
    pour = box(0, 0, 20, 20)
    snap = pour_snapshot(pour)
    plan = PourStitcher()(snap, options(), [])
    vias = [
        Via(f"v{i}", NET, u.position, u.size.diameter, u.size.drill)
        for i, u in enumerate(plan.units)
    ]

    again = PourStitcher()(pour_snapshot(pour, vias=vias), options(), [])

    assert again.units == ()


def test_other_net_pad_clearance_enforced_inside_own_fill():
    """Regression: vias landed 0.147 mm from a no-net pad whose fill gap was small."""
    pads = []
    fill_gap = 0.15
    pad_shapes = []
    for i, site in enumerate(lattice_sites((0, 0, 20, 20), options().pitch, 0)):
        shape = box(site[0] - 0.4, site[1] - 0.4, site[0] + 0.4, site[1] + 0.4)
        pad_shapes.append(shape)
        pads.append(
            Pad(f"p{i}", None, "", PadKind.SMD, "1", site, 0.0, {"F.Cu": shape})
        )
    pour = box(0, 0, 20, 20)
    front = pour.difference(unary_union(pad_shapes).buffer(fill_gap))
    zone = Zone(NET, TWO_LAYERS, 0, pour, {"F.Cu": front, "B.Cu": pour})
    snap = Snapshot(
        copper_layers=TWO_LAYERS,
        zones=[zone],
        pads=pads,
        edge_paths=outline_paths(pour),
        via_sizes={NET: VIA},
    )

    plan = PourStitcher()(snap, options(), [])

    assert plan.units
    copper = unary_union(pad_shapes)
    for u in plan.units:
        gap = Point(u.position).distance(copper) - u.size.diameter / 2
        assert gap >= CLEARANCE


def test_pad_clearance_override_is_honoured():
    hole_pos = (10.0, 10.0)
    mounting = Pad(
        "mh",
        None,
        "",
        PadKind.NPTH,
        "",
        hole_pos,
        3.2,
        {},
        clearance_override=1.6,
    )
    snap = pour_snapshot(box(0, 0, 20, 20), pads=[mounting])

    plan = PourStitcher()(snap, options(), [])

    hole = Point(hole_pos).buffer(1.6)
    for u in plan.units:
        assert (
            Point(u.position).distance(hole) - u.size.diameter / 2 >= 1.6 - QUANTISATION
        )


def test_net_with_one_zone_layer_is_skipped():
    zone = Zone(NET, ("F.Cu",), 0, box(0, 0, 10, 10), {"F.Cu": box(0, 0, 10, 10)})
    snap = Snapshot(
        copper_layers=TWO_LAYERS,
        zones=[zone],
        edge_paths=outline_paths(box(0, 0, 10, 10)),
    )

    plan = PourStitcher()(snap, options(), [])

    assert plan.units == ()


def test_via_override_is_used_for_lattice():
    snap = pour_snapshot(box(0, 0, 20, 20))
    big = ViaSize(0.8, 0.4)
    plan = PourStitcher()(snap, options(via=big), [])

    assert {u.size for u in plan.units} == {big}


def test_coverage_counts_planned_units():
    snap = pour_snapshot(box(0, 0, 4, 4))
    stitcher = PourStitcher()
    unit = PlannedUnit((2.0, 2.0), NET, VIA)

    without = stitcher.coverage(snap, options(), [])
    with_unit = stitcher.coverage(snap, options(), [unit])

    assert all(np.isinf(np.nanmax(m.distance)) for m in without)
    assert all(np.nanmax(m.distance) <= D for m in with_unit)
