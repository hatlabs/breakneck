import pytest
from shapely.geometry import LineString, box

from breakneck.clearvias import stitching_vias
from breakneck.snapshot import Pad, PadKind, Snapshot, Track, Via, Zone

TWO_LAYERS = ("F.Cu", "B.Cu")
GND_AREA = box(0, 0, 20, 20)
POWER_AREA = box(20, 0, 40, 20)


def zone(net, outline):
    return Zone(
        net=net,
        layers=TWO_LAYERS,
        priority=0,
        outline=outline,
        fills={L: outline for L in TWO_LAYERS},
    )


ZONES = (zone("GND", GND_AREA), zone("+3.3V", POWER_AREA))


def via(net, x, y, locked=False):
    return Via(
        id=f"{net}@{x},{y}",
        net=net,
        position=(x, y),
        diameter=0.6,
        drill=0.3,
        locked=locked,
    )


def snapshot(vias, tracks=(), pads=(), zones=ZONES):
    return Snapshot(
        copper_layers=TWO_LAYERS,
        vias=list(vias),
        tracks=list(tracks),
        pads=list(pads),
        zones=list(zones),
    )


def test_unattached_via_in_its_pour_is_removed():
    v = via("GND", 10, 10)

    assert stitching_vias(snapshot([v])) == [v]


def test_via_with_a_same_net_track_ending_on_it_is_kept():
    v = via("GND", 10, 10)
    t = Track("t", "GND", "F.Cu", 0.2, LineString([(5, 10), (10, 10)]))

    assert stitching_vias(snapshot([v], tracks=[t])) == []


@pytest.mark.parametrize(
    ("pad_net", "removed"), [("GND", False), ("+3.3V", True)], ids=["same", "other"]
)
def test_only_a_same_net_pad_keeps_the_via(pad_net, removed):
    v = via("GND", 10, 10)
    p = Pad(
        id="p",
        footprint_id=None,
        net=pad_net,
        kind=PadKind.SMD,
        number="1",
        position=(10, 10),
        drill=0.0,
        shapes={"F.Cu": box(9, 9, 11, 11)},
    )

    assert stitching_vias(snapshot([v], pads=[p])) == ([v] if removed else [])


def test_locked_via_is_kept():
    assert stitching_vias(snapshot([via("GND", 10, 10, locked=True)])) == []


def test_via_without_a_zone_or_outside_its_fill_is_kept():
    no_zone = via("SIG", 10, 10)
    outside = via("GND", 30, 10)

    assert stitching_vias(snapshot([no_zone, outside])) == []


def test_nets_limit_the_removal():
    gnd, power = via("GND", 10, 10), via("+3.3V", 30, 10)

    assert stitching_vias(snapshot([gnd, power])) == [gnd, power]
    assert stitching_vias(snapshot([gnd, power]), nets=["GND"]) == [gnd]
