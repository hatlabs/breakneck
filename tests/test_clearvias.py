import sys

import kipy.board_types
import pytest
from kipy.errors import ConnectionError as KiCadConnectionError
from loguru import logger
from shapely.geometry import LineString, box

from breakneck import cli
from breakneck.clearvias import stitching_vias
from breakneck.commit import eco2_segment
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


# ---- breakneck clear --vias ------------------------------------------------------


class ViaBoard:
    """A board without images whose vias `clear --vias` may remove."""

    def __init__(self, vias, shapes=(), push_error=None):
        self.vias = vias
        self.shapes = list(shapes)
        self.push_error = push_error
        self.calls: list = []
        self.labels: list[str] = []

    def get_shapes(self):
        return self.shapes

    def get_vias(self):
        return self.vias

    def begin_commit(self):
        self.calls.append("begin_commit")
        return "c"

    def remove_items(self, items):
        self.calls.append(("remove_items", [i.id.value for i in items]))

    def push_commit(self, c, message=""):
        self.calls.append("push_commit")
        self.labels.append(message)
        if self.push_error is not None:
            raise self.push_error

    def drop_commit(self, c):
        self.calls.append("drop_commit")

    def refill_zones(self, block=True, max_poll_seconds=0):
        self.calls.append("refill_zones")


class FakeKiCad:
    def __init__(self, board):
        self.board = board

    def get_board(self):
        return self.board

    def ping(self):
        pass


def kipy_via(via_id):
    v = kipy.board_types.Via()
    v.proto.id.value = via_id
    return v


@pytest.fixture
def log():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{message}")
    yield lines
    logger.remove(sink)


def clear(monkeypatch, snap, argv, **board_args):
    board = ViaBoard([kipy_via(v.id) for v in snap.vias], **board_args)
    monkeypatch.setattr(cli.kipy, "KiCad", lambda: FakeKiCad(board))
    monkeypatch.setattr(cli.Snapshot, "from_board", staticmethod(lambda b: snap))
    monkeypatch.setattr(sys, "argv", ["breakneck", *argv])
    cli.execute_clear(cli.parse_args())
    return board


TWO_GND_ONE_POWER = (via("GND", 5, 5), via("GND", 15, 5), via("+3.3V", 30, 5))


def test_clear_vias_removes_them_in_one_commit_then_refills(monkeypatch, log):
    board = clear(monkeypatch, snapshot(TWO_GND_ONE_POWER), ["clear", "--vias"])

    assert any("Removing stitching vias: 2 on GND, 1 on +3.3V" in m for m in log)
    assert board.calls == [
        "begin_commit",
        ("remove_items", [v.id for v in TWO_GND_ONE_POWER]),
        "push_commit",
        "refill_zones",
    ]
    assert board.labels[0].startswith("breakneck clear ")


def test_clear_vias_and_markers_go_in_one_remove(monkeypatch):
    marker = eco2_segment((1, 1), (2, 2))
    marker.proto.id.value = "marker"
    snap = snapshot(TWO_GND_ONE_POWER[:1])

    board = clear(monkeypatch, snap, ["clear", "--vias"], shapes=[marker])

    assert board.calls[:3] == [
        "begin_commit",
        ("remove_items", ["marker", TWO_GND_ONE_POWER[0].id]),
        "push_commit",
    ]


def test_clear_vias_still_refills_when_the_push_times_out(monkeypatch, log):
    board = clear(
        monkeypatch,
        snapshot(TWO_GND_ONE_POWER),
        ["clear", "--vias"],
        push_error=KiCadConnectionError("timed out"),
    )

    assert board.calls[-1] == "refill_zones"
    assert any("it may have landed" in m for m in log)


def test_clear_nets_without_vias_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["breakneck", "clear", "--nets", "GND"])

    with pytest.raises(SystemExit) as exc:
        cli.parse_args()

    assert exc.value.code == 2
    assert "--nets requires --vias" in capsys.readouterr().err


def test_clear_vias_warns_about_nets_without_a_zone(monkeypatch, log):
    clear(
        monkeypatch,
        snapshot(TWO_GND_ONE_POWER),
        ["clear", "--vias", "--nets", "GND", "GDN"],
    )

    assert any("No copper zone on net GDN" in m for m in log)
    assert not any("No copper zone on net GND" in m for m in log)


@pytest.mark.parametrize("nets", [[], ["--nets", "GND"]], ids=["all", "selected"])
def test_clear_vias_warns_about_unfilled_zones(monkeypatch, log, nets):
    unfilled = Zone(
        net="GND", layers=TWO_LAYERS, priority=0, outline=GND_AREA, fills={}
    )
    snap = snapshot(TWO_GND_ONE_POWER, zones=(unfilled, ZONES[1]))

    clear(monkeypatch, snap, ["clear", "--vias", *nets])

    assert any("1 zone of GND has no fill" in m for m in log)


def test_clear_vias_ignores_unfilled_zones_of_other_nets(monkeypatch, log):
    unfilled = Zone(
        net="GND", layers=TWO_LAYERS, priority=0, outline=GND_AREA, fills={}
    )
    snap = snapshot(TWO_GND_ONE_POWER, zones=(unfilled, ZONES[1]))

    clear(monkeypatch, snap, ["clear", "--vias", "--nets", "+3.3V"])

    assert not any("has no fill" in m for m in log)


def test_clear_vias_nets_limit_the_removal(monkeypatch):
    board = clear(
        monkeypatch, snapshot(TWO_GND_ONE_POWER), ["clear", "--vias", "--nets", "GND"]
    )

    assert ("remove_items", [v.id for v in TWO_GND_ONE_POWER[:2]]) in board.calls


def test_clear_vias_dry_run_removes_nothing_and_reports_per_net(monkeypatch, log):
    board = clear(monkeypatch, snapshot(TWO_GND_ONE_POWER), ["-n", "clear", "--vias"])

    assert "push_commit" not in board.calls
    assert board.calls[-1] == "drop_commit"
    assert any("Would remove stitching vias: 2 on GND, 1 on +3.3V" in m for m in log)


def test_clear_without_vias_leaves_vias_and_zones_alone(monkeypatch, log):
    board = clear(monkeypatch, snapshot(TWO_GND_ONE_POWER), ["clear"])

    assert board.calls == []
    assert any("Nothing to clear" in m for m in log)
