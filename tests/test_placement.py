from datetime import datetime

import kipy.board_types
import pytest
from kipy.board import BoardLayer as BL
from kipy.board_types import unwrap
from kipy.errors import ApiError, ConnectionError
from kipy.proto.common.commands.editor_commands_pb2 import (
    CreateItemsResponse,
    ItemStatusCode,
)
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from kipy.proto.common.types import DocumentSpecifier

from breakneck import commit, drc, placement
from breakneck.commit import Marker, MarkerKind
from breakneck.plan import Plan, PlannedUnit, Stub
from breakneck.snapshot import ViaSize

VIA = ViaSize(0.6, 0.3)
BOARD_CHANGES = {
    "refill_zones",
    "begin_commit",
    "create_items",
    "push_commit",
    "remove_items",
}


def net(name):
    n = kipy.board_types.Net()
    n.name = name
    return n


class FakeClient:
    """Answers CreateItems the way KiCad does: one status per item."""

    def __init__(self, board):
        self.board = board

    def send(self, command, response_type):
        assert response_type is CreateItemsResponse
        self.board.calls.append("create_items")
        items = [unwrap(a) for a in command.items]
        self.board.created.append(items)
        response = CreateItemsResponse()
        for packed, item in zip(command.items, items):
            result = response.created_items.add()
            result.item.CopyFrom(packed)
            if self.board.refuse(item):
                result.status.code = ItemStatusCode.ISC_INVALID_DATA
                result.status.error_message = "no overlapping layers"
            else:
                result.status.code = ItemStatusCode.ISC_OK
        return response


class RecordingBoard:
    name = "b.kicad_pcb"
    document = DocumentSpecifier()

    def __init__(self, refuse=lambda item: False, project_dir=""):
        self.calls = []
        self.refuse = refuse
        self.project_dir = project_dir
        self.created = []
        self.client = FakeClient(self)

    def __getattr__(self, attr):
        def record(*args, **kwargs):
            self.calls.append(attr)

        return record

    def get_nets(self):
        return [net("GND"), net("+3.3V")]

    def get_layer_by_name(self, name):
        return {"F.Cu": BL.BL_F_Cu, "B.Cu": BL.BL_B_Cu}[name]

    def get_project(self):
        return type("P", (), {"path": self.project_dir})()

    def begin_commit(self):
        self.calls.append("begin_commit")
        return "commit"

    def push_commit(self, c, message=""):
        self.calls.append(("push_commit", message))

    def drop_commit(self, c):
        self.calls.append("drop_commit")

    def refill_zones(self, block=True, max_poll_seconds=0):
        self.calls.append("refill_zones")


class FakeKiCad:
    def __init__(self, board, cli_path=""):
        self.board = board
        self.cli_path = cli_path
        self.pings = []

    def get_board(self):
        return self.board

    def get_kicad_binary_path(self, name):
        return self.cli_path

    def ping(self):
        result = self.pings.pop(0) if self.pings else None
        if isinstance(result, Exception):
            raise result


def unit(x, y, net="GND", stub=None):
    return PlannedUnit(position=(x, y), net=net, size=VIA, stub=stub)


def never_plan(snapshot, options, forbidden):
    raise AssertionError("planner must not run")


def board_changes(board):
    return [
        c for c in board.calls if (c[0] if isinstance(c, tuple) else c) in BOARD_CHANGES
    ]


# ---- fail before any board change ----------------------------------------------


def test_missing_kicad_cli_fails_before_any_board_change(monkeypatch):
    monkeypatch.setattr(drc.shutil, "which", lambda name: None)
    monkeypatch.setattr(drc, "KNOWN_KICAD_CLI_PATHS", ())
    board = RecordingBoard()

    with pytest.raises(drc.KiCadCliNotFound):
        placement.run_placement(
            never_plan, None, placement.RunSettings("test"), FakeKiCad(board)
        )

    assert board_changes(board) == []


def test_unsaved_board_fails_before_any_board_change(tmp_path):
    cli = tmp_path / "kicad-cli"
    cli.write_text("#!/bin/sh\necho 10.0.7\n")
    cli.chmod(0o755)
    board = RecordingBoard(project_dir="")

    with pytest.raises(drc.ProjectFilesError):
        placement.run_placement(
            never_plan, None, placement.RunSettings("test"), FakeKiCad(board, str(cli))
        )

    assert board_changes(board) == []


# ---- commit ------------------------------------------------------------------------


def test_commit_creates_items_and_markers_in_one_commit():
    board = RecordingBoard()
    stub = Stub(start=(0, 0), end=(1, 0), width=0.3, layer="F.Cu")
    units = [unit(1, 0, stub=stub), unit(5, 5, net="+3.3V")]
    markers = [Marker((9, 9), MarkerKind.UNFIXABLE)]

    outcome = commit.commit_units(FakeKiCad(board), board, "pourstitch", units, markers)

    placed, marks = board.created
    assert [type(i).__name__ for i in placed] == ["Via", "Track", "Via"]
    via, track, _ = placed
    assert (via.position.x, via.position.y) == (1_000_000, 0)
    assert via.diameter == 600_000 and via.drill_diameter == 300_000
    assert via.net.name == "GND"
    assert track.layer == BL.BL_F_Cu and track.width == 300_000
    assert all(m.layer == BL.BL_Eco2_User for m in marks)
    pushes = [c for c in board.calls if isinstance(c, tuple)]
    assert len(pushes) == 1 and pushes[0][1].startswith("breakneck pourstitch ")
    assert board.calls.count("begin_commit") == 1
    assert outcome.created == 3


def test_commit_refused_item_raises_and_drops_the_commit():
    # KiCad reports refusals per item instead of failing the request, and
    # kipy's create_items discards those statuses.
    board = RecordingBoard(refuse=lambda item: isinstance(item, kipy.board_types.Track))
    stub = Stub(start=(0, 0), end=(1, 0), width=0.3, layer="F.Cu")

    with pytest.raises(commit.CreateItemsError, match="no overlapping layers"):
        commit.commit_units(
            FakeKiCad(board), board, "padvia", [unit(1, 0, stub=stub)], []
        )

    assert "drop_commit" in board.calls
    assert not any(isinstance(c, tuple) and c[0] == "push_commit" for c in board.calls)


def test_commit_with_unknown_net_fails_before_opening_a_commit():
    board = RecordingBoard()

    with pytest.raises(ValueError, match="NOPE"):
        commit.commit_units(FakeKiCad(board), board, "x", [unit(1, 1, net="NOPE")], [])

    assert board_changes(board) == []


def test_commit_label_carries_command_and_timestamp():
    assert (
        commit.commit_label("padvia", datetime(2026, 10, 26, 14, 5, 9))
        == "breakneck padvia 2026-10-26 14:05:09"
    )


@pytest.mark.parametrize(
    "kind, segments",
    [(MarkerKind.UNFIXABLE, 2), (MarkerKind.FORCED, 2), (MarkerKind.UNATTRIBUTED, 4)],
)
def test_marker_glyphs_are_centred_on_the_spot(kind, segments):
    items = commit.marker_items([Marker((10, 20), kind)])

    assert len(items) == segments
    xs = [p.x for s in items for p in (s.start, s.end)]
    ys = [p.y for s in items for p in (s.start, s.end)]
    assert (min(xs) + max(xs)) / 2 == 10_000_000
    assert (min(ys) + max(ys)) / 2 == 20_000_000


def test_markers_for_unfixable_forced_and_unattributed():
    forced = PlannedUnit((2, 2), "GND", VIA, forced=True)
    from breakneck.plan import UnfixableSpot

    result = drc.PruneResult(
        plan=Plan(
            units=(unit(1, 1), forced), unfixable=(UnfixableSpot((3, 3), "GND", "x"),)
        ),
        dropped=(),
        forced=(),
        unattributed=(
            drc.Violation(
                "copper_sliver", "s", "warning", (drc.ViolationItem("Zone", (4, 4)),)
            ),
        ),
        rounds=1,
    )

    markers = placement.markers_for(result)

    assert [(m.position, m.kind) for m in markers] == [
        ((3, 3), MarkerKind.UNFIXABLE),
        ((2, 2), MarkerKind.FORCED),
        ((4, 4), MarkerKind.UNATTRIBUTED),
    ]

    without_unfixable = placement.markers_for(result, mark_unfixable=False)

    assert [m.kind for m in without_unfixable] == [
        MarkerKind.FORCED,
        MarkerKind.UNATTRIBUTED,
    ]


# ---- refill --------------------------------------------------------------------


class RefillBoard:
    def __init__(self, error):
        self.error = error

    def refill_zones(self, block=True, max_poll_seconds=0):
        if self.error:
            raise self.error


def test_refill_success():
    assert commit.refill_tolerant(
        FakeKiCad(None), RefillBoard(None), sleep=lambda s: None
    )


def test_refill_timeout_waits_until_kicad_answers():
    kicad = FakeKiCad(None)
    kicad.pings = [
        ConnectionError("timeout"),
        ApiError("busy", code=ApiStatusCode.AS_BUSY),
        None,
    ]
    sleeps = []

    ok = commit.refill_tolerant(
        kicad, RefillBoard(ConnectionError("timeout")), sleep=sleeps.append
    )

    assert ok
    assert len(sleeps) == 3


def test_refill_gives_up_after_timeout():
    kicad = FakeKiCad(None)
    kicad.pings = [ConnectionError("timeout")] * 10

    ok = commit.refill_tolerant(
        kicad,
        RefillBoard(ConnectionError("timeout")),
        timeout_s=3,
        poll_s=1,
        sleep=lambda s: None,
    )

    assert not ok


def test_refill_reraises_real_errors():
    with pytest.raises(ApiError):
        commit.refill_tolerant(
            FakeKiCad(None),
            RefillBoard(ApiError("bad", code=ApiStatusCode.AS_BAD_REQUEST)),
            sleep=lambda s: None,
        )
