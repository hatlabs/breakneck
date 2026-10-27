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
from loguru import logger

from breakneck import commit, drc, placement
from breakneck.commit import Marker, MarkerKind
from breakneck.plan import Plan, PlannedUnit, Stub, UnfixableSpot
from breakneck.snapshot import Snapshot, ViaSize

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
        # Successive get_as_string results; the last one repeats.
        self.texts = ["(kicad_pcb)"]

    def __getattr__(self, attr):
        def record(*args, **kwargs):
            self.calls.append(attr)

        return record

    def get_nets(self):
        return [net("GND"), net("+3.3V")]

    def get_as_string(self):
        return self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]

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


# ---- run ------------------------------------------------------------------------


class FakeDrcRunner:
    """Reports a clearance violation on every planned via at a bad position,
    on top of the `base` report."""

    base: tuple[drc.Violation, ...] = ()

    def __init__(self, kicad_cli, project, board_text, workdir):
        self.bad = {(5, 5)}
        self.planned: list[drc.Violation] = []

    def baseline(self):
        return list(self.base)

    def report(self, plan):
        return [
            *self.base,
            *(
                drc.Violation(
                    "clearance",
                    "c",
                    "error",
                    (drc.ViolationItem("Via [GND]", u.position),),
                )
                for u in plan.units
                if u.position in self.bad
            ),
        ]

    def new_violations(self, plan):
        self.planned = self.report(plan)
        return drc.new_violations(self.base, self.planned)


def plan_at(*positions, unfixable=()):
    def planner(snapshot, options, forbidden):
        return Plan(
            units=tuple(unit(*p) for p in positions if p not in forbidden),
            unfixable=unfixable,
        )

    return planner


@pytest.fixture
def saved_board(tmp_path, monkeypatch):
    """A RecordingBoard saved to disk, with kicad-cli and DRC faked."""
    cli = tmp_path / "kicad-cli"
    cli.write_text("#!/bin/sh\necho 10.0.7\n")
    cli.chmod(0o755)
    (tmp_path / "b.kicad_pcb").write_text("")
    (tmp_path / "b.kicad_pro").write_text("{}")
    monkeypatch.setattr(
        placement.Snapshot,
        "from_board",
        staticmethod(lambda board: Snapshot(copper_layers=("F.Cu", "B.Cu"))),
    )
    monkeypatch.setattr(drc, "DrcRunner", FakeDrcRunner)
    board = RecordingBoard(project_dir=str(tmp_path))
    return board, FakeKiCad(board, str(cli))


def pushes(board):
    return [c for c in board.calls if isinstance(c, tuple) and c[0] == "push_commit"]


def test_run_commits_the_pruned_units_in_one_commit(saved_board):
    board, kicad = saved_board

    result = placement.run_placement(
        plan_at((1, 1), (5, 5)), None, placement.RunSettings("test"), kicad
    )

    assert [u.position for u in result.prune.plan.units] == [(1, 1)]
    assert [d.unit.position for d in result.prune.dropped] == [(5, 5)]
    assert board.calls.count("begin_commit") == 1 and len(pushes(board)) == 1
    (placed,) = board.created
    assert [(v.position.x, v.position.y) for v in placed] == [(1_000_000, 1_000_000)]


def test_dry_run_commits_nothing(saved_board):
    board, kicad = saved_board

    result = placement.run_placement(
        plan_at((1, 1)), None, placement.RunSettings("test", dry_run=True), kicad
    )

    assert [u.position for u in result.prune.plan.units] == [(1, 1)]
    assert "begin_commit" not in board.calls and pushes(board) == []
    assert board.created == []


def test_empty_plan_without_markers_commits_nothing(saved_board):
    board, kicad = saved_board
    unfixable = (UnfixableSpot((3, 3), "GND", "no room"),)

    placement.run_placement(
        plan_at(unfixable=unfixable),
        None,
        placement.RunSettings("test", mark_unfixable=False),
        kicad,
    )

    assert "begin_commit" not in board.calls and pushes(board) == []


def test_board_edited_during_the_run_is_not_committed(saved_board):
    board, kicad = saved_board
    board.texts = ["(kicad_pcb)", "(kicad_pcb (edited))"]

    with pytest.raises(placement.BoardChangedError, match="run again"):
        placement.run_placement(
            plan_at((1, 1)), None, placement.RunSettings("test"), kicad
        )

    assert "begin_commit" not in board.calls and board.created == []


def test_board_changed_after_an_unconfirmed_refill_names_the_refill(
    saved_board, monkeypatch
):
    board, kicad = saved_board
    board.texts = ["(kicad_pcb)", "(kicad_pcb (refilled))"]
    monkeypatch.setattr(placement, "refill_tolerant", lambda kicad, board: False)

    with pytest.raises(placement.BoardChangedError, match="zone refill"):
        placement.run_placement(
            plan_at((1, 1)), None, placement.RunSettings("test"), kicad
        )


def test_board_changed_after_a_confirmed_refill_does_not_blame_it(saved_board):
    board, kicad = saved_board
    board.texts = ["(kicad_pcb)", "(kicad_pcb (edited))"]

    with pytest.raises(placement.BoardChangedError) as error:
        placement.run_placement(
            plan_at((1, 1)), None, placement.RunSettings("test"), kicad
        )

    assert "refill" not in str(error.value)


def split_runner(violation_type):
    """A DRC runner reporting a connectivity violation no planned via is named in."""

    class SplitRunner(FakeDrcRunner):
        def report(self, plan):
            return [
                drc.Violation(
                    violation_type,
                    "split",
                    "error",
                    (drc.ViolationItem("Pad 1 [+5V] of J1 on F.Cu", (20, 20)),),
                )
            ]

    return SplitRunner


@pytest.mark.parametrize("violation_type", ["unconnected_items", "isolated_copper"])
def test_unattributed_split_refuses_to_commit(saved_board, monkeypatch, violation_type):
    monkeypatch.setattr(drc, "DrcRunner", split_runner(violation_type))
    board, kicad = saved_board

    with pytest.raises(placement.SplitError, match="--allow-splits"):
        placement.run_placement(
            plan_at((1, 1)), None, placement.RunSettings("test"), kicad
        )

    assert "begin_commit" not in board.calls and board.created == []


def test_allow_splits_commits_with_a_marker_at_the_split(saved_board, monkeypatch):
    monkeypatch.setattr(drc, "DrcRunner", split_runner("unconnected_items"))
    board, kicad = saved_board

    result = placement.run_placement(
        plan_at((1, 1)), None, placement.RunSettings("test", allow_splits=True), kicad
    )

    assert len(pushes(board)) == 1
    _, marks = board.created
    assert len(marks) == 4  # one square
    assert placement.markers_for(result.prune) == (
        Marker((20, 20), MarkerKind.UNATTRIBUTED),
    )


def test_dry_run_reports_a_split_without_failing(saved_board, monkeypatch):
    monkeypatch.setattr(drc, "DrcRunner", split_runner("unconnected_items"))
    board, kicad = saved_board

    result = placement.run_placement(
        plan_at((1, 1)), None, placement.RunSettings("test", dry_run=True), kicad
    )

    assert [v.type for v in result.prune.unattributed] == ["unconnected_items"]
    assert "begin_commit" not in board.calls


def test_unattributed_non_connectivity_violation_still_commits(
    saved_board, monkeypatch
):
    monkeypatch.setattr(drc, "DrcRunner", split_runner("copper_sliver"))
    board, kicad = saved_board

    placement.run_placement(plan_at((1, 1)), None, placement.RunSettings("test"), kicad)

    assert len(pushes(board)) == 1


def unconnected(*pads):
    return drc.Violation(
        "unconnected_items",
        "Missing connection",
        "error",
        tuple(drc.ViolationItem(f"Pad {n} [+3V3] of U1", pos) for n, pos in pads),
    )


BASE_UNCONNECTED = unconnected(("A", (1, 2)), ("B", (3, 4)))


def reporting(base, planned):
    class ReportRunner(FakeDrcRunner):
        def report(self, plan):
            return list(planned)

    ReportRunner.base = tuple(base)
    return ReportRunner


def test_existing_unconnected_item_that_moved_is_not_a_split(saved_board, monkeypatch):
    # A refill around the new vias re-anchors an existing unconnected item:
    # same net, same count, different items and positions.
    moved = unconnected(("C", (7, 8)), ("D", (9, 10)))
    monkeypatch.setattr(drc, "DrcRunner", reporting([BASE_UNCONNECTED], [moved]))
    board, kicad = saved_board

    result = placement.run_placement(
        plan_at((1, 1)), None, placement.RunSettings("test"), kicad
    )

    assert result.prune.unattributed == (moved,)
    assert len(pushes(board)) == 1
    assert placement.markers_for(result.prune) == (
        Marker((7, 8), MarkerKind.UNATTRIBUTED),
    )


def test_second_unconnected_item_on_the_same_net_is_a_split(saved_board, monkeypatch):
    extra = unconnected(("C", (7, 8)), ("D", (9, 10)))
    monkeypatch.setattr(
        drc, "DrcRunner", reporting([BASE_UNCONNECTED], [BASE_UNCONNECTED, extra])
    )
    board, kicad = saved_board

    with pytest.raises(placement.SplitError, match=r"\+3V3"):
        placement.run_placement(
            plan_at((1, 1)), None, placement.RunSettings("test"), kicad
        )

    assert "begin_commit" not in board.calls


# ---- commit ------------------------------------------------------------------------


def test_commit_creates_items_and_markers_in_one_commit():
    board = RecordingBoard()
    stub = Stub(start=(0, 0), end=(1, 0), width=0.3, layer="F.Cu")
    units = [unit(1, 0, stub=stub), unit(5, 5, net="+3.3V")]
    markers = [Marker((9, 9), MarkerKind.UNFIXABLE)]

    created = commit.commit_units(FakeKiCad(board), board, "pourstitch", units, markers)

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
    assert created == 3


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


@pytest.fixture
def log():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{level} {message}")
    yield lines
    logger.remove(sink)


class PushFailingBoard(RecordingBoard):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def push_commit(self, c, message=""):
        self.calls.append(("push_commit", message))
        raise self.error


def test_push_timeout_keeps_the_commit_and_warns(log):
    # The commit may have landed: dropping it then would be wrong.
    board = PushFailingBoard(ConnectionError("timed out"))

    commit.commit_units(FakeKiCad(board), board, "x", [unit(1, 1)], [])  # type: ignore[arg-type]

    assert "drop_commit" not in board.calls
    assert any(
        line.startswith("WARNING") and "lost the connection" in line and "undo" in line
        for line in log
    )


def test_push_refused_by_kicad_drops_the_commit_and_raises():
    # An ApiError is KiCad's answer: nothing landed, so the commit is dropped.
    board = PushFailingBoard(ApiError("bad", code=ApiStatusCode.AS_BAD_REQUEST))

    with pytest.raises(ApiError):
        commit.commit_units(FakeKiCad(board), board, "x", [unit(1, 1)], [])  # type: ignore[arg-type]

    assert board.calls[-1] == "drop_commit"


def test_push_failing_otherwise_is_raised_without_dropping_the_commit():
    board = PushFailingBoard(RuntimeError("unexpected"))

    with pytest.raises(RuntimeError):
        commit.commit_units(FakeKiCad(board), board, "x", [unit(1, 1)], [])  # type: ignore[arg-type]

    assert "drop_commit" not in board.calls


def test_refill_error_after_the_commit_is_only_a_warning(log):
    class RefillFailingBoard(RecordingBoard):
        def refill_zones(self, block=True, max_poll_seconds=0):
            raise ApiError("bad", code=ApiStatusCode.AS_BAD_REQUEST)

    board = RefillFailingBoard()

    commit.commit_units(FakeKiCad(board), board, "x", [unit(1, 1)], [])  # type: ignore[arg-type]

    assert len(pushes(board)) == 1
    assert any(line.startswith("WARNING") and "refill" in line for line in log)


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
    """Raises `error` from every refill, or each of `errors` in turn."""

    def __init__(self, error=None, errors=()):
        self.error = error
        self.errors = list(errors)
        self.refills = 0
        self.blocking = []

    def refill_zones(self, block=True, max_poll_seconds=0):
        self.refills += 1
        self.blocking.append(block)
        error = self.errors.pop(0) if self.errors else self.error
        if error:
            raise error


def busy():
    return ApiError("busy", code=ApiStatusCode.AS_BUSY)


class FakeClock:
    """Time that moves only when the code under test sleeps or a ping waits."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_refill_success():
    board = RefillBoard(None)

    assert commit.refill_tolerant(FakeKiCad(None), board, sleep=lambda s: None)
    # kipy's blocking refill has no working time limit, so ours bounds the wait.
    assert board.blocking == [False]


def test_refill_gives_up_when_kicad_stays_busy():
    kicad = FakeKiCad(None)
    kicad.pings = [busy()] * 10
    clock = FakeClock()

    ok = commit.refill_tolerant(
        kicad, RefillBoard(None), timeout_s=3, poll_s=1, sleep=clock.sleep, clock=clock
    )

    assert not ok
    assert len(kicad.pings) == 7  # three polls, then the deadline


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
    clock = FakeClock()

    ok = commit.refill_tolerant(
        kicad,
        RefillBoard(ConnectionError("timeout")),
        timeout_s=3,
        poll_s=1,
        sleep=clock.sleep,
        clock=clock,
    )

    assert not ok


def test_refill_wait_is_bounded_in_time_not_polls():
    # A ping to a KiCad that is filling waits out the client timeout, so a
    # poll count would stretch the wait to several times timeout_s.
    clock = FakeClock()
    pings = []

    class SlowKiCad:
        def ping(self):
            pings.append(clock.now)
            clock.sleep(2)
            raise ConnectionError("timeout")

    ok = commit.refill_tolerant(
        SlowKiCad(),  # type: ignore[arg-type]
        RefillBoard(None),
        timeout_s=6,
        poll_s=1,
        sleep=clock.sleep,
        clock=clock,
    )

    assert not ok
    assert len(pings) == 2
    assert clock.now <= 6 + 1 + 2  # at most one poll and one ping past it


def test_refill_accepted_at_the_deadline_still_asks_kicad_once():
    kicad = FakeKiCad(None)
    kicad.pings = [None]
    clock = FakeClock()

    ok = commit.refill_tolerant(
        kicad,
        RefillBoard(errors=[busy()] * 3),
        timeout_s=3,
        poll_s=1,
        sleep=clock.sleep,
        clock=clock,
    )

    assert ok
    assert kicad.pings == []


def test_refill_reraises_real_errors():
    with pytest.raises(ApiError):
        commit.refill_tolerant(
            FakeKiCad(None),
            RefillBoard(ApiError("bad", code=ApiStatusCode.AS_BAD_REQUEST)),
            sleep=lambda s: None,
        )


def test_refill_refused_as_busy_is_retried():
    # AS_BUSY means KiCad refused to start the refill, not that it is running.
    board = RefillBoard(errors=[busy(), busy()])
    sleeps = []

    ok = commit.refill_tolerant(FakeKiCad(None), board, sleep=sleeps.append)

    assert ok
    assert board.refills == 3
    assert len(sleeps) == 3  # two retries, then one wait for KiCad to answer


def test_refill_busy_is_logged_once_naming_the_likely_cause(log):
    board = RefillBoard(errors=[busy(), busy(), busy()])

    commit.refill_tolerant(FakeKiCad(None), board, sleep=lambda s: None)  # type: ignore[arg-type]

    busy_lines = [line for line in log if "busy" in line]
    assert len(busy_lines) == 1
    assert busy_lines[0].startswith("WARNING") and "dialog" in busy_lines[0]


def test_refill_busy_until_the_deadline_is_not_success():
    kicad = FakeKiCad(None)
    board = RefillBoard(busy())
    clock = FakeClock()

    ok = commit.refill_tolerant(
        kicad, board, timeout_s=3, poll_s=1, sleep=clock.sleep, clock=clock
    )

    assert not ok
    assert board.refills == 4  # the first call plus one per poll


@pytest.mark.parametrize("refilled", [True, False])
def test_measure_only_refills_and_snapshots_with_an_empty_plan(
    monkeypatch, log, refilled
):
    board, kicad, snap = object(), object(), Snapshot(copper_layers=("F.Cu",))
    refills = []
    monkeypatch.setattr(
        placement,
        "refill_tolerant",
        lambda k, b: refills.append((k, b)) or refilled,
    )
    monkeypatch.setattr(Snapshot, "from_board", staticmethod(lambda b: snap))

    result = placement.measure_only(kicad, board)  # type: ignore[arg-type]

    assert result == (snap, Plan())
    assert refills == [(kicad, board)]
    stale = [m for m in log if "fills may be stale" in m]
    assert len(stale) == (0 if refilled else 1)
    assert all(m.startswith("WARNING") for m in stale)
