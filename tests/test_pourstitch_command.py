from pathlib import Path

import pytest
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from loguru import logger
from shapely.geometry import box

from breakneck import drc, overlay, pourstitch_command
from breakneck.placement import RunResult, RunSettings
from breakneck.pourstitch import Options
from breakneck.snapshot import Snapshot, ViaSize, Zone

LAYERS = ("F.Cu", "B.Cu")
POUR = box(0, 0, 10, 10)
POUR_MM2 = 100.0


def snapshot() -> Snapshot:
    zone = Zone("GND", LAYERS, 0, POUR, {L: POUR for L in LAYERS})
    return Snapshot(
        copper_layers=LAYERS,
        zones=[zone],
        edge_paths=[[(x, y) for x, y in POUR.exterior.coords]],
        via_sizes={"GND": ViaSize(0.6, 0.3)},
    )


class Board:
    def __init__(self, enabled=(BL.BL_User_1, BL.BL_User_2)):
        self.calls: list = []
        self.enabled = [BL.BL_F_Cu, BL.BL_B_Cu, *enabled]

    def get_copper_layer_count(self):
        return 2

    def get_enabled_layers(self):
        return self.enabled

    def get_reference_images(self):
        return []

    def refill_zones(self, block=True, max_poll_seconds=0):
        self.calls.append("refill_zones")

    def begin_commit(self):
        self.calls.append("begin_commit")
        return "c"

    def remove_items(self, items):
        self.calls.append("remove_items")

    def push_commit(self, c, message=""):
        self.calls.append(("push_commit", message))

    def drop_commit(self, c):
        self.calls.append("drop_commit")


class KiCad:
    def __init__(self, board):
        self.board = board

    def get_board(self):
        return self.board


@pytest.fixture
def fakes(monkeypatch):
    snap = snapshot()
    created: list = []
    placements: list[RunSettings] = []

    def fake_run(planner, options, settings, kicad):
        placements.append(settings)
        plan = planner(snap, options, [])
        return RunResult(snap, drc.PruneResult(plan, (), (), (), 1))

    monkeypatch.setattr(Snapshot, "from_board", staticmethod(lambda board: snap))
    monkeypatch.setattr(
        overlay, "create_checked", lambda b, items: created.extend(items)
    )
    monkeypatch.setattr(pourstitch_command, "run_placement", fake_run)
    return created, placements


@pytest.fixture
def log():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{message}")
    yield lines
    logger.remove(sink)


OPTIONS = Options(nets=("GND",))


def test_check_reports_whole_unstitched_pour_and_commits_nothing(fakes, log):
    created, placements = fakes
    board = Board()

    pourstitch_command.execute(
        OPTIONS,
        pourstitch_command.Outputs(),
        check=True,
        kicad=KiCad(board),  # type: ignore[arg-type]
    )

    assert placements == []
    assert board.calls == ["refill_zones"]
    assert created == []
    reports = [line for line in log if "GND on" in line]
    assert len(reports) == len(LAYERS)
    assert all(f"{POUR_MM2:.2f} mm²" in line for line in reports)
    assert all("0 existing stitches" in line for line in reports)


def test_check_with_overlay_creates_only_the_image_commit(fakes, tmp_path: Path):
    created, placements = fakes
    board = Board()
    outputs = pourstitch_command.Outputs(overlay_dir=tmp_path)

    pourstitch_command.execute(OPTIONS, outputs, check=True, kicad=KiCad(board))  # type: ignore[arg-type]

    assert placements == []
    assert board.calls == [
        "refill_zones",
        "begin_commit",
        ("push_commit", overlay.OVERLAY_COMMIT),
    ]
    assert len(created) == len(LAYERS)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "pourstitch-B_Cu.png",
        "pourstitch-F_Cu.png",
    ]


def test_dry_run_still_places_overlay(fakes, tmp_path: Path):
    created, placements = fakes
    board = Board()
    outputs = pourstitch_command.Outputs(
        overlay_dir=tmp_path, render=tmp_path / "heat.png"
    )

    pourstitch_command.execute(OPTIONS, outputs, dry_run=True, kicad=KiCad(board))  # type: ignore[arg-type]

    assert [s.dry_run for s in placements] == [True]
    assert board.calls == ["begin_commit", ("push_commit", overlay.OVERLAY_COMMIT)]
    assert len(created) == len(LAYERS)
    assert (tmp_path / "heat.png").exists()


def test_disabled_overlay_layer_fails_before_any_board_change(fakes, tmp_path: Path):
    _, placements = fakes
    board = Board(enabled=(BL.BL_User_1,))
    outputs = pourstitch_command.Outputs(overlay_dir=tmp_path)

    with pytest.raises(pourstitch_command.OverlayLayersError, match="User.2"):
        pourstitch_command.execute(OPTIONS, outputs, kicad=KiCad(board))  # type: ignore[arg-type]

    assert placements == []
    assert board.calls == []


def test_disabled_overlay_layer_in_dry_run_writes_pngs_only(fakes, tmp_path: Path):
    created, _ = fakes
    board = Board(enabled=())
    outputs = pourstitch_command.Outputs(overlay_dir=tmp_path)

    pourstitch_command.execute(OPTIONS, outputs, dry_run=True, kicad=KiCad(board))  # type: ignore[arg-type]

    assert board.calls == []
    assert created == []
    assert len(list(tmp_path.glob("*.png"))) == len(LAYERS)


def test_too_few_overlay_layers_is_an_error(fakes, tmp_path: Path):
    outputs = pourstitch_command.Outputs(
        overlay_dir=tmp_path, overlay_layers=("User.1",)
    )

    with pytest.raises(pourstitch_command.OverlayLayersError):
        pourstitch_command.execute(OPTIONS, outputs, kicad=KiCad(Board()))  # type: ignore[arg-type]


def test_planned_run_reports_coverage_of_the_plan(fakes, log):
    pourstitch_command.execute(
        OPTIONS,
        pourstitch_command.Outputs(),
        dry_run=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )

    reports = [line for line in log if "GND on" in line]
    assert reports and all("0.00 mm²" in line for line in reports)


def test_planned_run_leaves_uncovered_spots_unmarked(fakes):
    # The heatmap shows uncovered areas; per-spot Eco2 markers would be noise.
    _, placements = fakes

    pourstitch_command.execute(
        OPTIONS,
        pourstitch_command.Outputs(),
        dry_run=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )

    assert [s.mark_unfixable for s in placements] == [False]
