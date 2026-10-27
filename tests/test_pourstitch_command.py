import argparse
from itertools import pairwise
from pathlib import Path

import pytest
from kipy.geometry import Vector2
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from shapely.geometry import box

from breakneck import cli, drc, overlay, pourstitch_command, render
from breakneck.commit import eco2_segment
from breakneck.placement import RunResult, RunSettings
from breakneck.plan import POURSTITCH_POWER_DISTANCE, StitchClass
from breakneck.pourstitch import Options
from breakneck.snapshot import NM_PER_MM, Snapshot, ViaSize, Zone

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

    def ping(self):
        pass


@pytest.fixture
def fakes(monkeypatch):
    snap = snapshot()
    created: list = []
    placements: list[RunSettings] = []

    def fake_run(planner, options, settings, kicad):
        placements.append(settings)
        planned = Snapshot.from_board(kicad.get_board())
        plan = planner(planned, options, [])
        return RunResult(planned, drc.PruneResult(plan, (), (), (), 1))

    monkeypatch.setattr(Snapshot, "from_board", staticmethod(lambda board: snap))
    monkeypatch.setattr(
        overlay, "create_checked", lambda b, items: created.extend(items)
    )
    monkeypatch.setattr(pourstitch_command, "run_placement", fake_run)
    return created, placements


OPTIONS = Options(nets=("GND",))


def test_check_reports_whole_unstitched_pour_and_commits_nothing(fakes, log):
    created, placements = fakes
    board = Board()

    pourstitch_command.execute(
        OPTIONS,
        overlay.Outputs(),
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


def test_check_with_overlay_creates_only_the_image_commit(
    fakes, tmp_path: Path, png_dir: Path, monkeypatch
):
    created, placements = fakes
    board = Board()
    outputs = overlay.Outputs(overlay=True)
    monkeypatch.chdir(tmp_path)

    pourstitch_command.execute(OPTIONS, outputs, check=True, kicad=KiCad(board))  # type: ignore[arg-type]

    assert placements == []
    assert board.calls == [
        "refill_zones",
        "begin_commit",
        ("push_commit", pourstitch_command.OVERLAY_COMMIT),
    ]
    assert len(created) == len(LAYERS)
    # The placed images carry the PNG data; no file is left behind.
    assert all(i.image_data for i in created)
    assert list(tmp_path.iterdir()) == []


def test_dry_run_still_places_overlay(fakes, tmp_path: Path):
    created, placements = fakes
    board = Board()
    outputs = overlay.Outputs(overlay=True, render=tmp_path / "heat.png")

    pourstitch_command.execute(OPTIONS, outputs, dry_run=True, kicad=KiCad(board))  # type: ignore[arg-type]

    assert [s.dry_run for s in placements] == [True]
    assert board.calls == [
        "begin_commit",
        ("push_commit", pourstitch_command.OVERLAY_COMMIT),
    ]
    assert len(created) == len(LAYERS)
    assert (tmp_path / "heat.png").exists()


def test_disabled_overlay_layer_fails_before_any_board_change(fakes):
    _, placements = fakes
    board = Board(enabled=(BL.BL_User_1,))
    outputs = overlay.Outputs(overlay=True)

    with pytest.raises(overlay.OverlayLayersError, match="User.2"):
        pourstitch_command.execute(OPTIONS, outputs, kicad=KiCad(board))  # type: ignore[arg-type]

    assert placements == []
    assert board.calls == []


def test_disabled_overlay_layer_in_dry_run_keeps_the_pngs(fakes, png_dir, log):
    created, _ = fakes
    board = Board(enabled=())
    outputs = overlay.Outputs(overlay=True)

    pourstitch_command.execute(OPTIONS, outputs, dry_run=True, kicad=KiCad(board))  # type: ignore[arg-type]

    assert board.calls == []
    assert created == []
    pngs = sorted(png_dir.glob("*.png"))
    assert len(pngs) == len(LAYERS)
    assert all(any(str(p) in line for line in log) for p in pngs)


def test_too_few_overlay_layers_is_an_error(fakes):
    outputs = overlay.Outputs(overlay=True, overlay_layers=("User.1",))

    with pytest.raises(overlay.OverlayLayersError):
        pourstitch_command.execute(OPTIONS, outputs, kicad=KiCad(Board()))  # type: ignore[arg-type]


def test_planned_run_reports_coverage_of_the_plan(fakes, log):
    pourstitch_command.execute(
        OPTIONS,
        overlay.Outputs(),
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
        overlay.Outputs(),
        dry_run=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )

    assert [s.mark_unfixable for s in placements] == [False]


ISLAND = box(12, 2, 16, 6)
ISLAND_MM2 = 16.0


def island_snapshot() -> Snapshot:
    """The GND pour beside a GND local zone on F.Cu only, which no via can join."""
    return Snapshot(
        copper_layers=LAYERS,
        zones=[
            Zone("GND", LAYERS, 0, POUR, {L: POUR for L in LAYERS}),
            Zone("GND", ("F.Cu",), 0, ISLAND, {"F.Cu": ISLAND}),
        ],
        edge_paths=[[(x, y) for x, y in box(0, 0, 20, 10).exterior.coords]],
        via_sizes={"GND": ViaSize(0.6, 0.3)},
    )


def test_check_reports_unstitchable_copper_apart_from_uncovered(
    fakes, log, monkeypatch
):
    # No via can join the island, so it is not counted as uncovered.
    monkeypatch.setattr(
        Snapshot, "from_board", staticmethod(lambda board: island_snapshot())
    )

    pourstitch_command.execute(
        OPTIONS,
        overlay.Outputs(),
        check=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )

    (front,) = [line for line in log if "GND on F.Cu" in line]
    assert f"{POUR_MM2:.2f} mm²" in front
    assert f"{ISLAND_MM2:.2f} mm² can't be stitched" in front
    (back,) = [line for line in log if "GND on B.Cu" in line]
    assert "can't be stitched" not in back


class ClearBoard(Board):
    """Holds the images an overlay run created, inside the snapshot's outline."""

    def __init__(self, images):
        super().__init__()
        self.images = images
        self.removed: list = []

    def get_shapes(self):
        corners = [(x, y) for x, y in POUR.exterior.coords]
        edges = [eco2_segment(a, b) for a, b in pairwise(corners)]
        for e in edges:
            e.layer = BL.BL_Edge_Cuts
        return edges

    def get_reference_images(self):
        return self.images

    def remove_items(self, items):
        self.removed += items


@pytest.mark.parametrize("tagged", [True, False])
def test_clear_removes_the_overlays_pourstitch_placed(fakes, monkeypatch, tagged):
    # Earlier builds placed them untagged, without a legend, at the board
    # placement: only that placement matches.
    created, _ = fakes
    outputs = overlay.Outputs(overlay=True)
    pourstitch_command.execute(OPTIONS, outputs, check=True, kicad=KiCad(Board()))  # type: ignore[arg-type]
    assert created
    if not tagged:
        x, y = overlay.board_placement(POUR.bounds).centre
        for image in created:
            image.image_data = b""
            image.position = Vector2.from_xy(round(x * NM_PER_MM), round(y * NM_PER_MM))
    board = ClearBoard(list(created))
    monkeypatch.setattr(cli.kipy, "KiCad", lambda: KiCad(board))

    cli.execute_clear(
        argparse.Namespace(
            overlay_layers=list(overlay.DEFAULT_OVERLAY_LAYERS),
            dry_run=False,
            vias=False,
            nets=None,
        )
    )

    assert board.removed == created


def test_planning_run_logs_uncovered_area_without_unstitchable_copper(
    fakes, log, monkeypatch
):
    monkeypatch.setattr(
        Snapshot, "from_board", staticmethod(lambda board: island_snapshot())
    )

    pourstitch_command.execute(
        OPTIONS,
        overlay.Outputs(),
        dry_run=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )

    (planner,) = [line for line in log if line.startswith("pourstitch GND:")]
    assert "0.00 mm² beyond" in planner
    (front,) = [line for line in log if "GND on F.Cu" in line]
    assert "0.00 mm² (0 px)" in front
    assert f"{ISLAND_MM2:.2f} mm² can't be stitched" in front


class ImagesBoard(Board):
    def __init__(self, images):
        super().__init__()
        self.images = images

    def get_reference_images(self):
        return self.images


def test_run_without_maps_removes_the_previous_overlays(fakes):
    created, _ = fakes
    outputs = overlay.Outputs(overlay=True)
    pourstitch_command.execute(OPTIONS, outputs, check=True, kicad=KiCad(Board()))  # type: ignore[arg-type]
    board = ImagesBoard(list(created))

    pourstitch_command.execute(
        Options(nets=("+3.3V",)),
        outputs,
        check=True,
        kicad=KiCad(board),  # type: ignore[arg-type]
    )

    assert board.calls == [
        "refill_zones",
        "begin_commit",
        "remove_items",
        ("push_commit", pourstitch_command.OVERLAY_COMMIT),
    ]


def test_render_creates_the_missing_parent_directory(fakes, tmp_path: Path):
    path = tmp_path / "out" / "heat.png"
    outputs = overlay.Outputs(render=path)

    pourstitch_command.execute(OPTIONS, outputs, check=True, kicad=KiCad(Board()))  # type: ignore[arg-type]

    assert path.exists()


def test_failed_placement_leaves_no_temporary_directory(fakes, png_dir, monkeypatch):
    def drop(board, items):
        raise ConnectionError("KiCad went away")

    monkeypatch.setattr(overlay, "create_checked", drop)

    with pytest.raises(ConnectionError):
        pourstitch_command.execute(
            OPTIONS,
            overlay.Outputs(overlay=True),
            check=True,
            kicad=KiCad(Board()),  # type: ignore[arg-type]
        )

    assert not png_dir.exists()


def test_unplaceable_run_without_maps_leaves_no_temporary_directory(fakes, png_dir):
    pourstitch_command.execute(
        Options(nets=("+3.3V",)),
        overlay.Outputs(overlay=True),
        dry_run=True,
        kicad=KiCad(Board(enabled=())),  # type: ignore[arg-type]
    )

    assert not png_dir.exists()


def test_power_check_measures_and_renders_at_the_power_distance(
    fakes, log, monkeypatch, tmp_path: Path
):
    power_pour = box(20, 0, 30, 10)
    snap = Snapshot(
        copper_layers=LAYERS,
        zones=[
            Zone(n, LAYERS, 0, b, {L: b for L in LAYERS})
            for n, b in (("GND", POUR), ("+3.3V", power_pour))
        ],
        edge_paths=[[(x, y) for x, y in box(0, 0, 30, 10).exterior.coords]],
    )
    monkeypatch.setattr(Snapshot, "from_board", staticmethod(lambda board: snap))
    rendered = []
    monkeypatch.setattr(
        render,
        "render_panels",
        lambda path, snapshot, grid, maps, markers, distance, style: rendered.append(
            (maps, distance)
        ),
    )
    options = Options(stitch_class=StitchClass.POWER)

    pourstitch_command.execute(
        options,
        overlay.Outputs(render=tmp_path / "heat.png"),
        check=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )

    reports = [line for line in log if " on " in line and "existing stitches" in line]
    assert reports and all(line.strip().startswith("+3.3V on") for line in reports)
    assert all(f"beyond {POURSTITCH_POWER_DISTANCE:g} mm" in line for line in reports)
    ((maps, distance),) = rendered
    assert {m.net for m in maps} == {"+3.3V"}
    assert distance == POURSTITCH_POWER_DISTANCE


def test_render_without_maps_replaces_the_previous_heatmap(
    fakes, monkeypatch, tmp_path: Path
):
    snap = Snapshot(
        copper_layers=LAYERS,
        zones=[Zone("GND", LAYERS, 0, POUR, {L: POUR for L in LAYERS})],
        edge_paths=[[(x, y) for x, y in box(0, 0, 30, 10).exterior.coords]],
    )
    monkeypatch.setattr(Snapshot, "from_board", staticmethod(lambda board: snap))
    path = tmp_path / "heat.png"
    path.write_bytes(b"stale")

    pourstitch_command.execute(
        Options(stitch_class=StitchClass.POWER),
        overlay.Outputs(render=path),
        check=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )

    assert path.read_bytes().startswith(b"\x89PNG")
