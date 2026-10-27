import argparse
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from loguru import logger
from PIL import Image
from scipy import ndimage
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from breakneck import cli, drc, overlay, pourstitch_command, render, returnvia_command
from breakneck.commit import eco2_segment
from breakneck.placement import RunResult
from breakneck.plan import QUANTISATION, Plan, PlannedUnit
from breakneck.pourstitch import Options
from breakneck.raster import Grid
from breakneck.returnpath import NoBoardOutlineError
from breakneck.returnvia import NoGroundNetsError, ReturnViaOptions
from breakneck.snapshot import Snapshot, Track, Via, ViaSize, Zone

FOUR_LAYERS = ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")
INNER = ("In1.Cu", "In2.Cu")
USER = (BL.BL_User_1, BL.BL_User_2, BL.BL_User_3, BL.BL_User_4)
BOARD = box(0, 0, 20, 10)
GRID = Grid.covering(BOARD.bounds)
DISTANCE = 1.5
SIG = (5.0, 5.0)
ANTIPAD = 0.5
VIA = ViaSize(0.6, 0.3)
OPTIONS = ReturnViaOptions(distance=DISTANCE)
STYLE = returnvia_command.STYLE
WHITE = (255, 255, 255, 255)
# The panels' axes frames are the only black in a --render figure.
FRAME_MAX_RGB_SUM = 20
MIN_FRAME_PX = 100
# A ring is an antialiased hairline, a little dimmer than white and a pixel
# either side of where it is drawn; the scale's bright end is much further off.
NEAR_WHITE_RGB_DIFF = 120
RING_REACH_PX = 1


def via(net, pos):
    return Via(f"{net}@{pos}", net, pos, VIA.diameter, VIA.drill)


def antipads(*positions):
    return unary_union([Point(p).buffer(ANTIPAD) for p in positions])


def gnd_zone(fills, net="GND", outline=BOARD):
    return Zone(net, tuple(fills), 0, outline, fills)


def snapshot(signals=(SIG,), returns=(), zones=None):
    """Signal vias with F.Cu and B.Cu tracks: their ends reference In1 and In2.

    By default GND fills both inner layers but for the signal vias' antipads.
    """
    vias, tracks = [], []
    for s in signals:
        vias.append(via("SIG", s))
        tracks += [
            Track(f"SIG:{L}@{s}", "SIG", L, 0.2, LineString([s, (s[0] - 3, s[1])]))
            for L in ("F.Cu", "B.Cu")
        ]
    if zones is None:
        plain = BOARD.difference(antipads(*signals))
        zones = [gnd_zone({L: plain for L in INNER})]
    return Snapshot(
        copper_layers=FOUR_LAYERS,
        vias=vias + [via(n, p) for n, p in returns],
        tracks=tracks,
        zones=zones,
        edge_paths=[list(BOARD.exterior.coords)],
        via_sizes={"GND": VIA},
    )


def slotted(ret):
    """GND planes with a slot in In1 between SIG and a return east of it."""
    flush = BOARD.difference(antipads(SIG))
    slot = box(SIG[0] + 0.6, SIG[1] - 2, SIG[0] + 0.7, SIG[1] + 2)
    zones = [gnd_zone({"In1.Cu": flush.difference(slot), "In2.Cu": flush})]
    return snapshot(returns=[("GND", ret)], zones=zones)


class Board:
    def __init__(self, images=(), outline=BOARD, enabled=USER):
        self.calls: list = []
        self.images = list(images)
        self.removed: list = []
        self.outline = outline
        self.enabled = [BL.BL_F_Cu, BL.BL_In1_Cu, BL.BL_In2_Cu, BL.BL_B_Cu, *enabled]

    def get_copper_layer_count(self):
        return len(FOUR_LAYERS)

    def get_enabled_layers(self):
        return self.enabled

    def get_shapes(self):
        if self.outline is None:
            return []
        corners = list(self.outline.exterior.coords)
        edges = [eco2_segment(a, b) for a, b in pairwise(corners)]
        for e in edges:
            e.layer = BL.BL_Edge_Cuts
        return edges

    def get_reference_images(self):
        return self.images

    def refill_zones(self, block=True, max_poll_seconds=0):
        self.calls.append("refill_zones")

    def begin_commit(self):
        self.calls.append("begin_commit")
        return "c"

    def remove_items(self, items):
        self.calls.append("remove_items")
        self.removed += items

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


class Fakes:
    def __init__(self):
        self.snap = snapshot()
        self.created: list = []
        self.planned: list = []
        self.plan: Plan | None = None


@pytest.fixture
def fakes(monkeypatch):
    f = Fakes()

    def fake_run(planner, options, settings, kicad):
        f.planned.append(settings)
        plan = f.plan or planner(f.snap, options, [])
        return RunResult(f.snap, drc.PruneResult(plan, (), (), (), 1))

    monkeypatch.setattr(Snapshot, "from_board", staticmethod(lambda board: f.snap))
    monkeypatch.setattr(
        overlay, "create_checked", lambda b, items: f.created.extend(items)
    )
    monkeypatch.setattr(returnvia_command, "run_placement", fake_run)
    monkeypatch.setattr(pourstitch_command, "run_placement", fake_run)
    return f


@pytest.fixture
def log():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{message}")
    yield lines
    logger.remove(sink)


@pytest.fixture
def info_log():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{message}", level="INFO")
    yield lines
    logger.remove(sink)


def run(board, tmp_path=None, check=True, dry_run=False, **outputs):
    if tmp_path is not None:
        outputs.setdefault("overlay_dir", tmp_path)
    returnvia_command.execute(
        OPTIONS,
        overlay.Outputs(**outputs),
        check=check,
        dry_run=dry_run,
        kicad=KiCad(board),  # type: ignore[arg-type]
    )


def rgba(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def at(image: np.ndarray, point) -> tuple[int, ...]:
    return tuple(int(v) for v in image[GRID.index(point)])


def scale_colour(fraction: float) -> tuple[int, ...]:
    return tuple(
        int(v) for v in render.plt.get_cmap(STYLE.colormap)(fraction, bytes=True)
    )


def colour(name: str) -> tuple[int, ...]:
    return tuple(round(c * 255) for c in render.matplotlib.colors.to_rgba(name))


def panel_frames(path: Path) -> tuple[np.ndarray, list[tuple[slice, slice]]]:
    """The --render figure's RGB and its panels' frames, left to right."""
    image = np.asarray(Image.open(path).convert("RGB")).astype(int)
    labels, _ = ndimage.label(image.sum(axis=2) < FRAME_MAX_RGB_SUM)
    frames = [
        f
        for f in ndimage.find_objects(labels)
        if f[0].stop - f[0].start > MIN_FRAME_PX
        and f[1].stop - f[1].start > MIN_FRAME_PX
    ]
    return image, sorted(frames, key=lambda f: f[1].start)


def white_near(image: np.ndarray, frame: tuple[slice, slice], point) -> bool:
    """Whether a panel shows near-white at board `point`."""
    x0, y0, x1, _ = BOARD.bounds
    px_per_mm = (frame[1].stop - frame[1].start) / (x1 - x0 + 2 * render.MARGIN_MM)
    r = frame[0].start + round((point[1] - y0 + render.MARGIN_MM) * px_per_mm)
    c = frame[1].start + round((point[0] - x0 + render.MARGIN_MM) * px_per_mm)
    k = RING_REACH_PX
    near = image[r - k : r + k + 1, c - k : c + k + 1] - np.array(WHITE[:3])
    return bool((np.abs(near).sum(axis=2) < NEAR_WHITE_RGB_DIFF).any())


def test_check_overlay_images_only_the_reference_planes(fakes, tmp_path: Path):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    board = Board()

    run(board, tmp_path)

    assert fakes.planned == []
    assert board.calls == [
        "refill_zones",
        "begin_commit",
        ("push_commit", returnvia_command.OVERLAY_COMMIT),
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "returnvia-In1_Cu.png",
        "returnvia-In2_Cu.png",
    ]
    assert sorted(i.layer for i in fakes.created) == [BL.BL_User_2, BL.BL_User_3]


def test_unserved_end_gets_a_ring_only_on_its_plane(fakes, tmp_path: Path):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))

    run(Board(), tmp_path)

    on_ring = (SIG[0] - DISTANCE, SIG[1])
    assert at(rgba(tmp_path / "returnvia-In1_Cu.png"), on_ring) == WHITE
    assert at(rgba(tmp_path / "returnvia-In2_Cu.png"), on_ring) != WHITE


def test_check_logs_each_unserved_via_plane_and_commits_nothing(fakes, log):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))
    board = Board()

    run(board)

    assert board.calls == ["refill_zones"]
    assert fakes.created == []
    (line,) = [m for m in log if "unserved:" in m]
    assert "SIG at (5.000, 5.000) on In1.Cu (GND):" in line
    distance = float(line.rsplit(":", 1)[1].split()[0])
    assert distance > 2 * DISTANCE
    assert any("1 of 1 vias unserved" in m and "1 of 2 via-plane" in m for m in log)


def test_unserved_lines_are_debug_and_the_summary_info(fakes, log, info_log):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))
    fakes.snap.vias = [*fakes.snap.vias, via("SIG", (15.0, 5.0))]

    run(Board())

    assert [m for m in log if "unserved:" in m]
    assert not [m for m in info_log if "unserved:" in m]
    assert any("1 vias exempt" in m for m in info_log)
    assert any("1 of 1 vias unserved" in m for m in info_log)


def test_no_ground_fill_line_is_debug(fakes, log, info_log):
    far = box(15, 0, 20, 10)
    flush = BOARD.difference(antipads(SIG))
    fakes.snap = snapshot(
        returns=[("GND", (6.0, 5.0))],
        zones=[gnd_zone({"In1.Cu": far, "In2.Cu": flush})],
    )

    run(Board())

    assert [m for m in log if "no GND fill at the via" in m]
    assert not [m for m in info_log if "no GND fill at the via" in m]


def test_via_with_no_path_to_a_return_logs_an_infinite_distance(fakes, log):
    fakes.snap = snapshot()

    run(Board())

    lines = [m for m in log if "unserved:" in m]
    assert len(lines) == len(INNER)
    assert all(line.rstrip().endswith("inf mm to a return") for line in lines)


def test_layer_outside_every_ground_zone_is_logged_and_counted(fakes, log, tmp_path):
    # The only ground at the via is on F.Cu: the B.Cu end references it,
    # the F.Cu end has no other layer to reference.
    left = box(0, 0, 8, 10)
    fills = {"F.Cu": left.difference(antipads(SIG))}
    fakes.snap = snapshot(
        returns=[("GND", (6.0, 5.0))], zones=[gnd_zone(fills, outline=left)]
    )

    run(Board(), tmp_path)

    (line,) = [m for m in log if "unserved:" in m]
    assert line.rstrip().endswith("outside every ground zone for F.Cu")
    assert any("1 of 1 vias unserved" in m for m in log)
    # No image to draw the F.Cu end's ring on; F.Cu itself is served.
    assert [p.name for p in tmp_path.iterdir()] == ["returnvia-F_Cu.png"]
    assert not (rgba(tmp_path / "returnvia-F_Cu.png") == WHITE).all(axis=2).any()


def test_every_via_served_logs_zero_and_draws_no_ring(fakes, log, tmp_path):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])

    run(Board(), tmp_path)

    assert not [m for m in log if "unserved:" in m]
    assert any("0 of 1 vias unserved" in m for m in log)
    for name in ("returnvia-In1_Cu.png", "returnvia-In2_Cu.png"):
        assert not (rgba(tmp_path / name) == WHITE).all(axis=2).any()


def test_exempt_vias_are_counted(fakes, log):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    fakes.snap.vias = [*fakes.snap.vias, via("SIG", (15.0, 5.0))]

    run(Board())

    assert any("1 vias exempt" in m for m in log)


def test_split_plane_composites_both_ground_nets(fakes, tmp_path: Path):
    left, right = box(0, 0, 10, 10), box(10.3, 0, 20, 10)
    iso_sig, iso_ret = (15.0, 5.0), (16.0, 5.0)
    holes = antipads(SIG, iso_sig)
    zones = [
        gnd_zone({"In1.Cu": left.difference(holes)}, outline=left),
        gnd_zone({"In1.Cu": right.difference(holes)}, "GND_ISO", right),
        gnd_zone({"In2.Cu": BOARD.difference(holes)}),
    ]
    fakes.snap = snapshot(
        signals=(SIG, iso_sig),
        returns=[("GND", (6.0, 5.0)), ("GND_ISO", iso_ret)],
        zones=zones,
    )

    returnvia_command.execute(
        ReturnViaOptions(distance=DISTANCE, ground_nets=("GND", "GND_ISO")),
        overlay.Outputs(overlay_dir=tmp_path),
        check=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )

    image = rgba(tmp_path / "returnvia-In1_Cu.png")
    near = scale_colour(0.0)
    assert at(image, (6.0, 5.0)) == near
    assert at(image, iso_ret) == near
    assert at(image, (10.15, 5.0))[3] == 0


def test_no_reference_plane_removes_the_previous_set(fakes, tmp_path: Path):
    pourstitch_command.execute(
        Options(nets=("GND",)),
        overlay.Outputs(overlay_dir=tmp_path / "ps"),
        check=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )
    previous = list(fakes.created)
    assert previous
    fakes.snap = snapshot(signals=())
    board = Board(previous)

    run(board, tmp_path / "rv")

    assert not (tmp_path / "rv").exists() or not list((tmp_path / "rv").iterdir())
    assert board.calls == [
        "refill_zones",
        "begin_commit",
        "remove_items",
        ("push_commit", returnvia_command.OVERLAY_COMMIT),
    ]
    assert board.removed == previous


def test_no_reference_plane_and_no_previous_set_commits_nothing(fakes, tmp_path):
    fakes.snap = snapshot(signals=())
    board = Board()

    run(board, tmp_path)

    assert board.calls == ["refill_zones"]


def test_without_a_ground_net_the_previous_set_stays(fakes, tmp_path: Path):
    previous = [object()]
    fills = {"In1.Cu": BOARD.difference(antipads(SIG))}
    fakes.snap = snapshot(zones=[gnd_zone(fills, net="PWR")])
    board = Board(previous)

    with pytest.raises(NoGroundNetsError):
        run(board, tmp_path)

    assert board.calls == ["refill_zones"]
    assert fakes.created == []


def test_dry_run_overlay_shows_the_planned_vias(fakes, log, tmp_path: Path):
    fakes.snap = snapshot()
    board = Board()

    run(board, tmp_path, check=False, dry_run=True)

    assert [s.dry_run for s in fakes.planned] == [True]
    assert board.calls == [
        "begin_commit",
        ("push_commit", returnvia_command.OVERLAY_COMMIT),
    ]
    assert any("0 of 1 vias unserved" in m for m in log)
    assert fakes.created
    image = rgba(tmp_path / "returnvia-In1_Cu.png")
    # Without the planned via no copper reaches a return at all.
    unreachable = colour(STYLE.unreachable_colour)
    assert at(image, (1.0, 1.0)) != unreachable
    assert not (image == WHITE).all(axis=2).any()


def test_forced_vias_serve_nothing(fakes, log):
    fakes.snap = snapshot()
    fakes.plan = Plan(units=(PlannedUnit((6.0, 5.0), "GND", VIA, forced=True),))

    run(Board(), check=False, dry_run=True)

    assert any("1 of 1 vias unserved" in m for m in log)


def test_returnvia_and_pourstitch_replace_each_others_overlays(fakes, tmp_path):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    pourstitch_command.execute(
        Options(nets=("GND",)),
        overlay.Outputs(overlay_dir=tmp_path / "ps"),
        check=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )
    pourstitch_set = list(fakes.created)
    fakes.created.clear()

    board = Board(pourstitch_set)
    run(board, tmp_path / "rv")
    assert board.removed == pourstitch_set
    returnvia_set = list(fakes.created)
    assert returnvia_set

    board = Board(returnvia_set)
    pourstitch_command.execute(
        Options(nets=("GND",)),
        overlay.Outputs(overlay_dir=tmp_path / "ps"),
        check=True,
        kicad=KiCad(board),  # type: ignore[arg-type]
    )
    assert board.removed == returnvia_set


def test_clear_removes_the_returnvia_overlays(fakes, tmp_path, monkeypatch):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    run(Board(), tmp_path)
    assert fakes.created
    board = Board(list(fakes.created))
    monkeypatch.setattr(cli.kipy, "KiCad", lambda: KiCad(board))

    cli.execute_clear(
        argparse.Namespace(
            overlay_layers=list(overlay.DEFAULT_OVERLAY_LAYERS), dry_run=False
        )
    )

    assert board.removed == fakes.created


def test_image_shows_the_threshold_contour_and_saturates(fakes, tmp_path: Path):
    ret = (4.0, 5.0)
    fakes.snap = snapshot(returns=[("GND", ret)])

    run(Board(), tmp_path)

    image = rgba(tmp_path / "returnvia-In2_Cu.png")
    contour = colour(STYLE.contour.colour)  # type: ignore[union-attr]
    row = GRID.index(ret)[0]
    xs = [GRID.xs[c] for c in range(GRID.width) if tuple(image[row, c]) == contour]
    threshold = DISTANCE + QUANTISATION
    # East of the return the path runs around SIG's antipad; west it is straight.
    west = [x for x in xs if x < ret[0]]
    assert west and all(
        math.isclose(ret[0] - x, threshold, abs_tol=2 * GRID.pixel) for x in west
    )
    far = (ret[0] + returnvia_command.SCALE * DISTANCE + 2.0, 2.0)
    assert at(image, far) == scale_colour(1.0)


def test_board_without_outline_fails_before_the_refill(fakes, tmp_path):
    board = Board(outline=None)

    with pytest.raises(NoBoardOutlineError):
        run(board, tmp_path)

    assert board.calls == []


def test_disabled_overlay_layer_fails_before_any_board_change(fakes, tmp_path):
    board = Board(enabled=USER[:2])

    with pytest.raises(overlay.OverlayLayersError, match="User.3"):
        run(board, tmp_path, check=False)

    assert fakes.planned == []
    assert board.calls == []


def test_disabled_overlay_layer_in_dry_run_writes_pngs_only(fakes, tmp_path):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    board = Board(enabled=())

    run(board, tmp_path, check=False, dry_run=True)

    assert board.calls == []
    assert fakes.created == []
    assert len(list(tmp_path.glob("*.png"))) == len(INNER)


def test_too_few_overlay_layers_is_an_error(fakes, tmp_path):
    with pytest.raises(overlay.OverlayLayersError):
        run(Board(), tmp_path, overlay_layers=("User.1", "User.2"))


def test_render_writes_the_panels(fakes, tmp_path):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))
    board = Board()

    run(board, render=tmp_path / "heat.png")

    assert (tmp_path / "heat.png").exists()
    assert board.calls == ["refill_zones"]


def test_render_panels_ring_the_unserved_end_on_its_plane(fakes, tmp_path):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))

    run(Board(), render=tmp_path / "heat.png")

    image, (in1, in2) = panel_frames(tmp_path / "heat.png")
    on_ring = (SIG[0] - DISTANCE, SIG[1])
    assert white_near(image, in1, on_ring)
    assert not white_near(image, in2, on_ring)


def test_no_ground_fill_end_is_logged_and_ringed(fakes, log, tmp_path: Path):
    # GND covers the via on both planes, but In1's fill lies beyond --distance.
    far = box(15, 0, 20, 10)
    flush = BOARD.difference(antipads(SIG))
    fakes.snap = snapshot(
        returns=[("GND", (6.0, 5.0))],
        zones=[gnd_zone({"In1.Cu": far, "In2.Cu": flush})],
    )

    run(Board(), tmp_path)

    (line,) = [m for m in log if "unserved:" in m]
    assert line.rstrip().endswith(
        "SIG at (5.000, 5.000): no GND fill at the via on In1.Cu"
    )
    on_ring = (SIG[0] - DISTANCE, SIG[1])
    assert at(rgba(tmp_path / "returnvia-In1_Cu.png"), on_ring) == WHITE
    assert at(rgba(tmp_path / "returnvia-In2_Cu.png"), on_ring) != WHITE


def test_render_creates_the_missing_parent_directory(fakes, tmp_path: Path):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))
    path = tmp_path / "out" / "heat.png"

    run(Board(), render=path)

    assert path.exists()
