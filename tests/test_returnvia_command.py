import argparse
import io
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
from breakneck.raster import CoverageMap, Grid
from breakneck.returnpath import NoBoardOutlineError, ReturnFields, measure_returns
from breakneck.returnvia import NoGroundNetsError, ReferencePlane, ReturnViaOptions
from breakneck.snapshot import (
    NM_PER_MM,
    Pad,
    PadKind,
    Snapshot,
    Track,
    Via,
    ViaSize,
    Zone,
)

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
RING = (0xFF, 0x2A, 0x6D, 0xFF)
# The panels' axes frames are the only black in a --render figure.
FRAME_MAX_RGB_SUM = 20
MIN_FRAME_PX = 100
# Off copper, a ring is an antialiased pink hairline over the near-black
# background, a pixel either side of where it is drawn: red well above the
# background and above green and blue.
RING_MIN_RED = 100
RING_RED_OVER_GREEN = 50
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
def info_log():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{message}", level="INFO")
    yield lines
    logger.remove(sink)


def run(board, check=True, dry_run=False, full_map=False, **outputs):
    returnvia_command.execute(
        OPTIONS,
        overlay.Outputs(**outputs),
        check=check,
        dry_run=dry_run,
        full_map=full_map,
        kicad=KiCad(board),  # type: ignore[arg-type]
    )


def placed(fakes, layer: str) -> Image.Image:
    """The overlay image placed for copper `layer`."""
    user = USER[fakes.snap.copper_layers.index(layer)]
    (image,) = [i for i in fakes.created if i.layer == user]
    return Image.open(io.BytesIO(image.image_data))


def rgba(fakes, layer: str) -> np.ndarray:
    """An overlay's board pixels, without the legend."""
    image = placed(fakes, layer).convert("RGBA")
    return np.asarray(image)[: GRID.height, : GRID.width]


def at(image: np.ndarray, point) -> tuple[int, ...]:
    return tuple(int(v) for v in image[GRID.index(point)])


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


def ring_near(image: np.ndarray, frame: tuple[slice, slice], point) -> bool:
    """Whether a panel shows a ring at board `point`."""
    x0, y0, x1, _ = BOARD.bounds
    px_per_mm = (frame[1].stop - frame[1].start) / (x1 - x0 + 2 * render.MARGIN_MM)
    r = frame[0].start + round((point[1] - y0 + render.MARGIN_MM) * px_per_mm)
    c = frame[1].start + round((point[0] - x0 + render.MARGIN_MM) * px_per_mm)
    k = RING_REACH_PX
    near = image[r - k : r + k + 1, c - k : c + k + 1]
    red, green = near[..., 0], near[..., 1]
    return bool(((red >= RING_MIN_RED) & (red - green >= RING_RED_OVER_GREEN)).any())


def test_check_overlay_images_only_the_via_ends_signal_layers(fakes):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    board = Board()

    run(board, overlay=True, full_map=True)

    assert fakes.planned == []
    assert board.calls == [
        "refill_zones",
        "begin_commit",
        ("push_commit", returnvia_command.OVERLAY_COMMIT),
    ]
    assert sorted(i.layer for i in fakes.created) == [BL.BL_User_1, BL.BL_User_4]


def test_overlay_board_pixels_sit_on_the_raster_and_the_legend_below_it(fakes):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])

    run(Board(), overlay=True, full_map=True)

    front = next(i for i in fakes.created if i.layer == BL.BL_User_1)
    width, height = placed(fakes, "F.Cu").size
    pixel = overlay.MM_PER_INCH / overlay.OVERLAY_PPI * front.image_scale
    left = front.position.x / NM_PER_MM - width * pixel / 2
    top = front.position.y / NM_PER_MM - height * pixel / 2
    x0, y0, _, y1 = BOARD.bounds
    assert (left, top) == pytest.approx((x0, y0))
    # The board's left column has no copper on F.Cu: its first opaque pixel
    # is the legend's.
    full = np.asarray(placed(fakes, "F.Cu").convert("RGBA"))
    first_legend_row = np.nonzero(full[:, 0, 3])[0][0]
    assert top + first_legend_row * pixel >= y1


def test_unserved_end_gets_a_ring_only_on_its_signal_layer(fakes):
    # In1's slot leaves the F.Cu end unserved; the B.Cu end references In2.
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))

    run(Board(), overlay=True, full_map=True)

    off_copper = (SIG[0], SIG[1] - DISTANCE)
    assert at(rgba(fakes, "F.Cu"), off_copper) == RING
    assert at(rgba(fakes, "B.Cu"), off_copper)[3] == 0


# On the F.Cu track, within --distance of SIG.
IN_RING = (SIG[0] - 1, SIG[1])


def test_map_shows_only_the_copper_inside_the_unserved_ends_rings(fakes):
    # In1's slot leaves the F.Cu end unserved; the B.Cu end is served.
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))

    run(Board(), overlay=True)

    front = rgba(fakes, "F.Cu")
    plane = ReferencePlane("In1.Cu", "GND")
    assert at(front, IN_RING) == field_colour(fakes.snap, plane, IN_RING)
    assert at(front, ON_TRACK)[3] == 0
    # No end is unserved on B.Cu: it gets no image.
    assert [i.layer for i in fakes.created] == [BL.BL_User_1]


def test_full_map_shows_all_copper(fakes):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))

    run(Board(), overlay=True, full_map=True)

    front = rgba(fakes, "F.Cu")
    assert at(front, IN_RING)[3] == 255
    assert at(front, ON_TRACK)[3] == 255
    assert at(rgba(fakes, "B.Cu"), ON_TRACK)[3] == 255


def test_no_unserved_via_gets_no_heatmap(fakes, tmp_path: Path):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    board = Board()

    run(board, overlay=True)

    assert fakes.created == []
    assert board.calls == ["refill_zones"]


def test_render_after_everything_is_served_replaces_the_old_heatmap(
    fakes, tmp_path: Path
):
    path = tmp_path / "heat.png"
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))
    run(Board(), render=path)
    unserved = path.read_bytes()

    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    run(Board(), render=path)

    assert path.read_bytes() != unserved
    # One panel, showing the board outline and no ring.
    image, frames = panel_frames(path)
    assert len(frames) == 1
    assert not ring_near(image, frames[0], (SIG[0], SIG[1] - DISTANCE))


def field_colour(snap, plane, point) -> tuple[int, ...]:
    """Colour of `plane`'s return field at `point`."""
    fields = ReturnFields(snap)
    (expected,) = palette_at(float(fields.field(plane)[GRID.index(point)]))
    return expected


ON_TRACK = (SIG[0] - 2, SIG[1])
# A via end's distance is painted this far past the via's edge.
PAINTED_PAST_VIA_MM = 0.3


def test_track_reads_its_reference_planes_field(fakes):
    ret = (SIG[0] + 1.4, SIG[1])
    fakes.snap = slotted(ret)

    run(Board(), overlay=True, full_map=True)

    front = at(rgba(fakes, "F.Cu"), ON_TRACK)
    back = at(rgba(fakes, "B.Cu"), ON_TRACK)
    assert front == field_colour(fakes.snap, ReferencePlane("In1.Cu", "GND"), ON_TRACK)
    assert back == field_colour(fakes.snap, ReferencePlane("In2.Cu", "GND"), ON_TRACK)
    assert front != back


def test_track_over_a_slot_in_its_plane_is_unreachable(fakes):
    flush = BOARD.difference(antipads(SIG))
    slot = Point(ON_TRACK).buffer(0.5)
    zones = [gnd_zone({"In1.Cu": flush.difference(slot), "In2.Cu": flush})]
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))], zones=zones)

    run(Board(), overlay=True, full_map=True)

    unreachable = colour(STYLE.unreachable_colour)
    assert at(rgba(fakes, "F.Cu"), ON_TRACK) == unreachable
    assert at(rgba(fakes, "B.Cu"), ON_TRACK) != unreachable


def test_track_outside_every_ground_outline_is_blank(fakes):
    right = box(4, 0, 20, 10)
    fills = right.difference(antipads(SIG))
    zones = [gnd_zone({L: fills for L in INNER}, outline=right)]
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))], zones=zones)

    run(Board(), overlay=True, full_map=True)

    image = rgba(fakes, "F.Cu")
    assert at(image, ON_TRACK)[3] == 0
    assert at(image, (4.3, SIG[1]))[3] == 255
    assert at(image, (4.3, SIG[1])) != colour(STYLE.unreachable_colour)


def test_two_layer_board_leaves_the_unreferenced_layer_blank(fakes):
    # GND pours only B.Cu: F.Cu references it, B.Cu has no reference layer.
    pour = BOARD.difference(antipads(SIG))
    snap = snapshot(returns=[("GND", (4.0, 5.0))], zones=[gnd_zone({"B.Cu": pour})])
    snap.copper_layers = ("F.Cu", "B.Cu")
    fakes.snap = snap

    run(Board(), overlay=True, full_map=True)

    back = rgba(fakes, "B.Cu")
    assert at(back, (15.0, 8.0))[3] == 0
    assert at(back, ON_TRACK)[3] == 0
    assert at(back, (SIG[0], SIG[1] - DISTANCE)) == RING
    front = rgba(fakes, "F.Cu")
    assert at(front, ON_TRACK) != colour(STYLE.unreachable_colour)
    assert at(front, ON_TRACK)[3] == 255


def test_track_narrower_than_a_pixel_stays_unbroken(fakes):
    start, end = (8.0, 2.0), (18.0, 8.5)
    snap = snapshot(returns=[("GND", (6.0, 5.0))])
    thin = Track("thin", "X", "F.Cu", GRID.pixel / 4, LineString([start, end]))
    snap.tracks = [*snap.tracks, thin]
    fakes.snap = snap

    run(Board(), overlay=True, full_map=True)

    image = rgba(fakes, "F.Cu")
    steps = np.linspace(0.0, 1.0, 200)
    along = [
        (start[0] + t * (end[0] - start[0]), start[1] + t * (end[1] - start[1]))
        for t in steps
    ]
    assert all(at(image, p)[3] == 255 for p in along)


def test_ground_via_in_its_clearance_reads_its_own_return(fakes):
    # The return sits in a clearance hole in both planes, bridged by thermal
    # spokes: its own copper is the return, not a void.
    ret = (6.0, 5.0)
    plain = BOARD.difference(antipads(SIG, ret))
    fakes.snap = snapshot(
        returns=[("GND", ret)], zones=[gnd_zone({L: plain for L in INNER})]
    )

    run(Board(), overlay=True, full_map=True)

    image = rgba(fakes, "F.Cu")
    assert at(image, ret) == palette_at(0.0)[0]


def test_every_nets_copper_is_shown(fakes):
    snap = snapshot(returns=[("GND", (6.0, 5.0))])
    gnd_pour, pwr_pour, pad_at = box(12, 0, 20, 4), box(12, 6, 20, 10), (8.0, 8.0)
    pad = Pad(
        "p",
        None,
        "X",
        PadKind.SMD,
        "1",
        pad_at,
        0.0,
        {"F.Cu": Point(pad_at).buffer(0.5)},
    )
    fakes.snap = Snapshot(
        copper_layers=FOUR_LAYERS,
        vias=snap.vias,
        tracks=snap.tracks,
        pads=[pad],
        zones=[
            *snap.zones,
            gnd_zone({"F.Cu": gnd_pour}, outline=gnd_pour),
            gnd_zone({"F.Cu": pwr_pour}, net="PWR", outline=pwr_pour),
        ],
        edge_paths=snap.edge_paths,
        via_sizes=snap.via_sizes,
    )

    run(Board(), overlay=True, full_map=True)

    image = rgba(fakes, "F.Cu")
    assert at(image, ON_TRACK)[3] == 255
    assert at(image, (6.0, 5.0))[3] == 255
    assert at(image, pad_at)[3] == 255
    assert at(image, (16.0, 8.0))[3] == 255
    assert at(image, (16.0, 2.0))[3] == 255
    assert at(image, (8.0, 2.0))[3] == 0


def front_return(snap) -> float:
    (front,) = [
        r for r in measure_returns(snap, None, DISTANCE) if r.layers == ("F.Cu",)
    ]
    return front.distance


def test_via_end_shows_its_measured_distance_over_its_antipad(fakes):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])

    run(Board(), overlay=True, full_map=True)

    image = rgba(fakes, "F.Cu")
    (expected,) = palette_at(front_return(fakes.snap))
    reach = VIA.diameter / 2 + PAINTED_PAST_VIA_MM
    just_inside = (SIG[0] - reach + GRID.pixel / 2, SIG[1])
    just_outside = (SIG[0] - reach - GRID.pixel / 2, SIG[1])
    off_copper = (SIG[0], SIG[1] + reach - GRID.pixel / 2)
    assert at(image, SIG) == expected
    assert at(image, just_inside) == expected
    assert at(image, just_outside) != expected
    assert at(image, off_copper)[3] == 0


def test_via_shows_only_on_layers_where_it_has_a_measured_end(fakes):
    # SIG2 connects on F.Cu and In2.Cu; it passes B.Cu without copper there.
    sig2 = (15.0, 5.0)
    snap = snapshot(signals=(SIG, sig2), returns=[("GND", (6.0, 5.0))])
    snap.tracks = [
        t
        for t in snap.tracks
        if not (t.net == "SIG" and t.layer == "B.Cu" and t.centreline.coords[0] == sig2)
    ]
    snap.tracks.append(
        Track("SIG2:In2", "SIG", "In2.Cu", 0.2, LineString([sig2, (18.0, 5.0)]))
    )
    fakes.snap = snap

    run(Board(), overlay=True, full_map=True)

    assert at(rgba(fakes, "B.Cu"), sig2)[3] == 0


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


def test_layer_outside_every_ground_zone_is_logged_and_counted(fakes, log):
    # The only ground at the via is on F.Cu: the B.Cu end references it,
    # the F.Cu end has no other layer to reference.
    left = box(0, 0, 8, 10)
    fills = {"F.Cu": left.difference(antipads(SIG))}
    fakes.snap = snapshot(
        returns=[("GND", (6.0, 5.0))], zones=[gnd_zone(fills, outline=left)]
    )

    run(Board(), overlay=True, full_map=True)

    (line,) = [m for m in log if "unserved:" in m]
    assert line.rstrip().endswith("outside every ground zone for F.Cu")
    assert any("1 of 1 vias unserved" in m for m in log)
    # Only the F.Cu end is unserved: B.Cu references F.Cu's ground.
    off_copper = (SIG[0], SIG[1] - DISTANCE)
    assert at(rgba(fakes, "F.Cu"), off_copper) == RING
    assert not (rgba(fakes, "B.Cu") == RING).all(axis=2).any()
    # The F.Cu end has no distance to paint: its copper stays blank.
    assert at(rgba(fakes, "F.Cu"), SIG)[3] == 0


def test_every_via_served_logs_zero_and_draws_no_ring(fakes, log):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])

    run(Board(), overlay=True, full_map=True)

    assert not [m for m in log if "unserved:" in m]
    assert any("0 of 1 vias unserved" in m for m in log)
    for name in ("F.Cu", "B.Cu"):
        assert not (rgba(fakes, name) == RING).all(axis=2).any()


def test_exempt_via_alone_on_a_layer_gets_no_map(fakes):
    snap = snapshot(returns=[("GND", (6.0, 5.0))])
    lone = (15.0, 5.0)
    snap.vias = [*snap.vias, via("SIG", lone)]
    snap.tracks = [
        *snap.tracks,
        Track("lone", "SIG", "In1.Cu", 0.2, LineString([lone, (18.0, 5.0)])),
    ]
    fakes.snap = snap

    run(Board(), overlay=True, full_map=True)

    assert sorted(i.layer for i in fakes.created) == [BL.BL_User_1, BL.BL_User_4]


def test_exempt_vias_are_counted(fakes, log):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    fakes.snap.vias = [*fakes.snap.vias, via("SIG", (15.0, 5.0))]

    run(Board())

    assert any("1 vias exempt" in m for m in log)


def test_split_plane_composites_both_ground_nets(fakes):
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
        overlay.Outputs(overlay=True),
        check=True,
        full_map=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )

    image = rgba(fakes, "F.Cu")
    for point, net in ((ON_TRACK, "GND"), ((iso_sig[0] - 2, 5.0), "GND_ISO")):
        plane = ReferencePlane("In1.Cu", net)
        assert at(image, point) == field_colour(fakes.snap, plane, point)


def test_no_reference_plane_removes_the_previous_set(fakes):
    pourstitch_command.execute(
        Options(nets=("GND",)),
        overlay.Outputs(overlay=True),
        check=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )
    previous = list(fakes.created)
    assert previous
    fakes.snap = snapshot(signals=())
    board = Board(previous)

    run(board, overlay=True)

    assert fakes.created == previous
    assert board.calls == [
        "refill_zones",
        "begin_commit",
        "remove_items",
        ("push_commit", returnvia_command.OVERLAY_COMMIT),
    ]
    assert board.removed == previous


def test_no_reference_plane_and_no_previous_set_commits_nothing(fakes):
    fakes.snap = snapshot(signals=())
    board = Board()

    run(board, overlay=True)

    assert board.calls == ["refill_zones"]


def test_without_a_ground_net_the_previous_set_stays(fakes):
    previous = [object()]
    fills = {"In1.Cu": BOARD.difference(antipads(SIG))}
    fakes.snap = snapshot(zones=[gnd_zone(fills, net="PWR")])
    board = Board(previous)

    with pytest.raises(NoGroundNetsError):
        run(board, overlay=True)

    assert board.calls == ["refill_zones"]
    assert fakes.created == []


def test_dry_run_overlay_shows_the_planned_vias(fakes, log):
    fakes.snap = snapshot()
    board = Board()

    run(board, overlay=True, check=False, dry_run=True, full_map=True)

    assert [s.dry_run for s in fakes.planned] == [True]
    assert board.calls == [
        "begin_commit",
        ("push_commit", returnvia_command.OVERLAY_COMMIT),
    ]
    assert any("0 of 1 vias unserved" in m for m in log)
    assert fakes.created
    image = rgba(fakes, "F.Cu")
    # Without the planned via no copper reaches a return at all.
    unreachable = colour(STYLE.unreachable_colour)
    assert at(image, ON_TRACK)[3] == 255
    assert at(image, ON_TRACK) != unreachable
    assert not (image == RING).all(axis=2).any()


def test_forced_vias_serve_nothing(fakes, log):
    fakes.snap = snapshot()
    fakes.plan = Plan(units=(PlannedUnit((6.0, 5.0), "GND", VIA, forced=True),))

    run(Board(), check=False, dry_run=True)

    assert any("1 of 1 vias unserved" in m for m in log)


def test_returnvia_and_pourstitch_replace_each_others_overlays(fakes):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    pourstitch_command.execute(
        Options(nets=("GND",)),
        overlay.Outputs(overlay=True),
        check=True,
        kicad=KiCad(Board()),  # type: ignore[arg-type]
    )
    pourstitch_set = list(fakes.created)
    fakes.created.clear()

    board = Board(pourstitch_set)
    run(board, overlay=True, full_map=True)
    assert board.removed == pourstitch_set
    returnvia_set = list(fakes.created)
    assert returnvia_set

    board = Board(returnvia_set)
    pourstitch_command.execute(
        Options(nets=("GND",)),
        overlay.Outputs(overlay=True),
        check=True,
        kicad=KiCad(board),  # type: ignore[arg-type]
    )
    assert board.removed == returnvia_set


def test_clear_removes_the_returnvia_overlays(fakes, monkeypatch):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    run(Board(), overlay=True, full_map=True)
    assert fakes.created
    board = Board(list(fakes.created))
    monkeypatch.setattr(cli.kipy, "KiCad", lambda: KiCad(board))

    cli.execute_clear(
        argparse.Namespace(
            overlay_layers=list(overlay.DEFAULT_OVERLAY_LAYERS),
            dry_run=False,
            vias=False,
            nets=None,
        )
    )

    assert board.removed == fakes.created


# Off by at most one of the palette's 768 steps, rounded to bytes.
PALETTE_RGB_TOLERANCE = 2
# 0.05 mm either side of the threshold, each ramp is a few levels from its end.
RAMP_RGB_TOLERANCE = 10


def palette_at(*distances: float) -> list[tuple[int, ...]]:
    dist = np.array([distances])
    maps = [
        CoverageMap(
            returnvia_command.COPPER_MAP_NET, "F.Cu", dist, np.zeros(dist.shape, bool)
        )
    ]
    grid = Grid(0, 0, len(distances), 1)
    rgba = render.overlay_rgba(grid, maps, {}, DISTANCE, STYLE)
    return [tuple(int(v) for v in px) for px in rgba[0]]


def close_to(
    px: tuple[int, ...], name: str, tolerance: int = PALETTE_RGB_TOLERANCE
) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(px, colour(name)))


def unserved_colour(fraction: float) -> str:
    cmap = render.plt.get_cmap(returnvia_command.UNSERVED_COLORMAP)
    return render.matplotlib.colors.to_hex(cmap(fraction))


def test_palette_steps_into_the_unserved_colormap_where_served_ends():
    # A via end is served up to the distance plus the quantisation margin.
    under, over = palette_at(DISTANCE + 0.05, DISTANCE + 0.15)

    assert close_to(under, returnvia_command.STILL_SERVED_COLOUR, RAMP_RGB_TOLERANCE)
    assert close_to(over, unserved_colour(0.0), RAMP_RGB_TOLERANCE)
    assert not close_to(under, unserved_colour(0.0), RAMP_RGB_TOLERANCE)


def test_palette_runs_quiet_then_through_the_unserved_colormap_and_saturates():
    top = returnvia_command.SCALE * (DISTANCE + QUANTISATION)
    served, at_top, beyond = palette_at(0.0, top, 2 * top)

    assert close_to(served, returnvia_command.SERVED_COLOUR)
    assert close_to(at_top, unserved_colour(1.0))
    assert beyond == at_top


def test_board_without_outline_fails_before_the_refill(fakes):
    board = Board(outline=None)

    with pytest.raises(NoBoardOutlineError):
        run(board, overlay=True)

    assert board.calls == []


def test_disabled_overlay_layer_fails_before_any_board_change(fakes):
    board = Board(enabled=USER[:2])

    with pytest.raises(overlay.OverlayLayersError, match="User.3"):
        run(board, overlay=True, check=False)

    assert fakes.planned == []
    assert board.calls == []


def test_disabled_overlay_layer_in_dry_run_keeps_the_pngs(fakes, png_dir):
    fakes.snap = snapshot(returns=[("GND", (6.0, 5.0))])
    board = Board(enabled=())

    run(board, overlay=True, check=False, dry_run=True, full_map=True)

    assert board.calls == []
    assert fakes.created == []
    assert len(list(png_dir.glob("*.png"))) == len(INNER)


def test_too_few_overlay_layers_is_an_error(fakes):
    with pytest.raises(overlay.OverlayLayersError):
        run(Board(), overlay=True, overlay_layers=("User.1", "User.2"))


def test_render_writes_the_panels(fakes, tmp_path):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))
    board = Board()

    run(board, render=tmp_path / "heat.png")

    assert (tmp_path / "heat.png").exists()
    assert board.calls == ["refill_zones"]


def test_render_panels_ring_the_unserved_end_on_its_layer(fakes, tmp_path):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))

    run(Board(), render=tmp_path / "heat.png", full_map=True)

    image, (front, back) = panel_frames(tmp_path / "heat.png")
    off_copper = (SIG[0], SIG[1] - DISTANCE)
    assert ring_near(image, front, off_copper)
    assert not ring_near(image, back, off_copper)


def test_no_ground_fill_end_is_logged_and_ringed(fakes, log):
    # GND covers the via on both planes, but In1's fill lies beyond --distance.
    far = box(15, 0, 20, 10)
    flush = BOARD.difference(antipads(SIG))
    fakes.snap = snapshot(
        returns=[("GND", (6.0, 5.0))],
        zones=[gnd_zone({"In1.Cu": far, "In2.Cu": flush})],
    )

    run(Board(), overlay=True, full_map=True)

    (line,) = [m for m in log if "unserved:" in m]
    assert line.rstrip().endswith(
        "SIG at (5.000, 5.000): no GND fill at the via on In1.Cu"
    )
    off_copper = (SIG[0], SIG[1] - DISTANCE)
    assert at(rgba(fakes, "F.Cu"), off_copper) == RING
    assert at(rgba(fakes, "B.Cu"), off_copper) != RING


def test_render_creates_the_missing_parent_directory(fakes, tmp_path: Path):
    fakes.snap = slotted((SIG[0] + 1.4, SIG[1]))
    path = tmp_path / "out" / "heat.png"

    run(Board(), render=path)

    assert path.exists()
