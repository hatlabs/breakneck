import argparse
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from kipy.board_types import BoardSegment
from kipy.geometry import Vector2
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from PIL import Image
from shapely.geometry import box

from breakneck import cli, overlay, render
from breakneck.commit import CreateItemsError
from breakneck.plan import Plan, PlannedUnit, UnfixableSpot
from breakneck.pourstitch import CoverageMap, Grid
from breakneck.snapshot import NM_PER_MM, Snapshot, ViaSize

D = 2.5


def coverage(net, layer, dist, unstitchable=None):
    if unstitchable is None:
        unstitchable = np.zeros(dist.shape, bool)
    return CoverageMap(net, layer, dist, unstitchable)


def test_placement_maps_pixel_zero_to_raster_origin():
    origin, size, pixel = (12.3, -4.5), (200, 80), 0.1

    p = overlay.image_placement(origin, size, pixel, ppi=300)

    native = overlay.MM_PER_INCH / 300
    drawn_pixel = native * p.scale
    assert drawn_pixel == pytest.approx(pixel)
    top_left = (
        p.centre[0] - size[0] * drawn_pixel / 2,
        p.centre[1] - size[1] * drawn_pixel / 2,
    )
    assert top_left == pytest.approx(origin)


def test_overlay_ppi_needs_no_scaling():
    assert overlay.image_placement((0, 0), (1, 1), 0.1).scale == pytest.approx(1.0)


def test_layer_map_follows_stack_order():
    copper = ("F.Cu", "In1.Cu", "In2.Cu", "B.Cu")

    assert overlay.layer_map(copper, overlay.DEFAULT_OVERLAY_LAYERS) == {
        "F.Cu": "User.1",
        "In1.Cu": "User.2",
        "In2.Cu": "User.3",
        "B.Cu": "User.4",
    }
    with pytest.raises(ValueError):
        overlay.layer_map(copper, ("User.1",))


def maps_for(layer_nets: dict[str, list[tuple[str, np.ndarray]]]) -> list[CoverageMap]:
    return [
        coverage(net, layer, dist)
        for layer, entries in layer_nets.items()
        for net, dist in entries
    ]


def test_overlay_image_has_raster_size_and_transparent_background(tmp_path: Path):
    a = np.full((30, 50), np.nan)
    a[:, :20] = 1.0
    b = np.full((30, 50), np.nan)
    b[:, 30:] = np.inf
    maps = maps_for({"F.Cu": [("GND", a), ("+3.3V", b)]})

    paths = render.write_overlays(tmp_path, maps, D)

    img = Image.open(paths["F.Cu"])
    assert img.size == (50, 30)
    assert img.info["dpi"] == pytest.approx((overlay.OVERLAY_PPI,) * 2, abs=0.01)
    rgba = np.asarray(img.convert("RGBA"))
    assert (rgba[:, 20:30, 3] == 0).all()
    assert (rgba[:, :20, 3] == 255).all()
    assert (rgba[:, 30:, 3] == 255).all()
    assert tuple(rgba[0, 40, :3]) == (0xFF, 0x2A, 0x6D)


def test_unstitchable_copper_is_grey_not_pink(tmp_path: Path):
    dist = np.full((10, 10), np.inf)
    unstitchable = np.zeros(dist.shape, bool)
    unstitchable[:, 5:] = True
    maps = [coverage("GND", "F.Cu", dist, unstitchable)]

    rgba = np.asarray(Image.open(render.write_overlays(tmp_path, maps, D)["F.Cu"]))

    assert tuple(rgba[0, 0, :3]) == (0xFF, 0x2A, 0x6D)
    grey = render.matplotlib.colors.to_rgb(render.UNSTITCHABLE_COLOUR)
    assert tuple(rgba[0, 9, :3]) == tuple(round(c * 255) for c in grey)
    assert rgba[0, 9, 3] == 255


def test_layers_differing_only_in_unstitchable_copper_get_own_panels():
    a = np.array([[np.inf, np.inf]])
    maps = [
        coverage("GND", "F.Cu", a),
        coverage("GND", "B.Cu", a.copy(), np.array([[False, True]])),
    ]

    assert len(render.panels(maps)) == 2


def test_one_overlay_per_copper_layer(tmp_path: Path):
    a = np.zeros((4, 4))
    maps = maps_for({"In1.Cu": [("GND", a)], "In2.Cu": [("GND", a)]})

    assert set(render.write_overlays(tmp_path, maps, D)) == {"In1.Cu", "In2.Cu"}


def test_identical_layers_share_a_panel():
    a = np.array([[1.0, np.nan]])
    maps = [
        coverage("GND", "In1.Cu", a),
        coverage("GND", "In2.Cu", a.copy()),
        coverage("GND", "F.Cu", np.array([[2.0, np.nan]])),
    ]

    shown = render.panels(maps)

    assert [(m.layer, twins) for m, twins in shown] == [
        ("In1.Cu", ["In2.Cu"]),
        ("F.Cu", []),
    ]


def test_render_panels_writes_png(tmp_path: Path):
    snap = Snapshot(
        copper_layers=("F.Cu", "B.Cu"),
        edge_paths=[list(box(0, 0, 3, 2).exterior.coords)],  # type: ignore[list-item]
    )
    grid = Grid(0, 0, 30, 20)
    dist = np.linspace(0, 4, 600).reshape(20, 30)
    plan = Plan(
        units=(PlannedUnit((1.0, 1.0), "GND", ViaSize(0.6, 0.3)),),
        unfixable=(UnfixableSpot((2.0, 1.0), "GND", "no fit", "F.Cu"),),
    )
    out = tmp_path / "heat.png"

    unstitchable = np.zeros(dist.shape, bool)
    unstitchable[:, 25:] = True
    maps = [coverage("GND", "F.Cu", dist, unstitchable)]

    render.render_panels(out, snap, grid, maps, plan, D)

    assert Image.open(out).size[0] > 0


class ImageBoard:
    def __init__(self, existing=(), refuse=False):
        self.calls: list = []
        self.existing = list(existing)
        self.refuse = refuse

    def get_reference_images(self):
        return self.existing

    def begin_commit(self):
        self.calls.append("begin_commit")
        return "c"

    def remove_items(self, items):
        self.calls.append(("remove_items", len(items)))
        self.removed = list(items)

    def push_commit(self, c, message=""):
        self.calls.append(("push_commit", message))

    def drop_commit(self, c):
        self.calls.append("drop_commit")


def image(tmp_path: Path) -> overlay.OverlayImage:
    path = tmp_path / "x.png"
    path.write_bytes(b"png")
    return overlay.OverlayImage(
        "F.Cu", "User.1", path, overlay.image_placement((1, 2), (10, 10), 0.1)
    )


PLACEMENT = overlay.image_placement((1, 2), (10, 10), 0.1)


class Existing:
    """A reference image as kipy reports it."""

    def __init__(self, layer, centre=PLACEMENT.centre, scale=PLACEMENT.scale):
        self.layer = layer
        self.position = Vector2.from_xy(
            round(centre[0] * NM_PER_MM), round(centre[1] * NM_PER_MM)
        )
        self.image_scale = scale


@pytest.mark.parametrize(
    "existing, ours",
    [
        (Existing(BL.BL_User_1), True),
        (Existing(BL.BL_User_1, (PLACEMENT.centre[0] + 0.0005, 2.5)), True),
        (Existing(BL.BL_User_1, (PLACEMENT.centre[0] + 0.01, 2.5)), False),
        (Existing(BL.BL_User_1, scale=2.0), False),
        (Existing(BL.BL_Cmts_User), False),
    ],
)
def test_only_images_at_the_overlay_placement_are_breakneck_overlays(existing, ours):
    assert overlay.is_overlay(existing, PLACEMENT, ("User.1", "User.2")) is ours  # type: ignore[arg-type]


def test_overlay_placement_follows_the_board_outline_raster():
    bounds = (1.0, 2.0, 2.04, 3.0)  # 10.4 pixels wide rounds up to 11

    placement = overlay.board_placement(bounds)

    assert placement == overlay.image_placement((1.0, 2.0), (11, 10), 0.1)
    grid = Grid.covering(bounds)
    assert placement == overlay.image_placement(
        (grid.x0, grid.y0), (grid.width, grid.height), grid.pixel
    )


def test_place_overlays_replaces_images_in_one_commit(tmp_path, monkeypatch):
    created = []
    monkeypatch.setattr(
        overlay, "create_checked", lambda board, items: created.extend(items)
    )
    users_own = Existing(BL.BL_User_1, (30, 30))
    board = ImageBoard(
        existing=[Existing(BL.BL_User_1), Existing(BL.BL_Cmts_User), users_own]
    )

    assert overlay.place_overlays(board, [image(tmp_path)], ("User.1", "User.2"))  # type: ignore[arg-type]

    assert board.calls == [
        "begin_commit",
        ("remove_items", 1),
        ("push_commit", overlay.OVERLAY_COMMIT),
    ]
    assert board.removed == [board.existing[0]]
    (ref,) = created
    assert ref.layer == BL.BL_User_1
    assert (ref.position.x, ref.position.y) == (1_500_000, 2_500_000)
    assert ref.image_scale == pytest.approx(1.0)
    assert ref.image_data == b"png"


def test_refused_overlay_falls_back_to_manual_placement(tmp_path, monkeypatch):
    def refuse(board, items):
        raise CreateItemsError("no overlapping layers")

    monkeypatch.setattr(overlay, "create_checked", refuse)
    board = ImageBoard()

    assert not overlay.place_overlays(board, [image(tmp_path)], ("User.1",))  # type: ignore[arg-type]

    assert board.calls == ["begin_commit", "drop_commit"]


class ClearBoard:
    """A board with an Edge.Cuts outline, Eco2 markers and reference images."""

    def __init__(self, outline, images):
        self.shapes = [*outline, eco2_shape()]
        self.images = images
        self.removed: list = []

    def get_shapes(self):
        return self.shapes

    def get_reference_images(self):
        return self.images

    def begin_commit(self):
        return "c"

    def remove_items(self, items):
        self.removed += items

    def push_commit(self, c, message=""):
        pass


def segment(layer, start, end):
    seg = BoardSegment()
    seg.layer = layer
    seg.start = Vector2.from_xy(
        round(start[0] * NM_PER_MM), round(start[1] * NM_PER_MM)
    )
    seg.end = Vector2.from_xy(round(end[0] * NM_PER_MM), round(end[1] * NM_PER_MM))
    return seg


def eco2_shape():
    return segment(BL.BL_Eco2_User, (1, 1), (2, 2))


def rectangle_outline(x0, y0, x1, y1):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return [segment(BL.BL_Edge_Cuts, a, b) for a, b in pairwise(corners)]


def test_clear_removes_breakneck_overlays_and_keeps_other_images(monkeypatch):
    bounds = (0.0, 0.0, 30.0, 20.0)
    placement = overlay.board_placement(bounds)
    ours = Existing(BL.BL_User_2, placement.centre, placement.scale)
    users_own = Existing(BL.BL_User_1, (5, 5), 1.0)
    board = ClearBoard(rectangle_outline(*bounds), [ours, users_own])
    monkeypatch.setattr(
        cli.kipy, "KiCad", lambda: type("K", (), {"get_board": lambda self: board})()
    )

    args = argparse.Namespace(
        overlay_layers=list(overlay.DEFAULT_OVERLAY_LAYERS), dry_run=False
    )
    cli.execute_clear(args)

    assert ours in board.removed
    assert users_own not in board.removed
    assert any(getattr(s, "layer", None) == BL.BL_Eco2_User for s in board.removed)
