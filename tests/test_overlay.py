from pathlib import Path

import numpy as np
import pytest
from kipy.proto.board.board_types_pb2 import BoardLayer as BL
from PIL import Image
from shapely.geometry import box

from breakneck import overlay, render
from breakneck.commit import CreateItemsError
from breakneck.plan import Plan, PlannedUnit, UnfixableSpot
from breakneck.pourstitch import CoverageMap, Grid
from breakneck.snapshot import Snapshot, ViaSize

D = 2.5


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
        CoverageMap(net, layer, dist)
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


def test_one_overlay_per_copper_layer(tmp_path: Path):
    a = np.zeros((4, 4))
    maps = maps_for({"In1.Cu": [("GND", a)], "In2.Cu": [("GND", a)]})

    assert set(render.write_overlays(tmp_path, maps, D)) == {"In1.Cu", "In2.Cu"}


def test_identical_layers_share_a_panel():
    a = np.array([[1.0, np.nan]])
    maps = [
        CoverageMap("GND", "In1.Cu", a),
        CoverageMap("GND", "In2.Cu", a.copy()),
        CoverageMap("GND", "F.Cu", np.array([[2.0, np.nan]])),
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

    render.render_panels(out, snap, grid, [CoverageMap("GND", "F.Cu", dist)], plan, D)

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


class Existing:
    def __init__(self, layer):
        self.layer = layer


def test_place_overlays_replaces_images_in_one_commit(tmp_path, monkeypatch):
    created = []
    monkeypatch.setattr(
        overlay, "create_checked", lambda board, items: created.extend(items)
    )
    board = ImageBoard(existing=[Existing(BL.BL_User_1), Existing(BL.BL_Cmts_User)])

    assert overlay.place_overlays(board, [image(tmp_path)], ("User.1", "User.2"))  # type: ignore[arg-type]

    assert board.calls == [
        "begin_commit",
        ("remove_items", 1),
        ("push_commit", overlay.OVERLAY_COMMIT),
    ]
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
