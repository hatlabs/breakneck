"""Heatmap overlays as KiCad reference images on user layers.

KiCad places a reference image by its centre and sizes it as
pixels × (25.4 mm / PPI) × scale, with the PPI read from the PNG. The
overlay PNGs are written at `OVERLAY_PPI` and scaled so that one image pixel
is one raster pixel.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import kipy.board
import kipy.board_types
from kipy.geometry import Vector2
from kipy.util.board_layer import canonical_name, layer_from_canonical_name
from loguru import logger

from breakneck.commit import CreateItemsError, create_checked
from breakneck.snapshot import NM_PER_MM, Point2

DEFAULT_OVERLAY_LAYERS = ("User.1", "User.2", "User.3", "User.4")
MM_PER_INCH = 25.4
# 254 PPI is exactly 0.1 mm per pixel and exactly 10000 pixels per metre in
# the PNG pHYs chunk, so neither side has to round.
OVERLAY_PPI = 254
OVERLAY_COMMIT = "breakneck pourstitch overlay"


@dataclass(frozen=True)
class ImagePlacement:
    """Where KiCad must put an image so pixel (0, 0) starts at the raster origin."""

    centre: Point2
    scale: float


def image_placement(
    origin: Point2,
    size_px: tuple[int, int],
    pixel_mm: float,
    ppi: float = OVERLAY_PPI,
) -> ImagePlacement:
    """Placement for a `size_px` (width, height) image of `pixel_mm` pixels."""
    width, height = size_px
    native_pixel_mm = MM_PER_INCH / ppi
    return ImagePlacement(
        centre=(origin[0] + width * pixel_mm / 2, origin[1] + height * pixel_mm / 2),
        scale=pixel_mm / native_pixel_mm,
    )


def layer_map(
    copper_layers: Sequence[str], overlay_layers: Sequence[str]
) -> dict[str, str]:
    """Overlay layer for each copper layer, in stack order."""
    if len(overlay_layers) < len(copper_layers):
        raise ValueError(
            f"{len(copper_layers)} copper layers need as many overlay layers; "
            f"got {', '.join(overlay_layers)}"
        )
    return dict(zip(copper_layers, overlay_layers))


def missing_layers(board: kipy.board.Board, layers: Sequence[str]) -> list[str]:
    enabled = {canonical_name(L) for L in board.get_enabled_layers()}
    return [L for L in layers if L not in enabled]


def enable_hint(missing: Sequence[str]) -> str:
    return (
        f"Overlay layers not enabled on the board: {', '.join(missing)}. Enable them "
        "in Board Setup → Board Editor Layers, or pass --overlay-layers"
    )


@dataclass(frozen=True)
class OverlayImage:
    copper_layer: str
    overlay_layer: str
    path: Path
    placement: ImagePlacement

    def describe(self) -> str:
        x, y = self.placement.centre
        return (
            f"{self.path} ({self.copper_layer}) on {self.overlay_layer}: centre "
            f"({x:.4f}, {y:.4f}) mm, scale {self.placement.scale:.6g}"
        )


def _reference_image(image: OverlayImage) -> kipy.board_types.ReferenceImage:
    ref = kipy.board_types.ReferenceImage()
    ref.layer = layer_from_canonical_name(image.overlay_layer)
    x, y = image.placement.centre
    ref.position = Vector2.from_xy(round(x * NM_PER_MM), round(y * NM_PER_MM))
    ref.image_scale = image.placement.scale
    ref.image_data = image.path.read_bytes()
    return ref


def images_on(
    board: kipy.board.Board, layers: Sequence[str]
) -> list[kipy.board_types.ReferenceImage]:
    wanted = set(layers)
    return [
        i for i in board.get_reference_images() if canonical_name(i.layer) in wanted
    ]


def place_overlays(
    board: kipy.board.Board,
    images: Sequence[OverlayImage],
    overlay_layers: Sequence[str],
) -> bool:
    """Replace the reference images on the overlay layers in one commit.

    If KiCad refuses an image, nothing changes and the log says where to
    place the PNGs by hand.
    """
    old = images_on(board, overlay_layers)
    commit = board.begin_commit()
    try:
        if old:
            board.remove_items(old)
        create_checked(board, [_reference_image(i) for i in images])
    except CreateItemsError as e:
        board.drop_commit(commit)
        logger.warning(f"KiCad refused the overlay images ({e}); place them by hand:")
        for i in images:
            logger.warning(f"  {i.describe()}")
        return False
    except BaseException:
        board.drop_commit(commit)
        raise
    board.push_commit(commit, OVERLAY_COMMIT)
    logger.info(
        f"Placed {len(images)} overlay images"
        + (f", replacing {len(old)}" if old else "")
    )
    for i in images:
        logger.info(f"  {i.describe()}")
    return True
