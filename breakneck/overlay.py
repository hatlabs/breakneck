"""Heatmap overlays as KiCad reference images on user layers.

KiCad places a reference image by its centre and sizes it as
pixels × (25.4 mm / PPI) × scale, with the PPI read from the PNG. The
overlay PNGs are written at `OVERLAY_PPI` and scaled so that one image pixel
is one raster pixel. Each carries a PNG text chunk that marks it as
breakneck's; KiCad keeps it in the image data it reports back.
"""

from __future__ import annotations

import math
import shutil
import struct
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import kipy.board
import kipy.board_types
from kipy.geometry import Vector2
from kipy.util.board_layer import canonical_name, layer_from_canonical_name
from loguru import logger

from breakneck.commit import CreateItemsError, create_checked
from breakneck.plan import POURSTITCH_PIXEL, raster_size
from breakneck.snapshot import NM_PER_MM, Point2, Snapshot

DEFAULT_OVERLAY_LAYERS = ("User.1", "User.2", "User.3", "User.4")
MM_PER_INCH = 25.4
# 254 PPI is exactly 0.1 mm per pixel and exactly 10000 pixels per metre in
# the PNG pHYs chunk, so neither side has to round.
OVERLAY_PPI = 254
OVERLAY_TAG = ("breakneck", "pourstitch-overlay")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_CHUNK_HEADER = struct.Struct(">I4s")
PNG_CRC_SIZE = 4
PNG_IHDR_SIZE = struct.Struct(">II")
# Untagged overlays from earlier builds are recognised by their placement:
# an image within these of it is taken as breakneck's.
POSITION_TOLERANCE_NM = 1000
SCALE_TOLERANCE = 1e-6
PNG_DIR_PREFIX = "breakneck-overlay-"


@dataclass(frozen=True)
class ImagePlacement:
    """Where KiCad must put an image so pixel (0, 0) starts at the raster origin."""

    centre: Point2
    scale: float


def board_placement(
    bounds: tuple[float, float, float, float], pixel: float = POURSTITCH_PIXEL
) -> ImagePlacement:
    """Placement of the overlays for a board whose outline has `bounds`."""
    return image_placement((bounds[0], bounds[1]), raster_size(bounds, pixel), pixel)


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
class Outputs:
    """Where a heatmap command writes its panels and overlays."""

    render: Path | None = None
    overlay: bool = False
    overlay_layers: tuple[str, ...] = DEFAULT_OVERLAY_LAYERS


class OverlayLayersError(RuntimeError):
    pass


def overlay_ready(board: kipy.board.Board, outputs: Outputs, dry_run: bool) -> bool:
    """Whether overlays can go on the board; raises before any board change if not."""
    count = board.get_copper_layer_count()
    if count > len(outputs.overlay_layers):
        raise OverlayLayersError(
            f"{count} copper layers need {count} overlay layers; got "
            + ", ".join(outputs.overlay_layers)
        )
    needed = outputs.overlay_layers[:count]
    missing = missing_layers(board, needed)
    if not missing:
        return True
    if dry_run:
        logger.warning(enable_hint(missing) + "; writing the PNGs only")
        return False
    raise OverlayLayersError(enable_hint(missing))


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


def png_text(data: bytes) -> dict[str, str]:
    """The tEXt entries of a PNG; empty if `data` is not a PNG."""
    if not data.startswith(PNG_SIGNATURE):
        return {}
    out = {}
    at = len(PNG_SIGNATURE)
    while at + PNG_CHUNK_HEADER.size <= len(data):
        length, kind = PNG_CHUNK_HEADER.unpack_from(data, at)
        body_at = at + PNG_CHUNK_HEADER.size
        if kind == b"IEND":
            break
        if kind == b"tEXt":
            key, _, value = data[body_at : body_at + length].partition(b"\0")
            out[key.decode("latin-1")] = value.decode("latin-1")
        at = body_at + length + PNG_CRC_SIZE
    return out


def png_size(data: bytes) -> tuple[int, int]:
    """(width, height) from a PNG's header chunk, which comes first."""
    at = len(PNG_SIGNATURE) + PNG_CHUNK_HEADER.size
    width, height = PNG_IHDR_SIZE.unpack_from(data, at)
    return width, height


def is_overlay(
    image: kipy.board_types.ReferenceImage,
    placement: ImagePlacement,
    layers: Sequence[str],
) -> bool:
    """Whether `image` is a breakneck overlay: on an overlay layer, and
    either tagged as one or at the overlay placement for this board.
    Anything else is the user's own."""
    if canonical_name(image.layer) not in layers:
        return False
    key, value = OVERLAY_TAG
    if png_text(image.image_data).get(key) == value:
        return True
    x, y = placement.centre
    return (
        abs(image.position.x - x * NM_PER_MM) <= POSITION_TOLERANCE_NM
        and abs(image.position.y - y * NM_PER_MM) <= POSITION_TOLERANCE_NM
        and math.isclose(image.image_scale, placement.scale, rel_tol=SCALE_TOLERANCE)
    )


def overlays_on(
    board: kipy.board.Board, layers: Sequence[str], placement: ImagePlacement
) -> list[kipy.board_types.ReferenceImage]:
    return [i for i in board.get_reference_images() if is_overlay(i, placement, layers)]


def place_overlays(
    board: kipy.board.Board,
    images: Sequence[OverlayImage],
    placement: ImagePlacement,
    overlay_layers: Sequence[str],
    commit_label: str,
) -> bool:
    """Replace breakneck's overlays on the overlay layers in one commit.

    `placement` is the board's overlay placement, which finds untagged
    overlays of earlier builds. With no images, the previous set is only
    removed; with nothing to remove either, nothing is committed. Other
    reference images on those layers are left alone. If KiCad refuses an
    image, nothing changes and the log says where to place the PNGs by hand.
    """
    old = overlays_on(board, overlay_layers, placement)
    if not images and not old:
        return True
    commit = board.begin_commit()
    try:
        if old:
            board.remove_items(old)
        if images:
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
    board.push_commit(commit, commit_label)
    if images:
        logger.info(
            f"Placed {len(images)} overlay images"
            + (f", replacing {len(old)}" if old else "")
        )
    else:
        logger.info(f"Removed {len(old)} overlay images")
    for i in images:
        logger.info(f"  {i.describe()}")
    return True


def show_overlays(
    board: kipy.board.Board,
    write_pngs: Callable[[Path], Mapping[str, Path]] | None,
    snapshot: Snapshot,
    outputs: Outputs,
    place: bool,
    commit_label: str,
) -> None:
    """Write the per-copper-layer PNGs and place them as this run's overlay
    set, or log where they would go. `write_pngs` None means no images: the
    previous set is only removed.

    The board embeds the image data, so the PNGs go to a temporary directory
    that is removed once they are placed. When they are not, they stay there
    for placing by hand. Each PNG's pixel (0, 0) goes at the board raster's
    origin; a PNG larger than the raster, as with a legend below it, extends
    right and down.
    """
    bounds = snapshot.board_outline.bounds
    placement = board_placement(bounds)
    if write_pngs is None:
        if place:
            place_overlays(board, [], placement, outputs.overlay_layers, commit_label)
        return
    out_dir = Path(tempfile.mkdtemp(prefix=PNG_DIR_PREFIX))
    try:
        pngs = write_pngs(out_dir)
        layers = layer_map(snapshot.copper_layers, outputs.overlay_layers)
        origin = (bounds[0], bounds[1])
        images = [
            OverlayImage(
                L,
                layers[L],
                path,
                image_placement(origin, png_size(path.read_bytes()), POURSTITCH_PIXEL),
            )
            for L, path in pngs.items()
        ]
        placed = place and place_overlays(
            board, images, placement, outputs.overlay_layers, commit_label
        )
    except BaseException:
        shutil.rmtree(out_dir)
        raise
    if placed:
        shutil.rmtree(out_dir)
    elif not place:
        for i in images:
            logger.info(f"Overlay not placed: {i.describe()}")
