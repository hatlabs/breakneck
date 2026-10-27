"""Board rasters: a pixel grid, distances through copper masks, coverage maps."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
import shapely
from scipy import ndimage
from shapely.geometry.base import BaseGeometry
from skimage.graph import MCP_Geometric

from breakneck.plan import POURSTITCH_PIXEL, raster_size
from breakneck.snapshot import Point2

BoolMap = npt.NDArray[np.bool_]
FloatMap = npt.NDArray[np.floating]
Pixel = tuple[int, int]

PIXEL = POURSTITCH_PIXEL
FAR = 1e9
COORD_DECIMALS = 3


@dataclass(frozen=True)
class Grid:
    """Raster of square pixels; pixel (0, 0) spans [x0, x0+pixel) × [y0, y0+pixel)."""

    x0: float
    y0: float
    width: int
    height: int
    pixel: float = PIXEL

    @classmethod
    def covering(cls, bounds: tuple[float, float, float, float]) -> Grid:
        width, height = raster_size(bounds, PIXEL)
        return cls(bounds[0], bounds[1], width, height, PIXEL)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @cached_property
    def xs(self) -> FloatMap:
        return self.x0 + (np.arange(self.width) + 0.5) * self.pixel

    @cached_property
    def ys(self) -> FloatMap:
        return self.y0 + (np.arange(self.height) + 0.5) * self.pixel

    def index(self, p: Point2) -> Pixel:
        return int((p[1] - self.y0) / self.pixel), int((p[0] - self.x0) / self.pixel)

    def point(self, px: Pixel) -> Point2:
        r, c = px
        return (
            round(float(self.xs[c]), COORD_DECIMALS),
            round(float(self.ys[r]), COORD_DECIMALS),
        )

    def contains(self, px: Pixel) -> bool:
        return 0 <= px[0] < self.height and 0 <= px[1] < self.width

    def window(self, px: Pixel, radius: float) -> tuple[slice, slice]:
        k = int(radius / self.pixel) + 1
        r, c = px
        return (
            slice(max(0, r - k), min(self.height, r + k + 1)),
            slice(max(0, c - k), min(self.width, c + k + 1)),
        )

    def bounds_window(
        self, bounds: tuple[float, float, float, float]
    ) -> tuple[slice, slice]:
        x0, y0, x1, y1 = bounds
        r0, c0 = self.index((x0, y0))
        r1, c1 = self.index((x1, y1))
        return (
            slice(max(0, r0), min(self.height, r1 + 1)),
            slice(max(0, c0), min(self.width, c1 + 1)),
        )

    def mask(
        self, geom: BaseGeometry, win: tuple[slice, slice] | None = None
    ) -> BoolMap:
        rs, cs = win or (slice(None), slice(None))
        gx, gy = np.meshgrid(self.xs[cs], self.ys[rs])
        if geom.is_empty:
            return np.zeros(gx.shape, bool)
        shapely.prepare(geom)
        return shapely.contains_xy(geom, gx, gy)

    def touching(self, geom: BaseGeometry) -> BoolMap:
        """Pixels that the geometry overlaps at all.

        Distances to these pixels' centres understate the true distance by at
        most half a pixel diagonal, which the quantisation margin absorbs.
        Centre-inside rasters can be off by a full diagonal at corners.
        """
        if geom.is_empty:
            return np.zeros(self.shape, bool)
        return self.mask(geom.buffer(self.pixel * math.sqrt(2) / 2))

    def disc(self, px: Pixel, radius: float) -> tuple[tuple[slice, slice], BoolMap]:
        """Pixels whose centres are closer than `radius` to pixel `px`'s centre."""
        win = self.window(px, radius)
        rr, cc = np.ogrid[win[0], win[1]]
        r, c = px
        return win, (rr - r) ** 2 + (cc - c) ** 2 < (radius / self.pixel) ** 2

    def distance_to(self, mask: BoolMap) -> FloatMap:
        """Distance in mm from each pixel to the nearest True pixel."""
        if not mask.any():
            return np.full(self.shape, FAR, np.float32)
        return edt(~mask) * np.float32(self.pixel)

    def depth(self, mask: BoolMap) -> FloatMap:
        """Distance in mm from each True pixel to the nearest False pixel."""
        return edt(mask) * np.float32(self.pixel)


def edt(mask: BoolMap) -> FloatMap:
    """Pixel distance from each True pixel to the nearest False pixel."""
    return np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float32)


def touch(grid: Grid, mask: BoolMap, geom: BaseGeometry) -> list[Pixel]:
    """Pixels of `mask` inside `geom`."""
    win = grid.bounds_window(geom.bounds)
    sub = grid.mask(geom, win) & mask[win]
    rr, cc = np.nonzero(sub)
    return list(zip((rr + win[0].start).tolist(), (cc + win[1].start).tolist()))


def geodesic(
    grid: Grid, mask: BoolMap, starts: Sequence[Pixel], limit: float | None = None
) -> FloatMap:
    """Distance (mm) through `mask` from the nearest start pixel; inf elsewhere.

    Starts outside the mask are ignored. With a limit, only the starts'
    bounding box grown by the limit is searched: a path no longer than the
    limit can't leave it.
    """
    starts = [s for s in starts if mask[s]]
    out = np.full(grid.shape, np.inf)
    if not starts:
        return out
    rs, cs = zip(*starts)
    k = grid.height + grid.width if limit is None else int(limit / grid.pixel) + 1
    win = (
        slice(max(0, min(rs) - k), min(grid.height, max(rs) + k + 1)),
        slice(max(0, min(cs) - k), min(grid.width, max(cs) + k + 1)),
    )
    local = [(r - win[0].start, c - win[1].start) for r, c in starts]
    out[win] = geodesic_in(mask[win], local, grid.pixel, limit)
    return out


def geodesic_in(
    mask: BoolMap, starts: Sequence[Pixel], pixel: float, limit: float | None = None
) -> FloatMap:
    """Distance (mm) through `mask` from the nearest of `starts`, all inside
    `mask`; inf elsewhere and, with a limit, beyond it."""
    mcp = MCP_Geometric(np.where(mask, 1.0, np.inf), fully_connected=True)
    cum, _ = mcp.find_costs(list(starts))
    dist = np.asarray(cum) * pixel
    if limit is not None:
        dist[dist > limit] = np.inf
    return dist


@dataclass(frozen=True)
class CoverageMap:
    """Distance (mm) through `net`'s fill on `layer` to the nearest seed.

    NaN outside the fill, inf where no seed reaches. `excluded` marks fill
    that is drawn apart from the scale and left out of the uncovered area:
    pourstitch marks copper it can't stitch, returnvia leaves it empty.
    """

    net: str
    layer: str
    distance: FloatMap
    excluded: BoolMap

    def uncovered(self, distance: float) -> BoolMap:
        """Fill farther than `distance` from a seed, excluded fill aside."""
        return uncovered(self.distance, self.excluded, distance)


def uncovered(dist: FloatMap, excluded: BoolMap, distance: float) -> BoolMap:
    """Pixels farther than `distance` from a seed, excluded ones aside.

    Outside the fill `dist` must be NaN, or the caller masks the result by it.
    """
    return (dist > distance) & ~excluded
