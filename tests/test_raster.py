import math

import numpy as np
import pytest
from shapely.geometry import Point

from breakneck import raster
from breakneck.raster import Grid, geodesic, touch

PIXEL = 0.1


def l_mask(size=31, arm=3):
    """An L of `arm`-pixel-wide bars: along the top row, then down the right."""
    mask = np.zeros((size, size), bool)
    mask[:arm, :] = True
    mask[:, size - arm :] = True
    return mask


def test_geodesic_goes_around_the_corner_of_an_l():
    grid = Grid(0, 0, 31, 31, PIXEL)
    start, end = (1, 1), (29, 29)

    dist = geodesic(grid, l_mask(), [start])

    straight = math.dist(start, end) * PIXEL
    # The shortest path through the L hugs its inner corner near pixel (3, 28).
    around = (math.dist(start, (3, 28)) + math.dist((3, 28), end)) * PIXEL
    assert dist[end] >= around
    assert dist[end] > straight * 1.3
    assert dist[end] == pytest.approx(around, rel=0.05)


def test_geodesic_is_infinite_outside_the_mask():
    grid = Grid(0, 0, 31, 31, PIXEL)

    dist = geodesic(grid, l_mask(), [(1, 1)])

    assert np.isinf(dist[20, 10])
    assert dist[1, 1] == 0


def test_geodesic_ignores_starts_outside_the_mask():
    grid = Grid(0, 0, 31, 31, PIXEL)

    assert np.isinf(geodesic(grid, l_mask(), [(20, 10)])).all()


def test_limited_geodesic_is_infinite_beyond_the_limit():
    grid = Grid(0, 0, 51, 51, PIXEL)
    mask = np.ones(grid.shape, bool)

    dist = geodesic(grid, mask, [(25, 25)], limit=1.0)

    assert dist[25, 34] == pytest.approx(0.9)
    assert np.isinf(dist[25, 36])
    assert np.isinf(dist[0, 0])


def test_limited_geodesic_searches_only_a_window_around_the_starts(monkeypatch):
    searched = []
    real = raster.MCP_Geometric

    def spy(costs, **kw):
        searched.append(costs.shape)
        return real(costs, **kw)

    monkeypatch.setattr(raster, "MCP_Geometric", spy)
    grid = Grid(0, 0, 200, 200, PIXEL)
    limit = 1.0

    geodesic(grid, np.ones(grid.shape, bool), [(100, 100)], limit=limit)

    reach = int(limit / PIXEL) + 1
    assert searched == [(2 * reach + 1, 2 * reach + 1)]


def test_touch_returns_mask_pixels_inside_the_geometry():
    grid = Grid(0, 0, 20, 20, PIXEL)
    mask = np.zeros(grid.shape, bool)
    mask[:, :10] = True

    pixels = touch(grid, mask, Point(1.0, 1.0).buffer(0.35))

    assert pixels
    assert all(mask[p] for p in pixels)
    assert all(math.dist(grid.point(p), (1.0, 1.0)) < 0.35 for p in pixels)
    assert {c for _, c in pixels} == {7, 8, 9}
