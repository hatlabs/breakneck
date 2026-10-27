"""Distance heatmaps: a panel figure and per-layer overlay PNGs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Circle
from scipy import ndimage

from breakneck.overlay import OVERLAY_PPI, OVERLAY_TAG
from breakneck.raster import BoolMap, CoverageMap, Grid
from breakneck.snapshot import Point2, Snapshot

BACKGROUND = "#101010"
OUTLINE_COLOUR = "#888888"
UNFIXABLE_COLOUR = "cyan"
MARKER_COLOUR = "white"
RING_LINE_WIDTH = 0.8
CONTOUR_LINE_WIDTH = 0.8
# A served pixel is on the overlay contour when an edge neighbour is not.
CONTOUR_NEIGHBOURS = ndimage.generate_binary_structure(2, 1)
CROSS_SIZE = 5
CROSS_LINE_WIDTH = 1.2
# Two pixels, so a ring stays visible on the board zoomed out.
OVERLAY_RING_WIDTH_MM = 0.2
# Neutral, so excluded copper reads apart from the distance scale.
EXCLUDED_COLOUR = "#a0a0a0"
PANEL_COLUMNS = 2
PANEL_INCHES = 8
PANEL_DPI = 100
MARGIN_MM = 0.5


@dataclass(frozen=True)
class Contour:
    """A line at `margin` beyond the distance, between copper on either side."""

    colour: str
    margin: float = 0.0


@dataclass(frozen=True)
class HeatmapStyle:
    """A command's look; `colourbar_label` may use `{distance}`.

    The colour scale runs from 0 to `scale` × the distance. Without an over
    colour, values above it saturate at its top. Unreachable copper
    (infinite distance) has its own colour.
    """

    title: str
    colourbar_label: str
    png_prefix: str
    colormap: str
    over_colour: str | None
    unreachable_colour: str
    scale: float = 1.0
    contour: Contour | None = None


Disc = tuple[Point2, float]
Cross = tuple[str, Point2]
MapKey = tuple[str, str]


@dataclass(frozen=True)
class Markers:
    """What is drawn over one map: outlined circles and filled circles as
    (centre, radius) in board mm, and crosses as (layer, position).

    Crosses are the net's, in drawing order; each map's panel draws those on
    its own layers, so a folded panel keeps the order across its layers.
    """

    rings: tuple[Disc, ...] = ()
    dots: tuple[Disc, ...] = ()
    crosses: tuple[Cross, ...] = ()


NO_MARKERS = Markers()


def _colormap(style: HeatmapStyle) -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap(style.colormap).copy()
    if style.over_colour is not None:
        cmap.set_over(style.over_colour)
    return cmap


def colourbar_extend(
    maps: Sequence[CoverageMap], distance: float, style: HeatmapStyle
) -> str:
    """Colourbar `extend`: "max" when its over triangle stands for copper.

    An over colour always gets its triangle, as the legend for it; a
    saturating scale gets one only where copper lies beyond its top.
    """
    if style.over_colour is not None:
        return "max"
    top = style.scale * distance
    beyond = any(
        np.any(np.isfinite(m.distance) & (m.distance > top) & ~m.excluded) for m in maps
    )
    return "max" if beyond else "neither"


def _contour_pixels(m: CoverageMap, level: float) -> BoolMap:
    """Copper within `level` with an edge neighbour beyond it."""
    served = (m.distance <= level) & ~m.excluded
    return served & ndimage.binary_dilation(m.uncovered(level), CONTOUR_NEIGHBOURS)


def _rgba_bytes(colour: str) -> npt.NDArray[np.uint8]:
    return np.round(matplotlib.colors.to_rgba_array(colour)[0] * 255).astype(np.uint8)


def _same(a: CoverageMap, b: CoverageMap) -> bool:
    return bool(
        np.array_equal(a.distance, b.distance, equal_nan=True)
        and np.array_equal(a.excluded, b.excluded)
    )


def panels(maps: Sequence[CoverageMap]) -> list[tuple[CoverageMap, list[str]]]:
    """One panel per net and layer; a layer identical to an earlier one of
    the same net is folded into it and listed alongside."""
    out: list[tuple[CoverageMap, list[str]]] = []
    for m in maps:
        twin = next(
            (p for p in out if p[0].net == m.net and _same(p[0], m)),
            None,
        )
        if twin is None:
            out.append((m, []))
        else:
            twin[1].append(m.layer)
    return out


def panel_markers(
    markers: Mapping[MapKey, Markers], m: CoverageMap, twins: Sequence[str]
) -> Markers:
    """Markers of a panel's map and of the layers folded into it.

    Circles repeat across a net's identical layers and are drawn once;
    crosses are those of the panel's layers.
    """
    out = markers.get((m.net, m.layer), NO_MARKERS)
    for layer in twins:
        twin = markers.get((m.net, layer), NO_MARKERS)
        out = Markers(
            out.rings + tuple(r for r in twin.rings if r not in out.rings),
            out.dots + tuple(d for d in twin.dots if d not in out.dots),
            out.crosses,
        )
    layers = (m.layer, *twins)
    return Markers(out.rings, out.dots, tuple(c for c in out.crosses if c[0] in layers))


def render_panels(
    path: Path,
    snapshot: Snapshot,
    grid: Grid,
    maps: Sequence[CoverageMap],
    markers: Mapping[MapKey, Markers],
    distance: float,
    style: HeatmapStyle,
) -> None:
    shown = panels(maps)
    if not shown:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(shown) / PANEL_COLUMNS)
    cols = min(PANEL_COLUMNS, len(shown))
    x0, y0 = grid.x0, grid.y0
    x1, y1 = x0 + grid.width * grid.pixel, y0 + grid.height * grid.pixel
    aspect = (y1 - y0) / (x1 - x0)
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(cols * PANEL_INCHES, rows * PANEL_INCHES * aspect + 1),
        squeeze=False,
    )
    fig.patch.set_facecolor(BACKGROUND)
    cmap = _colormap(style)
    grey = matplotlib.colors.ListedColormap([EXCLUDED_COLOUR])
    unreachable = matplotlib.colors.ListedColormap([style.unreachable_colour])
    image = None
    for ax in axs.ravel()[len(shown) :]:
        ax.set_visible(False)
    for ax, (m, twins) in zip(axs.ravel(), shown):
        ax.set_facecolor(BACKGROUND)
        extent = (x0, x1, y1, y0)
        # imshow draws non-finite values as background, so copper no seed
        # reaches (inf) gets a layer of its own.
        lost = np.isinf(m.distance) & ~m.excluded
        image = ax.imshow(
            np.ma.masked_where(m.excluded | lost | np.isnan(m.distance), m.distance),
            cmap=cmap,
            vmin=0,
            vmax=style.scale * distance,
            extent=extent,
            interpolation="nearest",
        )
        if style.contour is not None:
            ax.contour(
                grid.xs,
                grid.ys,
                np.ma.masked_where(~np.isfinite(m.distance) | m.excluded, m.distance),
                levels=[distance + style.contour.margin],
                colors=[style.contour.colour],
                linewidths=CONTOUR_LINE_WIDTH,
            )
        ax.imshow(
            np.ma.masked_where(~lost, lost),
            cmap=unreachable,
            extent=extent,
            interpolation="nearest",
        )
        ax.imshow(
            np.ma.masked_where(~m.excluded, m.excluded),
            cmap=grey,
            extent=extent,
            interpolation="nearest",
        )
        marks = panel_markers(markers, m, twins)
        for centre, radius in marks.rings:
            ax.add_patch(
                Circle(
                    centre, radius, fill=False, color=MARKER_COLOUR, lw=RING_LINE_WIDTH
                )
            )
        for centre, radius in marks.dots:
            ax.add_patch(Circle(centre, radius, color=MARKER_COLOUR))
        for _, position in marks.crosses:
            ax.plot(
                *position,
                "x",
                color=UNFIXABLE_COLOUR,
                ms=CROSS_SIZE,
                mew=CROSS_LINE_WIDTH,
            )
        outline = snapshot.board_outline
        for poly in getattr(outline, "geoms", [outline]):
            ax.plot(*poly.exterior.xy, color=OUTLINE_COLOUR, lw=0.8)
        ax.set_xlim(x0 - MARGIN_MM, x1 + MARGIN_MM)
        ax.set_ylim(y1 + MARGIN_MM, y0 - MARGIN_MM)
        title = f"{m.net} on {m.layer}" + (
            f" ({', '.join(twins)} identical)" if twins else ""
        )
        ax.set_title(title, color="white")
        ax.set_xticks([])
        ax.set_yticks([])
    assert image is not None
    cb = fig.colorbar(
        image,
        ax=axs,
        orientation="horizontal",
        fraction=0.03,
        pad=0.03,
        extend=colourbar_extend(maps, distance, style),
    )
    cb.set_label(style.colourbar_label.format(distance=distance), color="white")
    cb.ax.tick_params(colors="white")
    fig.suptitle(style.title, color="white")
    fig.savefig(path, dpi=PANEL_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _draw_ring(
    rgba: npt.NDArray[np.uint8], grid: Grid, centre: Point2, radius: float
) -> None:
    half = OVERLAY_RING_WIDTH_MM / 2
    rs, cs = grid.window(grid.index(centre), radius + half)
    dx = grid.xs[cs][np.newaxis, :] - centre[0]
    dy = grid.ys[rs][:, np.newaxis] - centre[1]
    on = np.abs(np.hypot(dx, dy) - radius) <= half
    rgba[rs, cs][on] = _rgba_bytes(MARKER_COLOUR)


def overlay_rgba(
    grid: Grid,
    maps: Sequence[CoverageMap],
    markers: Mapping[MapKey, Markers],
    distance: float,
    style: HeatmapStyle,
) -> npt.NDArray[np.uint8]:
    """Composite heatmap of one layer's pours; transparent outside copper.

    A layer's pours of different nets don't overlap, so later maps only fill
    pixels the earlier ones left empty. Only the markers' rings are drawn:
    on the board, dots and crosses are vias and markers of their own.
    """
    shape = maps[0].distance.shape
    rgba = np.zeros((*shape, 4), np.uint8)
    cmap = _colormap(style)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=style.scale * distance)
    for m in maps:
        inside = ~np.isnan(m.distance)
        colours = cmap(norm(np.where(inside, m.distance, 0)), bytes=True)
        rgba[inside] = colours[inside]
        rgba[np.isinf(m.distance)] = _rgba_bytes(style.unreachable_colour)
        rgba[m.excluded] = _rgba_bytes(EXCLUDED_COLOUR)
        if style.contour is not None:
            level = distance + style.contour.margin
            rgba[_contour_pixels(m, level)] = _rgba_bytes(style.contour.colour)
    rings = {r for m in maps for r in markers.get((m.net, m.layer), NO_MARKERS).rings}
    for centre, radius in rings:
        _draw_ring(rgba, grid, centre, radius)
    return rgba


def write_overlays(
    out_dir: Path,
    grid: Grid,
    maps: Sequence[CoverageMap],
    markers: Mapping[MapKey, Markers],
    distance: float,
    style: HeatmapStyle,
) -> dict[str, Path]:
    """One overlay PNG per copper layer that has maps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_layer: dict[str, list[CoverageMap]] = {}
    for m in maps:
        by_layer.setdefault(m.layer, []).append(m)
    paths = {}
    for layer, layer_maps in by_layer.items():
        path = out_dir / f"{style.png_prefix}-{layer.replace('.', '_')}.png"
        matplotlib.image.imsave(
            path,
            overlay_rgba(grid, layer_maps, markers, distance, style),
            dpi=OVERLAY_PPI,
            metadata=dict([OVERLAY_TAG]),
        )
        paths[layer] = path
    return paths
