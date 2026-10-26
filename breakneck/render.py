"""Pour-stitch distance heatmaps: a panel figure and per-layer overlay PNGs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.patches import Circle

from breakneck.overlay import OVERLAY_PPI, OVERLAY_TAG
from breakneck.plan import Plan
from breakneck.pourstitch import CoverageMap, Grid
from breakneck.snapshot import Snapshot

COLORMAP = "viridis"
OVER_COLOUR = "#ff2a6d"
BACKGROUND = "#101010"
OUTLINE_COLOUR = "#888888"
UNFIXABLE_COLOUR = "cyan"
# Neutral, so copper no via can reach reads apart from the distance scale.
UNSTITCHABLE_COLOUR = "#a0a0a0"
PANEL_COLUMNS = 2
PANEL_INCHES = 8
PANEL_DPI = 100
MARGIN_MM = 0.5


def _colormap() -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_over(OVER_COLOUR)
    return cmap


def _same(a: CoverageMap, b: CoverageMap) -> bool:
    return bool(
        np.array_equal(a.distance, b.distance, equal_nan=True)
        and np.array_equal(a.unstitchable, b.unstitchable)
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


def render_panels(
    path: Path,
    snapshot: Snapshot,
    grid: Grid,
    maps: Sequence[CoverageMap],
    plan: Plan,
    distance: float,
) -> None:
    shown = panels(maps)
    if not shown:
        return
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
    cmap = _colormap()
    grey = matplotlib.colors.ListedColormap([UNSTITCHABLE_COLOUR])
    image = None
    for ax in axs.ravel()[len(shown) :]:
        ax.set_visible(False)
    for ax, (m, twins) in zip(axs.ravel(), shown):
        ax.set_facecolor(BACKGROUND)
        extent = (x0, x1, y1, y0)
        image = ax.imshow(
            np.ma.masked_where(m.unstitchable | np.isnan(m.distance), m.distance),
            cmap=cmap,
            vmin=0,
            vmax=distance,
            extent=extent,
            interpolation="nearest",
        )
        ax.imshow(
            np.ma.masked_where(~m.unstitchable, m.unstitchable),
            cmap=grey,
            extent=extent,
            interpolation="nearest",
        )
        for v in snapshot.vias:
            if v.net == m.net:
                ax.add_patch(
                    Circle(
                        v.position, v.diameter / 2, fill=False, color="white", lw=0.8
                    )
                )
        for u in plan.units:
            if u.net == m.net:
                ax.add_patch(Circle(u.position, u.size.diameter / 2, color="white"))
        for s in plan.unfixable:
            if s.net == m.net and s.layer in (m.layer, *twins):
                ax.plot(*s.position, "x", color=UNFIXABLE_COLOUR, ms=5, mew=1.2)
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
        image, ax=axs, orientation="horizontal", fraction=0.03, pad=0.03, extend="max"
    )
    cb.set_label(
        f"distance to nearest stitch through the copper (mm); pink > {distance:g} mm",
        color="white",
    )
    cb.ax.tick_params(colors="white")
    fig.suptitle(
        "Pour stitching: filled = planned via, ring = existing via, "
        "cyan x = no via fits, grey = out of reach of any two-layer spot",
        color="white",
    )
    fig.savefig(path, dpi=PANEL_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def overlay_rgba(maps: Sequence[CoverageMap], distance: float) -> npt.NDArray[np.uint8]:
    """Composite heatmap of one layer's pours; transparent outside copper.

    A layer's pours of different nets don't overlap, so later maps only fill
    pixels the earlier ones left empty.
    """
    shape = maps[0].distance.shape
    rgba = np.zeros((*shape, 4), np.uint8)
    cmap = _colormap()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=distance)
    grey = matplotlib.colors.to_rgba_array(UNSTITCHABLE_COLOUR)[0]
    for m in maps:
        inside = ~np.isnan(m.distance)
        colours = cmap(norm(np.where(inside, m.distance, 0)), bytes=True)
        rgba[inside] = colours[inside]
        rgba[m.unstitchable] = np.round(grey * 255).astype(np.uint8)
    return rgba


def write_overlays(
    out_dir: Path, maps: Sequence[CoverageMap], distance: float
) -> dict[str, Path]:
    """One overlay PNG per copper layer that has stitched pours."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_layer: dict[str, list[CoverageMap]] = {}
    for m in maps:
        by_layer.setdefault(m.layer, []).append(m)
    paths = {}
    for layer, layer_maps in by_layer.items():
        path = out_dir / f"pourstitch-{layer.replace('.', '_')}.png"
        matplotlib.image.imsave(
            path,
            overlay_rgba(layer_maps, distance),
            dpi=OVERLAY_PPI,
            metadata=dict([OVERLAY_TAG]),
        )
        paths[layer] = path
    return paths
