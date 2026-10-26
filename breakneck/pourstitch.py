"""Pour stitching: every pixel of a net's pours within D of a stitch.

Distance is measured through the net's own copper on each layer, so narrow
fingers that a straight-line check would call covered still get a via. A
stitch is a same-net via or plated pad that sits in the net's fill on every
layer where one of the net's zone outlines contains it (the zone-layer rule,
`Snapshot.required_layers`).

Placement runs a square lattice of pitch D·√2 first, then fills the gaps
greedily, farthest uncovered pixel first.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt
import shapely
from loguru import logger
from scipy import ndimage
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from skimage.graph import MCP_Geometric

from breakneck.plan import (
    EDGE_GAP,
    FORBIDDEN_RADIUS,
    HOLE_TO_HOLE_GAP,
    OTHER_NET_CLEARANCE,
    POURSTITCH_DISTANCE,
    POURSTITCH_FALLBACK_VIA,
    POURSTITCH_PIXEL,
    SAME_NET_PAD_GAP,
    Plan,
    PlannedUnit,
    UnfixableSpot,
    pad_clearance,
    raster_size,
)
from breakneck.snapshot import MIN_STITCH_LAYERS, PadKind, Point2, Snapshot, ViaSize

BoolMap = npt.NDArray[np.bool_]
FloatMap = npt.NDArray[np.floating]
Pixel = tuple[int, int]

PIXEL = POURSTITCH_PIXEL
# Every clearance gets one pixel of margin for the rasterisation error.
QUANTISATION = 0.1
VIA_SPACING = 0.9
IN_FILL_MARGIN = 0.05
# Thermal spokes let an existing stitch reach fill a little beyond its copper.
THERMAL_REACH = 0.4
LATTICE_WINDOW = 0.6
# A new via only shortens distances near it; bounding the update keeps it fast.
UPDATE_REACH_FACTOR = 3
UNREACHABLE = 1e6
FAR = 1e9
COORD_DECIMALS = 3


@dataclass(frozen=True)
class Options:
    nets: tuple[str, ...]
    distance: float = POURSTITCH_DISTANCE
    via: ViaSize | None = None
    fallback_via: ViaSize = POURSTITCH_FALLBACK_VIA

    @property
    def pitch(self) -> float:
        """Square lattice pitch whose cell centre is `distance` from a site."""
        return self.distance * math.sqrt(2)


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
        return _edt(~mask) * np.float32(self.pixel)

    def depth(self, mask: BoolMap) -> FloatMap:
        """Distance in mm from each True pixel to the nearest False pixel."""
        return _edt(mask) * np.float32(self.pixel)


def _edt(mask: BoolMap) -> FloatMap:
    """Pixel distance from each True pixel to the nearest False pixel."""
    return np.asarray(ndimage.distance_transform_edt(mask), dtype=np.float32)


def lattice_sites(
    bounds: tuple[float, float, float, float], pitch: float, offset: float
) -> list[Point2]:
    x0, y0, x1, y1 = bounds
    return [
        (float(x), float(y))
        for y in np.arange(y0 + offset, y1, pitch)
        for x in np.arange(x0 + offset, x1, pitch)
    ]


def _union(geoms: Iterable[BaseGeometry]) -> BaseGeometry:
    return unary_union([g for g in geoms if not g.is_empty])


def via_sizes(snapshot: Snapshot, options: Options, net: str) -> tuple[ViaSize, ...]:
    """Sizes to try, preferred first."""
    default = options.via or snapshot.via_size(net)
    return tuple(dict.fromkeys((default, options.fallback_via)))


@dataclass(frozen=True)
class CoverageMap:
    """Distance (mm) through `net`'s fill on `layer` to the nearest stitch.

    NaN outside the fill, inf where no stitch reaches. `unstitchable` marks
    fill no via could ever bring within D (see `_Context.unstitchable`).
    """

    net: str
    layer: str
    distance: FloatMap
    unstitchable: BoolMap

    def uncovered(self, distance: float) -> BoolMap:
        """Stitchable fill farther than `distance` from a stitch."""
        return _uncovered(self.distance, self.unstitchable, distance)


def _uncovered(dist: FloatMap, unstitchable: BoolMap, distance: float) -> BoolMap:
    """Pixels farther than `distance` from a stitch, unstitchable ones aside.

    Outside the fill `dist` must be NaN, or the caller masks the result by it.
    """
    return (dist > distance) & ~unstitchable


class _Context:
    """Rasters that depend only on the snapshot and the options."""

    def __init__(self, snapshot: Snapshot, options: Options) -> None:
        if snapshot.board_outline.is_empty:
            raise ValueError("Board has no closed Edge.Cuts outline")
        self.snapshot = snapshot
        self.options = options
        self.grid = g = Grid.covering(snapshot.board_outline.bounds)
        present = [n for n in options.nets if snapshot.zone_layers(n)]
        for n in sorted(set(options.nets) - set(present)):
            logger.warning(f"Net {n} has no copper zones; skipping")
        self.nets, _ = snapshot.split_stitchable(present)

        self.layers: dict[str, list[str]] = {}
        self.fill: dict[tuple[str, str], BoolMap] = {}
        self.depth: dict[tuple[str, str], FloatMap] = {}
        self.outline: dict[tuple[str, str], BoolMap] = {}
        for net in self.nets:
            self.layers[net] = []
            for L in snapshot.copper_layers:
                fill = g.mask(snapshot.fill(net, L))
                self.outline[(net, L)] = g.mask(
                    _union(
                        z.outline
                        for z in snapshot.copper_zones
                        if z.net == net and L in z.layers
                    )
                )
                if fill.any():
                    self.layers[net].append(L)
                    self.fill[(net, L)] = fill
                    self.depth[(net, L)] = g.depth(fill)

        self.feasible: dict[tuple[str, ViaSize], BoolMap] = {}
        if self.nets:
            self._edge, self._keepout, self._hole = self._board_clearances()
        for net in self.nets:
            self.feasible.update(self._feasible(net))

    @cached_property
    def unstitchable(self) -> dict[tuple[str, str], BoolMap]:
        """Fill per net and layer that no stitch can ever bring within D.

        A pixel is unstitchable when, through the net's copper on its layer,
        it is farther than D from every spot that passes the zone-layer rule
        (the net's zone outlines cover two or more layers there, all filled)
        and from every existing stitch. Other-net clearances are ignored:
        a spot they block is uncovered, not unstitchable. Reach through the
        copper, not fill islands, decides: a one-layer local zone joined to
        a two-layer pour by a long neck is out of reach beyond the neck's
        first D, though it is one island with the pour.
        """
        d = self.options.distance
        out = {}
        for net in self.nets:
            joinable = self.joinable[net]
            seeds = self.stitch_seeds(net)
            for L in self.layers[net]:
                fill = self.fill[(net, L)]
                starts = [(int(r), int(c)) for r, c in np.argwhere(joinable & fill)]
                starts += seeds[L]
                out[(net, L)] = fill & (self.geodesic(net, L, starts, limit=d) > d)
        return out

    @cached_property
    def joinable(self) -> dict[str, BoolMap]:
        """Pixels per net where a via passes the zone-layer rule."""
        return {net: self._zone_layer_rule(net) for net in self.nets}

    def _zone_layer_rule(self, net: str, min_depth: float | None = None) -> BoolMap:
        """Pixels where the net's zone outlines cover two or more layers and
        a via sits in the fill on each: anywhere in it, or at least
        `min_depth` from its edge."""
        n_required = np.zeros(self.grid.shape, np.int8)
        ok = np.ones(self.grid.shape, bool)
        for L in self.snapshot.copper_layers:
            outline = self.outline[(net, L)]
            n_required += outline
            if (net, L) not in self.fill:
                ok &= ~outline
            elif min_depth is None:
                ok &= ~outline | self.fill[(net, L)]
            else:
                ok &= ~outline | (self.depth[(net, L)] >= min_depth)
        return ok & (n_required >= MIN_STITCH_LAYERS)

    def in_fill(self, net: str, layer: str, px: Pixel, size: ViaSize) -> bool:
        depth = self.depth.get((net, layer))
        return depth is not None and depth[px] >= size.diameter / 2 + IN_FILL_MARGIN

    def connects(self, net: str, layer: str, px: Pixel, size: ViaSize) -> bool:
        """Whether a via at `px` joins the net's fill on `layer`."""
        return self.outline[(net, layer)][px] or self.in_fill(net, layer, px, size)

    def _board_clearances(self) -> tuple[FloatMap, FloatMap, FloatMap]:
        """Distances to the board edge, via keepouts and drilled holes."""
        snap, g = self.snapshot, self.grid
        # The False border makes the board edge count as outside.
        inside = np.pad(g.mask(snap.board_outline), 1)
        edge = _edt(inside)[1:-1, 1:-1] * np.float32(g.pixel)
        holes = [Point(p.position).buffer(p.drill / 2) for p in snap.pads if p.drill]
        holes += [Point(v.position).buffer(v.drill / 2) for v in snap.vias]
        return (
            edge,
            g.distance_to(g.touching(snap.via_keepouts)),
            g.distance_to(g.touching(_union(holes))),
        )

    def _feasible(self, net: str) -> dict[tuple[str, ViaSize], BoolMap]:
        """Pixels where a via of each candidate size on `net` may sit."""
        snap, g = self.snapshot, self.grid
        sizes = via_sizes(snap, self.options, net)
        radius = {s: s.diameter / 2 + QUANTISATION for s in sizes}
        ok = {
            s: (self._edge >= r + EDGE_GAP)
            & (self._keepout >= r)
            & (self._hole >= s.drill / 2 + QUANTISATION + HOLE_TO_HOLE_GAP)
            & self._zone_layer_rule(net, r + IN_FILL_MARGIN)
            for s, r in radius.items()
        }
        for L in snap.copper_layers:
            for gap, geoms in self._obstacles(net, L).items():
                dist = g.distance_to(g.touching(_union(geoms)))
                for s, r in radius.items():
                    ok[s] &= dist >= r + gap
        return {(net, s): m for s, m in ok.items()}

    def _obstacles(self, net: str, layer: str) -> dict[float, list[BaseGeometry]]:
        """Copper a via of `net` must clear on `layer`, grouped by clearance.

        Other-net clearance applies on every layer, inside the net's own fill
        too: the fill's edge is not a reliable proxy for it.
        """
        snap = self.snapshot
        out: dict[float, list[BaseGeometry]] = {}

        def add(gap: float, geom: BaseGeometry) -> None:
            out.setdefault(gap, []).append(geom)

        for p in snap.pads:
            other = p.net != net
            gap = pad_clearance(p.clearance_override) if other else SAME_NET_PAD_GAP
            if layer in p.shapes:
                add(gap, p.shapes[layer])
            elif p.kind is PadKind.NPTH and p.drill and other:
                add(gap, Point(p.position).buffer(p.drill / 2))
        for t in snap.tracks:
            if t.layer == layer and t.net != net:
                add(OTHER_NET_CLEARANCE, t.geometry)
        for v in snap.vias:
            if v.net != net:
                add(OTHER_NET_CLEARANCE, Point(v.position).buffer(v.diameter / 2))
        # Other nets' pours are not obstacles: they refill around new vias.
        return out

    def touch(self, net: str, layer: str, geom: BaseGeometry) -> list[Pixel]:
        """Fill pixels of `net` on `layer` inside `geom`."""
        g = self.grid
        win = g.bounds_window(geom.bounds)
        sub = g.mask(geom, win) & self.fill[(net, layer)][win]
        rr, cc = np.nonzero(sub)
        return list(zip((rr + win[0].start).tolist(), (cc + win[1].start).tolist()))

    @cached_property
    def stitches(self) -> dict[str, list[dict[str, list[Pixel]]]]:
        """Existing stitches per net, as the fill pixels each touches per layer."""
        return {net: self._stitches(net) for net in self.nets}

    def _stitches(self, net: str) -> list[dict[str, list[Pixel]]]:
        snap = self.snapshot
        candidates: list[tuple[Point2, dict[str, BaseGeometry]]] = []
        for v in snap.vias:
            if v.net == net:
                disc = Point(v.position).buffer(v.diameter / 2)
                candidates.append((v.position, dict.fromkeys(snap.copper_layers, disc)))
        for p in snap.pads:
            if p.net == net and p.drill and p.kind is PadKind.PTH:
                candidates.append((p.position, dict(p.shapes)))

        out = []
        for pos, shapes in candidates:
            required = snap.required_layers(net, pos)
            touched = {
                L: self.touch(net, L, shapes[L].buffer(THERMAL_REACH))
                for L in self.layers[net]
                if L in shapes
            }
            if len(required) >= MIN_STITCH_LAYERS and all(
                touched.get(L) for L in required
            ):
                out.append(touched)
        return out

    def stitch_seeds(
        self, net: str, extra: Sequence[PlannedUnit] = ()
    ) -> dict[str, list[Pixel]]:
        """Fill pixels per layer touched by a stitch of `net`."""
        seeds: dict[str, list[Pixel]] = {L: [] for L in self.layers[net]}
        touched = self.stitches[net] + [
            self.unit_touch(u) for u in extra if u.net == net
        ]
        for t in touched:
            for L, pxs in t.items():
                seeds[L] += pxs
        return seeds

    def unit_touch(self, u: PlannedUnit) -> dict[str, list[Pixel]]:
        px = self.grid.index(u.position)
        disc = Point(u.position).buffer(u.size.diameter / 2 + self.grid.pixel)
        return {
            L: self.touch(u.net, L, disc)
            for L in self.layers[u.net]
            if self.connects(u.net, L, px, u.size)
        }

    def geodesic(
        self, net: str, layer: str, starts: Sequence[Pixel], limit: float | None = None
    ) -> FloatMap:
        """Distance (mm) through the fill from the nearest start pixel.

        With a limit, only the starts' bounding box grown by the limit is
        searched: a path no longer than the limit can't leave it.
        """
        g = self.grid
        fill = self.fill[(net, layer)]
        starts = [s for s in starts if fill[s]]
        out = np.full(g.shape, np.inf)
        if not starts:
            return out
        rs, cs = zip(*starts)
        k = g.height + g.width if limit is None else int(limit / g.pixel) + 1
        win = (
            slice(max(0, min(rs) - k), min(g.height, max(rs) + k + 1)),
            slice(max(0, min(cs) - k), min(g.width, max(cs) + k + 1)),
        )
        mcp = MCP_Geometric(np.where(fill[win], 1.0, np.inf), fully_connected=True)
        cum, _ = mcp.find_costs(
            [(r - win[0].start, c - win[1].start) for r, c in starts]
        )
        dist = np.asarray(cum) * g.pixel
        if limit is not None:
            dist[dist > limit] = np.inf
        out[win] = dist
        return out


class _NetRun:
    """Placement state for one net within one planner call."""

    def __init__(
        self, ctx: _Context, net: str, blocked: BoolMap, plan: _PlanBuilder
    ) -> None:
        self.ctx, self.net, self.blocked, self.out = ctx, net, blocked, plan
        self.sizes = via_sizes(ctx.snapshot, ctx.options, net)
        seeds = ctx.stitch_seeds(net)
        self.dist = {L: ctx.geodesic(net, L, seeds[L]) for L in ctx.layers[net]}

    def add(self, px: Pixel, size: ViaSize) -> None:
        ctx = self.ctx
        unit = PlannedUnit(ctx.grid.point(px), self.net, size)
        limit = UPDATE_REACH_FACTOR * ctx.options.distance
        for L, touched in ctx.unit_touch(unit).items():
            self.dist[L] = np.minimum(
                self.dist[L], ctx.geodesic(self.net, L, touched, limit)
            )
        self.out.units.append(unit)
        win, near = ctx.grid.disc(px, VIA_SPACING)
        self.blocked[win] |= near

    def lattice(self, offset: float) -> int:
        ctx, g = self.ctx, self.ctx.grid
        pitch = ctx.options.pitch
        placed = 0
        for site in lattice_sites(ctx.snapshot.board_outline.bounds, pitch, offset):
            px = g.index(site)
            if not g.contains(px):
                continue
            if not ctx.joinable[self.net][px]:
                continue
            covered = [
                self.dist[L][px]
                for L in ctx.layers[self.net]
                if ctx.fill[(self.net, L)][px]
            ]
            if min(covered) <= pitch / 2:
                continue
            win = g.window(px, LATTICE_WINDOW)
            for size in self.sizes:
                cand = ctx.feasible[(self.net, size)][win] & ~self.blocked[win]
                if not cand.any():
                    continue
                rr, cc = np.nonzero(cand)
                rr, cc = rr + win[0].start, cc + win[1].start
                j = int(np.argmin((rr - px[0]) ** 2 + (cc - px[1]) ** 2))
                self.add((int(rr[j]), int(cc[j])), size)
                placed += 1
                break
        return placed

    def _worst(self, unfixable: dict[str, BoolMap]) -> tuple[str, Pixel] | None:
        d = self.ctx.options.distance
        worst: tuple[float, str, Pixel] | None = None
        for L, dist in self.dist.items():
            bad = self.ctx.fill[(self.net, L)] & ~unfixable[L] & (dist > d)
            if not bad.any():
                continue
            score = np.where(bad, np.where(np.isinf(dist), UNREACHABLE, dist), -1)
            idx = np.unravel_index(int(np.argmax(score)), score.shape)
            px = (int(idx[0]), int(idx[1]))
            if worst is None or score[px] > worst[0]:
                worst = (float(score[px]), L, px)
        return None if worst is None else (worst[1], worst[2])

    def greedy(self) -> int:
        ctx, d = self.ctx, self.ctx.options.distance
        # No via can reach unstitchable copper; searching it would only add
        # spurious unfixable spots.
        unfixable = {L: ctx.unstitchable[(self.net, L)].copy() for L in self.dist}
        placed = 0
        while (worst := self._worst(unfixable)) is not None:
            layer, px = worst
            reach = ctx.geodesic(self.net, layer, [px], limit=d) <= d
            choice = self._best_in(reach, layer)
            if choice is None:
                unfixable[layer] |= reach & (self.dist[layer] > d)
                self.out.unfixable.append(
                    UnfixableSpot(
                        ctx.grid.point(px),
                        self.net,
                        f"no via fits within {d:g} mm through the copper",
                        layer,
                    )
                )
                continue
            self.add(*choice)
            placed += 1
        return placed

    def _best_in(self, reach: BoolMap, layer: str) -> tuple[Pixel, ViaSize] | None:
        """The candidate in `reach` farthest from existing stitches, preferred size first."""
        ctx = self.ctx
        depth = ctx.depth[(self.net, layer)]
        dist = self.dist[layer]
        for size in self.sizes:
            joins = depth >= size.diameter / 2 + QUANTISATION + IN_FILL_MARGIN
            cand = reach & joins & ctx.feasible[(self.net, size)] & ~self.blocked
            if cand.any():
                score = np.where(cand, np.where(np.isinf(dist), UNREACHABLE, dist), -1)
                idx = np.unravel_index(int(np.argmax(score)), score.shape)
                return (int(idx[0]), int(idx[1])), size
        return None

    def uncovered_mm2(self) -> float:
        ctx = self.ctx
        d, px = ctx.options.distance, ctx.grid.pixel
        return sum(
            float(
                np.sum(
                    ctx.fill[(self.net, L)]
                    & _uncovered(dist, ctx.unstitchable[(self.net, L)], d)
                )
            )
            * px
            * px
            for L, dist in self.dist.items()
        )


@dataclass
class _PlanBuilder:
    units: list[PlannedUnit]
    unfixable: list[UnfixableSpot]


class PourStitcher:
    """Planner for `run_placement`; caches the static rasters across rounds."""

    def __init__(self) -> None:
        self._ctx: _Context | None = None

    def context(self, snapshot: Snapshot, options: Options) -> _Context:
        ctx = self._ctx
        if ctx is None or ctx.snapshot is not snapshot or ctx.options != options:
            ctx = self._ctx = _Context(snapshot, options)
        return ctx

    def __call__(
        self, snapshot: Snapshot, options: Options, forbidden: Sequence[Point2]
    ) -> Plan:
        ctx = self.context(snapshot, options)
        blocked = np.zeros(ctx.grid.shape, bool)
        for pos in forbidden:
            px = ctx.grid.index(pos)
            # Pad by one pixel so the snapped placement stays outside the radius.
            win, near = ctx.grid.disc(px, FORBIDDEN_RADIUS + ctx.grid.pixel)
            blocked[win] |= near
        out = _PlanBuilder([], [])
        for i, net in enumerate(ctx.nets):
            run = _NetRun(ctx, net, blocked, out)
            # Alternate nets sit half a pitch apart so their lattices interleave.
            n_lattice = run.lattice(offset=(i % 2) * options.pitch / 2)
            n_greedy = run.greedy()
            logger.info(
                f"pourstitch {net}: {n_lattice} lattice + {n_greedy} gap vias, "
                f"{run.uncovered_mm2():.2f} mm² beyond {options.distance:g} mm"
            )
        return Plan(units=tuple(out.units), unfixable=tuple(out.unfixable))

    def coverage(
        self, snapshot: Snapshot, options: Options, units: Sequence[PlannedUnit]
    ) -> list[CoverageMap]:
        """Distance maps with `units` added to the board's existing stitches."""
        ctx = self.context(snapshot, options)
        maps = []
        for net in ctx.nets:
            seeds = ctx.stitch_seeds(net, units)
            for L in ctx.layers[net]:
                dist = ctx.geodesic(net, L, seeds[L])
                maps.append(
                    CoverageMap(
                        net,
                        L,
                        np.where(ctx.fill[(net, L)], dist, np.nan),
                        ctx.unstitchable[(net, L)],
                    )
                )
        return maps

    def grid(self, snapshot: Snapshot, options: Options) -> Grid:
        return self.context(snapshot, options).grid

    def stitch_counts(self, snapshot: Snapshot, options: Options) -> dict[str, int]:
        """Existing stitches per stitchable net."""
        ctx = self.context(snapshot, options)
        return {net: len(ctx.stitches[net]) for net in ctx.nets}


@dataclass(frozen=True)
class LayerCoverage:
    net: str
    layer: str
    uncovered_pixels: int
    uncovered_mm2: float
    unstitchable_mm2: float


def summarise(
    maps: Sequence[CoverageMap], distance: float, pixel: float
) -> list[LayerCoverage]:
    """Stitchable fill area per net and layer farther than `distance` from a
    stitch, and the fill area no stitch can bring within it."""
    out = []
    area = pixel * pixel
    for m in maps:
        n = int(np.sum(m.uncovered(distance)))
        unstitchable = int(np.sum(m.unstitchable)) * area
        out.append(LayerCoverage(m.net, m.layer, n, n * area, unstitchable))
    return out
