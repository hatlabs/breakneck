"""Planner interface shared by the via-placing commands.

A planner is a plain callable `(snapshot, options, forbidden) -> Plan`. It
must not place a via within `FORBIDDEN_RADIUS` of any forbidden position; the
DRC prune loop feeds back the positions of units it dropped.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from breakneck.snapshot import Point2, Snapshot, ViaSize

FORBIDDEN_RADIUS = 0.5

# Spacing rules the planners keep; kicad-cli DRC remains the authority.
OTHER_NET_CLEARANCE = 0.3
# Keeps vias out of same-net pads (no via-in-pad).
SAME_NET_PAD_GAP = 0.15
HOLE_TO_HOLE_GAP = 0.3
EDGE_GAP = 0.5

# pourstitch defaults live here so the CLI can show them without the place extra.
POURSTITCH_DISTANCE = 2.5
POURSTITCH_FALLBACK_VIA = ViaSize(0.5, 0.3)
POURSTITCH_PIXEL = 0.1


def raster_size(
    bounds: tuple[float, float, float, float], pixel: float
) -> tuple[int, int]:
    """(width, height) in pixels of the pourstitch raster covering `bounds`."""
    x0, y0, x1, y1 = bounds
    return math.ceil((x1 - x0) / pixel), math.ceil((y1 - y0) / pixel)


@dataclass(frozen=True)
class Stub:
    """Track from a pad to its planned via."""

    start: Point2
    end: Point2
    width: float
    layer: str


@dataclass(frozen=True)
class PlannedUnit:
    """One planned via plus its optional stub; pruned and committed whole.

    A forced unit is a best-effort placement: DRC clearance violations it
    causes are reported instead of dropping it.
    """

    position: Point2
    net: str
    size: ViaSize
    stub: Stub | None = None
    forced: bool = False


@dataclass(frozen=True)
class UnfixableSpot:
    position: Point2
    net: str
    reason: str
    layer: str | None = None


@dataclass(frozen=True)
class Plan:
    units: tuple[PlannedUnit, ...] = ()
    unfixable: tuple[UnfixableSpot, ...] = ()


OptionsT = TypeVar("OptionsT")

Planner = Callable[[Snapshot, OptionsT, Sequence[Point2]], Plan]
