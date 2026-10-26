"""Planner interface shared by the via-placing commands.

A planner is a plain callable `(snapshot, options, forbidden) -> Plan`. It
must not place a via within `FORBIDDEN_RADIUS` of any forbidden position; the
DRC prune loop feeds back the positions of units it dropped.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from breakneck.snapshot import Point2, Snapshot, ViaSize

FORBIDDEN_RADIUS = 0.5


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
