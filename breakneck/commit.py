"""Land a placement run in KiCad as one undoable commit."""

from __future__ import annotations

import enum
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

import kipy
import kipy.board
import kipy.board_types
from kipy.common_types import Commit
from kipy.errors import ApiError
from kipy.errors import ConnectionError as KiCadConnectionError
from kipy.geometry import Vector2
from kipy.proto.board import board_types_pb2
from kipy.proto.board.board_types_pb2 import BoardLayer
from kipy.proto.common import base_types_pb2
from kipy.proto.common.commands.editor_commands_pb2 import (
    CreateItems,
    CreateItemsResponse,
    ItemStatusCode,
)
from kipy.proto.common.envelope_pb2 import ApiStatusCode
from kipy.util.proto import pack_any
from kipy.wrapper import Wrapper
from loguru import logger

from breakneck.plan import PlannedUnit
from breakneck.snapshot import NM_PER_MM, Point2

MARKER_LAYER = BoardLayer.BL_Eco2_User
MARKER_HALF_SIZE = 0.5
MARKER_WIDTH = 0.1
REFILL_TIMEOUT_S = 300.0
REFILL_POLL_S = 1.0
COMMIT_PREFIX = "breakneck"


class MarkerKind(enum.Enum):
    UNFIXABLE = "unfixable"
    FORCED = "forced"
    UNATTRIBUTED = "unattributed"


@dataclass(frozen=True)
class Marker:
    position: Point2
    kind: MarkerKind


# Each kind gets its own glyph so the spots can be told apart on Eco2.
_GLYPHS: dict[MarkerKind, list[tuple[Point2, Point2]]] = {
    MarkerKind.UNFIXABLE: [((-1, -1), (1, 1)), ((-1, 1), (1, -1))],
    MarkerKind.FORCED: [((-1, 0), (1, 0)), ((0, -1), (0, 1))],
    MarkerKind.UNATTRIBUTED: [
        ((-1, -1), (1, -1)),
        ((1, -1), (1, 1)),
        ((1, 1), (-1, 1)),
        ((-1, 1), (-1, -1)),
    ],
}


def _nm(v: float) -> int:
    return round(v * NM_PER_MM)


def _vec(p: Point2) -> Vector2:
    return Vector2.from_xy(_nm(p[0]), _nm(p[1]))


def eco2_segment(start: Point2, end: Point2) -> kipy.board_types.BoardSegment:
    shape = base_types_pb2.GraphicShape(
        segment=base_types_pb2.GraphicSegmentAttributes(
            start=_vec(start).proto, end=_vec(end).proto
        )
    )
    seg = kipy.board_types.BoardSegment(
        proto=board_types_pb2.BoardGraphicShape(shape=shape)
    )
    seg.layer = MARKER_LAYER
    seg.attributes.stroke.width = _nm(MARKER_WIDTH)
    return seg


def marker_shapes(
    shapes: Iterable[kipy.board_types.BoardShape],
) -> list[kipy.board_types.BoardShape]:
    """Every shape on the marker layer, whichever run or user drew it."""
    return [s for s in shapes if s.layer == MARKER_LAYER]


def marker_items(markers: Sequence[Marker]) -> list[kipy.board_types.BoardSegment]:
    out = []
    for m in markers:
        x, y = m.position
        h = MARKER_HALF_SIZE
        for (ax, ay), (bx, by) in _GLYPHS[m.kind]:
            out.append(eco2_segment((x + ax * h, y + ay * h), (x + bx * h, y + by * h)))
    return out


def unit_items(
    units: Sequence[PlannedUnit],
    nets: dict[str, kipy.board_types.Net],
    layer_ids: Mapping[str, BoardLayer.ValueType],
) -> list[kipy.board_types.Via | kipy.board_types.Track]:
    out: list[kipy.board_types.Via | kipy.board_types.Track] = []
    for u in units:
        via = kipy.board_types.Via()
        via.position = _vec(u.position)
        via.diameter = _nm(u.size.diameter)
        via.drill_diameter = _nm(u.size.drill)
        via.net = nets[u.net]
        out.append(via)
        if u.stub is not None:
            track = kipy.board_types.Track()
            track.start = _vec(u.stub.start)
            track.end = _vec(u.stub.end)
            track.width = _nm(u.stub.width)
            track.layer = layer_ids[u.stub.layer]
            track.net = nets[u.net]
            out.append(track)
    return out


def commit_label(command: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    return f"{COMMIT_PREFIX} {command} {stamp}"


class CreateItemsError(RuntimeError):
    pass


def create_checked(board: kipy.board.Board, items: Sequence[Wrapper]) -> list[Wrapper]:
    """Create items and raise if KiCad refuses any of them.

    KiCad reports refusals per item, and kipy's create_items discards those
    statuses, so a refused item would otherwise vanish silently.
    """
    command = CreateItems()
    command.header.document.CopyFrom(board.document)
    command.items.extend(pack_any(i.proto) for i in items)
    response = board.client.send(command, CreateItemsResponse)
    refused = [
        r.status.error_message or ItemStatusCode.Name(r.status.code)
        for r in response.created_items
        if r.status.code != ItemStatusCode.ISC_OK
    ]
    if refused or len(response.created_items) != len(items):
        raise CreateItemsError(
            f"KiCad created {len(response.created_items) - len(refused)} of "
            f"{len(items)} items: {'; '.join(sorted(set(refused)))}"
        )
    return [kipy.board_types.unwrap(r.item) for r in response.created_items]


def commit_units(
    kicad: kipy.KiCad,
    board: kipy.board.Board,
    command: str,
    units: Sequence[PlannedUnit],
    markers: Sequence[Marker],
    stale: Sequence[kipy.board_types.BoardShape] = (),
) -> int:
    """Remove the stale markers and create the units plus the new Eco2 markers
    as one commit.

    Returns the number of placed vias and stubs.
    """
    nets = {n.name: n for n in board.get_nets()}
    missing = sorted({u.net for u in units} - nets.keys())
    if missing:
        raise ValueError(f"Nets not on the board: {', '.join(missing)}")
    layer_ids = {
        L: board.get_layer_by_name(L) for L in {u.stub.layer for u in units if u.stub}
    }
    placed = unit_items(units, nets, layer_ids)
    label = commit_label(command)

    commit = board.begin_commit()
    try:
        if stale:
            board.remove_items(stale)
        created = create_checked(board, placed) if placed else []
        if markers:
            create_checked(board, marker_items(markers))
    except BaseException:
        board.drop_commit(commit)
        raise

    # Outside the try above: once pushed, the commit can't be dropped.
    if push_tolerant(board, commit, label):
        logger.info(
            f"Committed {len(units)} vias as '{label}'"
            + (f" and {len(markers)} markers on User.Eco2" if markers else "")
            + (f", replacing {len(stale)} earlier marker shapes" if stale else "")
        )

    refill_after_commit(kicad, board)
    return len(created)


def push_tolerant(board: kipy.board.Board, commit: Commit, label: str) -> bool:
    """Push the commit; True once KiCad confirms it landed.

    A lost or timed-out connection doesn't mean KiCad didn't land it, so that
    only warns. An ApiError is KiCad's answer, so nothing landed and the
    commit is dropped.
    """
    try:
        board.push_commit(commit, label)
    except KiCadConnectionError as e:
        logger.warning(
            f"Committing '{label}' timed out or lost the connection to KiCad "
            f"({e}); it may have landed. Check KiCad's undo history before "
            "running again"
        )
        return False
    except ApiError as e:
        logger.error(f"KiCad refused to commit '{label}': {e}")
        board.drop_commit(commit)
        raise
    except Exception as e:
        logger.error(f"Committing '{label}' failed: {e}")
        raise
    return True


def refill_after_commit(kicad: kipy.KiCad, board: kipy.board.Board) -> None:
    """Refill zones; a failure only warns, since the commit has landed."""
    try:
        refilled = refill_tolerant(kicad, board)
    except (ApiError, KiCadConnectionError) as e:
        logger.warning(f"Zone refill after the commit failed ({e})")
        refilled = False
    if not refilled:
        logger.warning("Zone refill not confirmed; press B in KiCad to refill zones")


def _busy(e: Exception) -> bool:
    return isinstance(e, KiCadConnectionError) or (
        isinstance(e, ApiError) and e.code == ApiStatusCode.AS_BUSY
    )


def refill_tolerant(
    kicad: kipy.KiCad,
    board: kipy.board.Board,
    timeout_s: float = REFILL_TIMEOUT_S,
    poll_s: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Refill zones and wait, until `timeout_s` has passed, for KiCad to answer.

    kipy's blocking refill never applies its own time limit (its poll loop
    doesn't count), so the refill is started without blocking and the
    deadline is ours. A ping to a KiCad busy filling waits out the client
    timeout, so the deadline is wall-clock, not a poll count. A ping reply
    suggests, but doesn't prove, that the fill finished. AS_BUSY on the
    refill itself means KiCad refused to start it (a tool or dialog is
    active), so it is retried.
    """
    if poll_s is None:
        poll_s = REFILL_POLL_S
    deadline = clock() + timeout_s
    busy_logged = False
    while True:
        try:
            board.refill_zones(block=False)
            break
        except KiCadConnectionError as e:
            logger.warning(f"Zone refill request timed out ({e}); waiting for KiCad")
            break
        except ApiError as e:
            if e.code != ApiStatusCode.AS_BUSY:
                raise
            if not busy_logged:
                logger.warning(
                    "KiCad is busy — a tool or dialog may be open in the PCB "
                    "editor; retrying the zone refill"
                )
                busy_logged = True
            if clock() >= deadline:
                logger.warning("KiCad stayed busy; zones were not refilled")
                return False
            sleep(poll_s)

    # Once KiCad has the refill, ask at least once whether it is done.
    while True:
        sleep(poll_s)
        try:
            kicad.ping()
            return True
        except (ApiError, KiCadConnectionError) as e:
            if not _busy(e):
                raise
        if clock() >= deadline:
            logger.warning(
                f"KiCad did not answer within {timeout_s:.0f} s of the zone refill"
            )
            return False
