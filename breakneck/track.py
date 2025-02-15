import math
import uuid
from collections.abc import Iterable, Sequence

import kipy.board
import kipy.board_types
import kipy.common_types
import kipy.proto.board.board_types_pb2 as board_types_pb2
import kipy.proto.common.types.base_types_pb2 as base_types_pb2
import shapely
import shapely.geometry as sg

import breakneck.conversions
import breakneck.footprint
from breakneck.base import Coords2D


def round_track_width(width_nm: int, tol_nm=10000) -> int:
    return tol_nm * int(round(width_nm / tol_nm))


def get_unique_track_widths(
    tracks: Iterable[kipy.board_types.Track | kipy.board_types.ArcTrack], tol_nm=10000
) -> list[int]:
    """Get tolerance-rounded unique track widths from a list of tracks."""
    track_widths = set()

    for track in tracks:
        track_widths.add(round_track_width(track.width, tol_nm))

    return sorted(list(track_widths))


def get_max_track_width(
    tracks: Iterable[kipy.board_types.Track | kipy.board_types.ArcTrack],
) -> int:
    return max([track.width for track in tracks])


def break_track_segment(
    track: kipy.board_types.Track, points: list[Coords2D]
) -> list[kipy.board_types.Track]:
    """
    Break a track segment at points
    """

    if not points:
        raise ValueError("No points to break track")

    start = breakneck.conversions.vector2_as_coords(track.start)
    end = breakneck.conversions.vector2_as_coords(track.end)

    current = start

    new_tracks = []

    even = False
    for point in points + [end]:
        new_track_proto = board_types_pb2.Track(
            id=base_types_pb2.KIID(value=str(uuid.uuid4())),
        )
        new_track = kipy.board_types.Track(new_track_proto)

        new_track.start = breakneck.conversions.coords_as_vector2(current)
        new_track.end = breakneck.conversions.coords_as_vector2(point)

        new_track.width = track.width
        new_track.net = track.net
        new_track.layer = track.layer

        # bump even segment widths by 1 nm down
        if even:
            new_track.width -= 1
        even = not even

        if (abs(new_track.start.x - new_track.end.x) > 10) or (
            abs(new_track.start.y - new_track.end.y) > 10
        ):
            new_tracks.append(new_track)

        current = point

    return new_tracks


def _calculate_angle(center: Coords2D, point: Coords2D) -> float:
    return math.atan2(point.y - center.y, point.x - center.x)


def _calculate_midpoint(
    center: Coords2D,
    start_angle: float,
    end_angle: float,
    radius: float,
    clockwise: bool,
) -> Coords2D:
    """
    Calculate the midpoint of an arc
    """
    if clockwise:
        if start_angle < end_angle:
            start_angle += 2 * math.pi
    else:
        if end_angle < start_angle:
            end_angle += 2 * math.pi
    mid_angle = (start_angle + end_angle) / 2
    return Coords2D(
        int(center.x + radius * math.cos(mid_angle)),
        int(center.y + radius * math.sin(mid_angle)),
    )


def _is_arc_clockwise(
    start: Coords2D,
    mid: Coords2D,
    end: Coords2D,
    center: Coords2D,
) -> bool:
    """
    Determine if an arc is clockwise or counterclockwise
    """
    # Calculate vectors from center to start, mid, and end points
    vector_start = (start.x - center.x, start.y - center.y)
    vector_mid = (mid.x - center.x, mid.y - center.y)
    vector_end = (end.x - center.x, end.y - center.y)

    # Calculate cross product of vectors (start to mid) and (mid to end)
    cross_product = (vector_mid[0] - vector_start[0]) * (
        vector_end[1] - vector_mid[1]
    ) - (vector_mid[1] - vector_start[1]) * (vector_end[0] - vector_mid[0])

    # If cross product is negative, the arc is clockwise
    return cross_product < 0


def break_arc_track(
    track: kipy.board_types.ArcTrack, points: list[Coords2D]
) -> list[kipy.board_types.Track]:
    """
    Break an arc track at points
    """

    if not points:
        raise ValueError("No points to break track")

    start = breakneck.conversions.vector2_as_coords(track.start)
    mid = breakneck.conversions.vector2_as_coords(track.mid)
    end = breakneck.conversions.vector2_as_coords(track.end)
    center_vector = track.center()
    assert center_vector is not None
    center = breakneck.conversions.vector2_as_coords(center_vector)
    radius = int(track.radius())
    clockwise = _is_arc_clockwise(start, mid, end, center)

    current = start
    new_tracks = []

    even = False
    for point in points + [end]:
        start_angle = _calculate_angle(center, current)
        end_angle = _calculate_angle(center, point)
        mid_point = _calculate_midpoint(
            center, start_angle, end_angle, radius, clockwise
        )
        new_track_proto = board_types_pb2.Arc(
            id=base_types_pb2.KIID(value=str(uuid.uuid4())),
        )

        new_track = kipy.board_types.ArcTrack(new_track_proto)
        new_track.start = breakneck.conversions.coords_as_vector2(current)
        new_track.mid = breakneck.conversions.coords_as_vector2(mid_point)
        new_track.end = breakneck.conversions.coords_as_vector2(point)
        new_track.width = track.width
        new_track.net = track.net
        new_track.layer = track.layer

        # bump even segment widths down by 1 nm
        if even:
            new_track.width -= 1
        even = not even

        if (abs(new_track.start.x - new_track.end.x) > 10) or (
            abs(new_track.start.y - new_track.end.y) > 10
        ):
            new_tracks.append(new_track)
        current = point

    return new_tracks


def break_track(
    track: kipy.board_types.Track | kipy.board_types.ArcTrack, points: list[Coords2D]
) -> Sequence[kipy.board_types.Track | kipy.board_types.ArcTrack]:
    """
    Break a track at points
    """
    if isinstance(track, kipy.board_types.ArcTrack):
        return break_arc_track(track, points)
    else:
        return break_track_segment(track, points)


class TrackTree:
    def __init__(
        self, tracks: Iterable[kipy.board_types.Track | kipy.board_types.ArcTrack]
    ):
        self._tracks = list(tracks)
        self._track_linestrings = [
            breakneck.conversions.track_as_linestring(track) for track in self._tracks
        ]
        self._track_tree = shapely.STRtree(self._track_linestrings)

    def intersects(
        self, courtyard: shapely.geometry.Polygon
    ) -> list[tuple[kipy.board_types.Track | kipy.board_types.ArcTrack, sg.LineString]]:
        """
        Return tracks that intersect with the given courtyard
        """
        intersect_indices = self._track_tree.query(courtyard, predicate="intersects")
        return [
            (self._tracks[i], self._track_linestrings[i]) for i in intersect_indices
        ]
