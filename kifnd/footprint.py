from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import kipy.board
import kipy.board_types
import shapely
import shapely.geometry as sg
from kipy.board import BoardLayer as BL

import kifnd.conversions
import kifnd.courtyard
import kifnd.tracks
from kifnd.base import Coords2D


@dataclass
class FootprintCrossings:
    ref: str
    footprint: kipy.board_types.FootprintInstance
    front_courtyards: list[sg.Polygon] = field(default_factory=list)
    back_courtyards: list[sg.Polygon] = field(default_factory=list)

    @classmethod
    def _buffer_layer_courtyards(
        cls,
        widths_nm: Iterable[int],
        padding_nm: int,
        courtyards: list[sg.Polygon],
    ) -> dict[int, sg.Polygon]:
        buffered_courtyards = {}
        for width in widths_nm:
            buffered_courtyards[width] = []
            distance_nm = width // 2 + padding_nm
            poly = shapely.union_all(courtyards)
            buffered_courtyards[width] = poly.buffer(
                distance_nm, cap_style="round", join_style="round", quad_segs=90
            )

        return buffered_courtyards

    def find_crossings(
        self,
        tracks: Iterable[kipy.board_types.Track | kipy.board_types.ArcTrack],
        linestrings: Iterable[sg.LineString],
        front: bool,
        width_tol_nm: int = 10000,
    ) -> dict[kipy.board_types.Track, list[Coords2D]]:
        crossings = defaultdict(list)

        if front:
            courtyards = self.front_courtyards
        else:
            courtyards = self.back_courtyards

        unique_widths = kifnd.tracks.get_unique_track_widths(tracks)

        padding = 10000

        buffered_courtyards = self._buffer_layer_courtyards(
            unique_widths, padding, courtyards
        )

        for track, linestring in zip(tracks, linestrings):
            track_points = []
            rounded_track_width = kifnd.tracks.round_track_width(
                track.width, width_tol_nm
            )
            if rounded_track_width not in buffered_courtyards:
                raise ValueError(
                    f"Track width {track.width} not in buffered courtyards: {buffered_courtyards}"
                )
            buffered = buffered_courtyards[rounded_track_width]
            points = buffered.boundary.intersection(linestring)
            if not points.is_empty:
                if isinstance(points, sg.Point):
                    points = [Coords2D(int(points.x), int(points.y))]
                elif isinstance(points, sg.MultiPoint):
                    points = [Coords2D(int(p.x), int(p.y)) for p in points.geoms]
                else:
                    raise ValueError(f"Unexpected geometry type: {type(points)}")
                track_points.extend(points)
            if not track_points:
                continue
            # Order track_points along track length
            track_points = sorted(
                track_points,
                key=lambda p: linestring.line_locate_point(sg.Point(p)),
            )
            crossings[track] = track_points
        return crossings

    def break_tracks(
        self,
        all_tracks: Sequence[kipy.board_types.Track | kipy.board_types.ArcTrack],
        max_width: int,
    ) -> tuple[
        list[kipy.board_types.Track | kipy.board_types.ArcTrack],
        list[kipy.board_types.Track | kipy.board_types.ArcTrack],
    ]:
        """
        Break tracks crossing the courtyards of this footprint.
        """

        items_to_remove = []
        items_to_create = []

        if self.front_courtyards:
            print("Breaking front tracks")
            front_tracks = [t for t in all_tracks if t.layer == BL.BL_F_Cu]
            front_track_tree = kifnd.tracks.TrackTree(front_tracks)
            print("Finding bounding box hits")
            front_hits = front_track_tree.bounding_box_hit(
                self.front_courtyards, max_width
            )
            if front_hits:
                tracks, linestrings = zip(*front_hits)
                print("Finding crossings")
                front_crossings = self.find_crossings(tracks, linestrings, front=True)
                for track, points in front_crossings.items():
                    new_tracks = kifnd.tracks.break_track(track, points)
                    items_to_remove.append(track)
                    items_to_create.extend(new_tracks)

        if self.back_courtyards:
            print("Breaking back tracks")
            back_tracks = [t for t in all_tracks if t.layer == BL.BL_B_Cu]
            back_track_tree = kifnd.tracks.TrackTree(back_tracks)
            print("Finding bounding box hits")
            back_hits = back_track_tree.bounding_box_hit(
                self.back_courtyards, max_width
            )
            if back_hits:
                tracks, linestrings = zip(*back_hits)
                print("Finding crossings")
                back_crossings = self.find_crossings(tracks, linestrings, front=False)
                for track, points in back_crossings.items():
                    new_tracks = kifnd.tracks.break_track(track, points)
                    items_to_remove.append(track)
                    items_to_create.extend(new_tracks)

        return items_to_remove, items_to_create


def break_tracks(board: kipy.board.Board) -> None:
    """
    Break all tracks crossing the courtyards of all footprints on a board.
    """
    print("Getting all tracks")
    all_tracks = board.get_tracks()
    print("Getting all courtyards")
    all_courtyards = kifnd.courtyard.get_all_courtyards(board)

    all_tracks = {t.id.value: t for t in board.get_tracks()}
    max_width = max(t.width for t in all_tracks.values())
    remove_dict = {}
    create_dict = {}

    for fpc in all_courtyards:
        print(f"Breaking tracks for {fpc.ref}")
        items_to_remove, items_to_create = fpc.break_tracks(
            list(all_tracks.values()), max_width
        )

        # remove items from all_tracks and create_list
        for item in items_to_remove:
            all_tracks.pop(item.id.value)
            create_dict.pop(item.id.value, None)

        # add new items to all_tracks and create_list
        for item in items_to_create:
            all_tracks[item.id.value] = item
            create_dict[item.id.value] = item

        # add items to remove_list
        for item in items_to_remove:
            remove_dict[item.id.value] = item

    commit = board.begin_commit()
    board.remove_items(list(remove_dict.values()))
    board.create_items(list(create_dict.values()))
    board.push_commit(commit)
