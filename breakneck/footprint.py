from collections.abc import Sequence
from enum import Enum

import kipy.board
import kipy.board_types
import shapely
import shapely.geometry as sg
from kipy.board import BoardLayer as BL
from kipy.proto.board import board_types_pb2
from loguru import logger

import breakneck.conversions
import breakneck.track
from breakneck.conversions import Coords2D


def get_courtyard_shapes(
    footprint: kipy.board_types.FootprintInstance,
    layer: board_types_pb2.BoardLayer.ValueType,
) -> list[kipy.board_types.BoardShape]:
    """
    Get the front or back courtyard shapes of a footprint
    """
    return [sh for sh in footprint.definition.shapes if sh.layer == layer]


def get_courtyard_polygons(
    footprint: kipy.board_types.FootprintInstance,
) -> tuple[sg.Polygon, sg.Polygon]:
    """Get the front and back courtyard polygons of a footprint."""

    f_crtyd_shapes = get_courtyard_shapes(footprint, BL.BL_F_CrtYd)
    f_poly = shapely.union_all(breakneck.conversions.as_polygons(f_crtyd_shapes, 1000))
    if f_poly.is_empty:
        f_poly = sg.Polygon()
    assert isinstance(f_poly, sg.Polygon)
    b_crtyd_shapes = get_courtyard_shapes(footprint, BL.BL_B_CrtYd)
    b_poly = shapely.union_all(breakneck.conversions.as_polygons(b_crtyd_shapes, 1000))
    if b_poly.is_empty:
        b_poly = sg.Polygon()
    assert isinstance(b_poly, sg.Polygon)

    return f_poly, b_poly


class Sides(str, Enum):
    front = "front"
    back = "back"
    both = "both"


class BNFootprint:
    def __init__(self, ref: str, footprint: kipy.board_types.FootprintInstance):
        self.ref = ref
        self.footprint = footprint
        self.front_courtyard, self.back_courtyard = get_courtyard_polygons(footprint)
        self.is_tht = False
        pads = self.footprint.definition.pads
        non_gnd_pads = [p for p in pads if p.net.name != "GND"]
        if any(
            [p for p in non_gnd_pads if p.pad_type == kipy.board_types.PadType.PT_PTH]
        ):
            self.is_tht = True
        self.nets = list(set([p.net.name for p in pads]))
        self._buffered_courtyards: dict[tuple[int, Sides], sg.Polygon] = {}

    def buffer_courtyard(
        self,
        buffer_width_nm: int,
        side: Sides,
    ) -> sg.Polygon:
        if (buffer_width_nm, side) in self._buffered_courtyards:
            return self._buffered_courtyards[(buffer_width_nm, side)]

        if side == Sides.front:
            courtyard = self.front_courtyard
        elif side == Sides.back:
            courtyard = self.back_courtyard
        elif side == Sides.both:
            courtyard = shapely.union_all([self.front_courtyard, self.back_courtyard])
            assert isinstance(courtyard, sg.Polygon)

        buffered_courtyard = courtyard.buffer(
            buffer_width_nm, cap_style="round", join_style="round", quad_segs=90
        )

        self._buffered_courtyards[(buffer_width_nm, side)] = buffered_courtyard

        return buffered_courtyard

    def _is_likely_origin(
        self,
        track: kipy.board_types.Track | kipy.board_types.ArcTrack,
        linestring: sg.LineString,
    ) -> bool:
        """
        Check if a track is likely to originate from the footprint.

        A track is likely to originate from the footprint if it crosses the respective couryard and shares a net with a pad.
        """
        if track.net.name not in self.nets:
            return False

        if self.is_tht:
            courtyard = shapely.union_all([self.front_courtyard, self.back_courtyard])
            assert isinstance(courtyard, sg.Polygon)
        elif track.layer == BL.BL_F_Cu:
            courtyard = self.front_courtyard
        elif track.layer == BL.BL_B_Cu:
            courtyard = self.back_courtyard
        else:
            return False

        if courtyard.intersects(linestring):
            return True

        return False

    def _sort_cut_points(
        self, cut_points: list[sg.Point], linestring: sg.LineString
    ) -> list[sg.Point]:
        """
        Sort cut points along the linestring.
        """
        return sorted(
            cut_points,
            key=lambda p: linestring.line_locate_point(p),
        )

    def break_tracks(
        self,
        tracks: Sequence[kipy.board_types.Track | kipy.board_types.ArcTrack],
        bounding_box_width: int,
        neck_length: int,
        padding: int = 10000,
    ) -> tuple[
        list[kipy.board_types.Track | kipy.board_types.ArcTrack],
        list[kipy.board_types.Track | kipy.board_types.ArcTrack],
    ]:
        """
        Break tracks crossing the courtyards of this footprint.
        """

        items_to_remove = []
        items_to_create = []

        # Get the bounding box hits for tracks

        front_bounding_box = self.front_courtyard.envelope.buffer(bounding_box_width)
        back_bounding_box = self.back_courtyard.envelope.buffer(bounding_box_width)

        front_track_tree = breakneck.track.TrackTree(
            [t for t in tracks if t.layer == BL.BL_F_Cu]
        )
        back_track_tree = breakneck.track.TrackTree(
            [t for t in tracks if t.layer == BL.BL_B_Cu]
        )
        front_hits = front_track_tree.intersects(front_bounding_box)
        back_hits = back_track_tree.intersects(back_bounding_box)
        if self.is_tht:
            back_hits += back_track_tree.intersects(front_bounding_box)
            front_hits += front_track_tree.intersects(back_bounding_box)

        hits = front_hits + back_hits

        logger.debug(f"Found {len(hits)} tracks crossing the courtyards.")

        # If there are no hits, return early

        if not hits:
            return items_to_remove, items_to_create

        bb_tracks, linestrings = zip(*hits)
        for track, linestring in zip(bb_tracks, linestrings):
            assert isinstance(linestring, sg.LineString)
            assert isinstance(track, kipy.board_types.Track) or isinstance(
                track, kipy.board_types.ArcTrack
            )
            if self.is_tht:
                side = Sides.both
            elif track.layer == BL.BL_F_Cu:
                side = Sides.front
            elif track.layer == BL.BL_B_Cu:
                side = Sides.back
            else:
                raise ValueError(f"Unexpected track layer: {track.layer}")

            if self._is_likely_origin(track, linestring):
                # Find possible cut points at the neck length
                neck_buffer = self.buffer_courtyard(
                    neck_length + track.width // 2 + padding, side
                )
                cut_points = neck_buffer.boundary.intersection(linestring)
                logger.debug(f"Cut points: {cut_points}")
            else:
                # Find possible cut points at the courtyard boundary
                cut_points = self.buffer_courtyard(
                    track.width // 2 + padding, side
                ).boundary.intersection(linestring)

            # Order the found points along the track
            track_points = []  # crossing points
            if not cut_points.is_empty:
                if isinstance(cut_points, sg.Point):
                    points = [Coords2D(int(cut_points.x), int(cut_points.y))]
                elif isinstance(cut_points, sg.MultiPoint):
                    points = [Coords2D(int(p.x), int(p.y)) for p in cut_points.geoms]
                else:
                    raise ValueError(f"Unexpected geometry type: {type(cut_points)}")
                track_points.extend(points)
            if not track_points:
                continue

            logger.debug(f"Track points: {track_points}")

            new_tracks = breakneck.track.break_track(track, track_points)
            items_to_remove.append(track)
            items_to_create.extend(new_tracks)

        return items_to_remove, items_to_create


def get_bn_footprints(
    footprints: Sequence[kipy.board_types.FootprintInstance],
) -> list[BNFootprint,]:
    """Get BNFootprint objects of all footprints on a board."""

    fpcs: list[BNFootprint] = []

    for fp in footprints:
        fpcs.append(
            BNFootprint(
                ref=fp.reference_field.text.value,
                footprint=fp,
            )
        )

    return fpcs
