import kipy
import kipy.board
import kipy.board_types
import shapely.geometry as sg
from kipy.board import BoardLayer as BL
from kipy.proto.board import board_types_pb2

import breakneck.conversions
import breakneck.footprint


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
) -> tuple[list[sg.Polygon], list[sg.Polygon]]:
    """Get the front and back courtyard polygons of a footprint."""

    f_crtyd_shapes = get_courtyard_shapes(footprint, BL.BL_F_CrtYd)
    f_polys = breakneck.conversions.as_polygons(f_crtyd_shapes, 1000)
    b_crtyd_shapes = get_courtyard_shapes(footprint, BL.BL_B_CrtYd)
    b_polys = breakneck.conversions.as_polygons(b_crtyd_shapes, 1000)

    return f_polys, b_polys


def get_all_courtyards(
    board: kipy.board.Board,
) -> list[breakneck.footprint.FootprintCrossings,]:
    """Get the front and back courtyard polygons of all footprints on a board."""

    fpcs: list[breakneck.footprint.FootprintCrossings] = []

    for fp in board.get_footprints():
        f_polys, b_polys = get_courtyard_polygons(fp)
        fpcs.append(
            breakneck.footprint.FootprintCrossings(
                ref=fp.reference_field.text.value,
                footprint=fp,
                front_courtyards=f_polys,
                back_courtyards=b_polys,
            )
        )

    return fpcs
