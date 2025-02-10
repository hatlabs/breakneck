import kipy
import kipy.board
import kipy.board_types
import shapely.geometry as sg
from kipy.board import BoardLayer as BL
from kipy.proto.board import board_types_pb2

import kifnd.conversions


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

    f_crtyds = get_courtyard_shapes(footprint, BL.BL_F_CrtYd)
    f_polys = kifnd.conversions.as_polygons(f_crtyds, 1000)
    b_crtyds = get_courtyard_shapes(footprint, BL.BL_B_CrtYd)
    b_polys = kifnd.conversions.as_polygons(b_crtyds, 1000)

    return f_polys, b_polys


def get_all_courtyard_polygons(
    board: kipy.board.Board,
) -> list[
    tuple[kipy.board_types.FootprintInstance, list[sg.Polygon], list[sg.Polygon]]
]:
    """Get the front and back courtyard polygons of all footprints on a board."""

    return [(fp, *get_courtyard_polygons(fp)) for fp in board.get_footprints()]
