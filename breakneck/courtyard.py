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
