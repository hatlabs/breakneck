from collections import defaultdict

import kipy
import kipy.board_types as kbt
import kipy.common_types
import kipy.geometry
import kipy.util.units
import numpy as np
import shapely.geometry as sg

from breakneck.base import Coords2D

OpenShape = kbt.BoardSegment | kbt.BoardArc
ClosedShape = kbt.BoardRectangle | kbt.BoardCircle | kbt.BoardPolygon

OpenShapes = list[OpenShape]
ClosedShapes = list[ClosedShape]


def vector2_as_coords(point: kipy.geometry.Vector2) -> Coords2D:
    """
    Convert a point to coordinates in millimeters with y-up orientation.
    """
    # Shapely uses y-up coordinates
    return Coords2D(point.x, -point.y)


def coords_as_vector2(coords: Coords2D) -> kipy.geometry.Vector2:
    """
    Convert coordinates in millimeters with y-up orientation to a point.
    """
    return kipy.geometry.Vector2.from_xy(coords.x, -coords.y)


def track_as_linestring(track: kbt.Track | kbt.ArcTrack) -> sg.LineString:
    if isinstance(track, kbt.Track):
        return sg.LineString(
            [vector2_as_coords(track.start).as_tuple(), vector2_as_coords(track.end)]
        )
    else:
        return arc_as_linestring(track)


def _polyline_as_coords(polyline: kipy.geometry.PolyLine) -> list[Coords2D]:
    return [vector2_as_coords(node.point) for node in polyline.nodes]


def _polyline_as_tuples(polyline: kipy.geometry.PolyLine) -> list[tuple[int, int]]:
    return [coords.as_tuple() for coords in _polyline_as_coords(polyline)]


def polygon_as_shapely_polygon(polygon: kipy.geometry.PolygonWithHoles) -> sg.Polygon:
    coords = _polyline_as_tuples(polygon.outline)
    holes = [_polyline_as_tuples(hole) for hole in polygon.holes]
    return sg.Polygon(coords, holes)


def board_polygon_as_polygons(
    polygon: kbt.BoardPolygon,
) -> list[sg.Polygon]:
    polygons: list[sg.Polygon] = []
    for poly in polygon.polygons:
        polygons.append(polygon_as_shapely_polygon(poly))
    return polygons


def board_segment_as_linestring(
    segment: kbt.BoardSegment,
) -> sg.LineString:
    return sg.LineString(
        [vector2_as_coords(segment.start), vector2_as_coords(segment.end)]
    )


def board_segments_as_linestrings(
    boardsegments: list[kbt.BoardSegment],
) -> list[sg.LineString]:
    return [board_segment_as_linestring(bs) for bs in boardsegments]


def board_rectangle_as_polygon(
    rectangle: kbt.BoardRectangle,
) -> sg.Polygon:
    # Get corner coordinates
    top_left = rectangle.top_left
    bottom_right = rectangle.bottom_right
    coords = [
        (top_left.x, top_left.y),
        (bottom_right.x, top_left.y),
        (bottom_right.x, bottom_right.y),
        (top_left.x, bottom_right.y),
    ]
    return sg.Polygon(coords)


def arc_as_linestring(
    arc: kipy.common_types.Arc | kbt.ArcTrack,
) -> sg.LineString:
    # Get the start and end points of the arc
    start = vector2_as_coords(arc.start)
    mid = vector2_as_coords(arc.mid)

    # Calculate the center of the arc
    arc_center = arc.center()

    if arc_center is None:
        # Return an empty line string for degenerate arcs
        return sg.LineString([])

    center = vector2_as_coords(arc_center)

    # Calculate the radius of the arc
    radius = int(arc.radius())

    # Determine if the arc is CW or CCW
    cross_product = (start.x - center.x) * (mid.y - center.y) - (start.y - center.y) * (
        mid.x - center.x
    )

    # Calculate the start and end angles of the arc
    start_angle = arc.start_angle()
    end_angle = arc.end_angle()
    assert start_angle is not None
    assert end_angle is not None

    # kipy coordinates are y-down but we use y-up. Adjust angles accordingly
    start_angle = -start_angle
    end_angle = -end_angle

    if cross_product > 0:  # Counterclockwise
        if end_angle < start_angle:
            end_angle += 2 * np.pi
    else:  # Clockwise
        if end_angle > start_angle:
            end_angle -= 2 * np.pi

    # We want points at every degree

    # Calculate the angle of the arc
    angle = abs(end_angle - start_angle)

    num_points = int(np.degrees(angle)) + 1

    angles = np.linspace(start_angle, end_angle, num_points)

    arc_points = [
        Coords2D(int(center.x + radius * np.cos(a)), int(center.y + radius * np.sin(a)))
        for a in angles
    ]

    return sg.LineString(arc_points)


def board_circle_as_polygon(
    circle: kbt.BoardCircle, num_points: int = 360
) -> sg.Polygon:
    # Get the center of the circle
    center = vector2_as_coords(circle.center)

    # Get the radius of the circle
    radius = int(circle.radius())

    # We want points at every degree
    angles = np.linspace(0, 2 * np.pi, num_points)

    # Calculate the points of the circle
    circle_points = [
        Coords2D(int(center.x + radius * np.cos(a)), int(center.y + radius * np.sin(a)))
        for a in angles
    ]

    # Create the circle
    return sg.Polygon(circle_points)


def polyline_as_shapely(polyline: kipy.geometry.PolyLine) -> sg.LineString:
    return sg.LineString(_polyline_as_tuples(polyline))


def closed_shapes_as_polygon(shapes: ClosedShapes) -> list[sg.Polygon]:
    geometries = []
    for shape in shapes:
        if isinstance(shape, kbt.BoardRectangle):
            geometries.append(board_rectangle_as_polygon(shape))
        elif isinstance(shape, kbt.BoardCircle):
            geometries.append(board_circle_as_polygon(shape))
        elif isinstance(shape, kbt.BoardPolygon):
            geometries.extend(board_polygon_as_polygons(shape))
    return geometries


def open_shapes_as_polygon(shapes: OpenShapes, tol_nm: int) -> list[sg.Polygon]:
    """
    Convert a list of open shapes to a list of shapely polygons
    """
    chains = _chain_shapes(shapes, tol_nm)
    polygons = []
    for chain in chains:
        coords = []
        for shape in chain:
            if isinstance(shape, kbt.BoardSegment):
                coords.append(vector2_as_coords(shape.start).as_tuple())
            elif isinstance(shape, kbt.BoardArc):
                coords.append(vector2_as_coords(shape.start).as_tuple())
                coords.extend(arc_as_linestring(shape).coords[1:-1])
            else:
                raise ValueError(f"Shape {type(shape)} not supported")
        coords.append(coords[0])
        polygons.append(sg.Polygon(coords))
    return polygons


def _get_endpoints(shape: kbt.BoardShape, tol_nm: int) -> tuple[Coords2D, Coords2D]:
    """Return the start and end Coords2D of a BoardSegment."""

    if isinstance(shape, kbt.BoardSegment) or isinstance(shape, kbt.BoardArc):
        start = Coords2D(shape.start.x, shape.start.y).as_tol(tol_nm)
        end = Coords2D(shape.end.x, shape.end.y).as_tol(tol_nm)
    else:
        raise ValueError(f"Shape {type(shape)} not supported")

    return start, end


def build_shape_adjacency_graph(
    shapes: OpenShapes, tol_nm: int
) -> tuple[
    dict[Coords2D, list[Coords2D]],
    dict[Coords2D, OpenShapes],
]:
    """Build an adjacency graph for non-closed KiPy board shapes.

    The adjacency dict maps Coords2D to a list of connected Coords2D.
    The shape_map dict maps Coords2D to a list of connected BoardShapes.
    """
    adjacency: dict[Coords2D, list[Coords2D]] = defaultdict(list)
    shape_map: dict[Coords2D, list[kbt.BoardSegment | kbt.BoardArc]] = defaultdict(list)

    for shape in shapes:
        assert isinstance(shape, kbt.BoardSegment) or isinstance(shape, kbt.BoardArc)
        start, end = _get_endpoints(shape, tol_nm)

        adjacency[start].append(end)
        adjacency[end].append(start)
        shape_map[start].append(shape)
        shape_map[end].append(shape)

    return adjacency, shape_map


def _reverse_segment(segment: kbt.BoardSegment) -> None:
    segment.start, segment.end = segment.end, segment.start


def _reverse_arc(arc: kbt.BoardArc) -> None:
    arc.start, arc.end = arc.end, arc.start
    arc.start_angle, arc.end_angle = arc.end_angle, arc.start_angle


def _reverse_shape(shape: kbt.BoardShape) -> None:
    if isinstance(shape, kbt.BoardSegment):
        _reverse_segment(shape)
    elif isinstance(shape, kbt.BoardArc):
        _reverse_arc(shape)
    else:
        raise ValueError(f"Shape {type(shape)} not supported")


def _extract_chain(
    shape: OpenShape,
    shapes: OpenShapes,
    visited_shapes: set[OpenShape],
    tol_nm: int,
) -> OpenShapes:
    """Extract a chain of connected shapes from a starting point."""
    ordered_shapes = []

    visited_shapes.add(shape)
    chain_start, end = _get_endpoints(shape, tol_nm)
    ordered_shapes.append(shape)

    while True:
        # Find a connected shape: start or end point should equal to
        # the end point of the last shape

        found = False

        for next_shape in shapes:
            if next_shape in visited_shapes:
                continue

            next_start, next_end = _get_endpoints(next_shape, tol_nm)

            if end == next_start:
                ordered_shapes.append(next_shape)
                visited_shapes.add(next_shape)
                end = next_end
                found = True
                break
            elif end == next_end:
                _reverse_shape(next_shape)
                ordered_shapes.append(next_shape)
                visited_shapes.add(next_shape)
                end = next_start
                found = True
                break

        if not found:
            break

    return ordered_shapes


def _chain_shapes(shapes: OpenShapes, tol_nm: int) -> list[OpenShapes]:
    """Convert a list of unordered BoardShapes to chains of shapes."""

    # Courtyards are so simple that a graph is not needed

    # Start traversing the shapes
    ordered_chains: list[OpenShapes] = []
    visited_shapes = set()

    for shape in shapes:
        if shape in visited_shapes:
            continue
        chain = _extract_chain(shape, shapes, visited_shapes, tol_nm)
        if chain:
            ordered_chains.append(chain)

    return ordered_chains


def _chain_as_polygon(chain: OpenShapes) -> sg.Polygon:
    coords = []
    for shape in chain:
        if isinstance(shape, kbt.BoardSegment):
            coords.append(vector2_as_coords(shape.start).as_tuple())
        elif isinstance(shape, kbt.BoardArc):
            coords.append(vector2_as_coords(shape.start).as_tuple())
            coords.extend(arc_as_linestring(shape).coords[1:-1])
        else:
            raise ValueError(f"Shape {type(shape)} not supported")
    coords.append(coords[0])
    return sg.Polygon(coords)


def as_polygons(shapes: list[kbt.BoardShape], tol_nm: int) -> list[sg.Polygon]:
    """Convert a list of BoardShapes to a list of shapely Polygons."""
    closed_shapes: ClosedShapes = []
    open_shapes: OpenShapes = []

    for shape in shapes:
        if isinstance(shape, kbt.BoardRectangle) or isinstance(shape, kbt.BoardCircle):
            closed_shapes.append(shape)
        elif isinstance(shape, kbt.BoardPolygon):
            closed_shapes.append(shape)
        elif isinstance(shape, kbt.BoardSegment) or isinstance(shape, kbt.BoardArc):
            open_shapes.append(shape)
        else:
            raise ValueError(f"Shape {type(shape)} not supported")

    polygons = closed_shapes_as_polygon(closed_shapes)

    chains = _chain_shapes(open_shapes, tol_nm)

    for chain in chains:
        polygons.append(_chain_as_polygon(chain))

    return polygons
