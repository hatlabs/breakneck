from collections import defaultdict

import kipy
import kipy.board_types as kbt
import kipy.common_types
import kipy.geometry
import kipy.util.units
import numpy as np
import shapely.geometry as sg

from kifnd.base import Coords2D

OpenShape = kbt.BoardSegment | kbt.BoardArc
ClosedShape = kbt.BoardRectangle | kbt.BoardCircle | kbt.BoardPolygon

OpenShapes = list[OpenShape]
ClosedShapes = list[ClosedShape]


def _point_as_coords(point: kipy.geometry.Vector2) -> Coords2D:
    """
    Convert a point to coordinates in millimeters with y-up orientation.
    """
    # Shapely uses y-up coordinates
    return Coords2D(point.x, -point.y)


def _track_as_shapely_linestring(track: kbt.Track) -> sg.LineString:
    return sg.LineString(
        [_point_as_coords(track.start).as_tuple(), _point_as_coords(track.end)]
    )


def _polyline_as_coords(polyline: kipy.geometry.PolyLine) -> list[Coords2D]:
    return [_point_as_coords(node.point) for node in polyline.nodes]


def _polyline_as_tuples(polyline: kipy.geometry.PolyLine) -> list[tuple[int, int]]:
    return [coords.as_tuple() for coords in _polyline_as_coords(polyline)]


def _polygon_as_shapely(polygon: kipy.geometry.PolygonWithHoles) -> sg.Polygon:
    coords = _polyline_as_tuples(polygon.outline)
    holes = [_polyline_as_tuples(hole) for hole in polygon.holes]
    return sg.Polygon(coords, holes)


def _board_polygon_as_shapely_polygons(
    polygon: kbt.BoardPolygon,
) -> list[sg.Polygon]:
    polygons: list[sg.Polygon] = []
    for poly in polygon.polygons:
        polygons.append(_polygon_as_shapely(poly))
    return polygons


def _board_segment_as_linestring(
    segment: kbt.BoardSegment,
) -> sg.LineString:
    return sg.LineString(
        [_point_as_coords(segment.start), _point_as_coords(segment.end)]
    )


def _board_segments_as_linestrings(
    boardsegments: list[kbt.BoardSegment],
) -> list[sg.LineString]:
    return [_board_segment_as_linestring(bs) for bs in boardsegments]


def _board_rectangle_as_polygon(
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


def _arc_as_shapely_linestring(
    arc: kipy.common_types.Arc | kbt.ArcTrack,
) -> sg.LineString:
    # Get the start and end points of the arc
    start = _point_as_coords(arc.start)
    mid = _point_as_coords(arc.mid)

    # Calculate the center of the arc
    arc_center = arc.center()

    if arc_center is None:
        # Return an empty line string for degenerate arcs
        return sg.LineString([])

    center = _point_as_coords(arc_center)

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


def _board_circle_as_polygon(
    circle: kbt.BoardCircle, num_points: int = 360
) -> sg.Polygon:
    # Get the center of the circle
    center = _point_as_coords(circle.center)

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


def track_as_shapely(track: kbt.Track) -> sg.LineString:
    return _track_as_shapely_linestring(track)


def board_segment_as_shapely(segment: kbt.BoardSegment) -> sg.LineString:
    return _board_segment_as_linestring(segment)


def board_polygon_as_shapely(polygon: kbt.BoardPolygon) -> list[sg.Polygon]:
    return _board_polygon_as_shapely_polygons(polygon)


def board_rectangle_as_shapely(rectangle: kbt.BoardRectangle) -> sg.Polygon:
    return _board_rectangle_as_polygon(rectangle)


def arc_as_shapely(arc: kbt.BoardArc | kbt.ArcTrack) -> sg.LineString:
    return _arc_as_shapely_linestring(arc)


def board_circle_as_shapely(circle: kbt.BoardCircle) -> sg.Polygon:
    return _board_circle_as_polygon(circle)


def polyline_as_shapely(polyline: kipy.geometry.PolyLine) -> sg.LineString:
    return sg.LineString(_polyline_as_tuples(polyline))


def polygon_as_shapely(polygon: kipy.geometry.PolygonWithHoles) -> sg.Polygon:
    return _polygon_as_shapely(polygon)


def closed_shapes_as_shapely(shapes: ClosedShapes) -> list[sg.Polygon]:
    geometries = []
    for shape in shapes:
        if isinstance(shape, kbt.BoardRectangle):
            geometries.append(board_rectangle_as_shapely(shape))
        elif isinstance(shape, kbt.BoardCircle):
            geometries.append(board_circle_as_shapely(shape))
        elif isinstance(shape, kbt.BoardPolygon):
            geometries.extend(board_polygon_as_shapely(shape))
    return geometries


def open_shapes_as_shapely(shapes: OpenShapes, tol_nm: int) -> list[sg.Polygon]:
    """
    Convert a list of open shapes to a list of shapely polygons
    """
    chains = _chain_shapes(shapes, tol_nm)
    polygons = []
    for chain in chains:
        coords = []
        for shape in chain:
            if isinstance(shape, kbt.BoardSegment):
                coords.append(_point_as_coords(shape.start).as_tuple())
            elif isinstance(shape, kbt.BoardArc):
                coords.append(_point_as_coords(shape.start).as_tuple())
                coords.extend(_arc_as_shapely_linestring(shape).coords[1:-1])
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


def _validate_adjacencies(
    adjacency: dict[Coords2D, list[Coords2D]],
) -> bool:
    """Validate that the adjacencies form a closed shape.

    Each adjacency point must have exactly two neighbors.
    """
    for point, neighbors in adjacency.items():
        if len(neighbors) != 2:
            return False

    return True


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
    start: Coords2D,  # Tolerance-adjusted starting point
    shape_map: dict[Coords2D, list[OpenShape]],
    visited_shapes: set[OpenShape],
    tol_nm: int,
) -> OpenShapes:
    """Extract a chain of connected shapes from a starting point."""
    ordered_shapes = []
    current_point = start

    while current_point in shape_map:
        # Get unvisited segments connected to this point
        candidates = [
            seg for seg in shape_map[current_point] if seg not in visited_shapes
        ]
        if not candidates:
            break  # Polygon is complete or no valid path

        next_shape = candidates[0]
        visited_shapes.add(next_shape)

        next_start, next_end = _get_endpoints(next_shape, tol_nm)
        if next_start == current_point:
            current_point = next_end
        else:
            _reverse_shape(next_shape)
            current_point = next_start

        ordered_shapes.append(next_shape)

    return ordered_shapes


def _chain_shapes(shapes: OpenShapes, tol_nm: int) -> list[OpenShapes]:
    """Convert a list of unordered BoardShapes to chains of shapes."""
    # Build an adjacency graph
    adjacency, shape_map = build_shape_adjacency_graph(shapes, tol_nm)

    if not _validate_adjacencies(adjacency):
        raise ValueError("Shapes do not form a closed polygon.")

    # Start traversing the shapes
    ordered_chains: list[OpenShapes] = []
    visited_shapes = set()

    for point in adjacency.keys():
        if point in visited_shapes:
            continue
        chain = _extract_chain(point, shape_map, visited_shapes, tol_nm)
        ordered_chains.append(chain)

    return ordered_chains


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

    polygons = closed_shapes_as_shapely(closed_shapes)

    chains = _chain_shapes(open_shapes, tol_nm)

    for chain in chains:
        coords = []
        for shape in chain:
            if isinstance(shape, kbt.BoardSegment):
                coords.append(_point_as_coords(shape.start).as_tuple())
            elif isinstance(shape, kbt.BoardArc):
                coords.append(_point_as_coords(shape.start).as_tuple())
                coords.extend(_arc_as_shapely_linestring(shape).coords)
            else:
                raise ValueError(f"Shape {type(shape)} not supported")

            # Close the polygon
            coords.append(coords[0])
        polygons.append(sg.Polygon(coords))

    return polygons
