from collections import namedtuple
from typing import overload

import kipy
import kipy.board_types as kbt
import kipy.common_types
import kipy.geometry
import kipy.util.units
import numpy as np
import shapely.geometry as sg

OpenShape = kbt.BoardSegment | kbt.BoardArc

OpenShapes = list[OpenShape]

Coords2D = namedtuple("Coords2D", ["x", "y"])


def as_coords2d(vector: kipy.geometry.Vector2) -> Coords2D:
    return Coords2D(vector.x, -vector.y)


def as_vector2(coords: Coords2D) -> kipy.geometry.Vector2:
    return kipy.geometry.Vector2.from_xy(coords[0], -coords[1])


@overload
def get_endpoints(
    shape: kbt.BoardSegment, tol_nm: int
) -> tuple[Coords2D, Coords2D]: ...


@overload
def get_endpoints(shape: kbt.BoardArc, tol_nm: int) -> tuple[Coords2D, Coords2D]: ...


def get_endpoints(shape: kbt.BoardShape, tol_nm: int) -> tuple[Coords2D, Coords2D]:
    """Return the start and end Coords2D of a BoardShape."""

    def as_tol(c: Coords2D, tol_nm: int) -> Coords2D:
        """Convert to a tolerance-scaled Coords2D."""
        return Coords2D(round(c.x / tol_nm), round(c.y / tol_nm))

    match shape:
        case kbt.BoardSegment() | kbt.BoardArc():
            start = as_tol(as_coords2d(shape.start), tol_nm)
            end = as_tol(as_coords2d(shape.end), tol_nm)
        case _:
            raise ValueError(f"Shape {type(shape)} not supported")

    return start, end


@overload
def as_linestring(shape: kbt.Track) -> sg.LineString: ...


@overload
def as_linestring(shape: kbt.ArcTrack) -> sg.LineString: ...


@overload
def as_linestring(shape: kbt.BoardSegment) -> sg.LineString: ...


@overload
def as_linestring(shape: kipy.common_types.Arc) -> sg.LineString: ...


@overload
def as_linestring(shape: kipy.geometry.PolyLine) -> sg.LineString: ...


def as_linestring(
    shape: kbt.Track
    | kbt.ArcTrack
    | kbt.BoardSegment
    | kipy.common_types.Arc
    | kipy.geometry.PolyLine,
    num_circle_points: int = 360,
) -> sg.LineString:
    def _arc_as_linestring(
        arc: kipy.common_types.Arc | kbt.ArcTrack,
    ) -> sg.LineString:
        # Get the start and end points of the arc
        start = as_coords2d(arc.start)
        mid = as_coords2d(arc.mid)

        # Calculate the center of the arc
        arc_center_vector = arc.center()

        if arc_center_vector is None:
            # Return an empty line string for degenerate arcs
            return sg.LineString([])

        center = as_coords2d(arc_center_vector)

        # Calculate the radius of the arc
        radius = int(arc.radius())

        # Determine if the arc is CW or CCW
        cross_product = (start.x - center.x) * (mid.y - center.y) - (
            start.y - center.y
        ) * (mid.x - center.x)

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

        # We want num_circle_points points on a full circle

        # Calculate the angle of the arc
        angle = end_angle - start_angle

        num_points = abs(int(num_circle_points / 360.0 * np.degrees(angle))) + 1

        angles = np.linspace(start_angle, end_angle, num_points)

        arc_points = [
            Coords2D(
                int(center.x + radius * np.cos(a)), int(center.y + radius * np.sin(a))
            )
            for a in angles
        ]

        return sg.LineString(arc_points)

    match shape:
        case kbt.Track():
            return sg.LineString(
                [
                    as_coords2d(shape.start),
                    as_coords2d(shape.end),
                ]
            )
        case kbt.ArcTrack():
            return _arc_as_linestring(shape)
        case kipy.common_types.Arc():
            return _arc_as_linestring(shape)
        case kbt.BoardSegment():
            return sg.LineString(
                [
                    as_coords2d(shape.start),
                    as_coords2d(shape.end),
                ]
            )
        case kipy.geometry.PolyLine():
            return sg.LineString(as_coords(shape))
        case _:
            raise ValueError(f"Shape {type(shape)} not supported")


def as_coords(shape: kipy.geometry.PolyLine) -> list[Coords2D]:
    return [as_coords2d(node.point) for node in shape.nodes]


@overload
def as_polygon(shape: kipy.geometry.PolygonWithHoles) -> sg.Polygon: ...


@overload
def as_polygon(shape: kbt.BoardRectangle) -> sg.Polygon: ...


@overload
def as_polygon(shape: kbt.BoardCircle) -> sg.Polygon: ...


def as_polygon(
    shape: kipy.geometry.PolygonWithHoles | kbt.BoardRectangle | kbt.BoardCircle,
    num_points: int = 360,
) -> sg.Polygon:
    match shape:
        case kipy.geometry.PolygonWithHoles():
            coords = as_coords(shape.outline)
            holes = [as_coords(hole) for hole in shape.holes]
            return sg.Polygon(coords, holes)
        case kbt.BoardRectangle():
            top_left = shape.top_left
            bottom_right = shape.bottom_right
            coords = [
                (top_left.x, -top_left.y),
                (bottom_right.x, -top_left.y),
                (bottom_right.x, -bottom_right.y),
                (top_left.x, -bottom_right.y),
            ]
            return sg.Polygon(coords)
        case kbt.BoardCircle():
            center = as_coords2d(shape.center)
            radius = shape.radius()
            angles = np.linspace(0, 2 * np.pi, num_points)
            circle_points = [
                Coords2D(
                    int(center.x + radius * np.cos(a)),
                    int(center.y + radius * np.sin(a)),
                )
                for a in angles
            ]
            return sg.Polygon(circle_points)


def _as_polygons(
    shapes: kbt.BoardPolygon | list[kbt.BoardShape],
) -> list[sg.Polygon]:
    polygons: list[sg.Polygon] = []
    if not isinstance(shapes, list):
        shapes = [shapes]

    for shape in shapes:
        match shape:
            case kbt.BoardRectangle() | kbt.BoardCircle():
                polygons.append(as_polygon(shape))
            case kbt.BoardPolygon():
                for poly in shape.polygons:
                    polygons.append(as_polygon(poly))
            case _:
                raise ValueError(f"Shape {type(shapes)} not supported")
    return polygons


def _reverse_shape(shape: kbt.BoardShape) -> None:
    match shape:
        case kbt.BoardSegment():
            shape.start, shape.end = shape.end, shape.start
        case kbt.BoardArc():
            shape.start, shape.end = shape.end, shape.start
        case _:
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
    chain_start, end = get_endpoints(shape, tol_nm)
    ordered_shapes.append(shape)

    while True:
        # Find a connected shape: start or end point should equal to
        # the end point of the last shape

        found = False

        for next_shape in shapes:
            if next_shape in visited_shapes:
                continue

            next_start, next_end = get_endpoints(next_shape, tol_nm)

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
        match shape:
            case kbt.BoardSegment():
                coords.append(as_coords2d(shape.start))
            case kbt.BoardArc():
                coords.append(as_coords2d(shape.start))
                coords.extend(as_linestring(shape).coords[1:-1])
            case _:
                raise ValueError(f"Shape {type(shape)} not supported")
    coords.append(coords[0])
    return sg.Polygon(coords)


def as_polygons(shapes: list[kbt.BoardShape], tol_nm: int) -> list[sg.Polygon]:
    """Convert a list of BoardShapes to a list of shapely Polygons."""
    closed_shapes: list[kbt.BoardShape] = []
    open_shapes: OpenShapes = []

    for shape in shapes:
        match shape:
            case kbt.BoardRectangle() | kbt.BoardCircle() | kbt.BoardPolygon():
                closed_shapes.append(shape)
            case kbt.BoardSegment() | kbt.BoardArc():
                open_shapes.append(shape)
            case _:
                raise ValueError(f"Shape {type(shape)} not supported")
    polygons = _as_polygons(closed_shapes)

    chains = _chain_shapes(open_shapes, tol_nm)

    for chain in chains:
        polygons.append(_chain_as_polygon(chain))

    return polygons
