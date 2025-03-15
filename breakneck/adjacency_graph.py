from collections import defaultdict

import kipy.board_types as kbt

from breakneck.conversions import Coords2D, OpenShapes, get_endpoints


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
        start, end = get_endpoints(shape, tol_nm)

        adjacency[start].append(end)
        adjacency[end].append(start)
        shape_map[start].append(shape)
        shape_map[end].append(shape)

    return adjacency, shape_map
