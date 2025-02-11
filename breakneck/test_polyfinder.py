import pytest
import shapely.geometry as sg
from polyfinder import find_polygons


def test_find_polygons():
    segments = [
        # First ring
        sg.LineString([(0, 0), (1, 0)]),
        sg.LineString([(1, 0), (1, 1)]),
        sg.LineString([(1, 1), (0, 1)]),
        sg.LineString([(0, 1), (0, 0)]),
        # Second ring
        sg.LineString([(2, 2), (3, 2)]),
        sg.LineString([(3, 2), (3, 3)]),
        sg.LineString([(3, 3), (2, 3)]),
        sg.LineString([(2, 3), (2, 2)]),
    ]

    linear_rings = find_polygons(segments)
    expected_rings = [
        sg.LinearRing([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]),
        sg.LinearRing([(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]),
    ]

    assert len(linear_rings) == len(expected_rings)
    for ring, expected_ring in zip(linear_rings, expected_rings):
        assert ring.equals(expected_ring)


if __name__ == "__main__":
    pytest.main()
