"""Tests for cortical-surface geometry utilities."""

import numpy as np
import pytest


pytestmark = pytest.mark.unit


def _mesh_slice_segments(*args, **kwargs):
    """Import lazily so missing implementation produces test failures."""
    from utils.surface_geometry import mesh_slice_segments

    return mesh_slice_segments(*args, **kwargs)


def _canonical_segment(segment):
    """Return segment endpoints in deterministic lexicographic order."""
    points = [tuple(point) for point in np.asarray(segment)]
    return np.asarray(sorted(points), dtype=float)


class TestMeshSliceSegments:
    """Test exact triangle-plane intersection segments."""

    def test_triangle_crossing_slice_returns_segment(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, -1.0],
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
            ]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)

        segments = _mesh_slice_segments(
            vertices,
            faces,
            axis=2,
            slice_position=0.0,
        )

        assert segments.shape == (1, 2, 3)

        np.testing.assert_allclose(
            _canonical_segment(segments[0]),
            _canonical_segment(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        )

    def test_triangle_on_one_side_returns_no_segment(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 1.0],
            ]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)

        segments = _mesh_slice_segments(
            vertices,
            faces,
            axis=2,
            slice_position=0.0,
        )

        assert segments.shape == (0, 2, 3)

    def test_vertex_on_slice_with_opposite_vertices_returns_segment(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, -1.0],
                [0.0, 2.0, 1.0],
            ]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)

        segments = _mesh_slice_segments(
            vertices,
            faces,
            axis=2,
            slice_position=0.0,
        )

        assert segments.shape == (1, 2, 3)

        np.testing.assert_allclose(
            _canonical_segment(segments[0]),
            _canonical_segment(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
        )

    def test_edge_on_slice_returns_edge_segment(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 1.0],
            ]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)

        segments = _mesh_slice_segments(
            vertices,
            faces,
            axis=2,
            slice_position=0.0,
        )

        assert segments.shape == (1, 2, 3)

        np.testing.assert_allclose(
            _canonical_segment(segments[0]),
            _canonical_segment(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                ]
            ),
        )

    def test_single_vertex_touch_returns_no_segment(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)

        segments = _mesh_slice_segments(
            vertices,
            faces,
            axis=2,
            slice_position=0.0,
        )

        assert segments.shape == (0, 2, 3)

    def test_coplanar_triangle_is_ignored(self):
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)

        segments = _mesh_slice_segments(
            vertices,
            faces,
            axis=2,
            slice_position=0.0,
        )

        assert segments.shape == (0, 2, 3)

    def test_intersection_supports_other_axes(self):
        vertices = np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 0.0, 2.0],
            ]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)

        segments = _mesh_slice_segments(
            vertices,
            faces,
            axis=0,
            slice_position=0.0,
        )

        assert segments.shape == (1, 2, 3)

        np.testing.assert_allclose(
            _canonical_segment(segments[0]),
            _canonical_segment(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )
