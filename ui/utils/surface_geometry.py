"""Geometry utilities for cortical-surface visual QC."""

import numpy as np


def mesh_slice_segments(
    vertices,
    faces,
    axis,
    slice_position,
    *,
    atol=1e-8,
):
    """Intersect a triangular mesh with an axis-aligned slice plane.

    Parameters
    ----------
    vertices : array-like, shape (N, 3)
        Mesh vertices, typically expressed in MRI voxel coordinates.
    faces : array-like, shape (M, 3)
        Triangle vertex indices.
    axis : int
        Slice-normal axis: 0, 1, or 2.
    slice_position : float
        Position of the slice plane along ``axis``.
    atol : float, optional
        Absolute tolerance used to determine whether a vertex lies exactly
        on the slice plane.

    Returns
    -------
    numpy.ndarray, shape (K, 2, 3)
        Exact line segments produced by triangle-plane intersections.

    Notes
    -----
    Point-only contacts are ignored because they do not form contour
    segments. Fully coplanar triangles are also ignored because their
    intersection with the plane is an area rather than a unique contour.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")

    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    if faces.size == 0:
        return np.empty((0, 2, 3), dtype=float)

    if not np.issubdtype(faces.dtype, np.integer):
        if not np.all(np.equal(faces, np.floor(faces))):
            raise ValueError("faces must contain integer vertex indices")
        faces = faces.astype(np.int64)
    else:
        faces = faces.astype(np.int64, copy=False)

    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("faces contain out-of-range vertex indices")

    triangles = vertices[faces]
    distances = triangles[:, :, axis] - float(slice_position)

    on_plane = np.abs(distances) <= atol
    positive = distances > atol
    negative = distances < -atol

    n_on = on_plane.sum(axis=1)
    has_positive = positive.any(axis=1)
    has_negative = negative.any(axis=1)

    segments = []

    # ------------------------------------------------------------------
    # Case 1: exactly two vertices lie on the plane.
    #
    # Their shared edge is the triangle-plane intersection.
    # Fully coplanar triangles (three vertices on-plane) are intentionally
    # excluded.
    # ------------------------------------------------------------------
    edge_on_mask = n_on == 2

    if np.any(edge_on_mask):
        edge_triangles = triangles[edge_on_mask]
        edge_on_vertices = on_plane[edge_on_mask]

        edge_points = edge_triangles[edge_on_vertices].reshape(
            -1,
            2,
            3,
        )
        segments.append(edge_points)

    # ------------------------------------------------------------------
    # Case 2: exactly one vertex lies on the plane and the other two are
    # on opposite sides.
    #
    # The segment runs from the on-plane vertex to the intersection of
    # the edge joining the other two vertices.
    #
    # If both remaining vertices are on the same side, the triangle only
    # touches the plane at one point, so it is ignored.
    # ------------------------------------------------------------------
    one_on_cross_mask = (n_on == 1) & has_positive & has_negative

    if np.any(one_on_cross_mask):
        tri = triangles[one_on_cross_mask]
        dist = distances[one_on_cross_mask]
        on = on_plane[one_on_cross_mask]

        n_triangles = tri.shape[0]
        rows = np.arange(n_triangles)

        on_index = np.argmax(on, axis=1)
        on_point = tri[rows, on_index]

        # False values sort before True, so the first two indices are the
        # two vertices not lying on the slice plane.
        off_indices = np.argsort(on, axis=1)[:, :2]

        first_index = off_indices[:, 0]
        second_index = off_indices[:, 1]

        first_point = tri[rows, first_index]
        second_point = tri[rows, second_index]

        first_distance = dist[rows, first_index]
        second_distance = dist[rows, second_index]

        fraction = first_distance / (first_distance - second_distance)

        crossing_point = first_point + fraction[:, None] * (second_point - first_point)

        segments.append(
            np.stack(
                [on_point, crossing_point],
                axis=1,
            )
        )

    # ------------------------------------------------------------------
    # Case 3: no vertices lie exactly on the plane, but the triangle has
    # vertices on both sides.
    #
    # Exactly two of its three edges cross the plane.
    # ------------------------------------------------------------------
    regular_cross_mask = (n_on == 0) & has_positive & has_negative

    if np.any(regular_cross_mask):
        tri = triangles[regular_cross_mask]
        dist = distances[regular_cross_mask]

        edge_pairs = (
            (0, 1),
            (1, 2),
            (2, 0),
        )

        crossing_points = []
        crossing_masks = []

        for first_index, second_index in edge_pairs:
            first_distance = dist[:, first_index]
            second_distance = dist[:, second_index]

            crosses = ((first_distance > 0) & (second_distance < 0)) | ((first_distance < 0) & (second_distance > 0))

            first_point = tri[:, first_index]
            second_point = tri[:, second_index]

            denominator = first_distance - second_distance

            fraction = np.zeros_like(first_distance)
            fraction[crosses] = first_distance[crosses] / denominator[crosses]

            point = first_point + fraction[:, None] * (second_point - first_point)

            crossing_points.append(point)
            crossing_masks.append(crosses)

        crossing_points = np.stack(
            crossing_points,
            axis=1,
        )
        crossing_masks = np.stack(
            crossing_masks,
            axis=1,
        )

        # Every non-degenerate triangle in this case has exactly two
        # crossing edges. Boolean indexing preserves row-major order, so
        # the points can be reshaped directly into one segment per face.
        regular_segments = crossing_points[crossing_masks].reshape(-1, 2, 3)

        segments.append(regular_segments)

    if not segments:
        return np.empty((0, 2, 3), dtype=float)

    return np.concatenate(segments, axis=0)
