"""Tests for the cortical-surface QC viewer."""

import numpy as np

from components.surface_viewer import (
    _default_slice_positions,
    _segments_to_xy,
)


def test_default_slice_positions_use_surface_bbox_center():
    surfaces = [
        {
            "vertices_voxel": np.asarray(
                [
                    [10.0, 20.0, 30.0],
                    [30.0, 60.0, 90.0],
                ]
            )
        },
        {
            "vertices_voxel": np.asarray(
                [
                    [20.0, 40.0, 50.0],
                    [40.0, 80.0, 110.0],
                ]
            )
        },
    ]

    positions = _default_slice_positions(
        surfaces,
        volume_shape=(100, 120, 140),
    )

    assert positions == {
        0: 25,
        1: 50,
        2: 70,
    }


def test_default_slice_positions_are_clipped_to_volume():
    surfaces = [
        {
            "vertices_voxel": np.asarray(
                [
                    [-50.0, -20.0, -10.0],
                    [-10.0, -2.0, -1.0],
                ]
            )
        }
    ]

    positions = _default_slice_positions(
        surfaces,
        volume_shape=(100, 100, 100),
    )

    assert positions == {
        0: 0,
        1: 0,
        2: 0,
    }


def test_segments_to_xy_for_axial_plane():
    segments = np.asarray(
        [
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 30.0],
            ],
            [
                [12.0, 22.0, 30.0],
                [13.0, 23.0, 30.0],
            ],
        ]
    )

    x, y = _segments_to_xy(
        segments,
        axis=2,
    )

    assert x == [
        10.0,
        11.0,
        None,
        12.0,
        13.0,
        None,
    ]

    assert y == [
        20.0,
        21.0,
        None,
        22.0,
        23.0,
        None,
    ]


def test_segments_to_xy_for_coronal_plane():
    segments = np.asarray(
        [
            [
                [10.0, 20.0, 30.0],
                [11.0, 20.0, 31.0],
            ]
        ]
    )

    x, y = _segments_to_xy(
        segments,
        axis=1,
    )

    assert x == [
        10.0,
        11.0,
        None,
    ]

    assert y == [
        30.0,
        31.0,
        None,
    ]


def test_segments_to_xy_for_sagittal_plane():
    segments = np.asarray(
        [
            [
                [10.0, 20.0, 30.0],
                [10.0, 21.0, 31.0],
            ]
        ]
    )

    x, y = _segments_to_xy(
        segments,
        axis=0,
    )

    assert x == [
        20.0,
        21.0,
        None,
    ]

    assert y == [
        30.0,
        31.0,
        None,
    ]


def test_plane_names_match_lia_voxel_axes():
    from components.surface_viewer import PLANE_NAMES

    assert PLANE_NAMES == {
        0: "Sagittal",
        1: "Axial",
        2: "Coronal",
    }
