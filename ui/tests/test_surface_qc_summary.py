"""Tests for quantitative cortical-surface QC presentation helpers."""

from components.surface_viewer import (
    _collision_pair_rows,
    _surface_metric_rows,
)
from utils.surface_qc_metrics import COLLISION_PAIRS, SURFACE_KEYS


def _geometry_qc_fixture():
    surfaces = {}
    for index, key in enumerate(SURFACE_KEYS, start=1):
        surfaces[key] = {
            "total_faces": 1000,
            "sif": {
                "status": "OK",
                "face_count": index,
                "pct": index / 10,
            },
            "collision": {
                "available": True,
                "exact": True,
                "face_count": index * 2,
                "pct": index / 5,
            },
        }

    pairs = {}
    for index, (name, (surface_a, surface_b)) in enumerate(
        COLLISION_PAIRS.items(),
        start=1,
    ):
        pairs[name] = {
            "surface_A": surface_a,
            "surface_B": surface_b,
            "status": "OK",
            "collision_detected": True,
            "num_contacts": index * 10,
            "pct_faces_A": index / 10,
            "pct_faces_B": index / 20,
            "contact_count_exact": True,
        }

    return {
        "status": "OK",
        "error": "",
        "surfaces": surfaces,
        "pairs": pairs,
        "collision_pct_union_mean4": 0.5,
        "collision_pct_union_max4": 0.8,
        "collision_exact_all_pairs": True,
    }


def test_surface_metric_rows_include_all_four_surfaces():
    rows = _surface_metric_rows(_geometry_qc_fixture())

    assert [row["Surface"] for row in rows] == [
        "Left white",
        "Left pial",
        "Right white",
        "Right pial",
    ]

    assert rows[0]["Self-intersecting faces"] == 1
    assert rows[0]["SIF (%)"] == 0.1
    assert rows[0]["Colliding faces (union)"] == 2
    assert rows[0]["Collision union (%)"] == 0.2
    assert rows[0]["Collision exact"] is True


def test_surface_metric_rows_keep_unavailable_collision_as_none():
    qc = _geometry_qc_fixture()
    qc["surfaces"]["lh_white"]["collision"] = {
        "available": False,
        "exact": False,
        "face_count": 0,
        "pct": None,
    }

    row = _surface_metric_rows(qc)[0]

    assert row["Colliding faces (union)"] is None
    assert row["Collision union (%)"] is None
    assert row["Collision exact"] is None


def test_collision_pair_rows_follow_six_pair_policy():
    rows = _collision_pair_rows(_geometry_qc_fixture())

    assert len(rows) == 6
    assert [row["Surface pair"] for row in rows] == [
        "Left white ↔ left pial",
        "Right white ↔ right pial",
        "Left pial ↔ right pial",
        "Left white ↔ right white",
        "Left white ↔ right pial",
        "Right white ↔ left pial",
    ]

    assert rows[0]["Collision"] == "Yes"
    assert rows[0]["Contacts"] == 10
    assert rows[0]["Faces A (%)"] == 0.1
    assert rows[0]["Faces B (%)"] == 0.05
    assert rows[0]["Exact"] is True
    assert rows[0]["Status"] == "OK"


def test_collision_pair_rows_distinguish_no_collision_from_unavailable():
    qc = _geometry_qc_fixture()

    qc["pairs"]["white_pial_left"].update(
        {
            "collision_detected": False,
            "num_contacts": 0,
            "pct_faces_A": 0.0,
            "pct_faces_B": 0.0,
            "contact_count_exact": True,
        }
    )
    qc["pairs"]["white_pial_right"].update(
        {
            "status": "no_fcl",
            "collision_detected": None,
            "num_contacts": None,
            "pct_faces_A": None,
            "pct_faces_B": None,
            "contact_count_exact": False,
        }
    )

    rows = _collision_pair_rows(qc)

    assert rows[0]["Collision"] == "No"
    assert rows[0]["Contacts"] == 0
    assert rows[0]["Faces A (%)"] == 0.0
    assert rows[0]["Faces B (%)"] == 0.0
    assert rows[0]["Exact"] is True

    assert rows[1]["Collision"] == "Unavailable"
    assert rows[1]["Contacts"] is None
    assert rows[1]["Faces A (%)"] is None
    assert rows[1]["Faces B (%)"] is None
    assert rows[1]["Exact"] is None
    assert rows[1]["Status"] == "no_fcl"
