import numpy as np
import pytest

from utils import surface_qc_metrics as sqm


def _surface(name: str, n_faces: int = 4):
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    base_faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=int,
    )
    return {
        "name": name,
        "vertices_world": vertices,
        "faces": base_faces[:n_faces],
    }


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("lh.white", "lh_white"),
        ("lh.pial", "lh_pial"),
        ("lh.pial.T1", "lh_pial"),
        ("rh.white", "rh_white"),
        ("rh.pial", "rh_pial"),
        ("rh.pial.T1", "rh_pial"),
    ],
)
def test_canonical_surface_key_free_surfer_names(name, expected):
    assert sqm.canonical_surface_key(name) == expected


def test_index_cortical_surfaces_requires_unique_keys():
    surfaces = [_surface("lh.white"), _surface("lh.white")]
    with pytest.raises(ValueError, match="Duplicate cortical surface"):
        sqm.index_cortical_surfaces(surfaces)


class _FakePyMesh:
    def __init__(self, vertex_matrix, face_matrix):
        self.vertices = np.asarray(vertex_matrix)
        self.faces = np.asarray(face_matrix)
        self.selection = np.zeros(len(self.faces), dtype=bool)

    def face_number(self):
        return len(self.faces)

    def face_selection_array(self):
        return self.selection


class _FakeMeshSet:
    def __init__(self):
        self.mesh = None

    def add_mesh(self, mesh, _name):
        self.mesh = mesh

    def current_mesh(self):
        return self.mesh

    def apply_filter(self, name):
        assert name == "compute_selection_by_self_intersections_per_face"
        self.mesh.selection[:] = np.array([True, False, True, False], dtype=bool)


class _FakePyMeshLab:
    Mesh = _FakePyMesh
    MeshSet = _FakeMeshSet

    @staticmethod
    def filter_list():
        return ["compute_selection_by_self_intersections_per_face"]


def test_compute_self_intersections_returns_exact_face_ids_and_pct():
    surface = _surface("lh.white")
    out = sqm.compute_self_intersections(
        surface["vertices_world"],
        surface["faces"],
        pyml=_FakePyMeshLab,
    )

    assert out["status"] == "OK"
    assert out["filter"] == "compute_selection_by_self_intersections_per_face"
    assert out["total_faces"] == 4
    assert out["face_count"] == 2
    assert out["face_ids"] == [0, 2]
    assert out["pct"] == pytest.approx(50.0)


def test_summarize_collision_unions_matches_unique_face_definition():
    pair_results = {
        "white_pial_left": {
            "collision_detected": True,
            "contact_count_exact": True,
            "face_ids_A": [0, 1],
            "face_ids_B": [1],
        },
        "white_pial_right": {
            "collision_detected": False,
            "contact_count_exact": True,
            "face_ids_A": [],
            "face_ids_B": [],
        },
        "pial_lr": {
            "collision_detected": True,
            "contact_count_exact": True,
            "face_ids_A": [2],
            "face_ids_B": [0],
        },
        "white_lr": {
            "collision_detected": True,
            "contact_count_exact": True,
            "face_ids_A": [1, 3],
            "face_ids_B": [2],
        },
        "cross_lhwhite_rhpial": {
            "collision_detected": True,
            "contact_count_exact": True,
            "face_ids_A": [3],
            "face_ids_B": [0, 4],
        },
        "cross_rhwhite_lhpial": {
            "collision_detected": True,
            "contact_count_exact": True,
            "face_ids_A": [2, 5],
            "face_ids_B": [1, 2],
        },
    }
    total_faces = {key: 10 for key in sqm.SURFACE_KEYS}

    out = sqm.summarize_collision_unions(pair_results, total_faces)

    assert out["surfaces"]["lh_white"]["face_ids"] == [0, 1, 3]
    assert out["surfaces"]["lh_pial"]["face_ids"] == [1, 2]
    assert out["surfaces"]["rh_white"]["face_ids"] == [2, 5]
    assert out["surfaces"]["rh_pial"]["face_ids"] == [0, 4]

    assert out["surfaces"]["lh_white"]["pct"] == pytest.approx(30.0)
    assert out["surfaces"]["lh_pial"]["pct"] == pytest.approx(20.0)
    assert out["surfaces"]["rh_white"]["pct"] == pytest.approx(20.0)
    assert out["surfaces"]["rh_pial"]["pct"] == pytest.approx(20.0)
    assert out["collision_pct_union_mean4"] == pytest.approx(22.5)
    assert out["collision_pct_union_max4"] == pytest.approx(30.0)
    assert out["collision_faces_union_sum4"] == 9
    assert out["exact_all_pairs"] is True


def test_missing_pair_marks_affected_surfaces_inexact():
    total_faces = {key: 10 for key in sqm.SURFACE_KEYS}
    out = sqm.summarize_collision_unions({}, total_faces)

    assert out["exact_all_pairs"] is False
    assert all(not out["surfaces"][key]["exact"] for key in sqm.SURFACE_KEYS)


def test_compute_surface_geometry_qc_reports_missing_surface():
    surfaces = [
        _surface("lh.white"),
        _surface("lh.pial.T1"),
        _surface("rh.white"),
    ]
    out = sqm.compute_surface_geometry_qc(surfaces)

    assert out["status"] == "missing_surfaces"
    assert out["missing_surfaces"] == ["rh_pial"]


def test_unavailable_collision_pairs_do_not_report_zero_metrics():
    total_faces = {key: 10 for key in sqm.SURFACE_KEYS}
    pair_results = {}
    for pair_name, (surface_a, surface_b) in sqm.COLLISION_PAIRS.items():
        pair = sqm._collision_result_base(surface_a, surface_b, 10, 10)
        pair["status"] = "no_fcl"
        pair["error"] = "missing fcl"
        pair_results[pair_name] = pair

    out = sqm.summarize_collision_unions(pair_results, total_faces)

    assert out["collision_pct_union_mean4"] is None
    assert out["collision_pct_union_max4"] is None
    assert out["collision_faces_union_sum4"] is None
    assert out["exact_all_pairs"] is False
    for key in sqm.SURFACE_KEYS:
        assert out["surfaces"][key]["available"] is False
        assert out["surfaces"][key]["pct"] is None
        assert out["surfaces"][key]["face_count"] == 0
        assert out["surfaces"][key]["exact"] is False


def test_compute_surface_geometry_qc_no_fcl_keeps_collision_metrics_unavailable(monkeypatch):
    surfaces = [
        _surface("lh.white"),
        _surface("lh.pial.T1"),
        _surface("rh.white"),
        _surface("rh.pial.T1"),
    ]

    monkeypatch.setattr(
        sqm,
        "compute_self_intersections",
        lambda vertices, faces: {
            "status": "OK",
            "error": "",
            "filter": "fake",
            "total_faces": len(faces),
            "face_count": 0,
            "face_ids": [],
            "pct": 0.0,
        },
    )
    monkeypatch.setattr(sqm, "_load_fcl", lambda: (None, "missing fcl"))

    out = sqm.compute_surface_geometry_qc(surfaces)

    assert out["status"] == "partial"
    assert out["collision_pct_union_mean4"] is None
    assert out["collision_pct_union_max4"] is None
    assert out["collision_faces_union_sum4"] is None
    assert out["collision_exact_all_pairs"] is False
    for key in sqm.SURFACE_KEYS:
        collision = out["surfaces"][key]["collision"]
        assert collision["available"] is False
        assert collision["pct"] is None
        assert collision["exact"] is False
