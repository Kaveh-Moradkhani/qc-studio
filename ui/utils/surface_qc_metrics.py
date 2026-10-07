"""Pipeline-independent cortical-surface geometry QC metrics.

The scientific definitions mirror the SimCortex evaluation code supplied for
collision-union and self-intersection fraction (SIF):

- SIF: percentage of faces selected by PyMeshLab's self-intersection filter.
- Collision: six pairwise tests across lh/rh white/pial surfaces.
- Per-surface collision union: unique colliding face IDs across the other three
  surfaces divided by that surface's total face count.
- Case collision score: mean of the four per-surface union percentages.

PyMeshLab and python-fcl are optional at import time. If either backend is
unavailable, results report that status explicitly instead of treating the
metric as zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SURFACE_QC_SCHEMA_VERSION = "qc_studio_surface_geometry_qc_v1"

SURFACE_KEYS = ("lh_white", "lh_pial", "rh_white", "rh_pial")

COLLISION_PAIRS = {
    "white_pial_left": ("lh_white", "lh_pial"),
    "white_pial_right": ("rh_white", "rh_pial"),
    "pial_lr": ("lh_pial", "rh_pial"),
    "white_lr": ("lh_white", "rh_white"),
    "cross_lhwhite_rhpial": ("lh_white", "rh_pial"),
    "cross_rhwhite_lhpial": ("rh_white", "lh_pial"),
}

SIF_FILTER_CANDIDATES = (
    "compute_selection_by_self_intersections_per_face",
    "select_self_intersecting_faces",
)


def canonical_surface_key(name: str | Path | None) -> str | None:
    """Map FreeSurfer-style surface names to a pipeline-independent key."""
    if name is None:
        return None

    basename = Path(str(name)).name.lower()
    exact = {
        "lh.white": "lh_white",
        "lh.pial": "lh_pial",
        "lh.pial.t1": "lh_pial",
        "rh.white": "rh_white",
        "rh.pial": "rh_pial",
        "rh.pial.t1": "rh_pial",
    }
    if basename in exact:
        return exact[basename]

    normalized = basename.replace("-", "_").replace(".", "_")
    for key in SURFACE_KEYS:
        if normalized == key or normalized.endswith(f"_{key}"):
            return key
    return None


def index_cortical_surfaces(
    surfaces: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Index four cortical surfaces by canonical lh/rh white/pial keys."""
    if isinstance(surfaces, Mapping):
        items = list(surfaces.items())
    else:
        items = [(None, surface) for surface in surfaces]

    indexed: dict[str, Mapping[str, Any]] = {}
    for explicit_key, surface in items:
        if not isinstance(surface, Mapping):
            raise TypeError("Each cortical surface must be a mapping.")

        key = explicit_key if explicit_key in SURFACE_KEYS else None
        if key is None:
            name = surface.get("name") or surface.get("path")
            key = canonical_surface_key(name)

        if key is None:
            continue
        if key in indexed:
            raise ValueError(f"Duplicate cortical surface for {key!r}.")
        indexed[key] = surface

    return indexed


def _extract_mesh_arrays(surface: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    vertices = None
    for key in ("vertices_world", "vertices_surface_ras", "vertices"):
        if surface.get(key) is not None:
            vertices = surface[key]
            break
    if vertices is None:
        raise ValueError("Surface does not contain vertices_world, vertices_surface_ras, or vertices.")

    if surface.get("faces") is None:
        raise ValueError("Surface does not contain faces.")

    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(surface["faces"], dtype=np.int64)

    if vertices_np.ndim != 2 or vertices_np.shape[1] != 3:
        raise ValueError(f"Invalid vertex array shape: {vertices_np.shape}")
    if faces_np.ndim != 2 or faces_np.shape[1] != 3:
        raise ValueError(f"Invalid face array shape: {faces_np.shape}")
    if not np.isfinite(vertices_np).all():
        raise ValueError("Mesh contains non-finite vertices.")
    if faces_np.size and (faces_np.min() < 0 or faces_np.max() >= len(vertices_np)):
        raise ValueError("Mesh faces contain invalid vertex indices.")

    return vertices_np, faces_np


def _load_pymeshlab() -> tuple[Any | None, str]:
    try:
        import pymeshlab as pyml  # type: ignore

        return pyml, ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, repr(exc)


def _available_sif_filters(pyml: Any) -> list[str]:
    try:
        available = set(pyml.filter_list())
    except Exception:
        available = set()

    if not available:
        return list(SIF_FILTER_CANDIDATES)
    return [name for name in SIF_FILTER_CANDIDATES if name in available]


def _apply_pymeshlab_filter(mesh_set: Any, filter_name: str) -> None:
    try:
        mesh_set.apply_filter(filter_name)
        return
    except Exception as apply_exc:
        if hasattr(mesh_set, filter_name):
            try:
                getattr(mesh_set, filter_name)()
                return
            except Exception:
                pass
        raise apply_exc


def compute_self_intersections(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    pyml: Any | None = None,
) -> dict[str, Any]:
    """Return SIF percentage plus the exact self-intersecting face IDs."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    n_faces = int(len(faces))

    out: dict[str, Any] = {
        "status": "OK",
        "error": "",
        "filter": "",
        "total_faces": n_faces,
        "face_count": None,
        "face_ids": [],
        "pct": None,
    }

    if n_faces == 0:
        out["status"] = "no_faces"
        return out

    if pyml is None:
        pyml, import_error = _load_pymeshlab()
        if pyml is None:
            out["status"] = "no_pymeshlab"
            out["error"] = import_error
            return out

    candidates = _available_sif_filters(pyml)
    if not candidates:
        out["status"] = "no_sif_filter"
        out["error"] = "No supported PyMeshLab self-intersection filter is available."
        return out

    errors: list[str] = []
    for filter_name in candidates:
        try:
            mesh_set = pyml.MeshSet()
            mesh_set.add_mesh(
                pyml.Mesh(
                    vertex_matrix=vertices,
                    face_matrix=faces.astype(np.int32, copy=False),
                ),
                "mesh",
            )
            mesh = mesh_set.current_mesh()
            mesh_face_count = int(mesh.face_number())
            if mesh_face_count != n_faces:
                raise RuntimeError(f"PyMeshLab face count {mesh_face_count} does not match input face count {n_faces}.")

            _apply_pymeshlab_filter(mesh_set, filter_name)
            selected = np.asarray(
                mesh_set.current_mesh().face_selection_array(),
                dtype=bool,
            ).reshape(-1)
            if selected.size != n_faces:
                raise RuntimeError(f"face_selection_array size {selected.size} does not match face count {n_faces}.")

            face_ids = np.flatnonzero(selected).astype(np.int64).tolist()
            face_count = int(len(face_ids))
            out.update(
                {
                    "status": "OK",
                    "error": "",
                    "filter": filter_name,
                    "face_count": face_count,
                    "face_ids": face_ids,
                    "pct": float(face_count / n_faces * 100.0),
                }
            )
            return out
        except Exception as exc:
            errors.append(f"{filter_name}: {exc!r}")

    out["status"] = "error"
    out["error"] = " | ".join(errors)
    out["filter"] = candidates[0]
    return out


def _load_fcl() -> tuple[Any | None, str]:
    try:
        import fcl  # type: ignore

        return fcl, ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, repr(exc)


def _make_fcl_object(fcl_mod: Any, vertices: np.ndarray, faces: np.ndarray) -> Any:
    model = fcl_mod.BVHModel()
    model.beginModel(int(len(vertices)), int(len(faces)))
    model.addSubModel(
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
    )
    model.endModel()
    return fcl_mod.CollisionObject(model, fcl_mod.Transform())


def _collision_result_base(surface_a: str, surface_b: str, n_a: int, n_b: int) -> dict[str, Any]:
    return {
        "surface_A": surface_a,
        "surface_B": surface_b,
        "status": "OK",
        "error": "",
        "collision_detected": None,
        "num_contacts": None,
        "num_contacts_saturated": None,
        "contact_count_exact": False,
        "contact_index_failures": 0,
        "total_faces_A": int(n_a),
        "total_faces_B": int(n_b),
        "face_ids_A": [],
        "face_ids_B": [],
        "intersecting_faces_A": None,
        "intersecting_faces_B": None,
        "pct_faces_A": None,
        "pct_faces_B": None,
        "max_contacts_used": 0,
    }


def _collision_pair_from_objects(
    *,
    fcl_mod: Any,
    obj_a: Any,
    n_a: int,
    obj_b: Any,
    n_b: int,
    surface_a: str,
    surface_b: str,
    contact_caps: Sequence[int],
) -> dict[str, Any]:
    out = _collision_result_base(surface_a, surface_b, n_a, n_b)

    try:
        request = fcl_mod.CollisionRequest(num_max_contacts=1, enable_contact=False)
        result = fcl_mod.CollisionResult()
        returned = fcl_mod.collide(obj_a, obj_b, request, result)
        detected = bool(returned)
        is_collision = getattr(result, "is_collision", None)
        if isinstance(is_collision, bool):
            detected = bool(detected or is_collision)
    except Exception as exc:
        out["status"] = "bool_error"
        out["error"] = repr(exc)
        return out

    if not detected:
        out.update(
            {
                "collision_detected": False,
                "num_contacts": 0,
                "num_contacts_saturated": False,
                "contact_count_exact": True,
                "intersecting_faces_A": 0,
                "intersecting_faces_B": 0,
                "pct_faces_A": 0.0,
                "pct_faces_B": 0.0,
            }
        )
        return out

    last: dict[str, Any] | None = None
    for cap in sorted({int(x) for x in contact_caps if int(x) > 0}):
        current = _collision_result_base(surface_a, surface_b, n_a, n_b)
        current["collision_detected"] = True
        current["max_contacts_used"] = int(cap)

        try:
            request = fcl_mod.CollisionRequest(
                num_max_contacts=int(cap),
                enable_contact=True,
            )
            result = fcl_mod.CollisionResult()
            fcl_mod.collide(obj_a, obj_b, request, result)
            contacts = list(getattr(result, "contacts", []))
            n_contacts = int(len(contacts))

            face_ids_a: set[int] = set()
            face_ids_b: set[int] = set()
            index_failures = 0
            for contact in contacts:
                try:
                    face_ids_a.add(int(contact.b1))
                    face_ids_b.add(int(contact.b2))
                except Exception:
                    index_failures += 1

            ids_a = sorted(face_ids_a)
            ids_b = sorted(face_ids_b)
            n_intersect_a = int(len(ids_a))
            n_intersect_b = int(len(ids_b))
            saturated = bool(n_contacts >= int(cap))

            current.update(
                {
                    "status": "OK",
                    "error": "",
                    "collision_detected": bool(n_contacts > 0),
                    "num_contacts": n_contacts,
                    "num_contacts_saturated": saturated,
                    "contact_count_exact": bool(not saturated and index_failures == 0),
                    "contact_index_failures": int(index_failures),
                    "face_ids_A": ids_a,
                    "face_ids_B": ids_b,
                    "intersecting_faces_A": n_intersect_a,
                    "intersecting_faces_B": n_intersect_b,
                    "pct_faces_A": float(n_intersect_a / n_a * 100.0) if n_a else None,
                    "pct_faces_B": float(n_intersect_b / n_b * 100.0) if n_b else None,
                }
            )
            if n_contacts > 0 and n_intersect_a == 0 and n_intersect_b == 0:
                current["status"] = "contacts_without_indices"
                current["error"] = "FCL returned contacts but no usable face indices."

            last = current
            if not saturated:
                return current
        except Exception as exc:
            current["status"] = "count_error"
            current["error"] = repr(exc)
            last = current
            break

    if last is None:
        out["status"] = "invalid_contact_caps"
        out["error"] = "No positive contact caps were configured."
        return out

    if last.get("num_contacts_saturated") is True:
        last["status"] = "saturated"
        last["error"] = "FCL contact query reached the configured maximum; " "reported counts and face IDs are lower bounds."
        last["contact_count_exact"] = False
    return last


def summarize_collision_unions(
    pair_results: Mapping[str, Mapping[str, Any]],
    total_faces: Mapping[str, int],
) -> dict[str, Any]:
    """Union colliding face IDs per surface using the SimCortex definition.

    A percentage is only reported for a surface when every collision pair that
    involves that surface produced a determinate boolean result. This prevents
    a missing FCL backend or pair-query failure from being misreported as 0%.
    Saturated-but-determinate queries may still yield lower-bound percentages;
    ``exact`` distinguishes those from exact results.
    """
    union_faces: dict[str, set[int]] = {key: set() for key in SURFACE_KEYS}
    exact_by_surface = {key: True for key in SURFACE_KEYS}
    available_by_surface = {key: True for key in SURFACE_KEYS}

    for pair_name, (surface_a, surface_b) in COLLISION_PAIRS.items():
        pair = pair_results.get(pair_name)
        if pair is None:
            available_by_surface[surface_a] = False
            available_by_surface[surface_b] = False
            exact_by_surface[surface_a] = False
            exact_by_surface[surface_b] = False
            continue

        detected = pair.get("collision_detected")
        if detected not in (True, False):
            available_by_surface[surface_a] = False
            available_by_surface[surface_b] = False

        if detected is True:
            union_faces[surface_a].update(int(x) for x in pair.get("face_ids_A", []) if int(x) >= 0)
            union_faces[surface_b].update(int(x) for x in pair.get("face_ids_B", []) if int(x) >= 0)

        if pair.get("contact_count_exact") is not True:
            exact_by_surface[surface_a] = False
            exact_by_surface[surface_b] = False

    surfaces: dict[str, Any] = {}
    percentages: list[float] = []
    all_surfaces_available = True
    for surface_key in SURFACE_KEYS:
        ids = sorted(union_faces[surface_key])
        total = int(total_faces.get(surface_key, 0))
        available = bool(available_by_surface[surface_key] and total > 0)
        pct = float(len(ids) / total * 100.0) if available else None
        surfaces[surface_key] = {
            "face_ids": ids,
            "face_count": int(len(ids)),
            "total_faces": total,
            "pct": pct,
            "available": available,
            "exact": bool(available and exact_by_surface[surface_key]),
        }
        if pct is not None:
            percentages.append(pct)
        else:
            all_surfaces_available = False

    return {
        "surfaces": surfaces,
        "collision_pct_union_mean4": (float(np.mean(percentages)) if all_surfaces_available else None),
        "collision_pct_union_max4": (float(np.max(percentages)) if all_surfaces_available else None),
        "collision_faces_union_sum4": (int(sum(entry["face_count"] for entry in surfaces.values())) if all_surfaces_available else None),
        "exact_all_pairs": bool(
            all_surfaces_available and len(pair_results) == len(COLLISION_PAIRS) and all(bool(entry["exact"]) for entry in surfaces.values())
        ),
    }


def compute_surface_geometry_qc(
    surfaces: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    contact_caps: Sequence[int] = (50_000, 200_000, 500_000),
) -> dict[str, Any]:
    """Compute SIF and six-pair collision diagnostics for one four-surface case."""
    indexed = index_cortical_surfaces(surfaces)
    missing = [key for key in SURFACE_KEYS if key not in indexed]

    result: dict[str, Any] = {
        "schema_version": SURFACE_QC_SCHEMA_VERSION,
        "status": "OK",
        "error": "",
        "missing_surfaces": missing,
        "surfaces": {},
        "pairs": {},
        "collision_pct_union_mean4": None,
        "collision_pct_union_max4": None,
        "collision_faces_union_sum4": None,
        "collision_exact_all_pairs": False,
    }

    if missing:
        result["status"] = "missing_surfaces"
        result["error"] = f"Missing required cortical surfaces: {', '.join(missing)}"
        return result

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    total_faces: dict[str, int] = {}
    for key in SURFACE_KEYS:
        vertices, faces = _extract_mesh_arrays(indexed[key])
        arrays[key] = (vertices, faces)
        total_faces[key] = int(len(faces))

        sif = compute_self_intersections(vertices, faces)
        result["surfaces"][key] = {
            "total_faces": int(len(faces)),
            "sif": sif,
        }

    fcl_mod, fcl_error = _load_fcl()
    if fcl_mod is None:
        for pair_name, (surface_a, surface_b) in COLLISION_PAIRS.items():
            pair = _collision_result_base(
                surface_a,
                surface_b,
                total_faces[surface_a],
                total_faces[surface_b],
            )
            pair["status"] = "no_fcl"
            pair["error"] = fcl_error
            result["pairs"][pair_name] = pair
    else:
        objects: dict[str, Any] = {}
        try:
            for key in SURFACE_KEYS:
                vertices, faces = arrays[key]
                objects[key] = _make_fcl_object(fcl_mod, vertices, faces)
        except Exception as exc:
            result["status"] = "fcl_build_error"
            result["error"] = repr(exc)
            return result

        for pair_name, (surface_a, surface_b) in COLLISION_PAIRS.items():
            result["pairs"][pair_name] = _collision_pair_from_objects(
                fcl_mod=fcl_mod,
                obj_a=objects[surface_a],
                n_a=total_faces[surface_a],
                obj_b=objects[surface_b],
                n_b=total_faces[surface_b],
                surface_a=surface_a,
                surface_b=surface_b,
                contact_caps=contact_caps,
            )

    collision_summary = summarize_collision_unions(result["pairs"], total_faces)
    result["collision_pct_union_mean4"] = collision_summary["collision_pct_union_mean4"]
    result["collision_pct_union_max4"] = collision_summary["collision_pct_union_max4"]
    result["collision_faces_union_sum4"] = collision_summary["collision_faces_union_sum4"]
    result["collision_exact_all_pairs"] = collision_summary["exact_all_pairs"]

    for key in SURFACE_KEYS:
        result["surfaces"][key]["collision"] = collision_summary["surfaces"][key]

    sif_ok = all(result["surfaces"][key]["sif"]["status"] == "OK" for key in SURFACE_KEYS)
    collisions_available = all(result["pairs"][pair_name]["status"] != "no_fcl" for pair_name in COLLISION_PAIRS)
    collision_errors = [
        pair_name for pair_name, pair in result["pairs"].items() if pair["status"] in {"bool_error", "count_error", "invalid_contact_caps"}
    ]

    if collision_errors:
        result["status"] = "collision_error"
        result["error"] = "Collision failures: " + ", ".join(collision_errors)
    elif not sif_ok or not collisions_available:
        result["status"] = "partial"

    return result
