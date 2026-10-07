"""Interactive cortical-surface QC viewer."""

from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.data_loaders import load_surface_data
from utils.surface_geometry import mesh_slice_segments
from utils.surface_qc_metrics import (
    COLLISION_PAIRS,
    SURFACE_KEYS,
    compute_surface_geometry_qc,
)


# ---------------------------------------------------------------------
# Anatomical plane definitions
#
# For the FreeSurfer-conformed MRI used here, voxel axis codes are:
#     axis 0 = Left/Right
#     axis 1 = Inferior/Superior
#     axis 2 = Anterior/Posterior
#
# Therefore:
#     constant axis 0 -> Sagittal
#     constant axis 1 -> Axial
#     constant axis 2 -> Coronal
# ---------------------------------------------------------------------

PLANE_NAMES = {
    0: "Sagittal",
    1: "Axial",
    2: "Coronal",
}


# For each slicing axis, define the two voxel-coordinate axes that
# should be displayed horizontally and vertically.
PLANE_AXES = {
    0: (1, 2),  # Sagittal
    1: (0, 2),  # Axial
    2: (0, 1),  # Coronal
}


# ---------------------------------------------------------------------
# Surface display styles
# ---------------------------------------------------------------------

SURFACE_STYLES = {
    "lh.white": {
        "label": "Left white",
        "color": "#FFD700",
    },
    "lh.pial": {
        "label": "Left pial",
        "color": "#FF4040",
    },
    "lh.pial.T1": {
        "label": "Left pial",
        "color": "#FF4040",
    },
    "rh.white": {
        "label": "Right white",
        "color": "#00FFFF",
    },
    "rh.pial": {
        "label": "Right pial",
        "color": "#FF00FF",
    },
    "rh.pial.T1": {
        "label": "Right pial",
        "color": "#FF00FF",
    },
}


SURFACE_METRIC_LABELS = {
    "lh_white": "Left white",
    "lh_pial": "Left pial",
    "rh_white": "Right white",
    "rh_pial": "Right pial",
}

COLLISION_PAIR_LABELS = {
    "white_pial_left": "Left white ↔ left pial",
    "white_pial_right": "Right white ↔ right pial",
    "pial_lr": "Left pial ↔ right pial",
    "white_lr": "Left white ↔ right white",
    "cross_lhwhite_rhpial": "Left white ↔ right pial",
    "cross_rhwhite_lhpial": "Right white ↔ left pial",
}


def _surface_style(name: str) -> dict:
    """Return display metadata for a cortical surface."""
    if name in SURFACE_STYLES:
        return SURFACE_STYLES[name]

    return {
        "label": name,
        "color": "#00FF00",
    }


# ---------------------------------------------------------------------
# Default slice positions
# ---------------------------------------------------------------------


def _default_slice_positions(
    surfaces: list[dict],
    volume_shape,
) -> dict[int, int]:
    """Return default slices centered on the combined surface bounding box."""
    vertices = np.concatenate(
        [np.asarray(surface["vertices_voxel"]) for surface in surfaces],
        axis=0,
    )

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)

    center = (bbox_min + bbox_max) / 2.0

    return {
        axis: int(
            np.clip(
                round(center[axis]),
                0,
                int(volume_shape[axis]) - 1,
            )
        )
        for axis in (0, 1, 2)
    }


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------


def _segments_to_xy(
    segments,
    axis: int,
):
    """Convert 3D mesh-slice segments to Plotly x/y line coordinates."""
    segments = np.asarray(segments)

    horizontal_axis, vertical_axis = PLANE_AXES[axis]

    x = []
    y = []

    for segment in segments:
        x.extend(
            [
                float(segment[0, horizontal_axis]),
                float(segment[1, horizontal_axis]),
                None,
            ]
        )

        y.extend(
            [
                float(segment[0, vertical_axis]),
                float(segment[1, vertical_axis]),
                None,
            ]
        )

    return x, y


# ---------------------------------------------------------------------
# MRI display helpers
# ---------------------------------------------------------------------


def _normalized_slice(
    volume,
    axis: int,
    slice_position: int,
):
    """Extract and normalize one MRI slice into the range [0, 1]."""
    slice_data = np.asarray(
        np.take(
            volume,
            slice_position,
            axis=axis,
        ),
        dtype=np.float32,
    ).T

    finite = np.isfinite(slice_data)

    if not finite.any():
        return np.zeros_like(
            slice_data,
            dtype=np.float32,
        )

    values = slice_data[finite]

    low = float(
        np.percentile(
            values,
            1.0,
        )
    )

    high = float(
        np.percentile(
            values,
            99.0,
        )
    )

    if high <= low:
        high = low + 1.0

    normalized = (slice_data - low) / (high - low)

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


# ---------------------------------------------------------------------
# Plot construction
# ---------------------------------------------------------------------


def _build_slice_figure(
    volume,
    surfaces: list[dict],
    axis: int,
    slice_position: int,
):
    """Build one MRI slice with cortical-surface intersection contours."""
    image = _normalized_slice(
        volume,
        axis,
        slice_position,
    )

    # Display sagittal slices rotated 90 degrees clockwise.
    # This is a presentation transform only; the underlying voxel and
    # surface coordinates remain unchanged.
    sagittal_source_height = None
    if axis == 0:
        sagittal_source_height = int(image.shape[0])
        image = np.rot90(
            image,
            k=-1,
        )

    figure = go.Figure()

    # MRI background
    figure.add_trace(
        go.Heatmap(
            z=image,
            colorscale="Gray",
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            hoverinfo="skip",
        )
    )

    # Cortical-surface contours
    for surface in surfaces:
        segments = mesh_slice_segments(
            surface["vertices_voxel"],
            surface["faces"],
            axis=axis,
            slice_position=float(slice_position),
        )

        if len(segments) == 0:
            continue

        x, y = _segments_to_xy(
            segments,
            axis,
        )

        if axis == 0:
            rotated_x = []
            rotated_y = []

            for x_value, y_value in zip(x, y):
                if x_value is None or y_value is None:
                    rotated_x.append(None)
                    rotated_y.append(None)
                    continue

                rotated_x.append(sagittal_source_height - 1 - float(y_value))
                rotated_y.append(float(x_value))

            x = rotated_x
            y = rotated_y

        style = _surface_style(surface["name"])

        figure.add_trace(
            go.Scattergl(
                x=x,
                y=y,
                mode="lines",
                name=style["label"],
                line={
                    "color": style["color"],
                    "width": 1.5,
                },
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        margin={
            "l": 0,
            "r": 0,
            "t": 36,
            "b": 0,
        },
        title={
            "text": (f"{PLANE_NAMES[axis]} " f"slice {slice_position}"),
            "x": 0.5,
        },
        showlegend=False,
        height=420,
    )

    figure.update_xaxes(
        showgrid=False,
        zeroline=False,
        visible=False,
        constrain="domain",
    )

    figure.update_yaxes(
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        autorange="reversed",
    )

    return figure


# ---------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_surface_qc_data(
    dataset_dir,
    qc_config,
):
    """Load and cache MRI volume, cortical surfaces, and geometry diagnostics."""
    surface_data = load_surface_data(
        dataset_dir,
        qc_config,
    )

    if not surface_data:
        return None

    target_path = Path(surface_data["target_mri_image_path"])

    target_img = nib.load(str(target_path))

    volume = np.asanyarray(target_img.dataobj)

    try:
        geometry_qc = compute_surface_geometry_qc(
            surface_data["surfaces"],
        )
    except Exception as error:
        # Quantitative diagnostics must not prevent visual QC if a backend or
        # malformed mesh fails unexpectedly. Surface the failure explicitly.
        geometry_qc = {
            "status": "error",
            "error": repr(error),
            "missing_surfaces": [],
            "surfaces": {},
            "pairs": {},
            "collision_pct_union_mean4": None,
            "collision_pct_union_max4": None,
            "collision_faces_union_sum4": None,
            "collision_exact_all_pairs": False,
        }

    return {
        "surface_data": surface_data,
        "volume": volume,
        "geometry_qc": geometry_qc,
    }


# ---------------------------------------------------------------------
# Independent plane fragment
#
# Each plane is a Streamlit fragment. Moving one slider reruns only
# that plane instead of rebuilding all three MRI views.
# ---------------------------------------------------------------------


def _step_surface_slice(
    slider_key: str,
    default_position: int,
    step: int,
    max_position: int,
) -> None:
    """Move a surface-QC slice backward or forward by one step."""
    current_position = int(
        st.session_state.get(
            slider_key,
            default_position,
        )
    )

    st.session_state[slider_key] = max(
        0,
        min(
            max_position,
            current_position + step,
        ),
    )


@st.fragment
def _display_surface_plane(
    volume,
    surfaces,
    axis: int,
    default_position: int,
    subject_key: str,
):
    """Render one independently updating cortical-surface QC plane."""
    max_position = int(volume.shape[axis]) - 1

    slider_key = f"surface_slice_" f"{axis}_" f"{subject_key}"

    current_position = int(
        st.session_state.get(
            slider_key,
            default_position,
        )
    )

    st.markdown(f"**{PLANE_NAMES[axis]}**")

    previous_col, slider_col, next_col = st.columns([1.1, 7.8, 1.1])

    with previous_col:
        st.button(
            " ",
            icon=":material/chevron_left:",
            key=(f"surface_previous_" f"{axis}_" f"{subject_key}"),
            help="Previous slice",
            disabled=current_position <= 0,
            on_click=_step_surface_slice,
            args=(
                slider_key,
                default_position,
                -1,
                max_position,
            ),
        )

    with slider_col:
        slice_position = st.slider(
            PLANE_NAMES[axis],
            min_value=0,
            max_value=max_position,
            value=default_position,
            step=1,
            key=slider_key,
            label_visibility="collapsed",
        )

    with next_col:
        st.button(
            " ",
            icon=":material/chevron_right:",
            key=(f"surface_next_" f"{axis}_" f"{subject_key}"),
            help="Next slice",
            disabled=current_position >= max_position,
            on_click=_step_surface_slice,
            args=(
                slider_key,
                default_position,
                1,
                max_position,
            ),
        )

    figure = _build_slice_figure(
        volume,
        surfaces,
        axis,
        slice_position,
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "displaylogo": False,
        },
        key=(f"surface_plot_" f"{axis}_" f"{subject_key}"),
    )


# ---------------------------------------------------------------------
# Quantitative geometry diagnostics
# ---------------------------------------------------------------------


def _rounded_pct(value):
    """Round a percentage for display while preserving unavailable values."""
    if value is None:
        return None
    return round(float(value), 4)


def _surface_metric_rows(geometry_qc: dict) -> list[dict]:
    """Build display rows for per-surface SIF and collision-union metrics."""
    surface_results = geometry_qc.get("surfaces", {})
    rows = []

    for surface_key in SURFACE_KEYS:
        surface = surface_results.get(surface_key, {})
        sif = surface.get("sif", {})
        collision = surface.get("collision", {})
        collision_available = bool(collision.get("available", False))

        rows.append(
            {
                "Surface": SURFACE_METRIC_LABELS[surface_key],
                "Self-intersecting faces": sif.get("face_count"),
                "SIF (%)": _rounded_pct(sif.get("pct")),
                "Colliding faces (union)": (collision.get("face_count") if collision_available else None),
                "Collision union (%)": (_rounded_pct(collision.get("pct")) if collision_available else None),
                "Collision exact": (bool(collision.get("exact")) if collision_available else None),
            }
        )

    return rows


def _collision_pair_rows(geometry_qc: dict) -> list[dict]:
    """Build display rows for the six SimCortex-compatible collision pairs."""
    pair_results = geometry_qc.get("pairs", {})
    rows = []

    for pair_name in COLLISION_PAIRS:
        pair = pair_results.get(pair_name, {})
        detected = pair.get("collision_detected")

        if detected is True:
            detected_text = "Yes"
        elif detected is False:
            detected_text = "No"
        else:
            detected_text = "Unavailable"

        rows.append(
            {
                "Surface pair": COLLISION_PAIR_LABELS[pair_name],
                "Collision": detected_text,
                "Contacts": pair.get("num_contacts"),
                "Faces A (%)": _rounded_pct(pair.get("pct_faces_A")),
                "Faces B (%)": _rounded_pct(pair.get("pct_faces_B")),
                "Exact": (bool(pair.get("contact_count_exact")) if detected in (True, False) else None),
                "Status": pair.get("status", "unavailable"),
            }
        )

    return rows


def _display_surface_geometry_metrics(geometry_qc: dict) -> None:
    """Render numerical SIF and raw surface-contact diagnostics."""
    st.subheader("Surface integrity metrics")

    status = geometry_qc.get("status", "unknown")
    error = geometry_qc.get("error", "")

    if status != "OK":
        message = f"Geometry diagnostics status: {status}."
        if error:
            message += f" {error}"
        st.warning(message)

    st.dataframe(
        _surface_metric_rows(geometry_qc),
        hide_index=True,
        use_container_width=True,
    )

    mean4 = geometry_qc.get("collision_pct_union_mean4")
    max4 = geometry_qc.get("collision_pct_union_max4")
    exact = geometry_qc.get("collision_exact_all_pairs", False)

    mean_text = "unavailable" if mean4 is None else f"{float(mean4):.4f}%"
    max_text = "unavailable" if max4 is None else f"{float(max4):.4f}%"

    st.caption("Raw case collision union — " f"mean: {mean_text} · max: {max_text} · " f"all pair counts exact: {'yes' if exact else 'no'}")

    with st.expander(
        "Pairwise surface contacts",
        expanded=False,
    ):
        st.dataframe(
            _collision_pair_rows(geometry_qc),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "SIF is the percentage of faces flagged self-intersecting by "
        "PyMeshLab. Collision values are raw geometric contacts. Depending "
        "on the surface definition, coincident medial-wall white/pial faces "
        "may contribute. These diagnostics inform, but do not automatically "
        "determine, the PASS/FAIL/UNCERTAIN rating."
    )


# ---------------------------------------------------------------------
# Main Surface QC panel
# ---------------------------------------------------------------------


def display_surface_qc_panel(
    dataset_dir,
    qc_config,
    participant_id: str | None = None,
    session_id: str | None = None,
    task_suffix: str = "",
) -> None:
    """Render MRI slices with cortical-surface contours."""
    st.header("Cortical Surface QC")

    try:
        cached_data = _load_surface_qc_data(
            dataset_dir,
            qc_config,
        )

    except (
        OSError,
        ValueError,
    ) as error:
        st.error("Failed to load " f"surface-QC data: {error}")
        return

    if not cached_data:
        st.info("Cortical surfaces or " "their reference MRI " "could not be loaded.")
        return

    surface_data = cached_data["surface_data"]

    volume = cached_data["volume"]

    if volume.ndim != 3:
        st.info("Surface QC currently " "requires a 3D MRI volume.")
        return

    surfaces = surface_data["surfaces"]

    if not surfaces:
        st.info("No cortical surfaces " "could be loaded.")
        return

    default_positions = _default_slice_positions(
        surfaces,
        volume.shape,
    )

    subject_key = f"{participant_id or 'participant'}_" f"{session_id or 'session'}_" f"{task_suffix or 'task'}"

    columns = st.columns(3)

    for column, axis in zip(
        columns,
        (0, 1, 2),
    ):
        with column:
            _display_surface_plane(
                volume,
                surfaces,
                axis,
                default_positions[axis],
                subject_key,
            )

    st.caption("Yellow: left white · " "Red: left pial · " "Cyan: right white · " "Magenta: right pial")

    geometry_qc = cached_data.get("geometry_qc")

    if geometry_qc:
        st.divider()
        _display_surface_geometry_metrics(
            geometry_qc,
        )
