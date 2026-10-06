"""Landing page component for QC-Studio UI."""

import pandas as pd
import streamlit as st
from constants import (
    EXPERIENCE_LEVELS,
    FATIGUE_LEVELS,
    SCREEN_SIZES,
    PANEL_CONFIG,
    UPLOAD_FILE_TYPES,
    MESSAGES,
    ERROR_MESSAGES,
    SUCCESS_MESSAGES,
    INFO_MESSAGES,
    MONTAGE_HEIGHT,
    MIN_MONTAGE_GRID_SIZE,
    MAX_MONTAGE_GRID_SIZE,
    QC_DEDUP_KEYS,
    SESSION_KEYS,
)
from managers.session_manager import SessionManager
from models import QCRecord
from managers.panel_layout_manager import PanelLayoutManager
from managers.niivue_viewer_manager import NiivueViewerManager
from utils.config import list_qc_tasks_from_json, parse_qc_config
from utils.cohort import (
    build_qc_cohort,
    count_complete_cohort_pages,
    decided_rating_keys_from_df,
    invalid_upload_cohort_pairs,
    participant_ids_in_cohort_order,
)


def _normalize_participant_id(pid: str) -> str:
    """Normalize participant IDs for CSV/list comparisons."""
    pid_str = str(pid).strip()
    return pid_str[4:] if pid_str.startswith("sub-") else pid_str


def _maybe_apply_montage_defaults_from_qc_json(
    qc_config_path: str,
    qc_task: str,
    first_participant_raw: str,
) -> None:
    """Apply optional montage_max_rows / montage_max_cols from qc.json once per session.

    Pipeline authors can recommend a default grid for multi-image montages per QC task.
    Raters can still override via the landing-page montage form afterward.
    """
    applied_key = SESSION_KEYS["montage_defaults_applied_qc_task"]
    if st.session_state.get(applied_key) == qc_task:
        return
    task_for_defaults = qc_task
    if str(qc_task).strip().lower() == "all":
        tasks = list_qc_tasks_from_json(qc_config_path)
        task_for_defaults = tasks[0] if tasks else qc_task
    pid = str(first_participant_raw).strip()
    if not pid.startswith("sub-"):
        pid = f"sub-{pid}"
    cfg = parse_qc_config(
        qc_config_path,
        task_for_defaults,
        {"participant_id": pid, "session_id": "ses-01"},
    )
    if cfg.get("montage_max_rows") is not None:
        SessionManager.set_montage_max_rows(cfg["montage_max_rows"])
    if cfg.get("montage_max_cols") is not None:
        SessionManager.set_montage_max_cols(cfg["montage_max_cols"])
    st.session_state[applied_key] = qc_task


def _upload_qc_task_filter_keys(qc_task: str, qc_config_path: str) -> list[str] | None:
    """QC task keys to keep when filtering an uploaded results file.

    For ``--qc_task all``, exports store concrete task names (``anat_wf_qc``, etc.),
    not the literal CLI value ``all``.
    """
    if str(qc_task).strip().lower() == "all":
        tasks = list_qc_tasks_from_json(qc_config_path)
        return tasks if tasks else None
    return [str(qc_task).strip()]


def _filter_uploaded_df_for_qc_task(
    df: pd.DataFrame,
    qc_task: str,
    qc_config_path: str,
) -> pd.DataFrame:
    """Return uploaded rows that belong to the current QC workflow."""
    if "qc_task" not in df.columns:
        return df.copy()
    filter_keys = _upload_qc_task_filter_keys(qc_task, qc_config_path)
    if filter_keys is None:
        return df.iloc[0:0].copy()
    return df[df["qc_task"].astype(str).isin(filter_keys)].copy()


def _upload_filter_label(qc_task: str, qc_config_path: str) -> str:
    """Human-readable label for the upload filter caption."""
    if str(qc_task).strip().lower() == "all":
        tasks = list_qc_tasks_from_json(qc_config_path)
        if tasks:
            return f"all ({', '.join(tasks)})"
        return "all (no tasks found in qc.json)"
    return str(qc_task)


def _qc_task_display_labels(qc_config_path: str, task_keys: list[str]) -> list[str]:
    """Human-readable QC task names (qc.json ``display_name``, else the task key)."""
    dummy = {"participant_id": "sub-x", "session_id": "ses-01"}
    labels: list[str] = []
    for key in task_keys:
        key_s = str(key).strip()
        if not key_s:
            continue
        cfg = parse_qc_config(qc_config_path, key_s, dummy)
        labels.append(cfg.get("display_name") or key_s)
    return labels


def _landing_run_summary_lines(
    qc_pipeline: str,
    task_labels: list[str],
    n_subjects: int,
    n_pages: int,
    *,
    all_tasks: bool = False,
) -> tuple[str, str]:
    """Two compact lines: pipeline title, then task summary and cohort size."""
    labels = [str(label).strip() for label in task_labels if str(label).strip()]
    if all_tasks or len(labels) > 1:
        n = len(labels)
        task_word = "task" if n == 1 else "tasks"
        task_part = f"**Task:** all tasks ({n} {task_word})"
    elif len(labels) == 1:
        task_part = f"**Task:** {labels[0]}"
    else:
        task_part = "**Task:** —"
    line2 = f"{task_part} · **Subjects:** {n_subjects} · **Cohort pages:** {n_pages}"
    return qc_pipeline, line2


def show_landing_page(
    qc_pipeline,
    qc_task,
    out_dir,
    participant_list,
    qc_config_path: str,
    *,
    qc_cohort: list[dict] | None = None,
    session_ids: list[str] | None = None,
    entrypoint_rel_path: str | None = None,
) -> None:
    """Display the landing page with rater info, panel selection, and CSV upload.

    Args:
            qc_pipeline: QC pipeline name
            qc_task: QC task name
            out_dir: Output directory path
            participant_list: Path to participant list file
            qc_config_path: Absolute or cwd-relative path to qc.json (for optional montage defaults)
            qc_cohort: Optional pre-built (participant, session) page list from CLI context
            session_ids: BIDS session labels when ``qc_cohort`` is not provided
            entrypoint_rel_path: If set (for example ``"main.py"``), a successful Continue to QC
                    uses ``st.switch_page`` so the multipage sidebar hands off to the real app entrypoint.
                    Omit when the host is already ``main`` / ``app`` (normal ``st.rerun()``).
    """
    st.title(MESSAGES["welcome_title"])

    # Load participant list to get total unique participants
    try:
        participants_df = pd.read_csv(participant_list, delimiter="\t")
        raw_ids = participants_df["participant_id"].tolist()
        normalized_ids = [_normalize_participant_id(pid) for pid in raw_ids]
        total_participants_in_ds = len(set(normalized_ids))
        participant_ids_in_ds = set(normalized_ids)
        participant_ids_ordered = normalized_ids
        if qc_cohort is None:
            qc_cohort = build_qc_cohort(participants_df, session_ids or ["ses-01"])
        total_cohort_pages = len(qc_cohort)
    except Exception as e:
        st.error(ERROR_MESSAGES["participant_list_load_error"].format(error=e))
        return

    if raw_ids:
        _maybe_apply_montage_defaults_from_qc_json(qc_config_path, qc_task, raw_ids[0])

    qc_tasks_for_page = _upload_qc_task_filter_keys(qc_task, qc_config_path) or []
    task_keys = qc_tasks_for_page if qc_tasks_for_page else [str(qc_task).strip()]
    task_labels = _qc_task_display_labels(qc_config_path, task_keys)
    line1, line2 = _landing_run_summary_lines(
        qc_pipeline,
        task_labels,
        total_participants_in_ds,
        total_cohort_pages,
        all_tasks=str(qc_task).strip().lower() == "all",
    )
    st.header(line1)
    st.markdown(line2)

    st.markdown("---")

    # Three-column layout for rater info, panel selection, and CSV upload
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    # Left column: Rater Information
    with col1:
        _display_rater_form(entrypoint_rel_path=entrypoint_rel_path)

    # Middle column: Panel Selection and Montage Settings
    with col2:
        selected_panels = PanelLayoutManager.render_panel_header_with_controls()
        st.divider()
        _display_montage_settings()

    # Right column: CSV Upload
    with col3:
        _display_csv_upload(
            participant_ids_in_ds,
            total_cohort_pages,
            qc_cohort,
            qc_task,
            qc_config_path,
        )

    st.markdown("---")

    # Display panel layout preview based on selected panels
    _display_panel_layout_preview(selected_panels)


def _display_rater_form(entrypoint_rel_path: str | None = None) -> None:
    """Render rater information form in the landing page."""
    st.subheader(MESSAGES["rater_info_header"])
    with st.form("rater_form"):
        # Rater name/ID
        rater_id = st.text_input(MESSAGES["rater_id_prompt"], value=SessionManager.get_rater_id())

        # Remove spaces and normalize to lowercase so exported filenames do not collide by case.
        rater_id_clean = "".join(rater_id.split()).lower()

        # Experience level
        default_exp_idx = 0
        if SessionManager.get_rater_experience() in EXPERIENCE_LEVELS:
            default_exp_idx = EXPERIENCE_LEVELS.index(SessionManager.get_rater_experience())
        rater_experience = st.radio(MESSAGES["experience_prompt"], EXPERIENCE_LEVELS, index=default_exp_idx)

        # Fatigue level
        default_fatigue_idx = 0
        if SessionManager.get_rater_fatigue() in FATIGUE_LEVELS:
            default_fatigue_idx = FATIGUE_LEVELS.index(SessionManager.get_rater_fatigue())
        rater_fatigue = st.radio(MESSAGES["fatigue_prompt"], FATIGUE_LEVELS, index=default_fatigue_idx)

        # Screen size
        default_screen_idx = 0
        if SessionManager.get_rater_screen_size() in SCREEN_SIZES:
            default_screen_idx = SCREEN_SIZES.index(SessionManager.get_rater_screen_size())
        rater_screen_size = st.radio(MESSAGES["screen_size_prompt"], SCREEN_SIZES, index=default_screen_idx)

        # Autoplay countdown duration
        autoplay_duration = st.slider(
            "⏱️ Autoplay duration (seconds)", min_value=2, max_value=10, value=SessionManager.get_autoplay_duration(), step=1
        )

        submit_rater = st.form_submit_button(MESSAGES["rater_form_button"], width="stretch")

        if submit_rater:
            if not rater_id_clean:
                st.error(ERROR_MESSAGES["invalid_rater_id"])
            elif SessionManager.get_panel_count() == 0:
                st.error(ERROR_MESSAGES["no_panel_selected"])
            else:
                SessionManager.set_rater_id(rater_id_clean)
                SessionManager.set_rater_experience(rater_experience)
                SessionManager.set_rater_fatigue(rater_fatigue)
                SessionManager.set_rater_screen_size(rater_screen_size)
                SessionManager.set_autoplay_duration(autoplay_duration)
                SessionManager.set_landing_page_complete(True)
                if entrypoint_rel_path:
                    st.switch_page(entrypoint_rel_path)
                st.rerun()


def _display_csv_upload(
    participant_ids_in_ds: set,
    total_cohort_pages: int,
    qc_cohort: list[dict],
    qc_task: str,
    qc_config_path: str,
) -> None:
    """Render CSV upload section in the landing page.

    Args:
            participant_ids_in_ds: Set of normalized participant IDs in dataset
            total_cohort_pages: Total (participant, session) pages in this run
            qc_cohort: Ordered cohort rows for pagination
            qc_task: Current QC task name (used to filter uploaded CSV)
            qc_config_path: Path to qc.json (required when ``qc_task`` is ``all``)
    """
    qc_tasks = _upload_qc_task_filter_keys(qc_task, qc_config_path) or []
    st.subheader(MESSAGES["upload_header"])
    st.info(MESSAGES["upload_help"])

    uploaded_file = st.file_uploader(MESSAGES["csv_uploader_label"], type=UPLOAD_FILE_TYPES, key="qc_file_upload")

    if uploaded_file is not None:
        try:
            # Read the uploaded file while preserving zero-padded subject IDs.
            df = pd.read_csv(uploaded_file, sep=None, engine="python", dtype=str)

            # Deduplicate rows by QC_DEDUP_KEYS (keeping most recent record per participant)
            dedup_cols = [k for k in QC_DEDUP_KEYS if k in df.columns]
            if dedup_cols:
                df[dedup_cols] = df[dedup_cols].astype(str)
                df = df.drop_duplicates(subset=dedup_cols, keep="last").reset_index(drop=True)

            # Normalize participant IDs to support re-uploaded exports without sub- prefix.
            df["participant_id"] = df["participant_id"].astype(str)
            df["_participant_id_norm"] = df["participant_id"].map(_normalize_participant_id)

            df_task = _filter_uploaded_df_for_qc_task(df, qc_task, qc_config_path)
            filter_label = _upload_filter_label(qc_task, qc_config_path)
            decided = decided_rating_keys_from_df(df_task, qc_tasks)
            pages_reviewed = count_complete_cohort_pages(qc_cohort, qc_tasks, decided)
            records_reviewed = len(decided)
            total_qc_records = len(qc_cohort) * len(qc_tasks) if qc_cohort and qc_tasks else 0
            participant_ids_in_csv = {str(pid).strip() for pid in df_task["_participant_id_norm"].unique()}
            preview_df = df_task.drop(columns=["_participant_id_norm"], errors="ignore")

            st.success(SUCCESS_MESSAGES["csv_loaded"].format(count=len(df), filename=uploaded_file.name))
            st.caption(f"Current workflow filter: **{filter_label}**")

            # Validate: Check if CSV has participants not in the participant list
            invalid_participants = participant_ids_in_csv - participant_ids_in_ds
            if invalid_participants:
                st.error(
                    ERROR_MESSAGES["no_participants"].format(count=len(invalid_participants), participants=", ".join(sorted(invalid_participants)))
                )
                st.stop()

            invalid_pairs = invalid_upload_cohort_pairs(df_task, qc_cohort, participant_ids_in_ds)
            if invalid_pairs:
                pair_text = ", ".join(f"{pid} / {sid}" for pid, sid in sorted(invalid_pairs))
                st.error(f"❌ Error: The uploaded file contains {len(invalid_pairs)} " f"participant/session pair(s) not in this cohort: {pair_text}")
                st.stop()

            # Load participant list and show comparison
            try:
                # Create comparison display
                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    st.metric(label="Cohort pages reviewed", value=pages_reviewed)
                with col_comp2:
                    st.metric(label="QC records reviewed", value=records_reviewed)
                st.caption(
                    f"Totals: {total_cohort_pages} cohort pages · {total_qc_records} QC records "
                    f"across this workflow. When a page includes multiple tasks, records can exceed pages."
                )

                # Progress percentage is based on QC records, which reflects the actual per-task data points.
                progress_pct = (records_reviewed / total_qc_records) * 100 if total_qc_records > 0 else 0
                st.progress(min(progress_pct / 100, 1.0), text=f"{progress_pct:.1f}% of QC records complete")

            except Exception as e:
                st.warning(ERROR_MESSAGES["csv_comparison_error"].format(error=e))

            # Extract rater information from the current-task subset when available.
            source_df = df_task if len(df_task) > 0 else df
            if len(source_df) > 0:
                first_record = source_df.iloc[0]
                extracted_rater_id = str(first_record.get("rater_id", ""))
                extracted_experience = str(first_record.get("rater_experience", ""))
                extracted_fatigue = str(first_record.get("rater_fatigue", ""))
                extracted_screen_size = str(first_record.get("rater_screen_size", ""))

                # Update session state with extracted rater info
                SessionManager.set_rater_id(extracted_rater_id)
                SessionManager.set_rater_experience(extracted_experience)
                SessionManager.set_rater_fatigue(extracted_fatigue)
                SessionManager.set_rater_screen_size(extracted_screen_size)

                st.info(INFO_MESSAGES["rater_info_extracted"])
                st.write(INFO_MESSAGES["rater_id_prefix"].format(id=extracted_rater_id))
                st.write(INFO_MESSAGES["experience_prefix"].format(exp=extracted_experience))
                st.write(INFO_MESSAGES["fatigue_prefix"].format(fatigue=extracted_fatigue))
                st.write(INFO_MESSAGES["screen_size_prefix"].format(size=extracted_screen_size))

            # Display preview (filtered to current qc_task)
            st.subheader(INFO_MESSAGES["preview_header"])
            if preview_df.empty:
                st.warning(f"No records found for workflow **{filter_label}** in the uploaded file. " f"All {len(df)} records are for other tasks.")
            else:
                st.caption(f"Showing records for workflow: **{filter_label}**")
                st.dataframe(preview_df.head(10), width="stretch")

            # Option to load these records
            if st.button(INFO_MESSAGES["load_records_button"], width="stretch"):
                # Convert dataframe rows to QCRecord objects (current task only)
                loaded_records = []
                for _, row in df_task.iterrows():
                    record = QCRecord(
                        participant_id=str(row.get("participant_id", "")),
                        session_id=str(row.get("session_id", "")),
                        qc_task=str(row.get("qc_task", "")),
                        pipeline=str(row.get("pipeline", "")),
                        timestamp=str(row.get("timestamp", "")),
                        rater_id=str(row.get("rater_id", "")),
                        rater_experience=str(row.get("rater_experience", "")),
                        rater_fatigue=str(row.get("rater_fatigue", "")),
                        rater_screen_size=str(row.get("rater_screen_size", "")),
                        final_qc=str(row.get("final_qc", "")),
                        notes=str(row.get("notes", "")) if pd.notna(row.get("notes")) else "",
                    )
                    loaded_records.append(record)

                SessionManager.set_qc_records(loaded_records)
                SessionManager.set_qc_cohort_order(qc_cohort)
                SessionManager.set_participant_ids(participant_ids_in_cohort_order(qc_cohort))
                if SessionManager.all_qc_cohort_pages_complete_for_tasks(qc_tasks, qc_cohort):
                    target_page = total_cohort_pages + 1
                else:
                    next_page = SessionManager.first_qc_cohort_page_missing_for_tasks(qc_tasks, qc_cohort)
                    target_page = min(next_page, total_cohort_pages)
                SessionManager.set_current_page(target_page)
                st.success(SUCCESS_MESSAGES["records_loaded"].format(count=len(loaded_records)))
                st.info(INFO_MESSAGES["proceed_with_form"])

        except Exception as e:
            st.error(ERROR_MESSAGES["file_load_error"].format(error=e))

    st.divider()
    st.markdown(
        """
	**ℹ️ Tips:**
	- Save your work frequently using the **Save QC** button
	- Your session data persists within this application
	- Upload a previous file to resume or review work
	"""
    )


def _display_panel_layout_preview(selected_panels: dict) -> None:
    """Display a preview of the panel layout based on selected panels.

    When Niivue is selected: Shows 3-column layout (controls | Niivue | Montage/IQM)
    When Niivue is not selected: Shows full-width layout

    Args:
            selected_panels: Dictionary of selected panels
    """
    st.subheader("📐 Panel Layout Preview")

    show_niivue = selected_panels.get("niivue", False)
    show_montage = selected_panels.get("montage", False)
    show_iqm = selected_panels.get("iqm", False)
    show_surface = selected_panels.get("surface", False)

    # No panels selected
    if not (show_niivue or show_montage or show_iqm or show_surface):
        st.info("👉 Select panels above to see the layout preview")
        return

    # 3-column layout: Niivue with another panel
    if show_niivue and (show_montage or show_iqm):
        st.write("**Layout:** 3-column (Controls | Niivue Viewer | Secondary Panel)")
        ctrl_col, viewer_col, panel_col = st.columns([0.2, 0.4, 0.4], gap="small")

        with ctrl_col:
            st.info("🎮 **Controls**\n\n- View Mode\n- Overlay\n- Colormap\n- Opacity")

        with viewer_col:
            st.info("🧠 **Niivue Viewer**\n\n3D MRI data will be displayed here")

        with panel_col:
            secondary = "📊 **Montage**" if show_montage else "📈 **QC Metrics**"
            st.info(f"{secondary}\n\nSecondary visualization will be displayed here")

    # Full-width Niivue only
    elif show_niivue:
        st.write("**Layout:** 2-column (Controls | Niivue Viewer)")
        left_col, right_col = st.columns([0.32, 0.68], gap="small")

        with left_col:
            st.info("🎮 **Controls**\n\n- View Mode\n- Overlay\n- Colormap\n- Opacity")

        with right_col:
            st.info("🧠 **Niivue Viewer**\n\n3D MRI data will be displayed here")

    # Full-width Montage only
    elif show_montage:
        st.write("**Layout:** Full-width (Montage)")
        st.info("📊 **Montage**\n\nMontage visualization will be displayed across the full width")

    # Full-width IQM only
    elif show_iqm:
        st.write("**Layout:** Full-width (QC Metrics)")
        st.info("📈 **QC Metrics**\n\nQC metrics will be displayed across the full width")

    if show_surface:
        if show_niivue or show_montage or show_iqm:
            st.write("**Additional full-width panel:** Cortical Surface QC")
        else:
            st.write("**Layout:** Full-width (Cortical Surface QC)")

        st.info("🧠 **Cortical Surface QC**\n\n" "MRI slices with cortical-surface contours will be " "displayed across the full width")


def _display_montage_settings() -> None:
    """Render montage grid configuration settings.

    Allows users to specify maximum rows and columns for the montage grid.
    When both are set to None (auto), the montage will optimize for square aspect ratio.
    """
    st.markdown("#### 🎨 Montage Grid Settings")

    with st.form("montage_settings_form"):
        col1, col2 = st.columns(2)

        with col1:
            current_rows = SessionManager.get_montage_max_rows()
            montage_rows = st.number_input(
                "Max Rows (use checkbox for auto-calculation)",
                min_value=MIN_MONTAGE_GRID_SIZE,
                max_value=MAX_MONTAGE_GRID_SIZE,
                value=current_rows if current_rows else MIN_MONTAGE_GRID_SIZE,
                step=1,
                help="Maximum number of rows in the montage grid",
            )
            use_auto_rows = st.checkbox("Auto-calculate rows", value=(current_rows is None))

        with col2:
            current_cols = SessionManager.get_montage_max_cols()
            montage_cols = st.number_input(
                "Max Columns (use checkbox for auto-calculation)",
                min_value=MIN_MONTAGE_GRID_SIZE,
                max_value=MAX_MONTAGE_GRID_SIZE,
                value=current_cols if current_cols else MIN_MONTAGE_GRID_SIZE,
                step=1,
                help="Maximum number of columns in the montage grid",
            )
            use_auto_cols = st.checkbox("Auto-calculate columns", value=(current_cols is None))

        submit_montage = st.form_submit_button("Apply Montage Settings", width="stretch")

        if submit_montage:
            # Set to None if auto-calculate is checked, otherwise use the specified value
            rows_to_set = None if use_auto_rows else montage_rows
            cols_to_set = None if use_auto_cols else montage_cols

            SessionManager.set_montage_max_rows(rows_to_set)
            SessionManager.set_montage_max_cols(cols_to_set)

            # Display confirmation
            auto_text = "(auto)" if use_auto_rows else f"({montage_rows})"
            auto_text_cols = "(auto)" if use_auto_cols else f"({montage_cols})"
            st.success(f"✅ Montage settings updated: Max rows {auto_text}, Max columns {auto_text_cols}")

    # Show current settings
    current_rows = SessionManager.get_montage_max_rows()
    current_cols = SessionManager.get_montage_max_cols()
    rows_display = "Auto" if current_rows is None else str(current_rows)
    cols_display = "Auto" if current_cols is None else str(current_cols)
    st.info(f"📋 Current montage settings: Max rows = {rows_display}, Max columns = {cols_display}")
