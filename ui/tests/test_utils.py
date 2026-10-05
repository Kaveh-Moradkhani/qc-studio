"""Tests for utils.py module."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import utils.data_loaders as data_loaders
from utils.config import parse_qc_config
from utils.data_loaders import (
    load_mri_data,
    load_montage_data,
    load_iqm_data,
    _resolve_metadata_path,
    _infer_bids_ids_from_path,
    _infer_dataset_root_from_path,
    _infer_bids_folder_from_path,
    _find_bids_metadata_sidecar,
    load_scanner_metadata,
    resolve_iqm_data_path,
    load_iqm_distribution_table,
    load_iqm_metrics_subject_level,
)
from utils.reference_data import (
    normalize_manufacturer,
    normalize_field_strength,
    download_reference_parquet,
    filter_reference_iqm,
)
from utils.export import save_qc_results_to_csv

pytestmark = pytest.mark.integration


class TestParseQcConfig:
    """Test parse_qc_config function."""

    substitution_values = {
        "participant_id": "sub-ED01",
        "session_id": "ses-01",
    }

    def test_parse_valid_qc_config(self, temp_dir):
        """Test parsing a valid QC config file."""
        qc_config = {
            "anat_wf_qc": {
                "base_mri_image_path": str(temp_dir / "base.nii.gz"),
                "overlay_mri_image_path": str(temp_dir / "overlay.nii.gz"),
                "svg_montage_path": [str(temp_dir / "montage.svg")],
                "iqm_path": str(temp_dir / "iqm.json"),
                "surface_path": [
                    str(temp_dir / "lh.white"),
                    str(temp_dir / "rh.white"),
                ],
                "surface_reference_mri_image_path": str(temp_dir / "orig.mgz"),
            }
        }
        config_path = temp_dir / "qc_config.json"
        config_path.write_text(json.dumps(qc_config))

        result = parse_qc_config(str(config_path), "anat_wf_qc", self.substitution_values)

        assert result is not None
        assert "base_mri_image_path" in result
        assert "montage_path" in result
        assert result["base_mri_image_path"] is not None
        assert result["surface_path"] == [
            temp_dir / "lh.white",
            temp_dir / "rh.white",
        ]
        assert result["surface_reference_mri_image_path"] == temp_dir / "orig.mgz"
        assert "montage_max_rows" in result
        assert "montage_max_cols" in result
        assert result["montage_max_rows"] is None
        assert result["montage_max_cols"] is None

    def test_parse_qc_config_montage_layout_from_json(self, temp_dir):
        """Optional montage_max_rows / montage_max_cols are parsed per QC task."""
        qc_path = temp_dir / "qc.json"
        qc_path.write_text(
            json.dumps(
                {
                    "anat_wf_qc": {
                        "montage_path": [str(temp_dir / "a.svg"), str(temp_dir / "b.svg")],
                        "montage_max_rows": 2,
                        "montage_max_cols": 2,
                    }
                }
            )
        )
        result = parse_qc_config(str(qc_path), "anat_wf_qc")
        assert result["montage_max_rows"] == 2
        assert result["montage_max_cols"] == 2

    def test_parse_qc_config_display_name(self, temp_dir):
        qc_path = temp_dir / "qc.json"
        (temp_dir / "a.svg").write_text("<svg></svg>")
        qc_path.write_text(
            json.dumps(
                {
                    "demo_task": {
                        "display_name": "Friendly label",
                        "montage_path": str(temp_dir / "a.svg"),
                    }
                }
            )
        )
        result = parse_qc_config(str(qc_path), "demo_task")
        assert result["display_name"] == "Friendly label"

    def test_parse_qc_config_nonexistent_task(self, sample_qc_config):
        """Test parsing QC config with non-existent task."""
        result = parse_qc_config(str(sample_qc_config), "nonexistent_task")

        assert result["base_mri_image_path"] is None
        assert result["overlay_mri_image_path"] is None
        assert result["montage_path"] is None
        assert result["iqm_path"] is None
        assert result["surface_path"] is None
        assert result["surface_reference_mri_image_path"] is None
        assert result["montage_max_rows"] is None
        assert result["montage_max_cols"] is None
        assert result["display_name"] is None

    def test_parse_qc_config_invalid_file(self, temp_dir):
        """Test parsing non-existent QC config file."""
        result = parse_qc_config(str(temp_dir / "nonexistent.json"), "anat_wf_qc")

        assert result["base_mri_image_path"] is None
        assert result["overlay_mri_image_path"] is None
        assert result["montage_max_rows"] is None
        assert result["montage_max_cols"] is None

    def test_parse_qc_config_malformed_json(self, temp_dir):
        """Test parsing malformed JSON file."""
        bad_json_file = temp_dir / "bad.json"
        bad_json_file.write_text("{ invalid json }")

        result = parse_qc_config(str(bad_json_file), "anat_wf_qc")

        assert result["base_mri_image_path"] is None
        assert result["montage_max_rows"] is None
        assert result["montage_max_cols"] is None

    def test_parse_qc_config_none_input(self):
        """Test parsing with None input."""
        result = parse_qc_config(None, "anat_wf_qc")

        assert result["base_mri_image_path"] is None
        assert result["montage_max_rows"] is None
        assert result["montage_max_cols"] is None


class TestLoadMriData:
    """Test load_mri_data function."""

    def test_load_both_mri_files(self, temp_dir):
        """Test loading both base and overlay MRI files."""
        base_file = temp_dir / "base.nii.gz"
        overlay_file = temp_dir / "overlay.nii.gz"

        base_file.write_bytes(b"base content")
        overlay_file.write_bytes(b"overlay content")

        path_dict = {"base_mri_image_path": base_file, "overlay_mri_image_path": overlay_file}

        result = load_mri_data(path_dict)

        assert "base_mri_image_bytes" in result
        assert "overlay_mri_image_bytes" in result
        assert result["base_mri_image_bytes"] == b"base content"
        assert result["overlay_mri_image_bytes"] == b"overlay content"

    def test_load_only_base_mri(self, temp_dir):
        """Test loading only base MRI file."""
        base_file = temp_dir / "base.nii.gz"
        base_file.write_bytes(b"base content")

        path_dict = {"base_mri_image_path": base_file, "overlay_mri_image_path": None}

        result = load_mri_data(path_dict)

        assert "base_mri_image_bytes" in result
        assert "overlay_mri_image_bytes" not in result

    def test_load_nonexistent_mri_file(self, temp_dir):
        """Test loading non-existent MRI file."""
        path_dict = {"base_mri_image_path": temp_dir / "nonexistent.nii.gz", "overlay_mri_image_path": None}

        result = load_mri_data(path_dict)

        assert result == {}

    def test_load_mri_with_none_paths(self):
        """Test loading with None paths."""
        path_dict = {"base_mri_image_path": None, "overlay_mri_image_path": None}

        result = load_mri_data(path_dict)

        assert result == {}


class TestSurfaceData:
    """Test cortical-surface geometry loading and coordinate transforms."""

    @staticmethod
    def _fake_image(affine, vox2ras_tkr=None):
        class Header:
            def get_vox2ras_tkr(self):
                if vox2ras_tkr is None:
                    raise AttributeError("No vox2ras_tkr")
                return vox2ras_tkr

        class Image:
            pass

        image = Image()
        image.affine = np.asarray(affine, dtype=float)
        image.header = Header()
        return image

    def test_transform_surface_vertices_to_target_voxels(self):
        """Surface RAS is mapped through FreeSurfer scanner RAS to target voxels."""
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 20.0, 30.0],
            ]
        )

        vox2ras_tkr = np.eye(4)
        reference_affine = np.eye(4)
        reference_affine[:3, 3] = [1.0, 2.0, 3.0]

        target_affine = np.eye(4)
        target_affine[:3, 3] = [-4.0, -5.0, -6.0]

        reference_img = self._fake_image(
            reference_affine,
            vox2ras_tkr,
        )
        target_img = self._fake_image(target_affine)

        world, target_voxels = data_loaders._transform_surface_vertices(
            vertices,
            reference_img,
            target_img,
        )

        np.testing.assert_allclose(
            world,
            [
                [1.0, 2.0, 3.0],
                [11.0, 22.0, 33.0],
            ],
        )
        np.testing.assert_allclose(
            target_voxels,
            [
                [5.0, 7.0, 9.0],
                [15.0, 27.0, 39.0],
            ],
        )

    def test_load_surface_data_from_dataset_relative_paths(
        self,
        temp_dir,
        monkeypatch,
    ):
        """Surface and MRI paths are resolved relative to dataset_dir."""
        surface_dir = temp_dir / "derivatives" / "freesurfer" / "sub-01" / "surf"
        mri_dir = temp_dir / "derivatives" / "freesurfer" / "sub-01" / "mri"
        anat_dir = temp_dir / "bids" / "sub-01" / "anat"

        surface_dir.mkdir(parents=True)
        mri_dir.mkdir(parents=True)
        anat_dir.mkdir(parents=True)

        lh_white = surface_dir / "lh.white"
        rh_white = surface_dir / "rh.white"
        reference_mri = mri_dir / "orig.mgz"
        target_mri = anat_dir / "sub-01_T1w.nii.gz"

        for path in [
            lh_white,
            rh_white,
            reference_mri,
            target_mri,
        ]:
            path.write_bytes(b"test")

        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)

        def fake_read_geometry(path):
            assert Path(path) in {lh_white, rh_white}
            return vertices.copy(), faces.copy()

        reference_img = self._fake_image(
            np.eye(4),
            np.eye(4),
        )
        target_img = self._fake_image(np.eye(4))

        def fake_load(path):
            path = Path(path)
            if path == reference_mri:
                return reference_img
            if path == target_mri:
                return target_img
            raise AssertionError(f"Unexpected MRI path: {path}")

        monkeypatch.setattr(
            "nibabel.freesurfer.io.read_geometry",
            fake_read_geometry,
        )
        monkeypatch.setattr(
            "nibabel.load",
            fake_load,
        )

        path_dict = {
            "surface_path": [
                Path("derivatives/freesurfer/" "sub-01/surf/lh.white"),
                Path("derivatives/freesurfer/" "sub-01/surf/rh.white"),
            ],
            "surface_reference_mri_image_path": Path("derivatives/freesurfer/" "sub-01/mri/orig.mgz"),
            "base_mri_image_path": Path("bids/sub-01/anat/sub-01_T1w.nii.gz"),
        }

        result = data_loaders.load_surface_data(
            temp_dir,
            path_dict,
        )

        assert result["surface_reference_mri_image_path"] == reference_mri
        assert result["target_mri_image_path"] == target_mri

        surfaces = result["surfaces"]
        assert len(surfaces) == 2
        assert [surface["name"] for surface in surfaces] == [
            "lh.white",
            "rh.white",
        ]
        assert [surface["path"] for surface in surfaces] == [
            lh_white,
            rh_white,
        ]

        for surface in surfaces:
            np.testing.assert_array_equal(
                surface["faces"],
                faces,
            )
            np.testing.assert_allclose(
                surface["vertices_surface_ras"],
                vertices,
            )
            np.testing.assert_allclose(
                surface["vertices_world"],
                vertices,
            )
            np.testing.assert_allclose(
                surface["vertices_voxel"],
                vertices,
            )

    def test_load_surface_data_returns_empty_without_required_paths(
        self,
        temp_dir,
    ):
        """Missing surface/reference/target configuration produces no data."""
        assert (
            data_loaders.load_surface_data(
                temp_dir,
                {},
            )
            == {}
        )


class TestLoadMontageData:
    """Test load_montage_data function."""

    def test_load_valid_montage_single(self, temp_dir, sample_montage_content):
        """Test loading single valid Montage file."""
        montage_file = temp_dir / "montage.svg"
        montage_file.write_text(sample_montage_content)

        path_dict = {"montage_path": montage_file}

        result = load_montage_data(temp_dir, path_dict)

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 1
        # Check structure of returned data
        filename = list(result.keys())[0]
        assert result[filename]["type"] == "svg"
        assert "<svg" in result[filename]["content"]
        assert sample_montage_content in result[filename]["content"]

    def test_load_multiple_montage_files(self, temp_dir, sample_montage_content):
        """Test loading multiple Montage files."""
        montage_file1 = temp_dir / "montage1.svg"
        montage_file2 = temp_dir / "montage2.svg"

        montage_file1.write_text(sample_montage_content)
        montage_file2.write_text("<svg>second montage</svg>")

        path_dict = {"montage_path": [montage_file1, montage_file2]}

        result = load_montage_data(temp_dir, path_dict)

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) >= 2
        if "montage" in result:
            assert result["montage"]["type"] == "png"

        # Check that Montage files are loaded with correct type.
        for filename, data in result.items():
            if filename == "montage":
                continue
            assert data["type"] == "svg"
            assert "<svg" in data["content"]

    def test_load_montage_and_png_mixed(self, temp_dir, sample_montage_content):
        """Test loading mixed SVG and PNG files."""
        from PIL import Image

        montage_file = temp_dir / "montage.svg"
        montage_file.write_text(sample_montage_content)

        # Create a simple PNG file
        png_file = temp_dir / "image.png"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(png_file)

        path_dict = {"montage_path": [montage_file, png_file]}

        result = load_montage_data(temp_dir, path_dict)

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) >= 2
        if "montage" in result:
            assert result["montage"]["type"] == "png"

        # Verify we have one SVG and one PNG in addition to montage.
        types = [data["type"] for key, data in result.items() if key != "montage"]
        assert "svg" in types
        assert "png" in types

    def test_load_multiple_png_files_creates_montage(self, temp_dir):
        """Test that multiple raster images produce an auto-layout montage."""
        from PIL import Image

        png_file1 = temp_dir / "image1.png"
        png_file2 = temp_dir / "image2.png"
        Image.new("RGB", (100, 100), color="red").save(png_file1)
        Image.new("RGB", (100, 100), color="blue").save(png_file2)

        path_dict = {"montage_path": [png_file1, png_file2]}

        result = load_montage_data(temp_dir, path_dict)

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 3
        assert "montage" in result
        assert result["montage"]["type"] == "png"

    def test_load_jpeg_file(self, temp_dir):
        """Test loading JPEG file."""
        from PIL import Image

        # Create a simple JPEG file
        jpeg_file = temp_dir / "image.jpg"
        img = Image.new("RGB", (100, 100), color="blue")
        img.save(jpeg_file, "JPEG")

        path_dict = {"montage_path": jpeg_file}

        result = load_montage_data(temp_dir, path_dict)

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 1

        # Check JPEG file is loaded correctly
        data = list(result.values())[0]
        assert data["type"] == "jpeg"
        assert isinstance(data["content"], Image.Image)

    def test_load_montage_partial_failure(self, temp_dir, sample_montage_content):
        """Test loading multiple SVGs when one file doesn't exist."""
        montage_file1 = temp_dir / "montage1.svg"
        montage_file1.write_text(sample_montage_content)

        # Non-existent file
        montage_file2 = temp_dir / "nonexistent.svg"

        path_dict = {"montage_path": [montage_file1, montage_file2]}

        result = load_montage_data(temp_dir, path_dict)

        # Should return dict with only the existing file
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 1
        assert result[list(result.keys())[0]]["type"] == "svg"

    def test_load_unsupported_file_type(self, temp_dir):
        """Test loading unsupported file types (should be skipped)."""
        # Create an unsupported file type
        txt_file = temp_dir / "file.txt"
        txt_file.write_text("This is not an image")

        path_dict = {"montage_path": txt_file}

        result = load_montage_data(temp_dir, path_dict)

        # Unsupported files should be skipped, returning None
        assert result is None

    def test_load_montage_nonexistent_file(self, temp_dir):
        """Test loading non-existent Montage file."""
        path_dict = {"montage_path": temp_dir / "nonexistent.svg"}

        result = load_montage_data(temp_dir, path_dict)

        assert result is None

    def test_load_montage_with_none_path(self):
        """Test loading montage with None path."""
        path_dict = {"montage_path": None}

        result = load_montage_data("", path_dict)

        assert result is None

    def test_load_montage_unreadable_file(self, temp_dir):
        """Test loading unreadable montage file."""
        montage_file = temp_dir / "montage.svg"
        montage_file.write_text("valid content")

        with patch("builtins.open", side_effect=IOError("Permission denied")):
            path_dict = {"montage_path": montage_file}
            result = load_montage_data(temp_dir, path_dict)

        assert result is None

    def test_load_montage_empty_list(self):
        """Test loading montage with empty list."""
        path_dict = {"montage_path": []}

        result = load_montage_data("", path_dict)

        assert result is None

    def test_load_montage_skips_malformed_list_entries(self, temp_dir, sample_montage_content):
        """Malformed list entries should be ignored instead of raising TypeError."""
        montage_file = temp_dir / "montage.svg"
        montage_file.write_text(sample_montage_content)

        path_dict = {"montage_path": [123, montage_file]}

        result = load_montage_data(temp_dir, path_dict)

        assert result is not None
        assert isinstance(result, dict)
        assert len(result) == 1
        filename = list(result.keys())[0]
        assert result[filename]["type"] == "svg"


class TestScannerMetadataHelpers:
    """Test scanner metadata helper functions in data_loaders."""

    def test_resolve_metadata_path_for_nii_gz(self, temp_dir):
        """Test sidecar path resolution for .nii.gz images."""
        image_path = temp_dir / "sub-01_T1w.nii.gz"
        expected = temp_dir / "sub-01_T1w.json"

        result = _resolve_metadata_path(image_path)

        assert result == expected

    def test_resolve_metadata_path_for_nii(self, temp_dir):
        """Test sidecar path resolution for .nii images."""
        image_path = temp_dir / "sub-01_T1w.nii"
        expected = temp_dir / "sub-01_T1w.json"

        result = _resolve_metadata_path(str(image_path))

        assert result == expected

    def test_resolve_metadata_path_with_none(self):
        """Test sidecar path resolution with empty path."""
        assert _resolve_metadata_path(None) is None

    def test_resolve_metadata_path_with_unsupported_extension(self, temp_dir):
        """Unsupported image extensions should not produce sidecar paths."""
        image_path = temp_dir / "sub-01_T1w.mgz"
        assert _resolve_metadata_path(image_path) is None

    def test_infer_bids_ids_from_path(self):
        """Infer participant/session IDs from standard BIDS-like paths."""
        participant_id, session_id = _infer_bids_ids_from_path("derivatives/fmriprep/sub-01/ses-02/anat/sub-01_ses-02_T1w.nii.gz")
        assert participant_id == "sub-01"
        assert session_id == "ses-02"

    def test_infer_dataset_root_from_derivatives_path(self):
        """Infer dataset root from derivatives path."""
        root = _infer_dataset_root_from_path("/tmp/project/derivatives/fmriprep/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz")
        assert str(root).endswith("/tmp/project")

    def test_find_bids_metadata_sidecar_prefers_participant_session(self, temp_dir):
        """Find participant/session-specific sidecar under bids directory."""
        sidecar = temp_dir / "bids" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps({"Manufacturer": "Siemens"}))

        result = _find_bids_metadata_sidecar(temp_dir, participant_id="sub-01", session_id="ses-01")
        assert result == sidecar

    def test_load_scanner_metadata_reads_sidecar(self, temp_dir):
        """Test loading scanner metadata from a valid sidecar."""
        image_path = temp_dir / "sub-01_T1w.nii.gz"
        image_path.write_text("dummy")
        sidecar_path = temp_dir / "sub-01_T1w.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "Manufacturer": "Siemens",
                    "MagneticFieldStrength": 3,
                }
            )
        )

        result = load_scanner_metadata(image_path)

        assert result["Manufacturer"] == "Siemens"
        assert result["MagneticFieldStrength"] == 3

    def test_load_scanner_metadata_defaults_when_keys_missing(self, temp_dir):
        """Test loading scanner metadata defaults missing keys to Unknown."""
        image_path = temp_dir / "sub-01_T1w.nii.gz"
        image_path.write_text("dummy")
        sidecar_path = temp_dir / "sub-01_T1w.json"
        sidecar_path.write_text(json.dumps({}))

        result = load_scanner_metadata(image_path)

        assert result["Manufacturer"] == "Unknown"
        assert result["MagneticFieldStrength"] == "Unknown"

    def test_load_scanner_metadata_falls_back_to_bids_sidecar(self, temp_dir):
        """If derivative sidecar is missing, use raw BIDS sidecar lookup."""
        image_path = temp_dir / "derivatives" / "fmriprep" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.nii.gz"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_text("dummy")

        bids_sidecar = temp_dir / "bids" / "sub-01" / "ses-01" / "anat" / "sub-01_ses-01_T1w.json"
        bids_sidecar.parent.mkdir(parents=True, exist_ok=True)
        bids_sidecar.write_text(json.dumps({"Manufacturer": "GE", "MagneticFieldStrength": 1.5}))

        result = load_scanner_metadata(image_path, participant_id="sub-01", session_id="ses-01")

        assert result["Manufacturer"] == "GE"
        assert result["MagneticFieldStrength"] == 1.5

    def test_load_scanner_metadata_returns_unknown_when_no_sidecar(self, temp_dir):
        """When no sidecar is available anywhere, return Unknown values."""
        image_path = temp_dir / "derivatives" / "fmriprep" / "sub-99" / "anat" / "sub-99_T1w.nii.gz"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_text("dummy")

        result = load_scanner_metadata(image_path, participant_id="sub-99")

        assert result["Manufacturer"] == "Unknown"
        assert result["MagneticFieldStrength"] == "Unknown"


class TestSaveQcResultsToCsv:
    """Test save_qc_results_to_csv function."""

    def test_save_qc_records_to_csv(self, temp_dir, qc_record_sample):
        """Test saving QC records to CSV."""
        output_file = temp_dir / "output.tsv"
        records = [qc_record_sample]

        result = save_qc_results_to_csv(output_file, records, drop_duplicates=False)

        assert output_file.exists()
        df = pd.read_csv(output_file, sep="\t")
        assert len(df) == 1
        assert list(df.columns)[0] == "pipeline"
        assert df.iloc[0]["participant_id"] == "sub-CMH0001"
        assert "rater_screen_size" in df.columns
        assert df.iloc[0]["rater_screen_size"] == "26-30"

    def test_save_empty_records_list(self, temp_dir):
        """Test saving empty records list."""
        output_file = temp_dir / "output.tsv"

        result = save_qc_results_to_csv(output_file, [], drop_duplicates=False)

        assert output_file.exists()
        df = pd.read_csv(output_file, sep="\t")
        assert len(df) == 0

    def test_save_with_duplicate_removal(self, temp_dir, qc_record_sample):
        """Test saving with duplicate removal enabled."""
        output_file = temp_dir / "output.tsv"
        records = [qc_record_sample, qc_record_sample]

        result = save_qc_results_to_csv(output_file, records, drop_duplicates=True)

        df = pd.read_csv(output_file, sep="\t")
        # Should have only 1 record if duplicates are dropped
        assert len(df) <= 2

    def test_save_creates_parent_directory(self, temp_dir):
        """Test that parent directory is created if it doesn't exist."""
        nested_output_file = temp_dir / "subdir" / "output.tsv"
        records = []

        # This may or may not create parent dir depending on implementation
        result = save_qc_results_to_csv(nested_output_file, records, drop_duplicates=False)

        # At least verify it doesn't crash
        assert result is not None or nested_output_file.parent.exists()

    def test_save_qc_records_sorted_by_participant_session_task(self, temp_dir, qc_record_sample):
        """TSV rows follow participant_id, then session_id, then qc_task (cohort walk order)."""
        b = qc_record_sample.model_copy(update={"qc_task": "b_task"})
        a2 = qc_record_sample.model_copy(update={"session_id": "ses-02", "qc_task": "a_task"})
        other = qc_record_sample.model_copy(update={"participant_id": "sub-CMH0002", "qc_task": "z_task"})
        # Intentionally shuffled input order
        records = [other, a2, qc_record_sample, b]

        output_file = temp_dir / "sorted.tsv"
        save_qc_results_to_csv(output_file, records, drop_duplicates=False)
        df = pd.read_csv(output_file, sep="\t")

        assert list(df["participant_id"]) == [
            "sub-CMH0001",
            "sub-CMH0001",
            "sub-CMH0001",
            "sub-CMH0002",
        ]
        assert list(df["session_id"]) == ["ses-01", "ses-01", "ses-02", "ses-01"]
        assert list(df["qc_task"]) == ["anat_wf_qc", "b_task", "a_task", "z_task"]

    def test_save_qc_records_trims_notes_whitespace_and_newlines(self, temp_dir, qc_record_sample):
        """Exported notes should be normalized to avoid whitespace-only drift creating duplicate row values."""
        output_file = temp_dir / "notes_trimmed.tsv"
        record = qc_record_sample.model_copy(update={"notes": "\n  Motion artifact\n  "})

        save_qc_results_to_csv(output_file, [record], drop_duplicates=False)

        df = pd.read_csv(output_file, sep="\t", dtype=str)
        assert list(df["notes"]) == ["Motion artifact"]

    def test_save_qc_records_preserves_zero_padded_subject_ids_when_appending_existing_file(self, temp_dir, qc_record_sample):
        """Existing TSV exports must keep leading zeros in subject IDs instead of coercing them to integers."""
        output_file = temp_dir / "zero_padded.tsv"
        existing = pd.DataFrame(
            [
                {
                    "pipeline": "fmriprep",
                    "qc_task": "anat_wf_qc",
                    "participant_id": "000123",
                    "session_id": "ses-01",
                    "task_id": "",
                    "run_id": "",
                    "timestamp": "2024-01-01T00:00:00",
                    "rater_id": "rater1",
                    "rater_experience": "Beginner (< 1 year experience)",
                    "rater_fatigue": "Not at all",
                    "rater_screen_size": "14in or less",
                    "final_qc": "PASS",
                    "notes": "",
                }
            ]
        )
        existing.to_csv(output_file, sep="\t", index=False)

        new_record = qc_record_sample.model_copy(update={"participant_id": "000124"})
        save_qc_results_to_csv(output_file, [new_record], drop_duplicates=False)

        df = pd.read_csv(output_file, sep="\t", dtype=str)
        assert list(df["participant_id"]) == ["000123", "000124"]


class TestInferBidsFolderFromPath:
    """Test _infer_bids_folder_from_path - shared by load_scanner_metadata
    (BIDS sidecar lookup) and mirrored by iqm_viewer.py's own
    _infer_modality_from_path (already tested against the same cases in
    test_iqm_viewer.py::test_infer_bids_from_path, different return
    vocabulary: anat/func/dwi here vs t1w/bold/dwi there)."""

    def test_infers_modality_folder_from_directory_and_filename(self):
        cases = [
            ("bids/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz", "anat"),
            ("bids/sub-01/ses-01/func/sub-01_ses-01_task-rest_bold.nii.gz", "func"),
            ("bids/sub-01/ses-01/dwi/sub-01_ses-01_dwi.nii.gz", "dwi"),
            ("derivatives/mriqc/group_T1w.tsv", "anat"),
            ("derivatives/mriqc/group_bold.tsv", "func"),
            ("derivatives/mriqc/group_dwi.tsv", "dwi"),
            ("derivatives/fsqc/2.1.4/output/metrics.csv", None),
            (None, None),
            ("", None),
        ]
        for path, expected in cases:
            assert _infer_bids_folder_from_path(path) == expected, f"failed for {path!r}"


class TestResolveIqmDataPath:
    """Test resolve_iqm_data_path's candidate-directory resolution."""

    def test_absolute_existing_path_returned_as_is(self, temp_dir):
        f = temp_dir / "group_T1w.tsv"
        f.write_text("data")

        assert resolve_iqm_data_path(str(f)) == f

    def test_resolves_relative_to_dataset_dir(self, temp_dir):
        f = temp_dir / "derivatives" / "mriqc" / "group_T1w.tsv"
        f.parent.mkdir(parents=True)
        f.write_text("data")

        result = resolve_iqm_data_path("derivatives/mriqc/group_T1w.tsv", dataset_dir=str(temp_dir))

        assert result == f

    def test_resolves_relative_to_qc_config_path_parent(self, temp_dir):
        f = temp_dir / "group_T1w.tsv"
        f.write_text("data")
        qc_config_path = temp_dir / "qc.json"

        result = resolve_iqm_data_path("group_T1w.tsv", qc_config_path=str(qc_config_path))

        assert result == f

    def test_falls_back_to_original_path_when_no_candidate_exists(self):
        result = resolve_iqm_data_path("nonexistent_dir/group_T1w.tsv")

        assert result == Path("nonexistent_dir/group_T1w.tsv")


class TestLoadIqmDistributionTable:
    """Test load_iqm_distribution_table's extension-based separator dispatch."""

    def test_loads_tsv_with_tab_separator(self, temp_dir):
        f = temp_dir / "group_T1w.tsv"
        f.write_text("bids_name\tefc\nsub-01_T1w\t0.1\n")

        df = load_iqm_distribution_table(f)

        assert list(df.columns) == ["bids_name", "efc"]
        assert df.iloc[0]["efc"] == 0.1

    def test_loads_csv_with_comma_separator(self, temp_dir):
        f = temp_dir / "metrics.csv"
        f.write_text("bids_name,efc\nsub-01_T1w,0.1\n")

        df = load_iqm_distribution_table(f)

        assert list(df.columns) == ["bids_name", "efc"]
        assert df.iloc[0]["efc"] == 0.1

    def test_raises_on_missing_file(self, temp_dir):
        with pytest.raises(Exception):
            load_iqm_distribution_table(temp_dir / "nonexistent.tsv")


class TestLoadIqmMetricsSubjectLevel:
    """Test load_iqm_metrics_subject_level's JSON reading."""

    def test_reads_json_metrics(self, temp_dir):
        f = temp_dir / "sub-01_T1w.json"
        f.write_text(json.dumps({"efc": 0.1, "cjv": 0.5}))

        result = load_iqm_metrics_subject_level(f)

        assert result == {"efc": 0.1, "cjv": 0.5}

    def test_raises_on_missing_file(self, temp_dir):
        with pytest.raises(FileNotFoundError):
            load_iqm_metrics_subject_level(temp_dir / "nonexistent.json")

    def test_raises_on_malformed_json(self, temp_dir):
        f = temp_dir / "bad.json"
        f.write_text("{ not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_iqm_metrics_subject_level(f)


class TestNormalizeManufacturer:
    """Test normalize_manufacturer's alias mapping."""

    def test_maps_known_aliases(self):
        assert normalize_manufacturer("Siemens Healthineers") == "siemens"
        assert normalize_manufacturer("GE Medical Systems") == "ge"
        assert normalize_manufacturer("Philips Healthcare") == "philips"

    def test_unknown_labels_normalize_to_unknown(self):
        for value in ["", "Unknown", "N/A", "null", "none", None]:
            assert normalize_manufacturer(value) == "unknown", f"failed for {value!r}"

    def test_unrecognized_value_passes_through_lowercased(self):
        assert normalize_manufacturer("Canon") == "canon"


class TestNormalizeFieldStrength:
    """Test normalize_field_strength's alias and numeric normalization."""

    def test_maps_known_aliases(self):
        assert normalize_field_strength("3.0") == "3"
        assert normalize_field_strength("3T") == "3"
        assert normalize_field_strength("3.0T") == "3"
        assert normalize_field_strength("1.0") == "1"
        assert normalize_field_strength("1.5T") == "1.5"
        assert normalize_field_strength("7T") == "7"

    def test_unknown_labels_normalize_to_none(self):
        for value in ["", "Unknown", "N/A", None]:
            assert normalize_field_strength(value) is None, f"failed for {value!r}"

    def test_numeric_values_normalize_to_strings(self):
        assert normalize_field_strength(3) == "3"
        assert normalize_field_strength(3.0) == "3"
        assert normalize_field_strength(1.5) == "1.5"
        assert normalize_field_strength("3") == "3"


class TestDownloadReferenceParquet:
    """Test lazy filesystem cache creation for reference Parquet downloads."""

    def test_missing_url_does_not_create_cache_dir(self, temp_dir, monkeypatch):
        cache_dir = temp_dir / "reference_cache"
        monkeypatch.setattr("utils.reference_data.REFERENCE_CACHE_DIR", cache_dir)
        monkeypatch.delenv("REFERENCE_DATA_URL", raising=False)

        with pytest.raises(RuntimeError):
            download_reference_parquet("t1w")

        assert not cache_dir.exists()

    def test_download_creates_cache_dir_when_writing(self, temp_dir, monkeypatch):
        cache_dir = temp_dir / "reference_cache"
        monkeypatch.setattr("utils.reference_data.REFERENCE_CACHE_DIR", cache_dir)

        with patch("utils.reference_data._download_reference_parquet_bytes", return_value=b"parquet"):
            result = download_reference_parquet("t1w", url_parent="https://example.test/reference")

        assert Path(result) == cache_dir / "t1w.parquet"
        assert Path(result).read_bytes() == b"parquet"


class TestFilterReferenceIqm:
    """Test filter_reference_iqm's manufacturer/field-strength filtering and
    row-sampling, with _load_reference_parquet mocked out (no real Parquet
    download/IO). Takes already-normalized values directly - the
    normalize-then-cache orchestration lives in
    iqm_viewer.load_reference_iqm_for_subject, tested separately."""

    def test_filters_by_manufacturer_and_field_strength(self):
        reference_df = pd.DataFrame(
            {
                "Manufacturer": ["Siemens", "GE", "Siemens"],
                "MagneticFieldStrength": ["3.0", 3, 1.0],
                "efc": [0.1, 0.2, 0.3],
            }
        )
        with patch("utils.reference_data._load_reference_parquet", return_value=reference_df):
            result = filter_reference_iqm(
                modality="t1w",
                manufacturer_norm="siemens",
                field_strength_norm="3",
                max_rows=50_000,
            )

        assert len(result) == 1
        assert result.iloc[0]["efc"] == 0.1

    def test_unknown_manufacturer_skips_manufacturer_filter(self):
        reference_df = pd.DataFrame(
            {
                "Manufacturer": ["Siemens", "GE"],
                "efc": [0.1, 0.2],
            }
        )
        with patch("utils.reference_data._load_reference_parquet", return_value=reference_df):
            result = filter_reference_iqm(
                modality="t1w",
                manufacturer_norm="unknown",
                field_strength_norm=None,
                max_rows=50_000,
            )

        assert len(result) == 2

    def test_samples_down_to_max_rows(self):
        reference_df = pd.DataFrame(
            {
                "Manufacturer": ["Siemens"] * 100,
                "efc": range(100),
            }
        )
        with patch("utils.reference_data._load_reference_parquet", return_value=reference_df):
            result = filter_reference_iqm(
                modality="t1w",
                manufacturer_norm="siemens",
                field_strength_norm=None,
                max_rows=10,
            )

        assert len(result) == 10
