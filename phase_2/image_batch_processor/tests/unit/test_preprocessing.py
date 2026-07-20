"""Unit tests for the optional preprocessing subflow integration."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import from the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from exceptions import ConfigurationError
from preprocessing.exceptions import ConfigurationError as PreprocessingConfigurationError
from utils.preprocessing import run_preprocessing_if_needed


def _touch_image(directory: Path, name: str = "000-page.jpg") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"not a real image, just a placeholder")


class TestRunPreprocessingIfNeeded:
    """Tests for run_preprocessing_if_needed."""

    def test_skips_when_output_already_exists(self, tmp_path):
        """If output already has images, the pipeline is not run."""
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        _touch_image(output_dir)

        with patch(
            "utils.preprocessing.PreprocessingPipeline"
        ) as mock_pipeline_cls:
            result = run_preprocessing_if_needed(
                source_dir=str(source_dir),
                output_dir=str(output_dir),
            )

        mock_pipeline_cls.assert_not_called()
        assert result == output_dir

    def test_force_reruns_even_if_output_exists(self, tmp_path):
        """force=True re-runs the pipeline even when output already exists."""
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        _touch_image(output_dir)

        mock_report = MagicMock(successful=0, total_sources=0, total_output_files=0)
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = mock_report

        with patch(
            "utils.preprocessing.PreprocessingPipeline", return_value=mock_pipeline
        ) as mock_pipeline_cls:
            result = run_preprocessing_if_needed(
                source_dir=str(source_dir),
                output_dir=str(output_dir),
                force=True,
            )

        mock_pipeline_cls.assert_called_once()
        mock_pipeline.run.assert_called_once()
        assert result == output_dir

    def test_runs_pipeline_when_output_missing(self, tmp_path):
        """No existing output triggers the preprocessing pipeline run."""
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"  # does not exist yet
        source_dir.mkdir()

        captured_config = {}

        def _fake_pipeline(config, logger=None):
            captured_config["source_dir"] = config.source_dir
            captured_config["output_dir"] = config.output_dir
            mock_pipeline = MagicMock()
            mock_pipeline.run.return_value = MagicMock(
                successful=0, total_sources=0, total_output_files=0
            )
            return mock_pipeline

        with patch(
            "utils.preprocessing.PreprocessingPipeline", side_effect=_fake_pipeline
        ) as mock_pipeline_cls:
            result = run_preprocessing_if_needed(
                source_dir=str(source_dir),
                output_dir=str(output_dir),
            )

        assert mock_pipeline_cls.called
        assert captured_config["source_dir"] == str(source_dir)
        assert captured_config["output_dir"] == str(output_dir)
        assert result == output_dir

    def test_raises_on_preprocessing_configuration_error(self, tmp_path):
        """A ConfigurationError from the preprocessing pipeline is re-raised
        as the batch processor's own ConfigurationError."""
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()

        with patch(
            "utils.preprocessing._build_default_preprocessing_config",
            side_effect=PreprocessingConfigurationError("boom"),
        ):
            with pytest.raises(ConfigurationError):
                run_preprocessing_if_needed(
                    source_dir=str(source_dir),
                    output_dir=str(output_dir),
                )

    def test_restores_previous_env_vars_after_run(self, tmp_path, monkeypatch):
        """The PREPROCESS_* env vars are restored after the run, if they were
        previously set."""
        source_dir = tmp_path / "source"
        output_dir = tmp_path / "output"
        source_dir.mkdir()

        monkeypatch.setenv("PREPROCESS_SOURCE_DIR", "previous-source")
        monkeypatch.setenv("PREPROCESS_OUTPUT_DIR", "previous-output")

        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = MagicMock(
            successful=0, total_sources=0, total_output_files=0
        )

        with patch(
            "utils.preprocessing.PreprocessingPipeline", return_value=mock_pipeline
        ):
            run_preprocessing_if_needed(
                source_dir=str(source_dir),
                output_dir=str(output_dir),
            )

        import os

        assert os.environ["PREPROCESS_SOURCE_DIR"] == "previous-source"
        assert os.environ["PREPROCESS_OUTPUT_DIR"] == "previous-output"
