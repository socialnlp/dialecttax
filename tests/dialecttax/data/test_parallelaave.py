"""Tests for dialecttax.data.parallelaave."""

import os

import pytest

from dialecttax.data import parallelaave


###############
# CONSTANTS   #
###############


class TestConstants:
    def test_dialects(self):
        """DIALECTS contains sae and aave."""
        assert parallelaave.DIALECTS == ["sae", "aave"]

    def test_directory_name(self):
        """DIRECTORY_NAME is 'parallelaave'."""
        assert parallelaave.DIRECTORY_NAME == "parallelaave"

    def test_file_name_format(self):
        """FILE_NAME_FORMAT uses dialect placeholder."""
        assert parallelaave.FILE_NAME_FORMAT == "{dialect}_samples.txt"

    def test_file_name_format_substitution(self):
        """FILE_NAME_FORMAT produces expected filename."""
        result = parallelaave.FILE_NAME_FORMAT.format(dialect="aave")
        assert result == "aave_samples.txt"


################
# LOAD_DATASET #
################


class TestLoadDataset:
    def test_loads_lines_as_dicts(self, tmp_path):
        """Each non-empty line becomes a dict with text and unique_id."""
        data_dir = tmp_path / "parallelaave"
        data_dir.mkdir()
        (data_dir / "sae_samples.txt").write_text("Hello world\nGoodbye world\n")

        result = parallelaave.load_dataset(
            str(tmp_path), os.path.join("parallelaave", "sae_samples.txt")
        )

        assert len(result) == 2
        assert result[0] == {
            "text": "Hello world",
            "unique_id": "parallelaave-0",
        }
        assert result[1] == {
            "text": "Goodbye world",
            "unique_id": "parallelaave-1",
        }

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines are filtered out."""
        data_dir = tmp_path / "parallelaave"
        data_dir.mkdir()
        (data_dir / "aave_samples.txt").write_text("line one\n\n\nline two\n")

        result = parallelaave.load_dataset(
            str(tmp_path), os.path.join("parallelaave", "aave_samples.txt")
        )

        assert len(result) == 2
        assert result[0]["text"] == "line one"
        assert result[1]["text"] == "line two"

    def test_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        data_dir = tmp_path / "parallelaave"
        data_dir.mkdir()
        (data_dir / "sae_samples.txt").write_text("")

        result = parallelaave.load_dataset(
            str(tmp_path), os.path.join("parallelaave", "sae_samples.txt")
        )

        assert result == []

    def test_unique_ids_are_sequential(self, tmp_path):
        """unique_id values use sequential zero-based indices."""
        data_dir = tmp_path / "parallelaave"
        data_dir.mkdir()
        (data_dir / "sae_samples.txt").write_text("a\nb\nc\n")

        result = parallelaave.load_dataset(
            str(tmp_path), os.path.join("parallelaave", "sae_samples.txt")
        )

        ids = [r["unique_id"] for r in result]
        assert ids == ["parallelaave-0", "parallelaave-1", "parallelaave-2"]
