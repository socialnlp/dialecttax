"""Tests for dialecttax.data.multivalue."""

import os

import pytest

from dialecttax.data import multivalue


###############
# CONSTANTS   #
###############


class TestConstants:
    def test_dialects(self):
        """DIALECTS contains all six dialects."""
        assert multivalue.DIALECTS == [
            "sae", "aave", "appalachian", "chicano", "indian", "singapore",
        ]

    def test_directory_name(self):
        """DIRECTORY_NAME is 'multivalue'."""
        assert multivalue.DIRECTORY_NAME == "multivalue"

    def test_file_name_format(self):
        """FILE_NAME_FORMAT uses dialect placeholder."""
        assert multivalue.FILE_NAME_FORMAT == "coqa_{dialect}.txt"

    def test_file_name_format_substitution(self):
        """FILE_NAME_FORMAT produces expected filename."""
        result = multivalue.FILE_NAME_FORMAT.format(dialect="chicano")
        assert result == "coqa_chicano.txt"


################
# LOAD_DATASET #
################


class TestLoadDataset:
    def test_loads_lines_as_dicts(self, tmp_path):
        """Each non-empty line becomes a dict with text and unique_id."""
        data_dir = tmp_path / "multivalue"
        data_dir.mkdir()
        (data_dir / "coqa_sae.txt").write_text("First story\nSecond story\n")

        result = multivalue.load_dataset(
            str(tmp_path), os.path.join("multivalue", "coqa_sae.txt")
        )

        assert len(result) == 2
        assert result[0] == {
            "text": "First story",
            "unique_id": "multivalue-0",
        }
        assert result[1] == {
            "text": "Second story",
            "unique_id": "multivalue-1",
        }

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines are filtered out."""
        data_dir = tmp_path / "multivalue"
        data_dir.mkdir()
        (data_dir / "coqa_aave.txt").write_text("line one\n\n\nline two\n")

        result = multivalue.load_dataset(
            str(tmp_path), os.path.join("multivalue", "coqa_aave.txt")
        )

        assert len(result) == 2
        assert result[0]["text"] == "line one"
        assert result[1]["text"] == "line two"

    def test_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        data_dir = tmp_path / "multivalue"
        data_dir.mkdir()
        (data_dir / "coqa_sae.txt").write_text("")

        result = multivalue.load_dataset(
            str(tmp_path), os.path.join("multivalue", "coqa_sae.txt")
        )

        assert result == []

    def test_unique_ids_are_sequential(self, tmp_path):
        """unique_id values use sequential zero-based indices."""
        data_dir = tmp_path / "multivalue"
        data_dir.mkdir()
        (data_dir / "coqa_singapore.txt").write_text("a\nb\nc\n")

        result = multivalue.load_dataset(
            str(tmp_path), os.path.join("multivalue", "coqa_singapore.txt")
        )

        ids = [r["unique_id"] for r in result]
        assert ids == ["multivalue-0", "multivalue-1", "multivalue-2"]
