"""Tests for dialecttax.data.graders.algorithm."""

from unittest.mock import MagicMock, patch

import pytest

from dialecttax.data.graders.algorithm import (
    DockerSandbox,
    extract_answer,
    grade,
    grade_completions,
    make_test_function,
    normalize_answer,
)


###########
# EXTRACT #
###########

class TestExtractAnswer:
    def test_simple_function(self):
        text = "def python_function(x):\n    return x + 1\n"
        assert extract_answer(text) == "def python_function(x):\n    return x + 1"

    def test_with_imports(self):
        text = (
            "import math\n"
            "from collections import deque\n"
            "\n"
            "def python_function(x):\n"
            "    return math.sqrt(x)\n"
        )
        result = extract_answer(text)
        assert result.startswith("import math")
        assert "from collections import deque" in result
        assert "def python_function(x):" in result

    def test_stops_at_non_indented_line(self):
        text = (
            "def python_function(x):\n"
            "    return x\n"
            "print('after')\n"
        )
        result = extract_answer(text)
        assert "print" not in result

    def test_none_input(self):
        assert extract_answer(None) is None

    def test_no_function(self):
        assert extract_answer("x = 1\ny = 2\n") is None


#############
# NORMALIZE #
#############

class TestNormalizeAnswer:
    def test_strips_python_fence(self):
        assert normalize_answer("```python\ncode\n```") == "code"

    def test_strips_plain_fence(self):
        assert normalize_answer("```\ncode\n```") == "code"

    def test_no_fence(self):
        assert normalize_answer("  code  ") == "code"


################
# TEST BUILDER #
################

class TestMakeTestFunction:
    def test_generates_try_except_blocks(self):
        result = make_test_function([], ["assert 1 == 1", "assert 2 == 2"])
        assert "assert 1 == 1" in result
        assert "assert 2 == 2" in result
        assert result.count("try:") == 2
        assert "ALL TESTS PASSED" in result

    def test_includes_imports(self):
        result = make_test_function(["import math"], ["assert True"])
        assert "import math" in result

    def test_multiline_block_indented(self):
        """Multi-line test entries have all lines indented inside try."""
        block = "for x in range(3):\n    assert python_function(x) == x"
        result = make_test_function([], [block])
        lines = result.split("\n")
        # Find the try block
        try_idx = next(i for i, l in enumerate(lines) if l == "try:")
        assert lines[try_idx + 1] == "    for x in range(3):"
        assert lines[try_idx + 2] == "        assert python_function(x) == x"

    def test_single_quotes_in_assertion(self):
        """Assertions with single quotes don't break the error print."""
        test = "assert python_function('hello') == 'world'"
        result = make_test_function([], [test])
        # Should produce valid Python — compile it to verify
        compile(result, "<test>", "exec")

    def test_curly_braces_in_assertion(self):
        """Assertions with curly braces don't break the error print."""
        test = "assert python_function([4, {}, 'x']) == [4]"
        result = make_test_function([], [test])
        compile(result, "<test>", "exec")

    def test_double_quotes_in_assertion(self):
        """Assertions with double quotes produce valid Python."""
        test = 'assert python_function("hello") == "world"'
        result = make_test_function([], [test])
        compile(result, "<test>", "exec")

    def test_setup_plus_assert_block(self):
        """A multi-line block with setup + assert is properly indented."""
        block = "lst = list(range(10))\nassert python_function(lst) == 45"
        result = make_test_function([], [block])
        lines = result.split("\n")
        try_idx = next(i for i, l in enumerate(lines) if l == "try:")
        assert lines[try_idx + 1] == "    lst = list(range(10))"
        assert lines[try_idx + 2] == "    assert python_function(lst) == 45"


###########
# SANDBOX #
###########

class TestDockerSandbox:
    """Tests for DockerSandbox using mocked Docker client."""

    def _make_sandbox(self):
        """Create a DockerSandbox with a mocked Docker client."""
        with patch("dialecttax.data.graders.algorithm.docker") as mock_docker:
            mock_client = MagicMock()
            mock_docker.from_env.return_value = mock_client
            mock_container = MagicMock()
            mock_client.containers.run.return_value = mock_container
            sandbox = DockerSandbox(timeout=5)
        return sandbox, mock_container

    def test_run_passing_script(self):
        sandbox, mock_container = self._make_sandbox()
        stdout = b"####################\n# ALL TESTS PASSED #\n####################"
        mock_container.exec_run.return_value = (0, (stdout, b""))

        result = sandbox.run("print('hello')")
        assert result["correct"] is True
        assert result["error"] is None

    def test_run_failing_script(self):
        sandbox, mock_container = self._make_sandbox()
        mock_container.exec_run.return_value = (1, (b"0/1 TESTS PASSED", b"err"))

        result = sandbox.run("raise Exception")
        assert result["correct"] is False
        assert "exited with code: 1" in result["error"]

    def test_run_timeout(self):
        sandbox, mock_container = self._make_sandbox()
        mock_container.exec_run.return_value = (124, (b"", b""))

        result = sandbox.run("import time; time.sleep(999)")
        assert result["correct"] is False
        assert result["error"] == "Timed out"

    def test_run_exception(self):
        sandbox, mock_container = self._make_sandbox()
        mock_container.exec_run.side_effect = RuntimeError("Docker error")

        result = sandbox.run("print(1)")
        assert result["correct"] is False
        assert "Docker error" in result["error"]

    def test_close_removes_container(self):
        sandbox, mock_container = self._make_sandbox()
        sandbox.close()
        mock_container.remove.assert_called_once_with(force=True)

    def test_context_manager(self):
        with patch("dialecttax.data.graders.algorithm.docker") as mock_docker:
            mock_client = MagicMock()
            mock_docker.from_env.return_value = mock_client
            mock_container = MagicMock()
            mock_client.containers.run.return_value = mock_container

            with DockerSandbox() as sb:
                assert sb._container is mock_container
            mock_container.remove.assert_called_once_with(force=True)


#########
# GRADE #
#########

class TestGrade:
    def test_none_code(self):
        result = grade(None, [], [])
        assert result["correct"] is False
        assert result["error"] == "No code extracted"

    def test_uses_sandbox_when_provided(self):
        sandbox = MagicMock()
        sandbox.run.return_value = {
            "correct": True,
            "stdout": "ok",
            "stderr": "",
            "error": None,
        }
        result = grade(
            "def python_function(x):\n    return x",
            [],
            ["assert python_function(1) == 1"],
            sandbox=sandbox,
        )
        assert result["correct"] is True
        sandbox.run.assert_called_once()

    @patch("dialecttax.data.graders.algorithm.run_in_docker")
    def test_falls_back_to_run_in_docker(self, mock_run):
        mock_run.return_value = {
            "correct": True,
            "stdout": "",
            "stderr": "",
            "error": None,
        }
        result = grade(
            "def python_function(x):\n    return x",
            [],
            ["assert python_function(1) == 1"],
        )
        assert result["correct"] is True
        mock_run.assert_called_once()


#####################
# GRADE COMPLETIONS #
#####################

class TestGradeCompletions:
    def _mock_sandbox_cls(self, results_seq):
        """Return a patched DockerSandbox class whose .run() yields
        results from *results_seq* in call order."""
        mock_container = MagicMock()
        mock_client = MagicMock()
        mock_client.containers.run.return_value = mock_container

        call_idx = {"i": 0}
        def fake_run(script):
            idx = call_idx["i"]
            call_idx["i"] += 1
            return results_seq[idx % len(results_seq)]

        mock_container.exec_run.side_effect = None
        return mock_client, mock_container, fake_run

    @patch("dialecttax.data.graders.algorithm.DockerSandbox")
    def test_single_completion(self, MockSandbox):
        mock_sb = MagicMock()
        mock_sb.run.return_value = {
            "correct": True,
            "stdout": "1/1 TESTS PASSED",
            "stderr": "",
            "error": None,
        }
        MockSandbox.return_value = mock_sb

        results = grade_completions(
            "def python_function(x):\n    return x + 1",
            ([], ["assert python_function(1) == 2"]),
        )
        assert len(results) == 1
        assert results[0]["correct"] is True

    @patch("dialecttax.data.graders.algorithm.DockerSandbox")
    def test_multiple_completions_parallel(self, MockSandbox):
        mock_sb = MagicMock()
        mock_sb.run.return_value = {
            "correct": True,
            "stdout": "ok",
            "stderr": "",
            "error": None,
        }
        MockSandbox.return_value = mock_sb

        completions = [
            "def python_function(x):\n    return x + 1",
            "def python_function(x):\n    return x * 2",
            "def python_function(x):\n    return x - 1",
        ]
        gold = [
            ([], ["assert python_function(1) == 2"]),
            ([], ["assert python_function(3) == 6"]),
            ([], ["assert python_function(5) == 4"]),
        ]

        results = grade_completions(completions, gold, max_workers=2)
        assert len(results) == 3
        # Order is preserved
        assert results[0]["completion"] == completions[0]
        assert results[1]["completion"] == completions[1]
        assert results[2]["completion"] == completions[2]

    @patch("dialecttax.data.graders.algorithm.DockerSandbox")
    def test_max_workers_capped_by_completions(self, MockSandbox):
        """When completions < max_workers, only that many sandboxes are created."""
        mock_sb = MagicMock()
        mock_sb.run.return_value = {
            "correct": True,
            "stdout": "",
            "stderr": "",
            "error": None,
        }
        MockSandbox.return_value = mock_sb

        grade_completions(
            ["def python_function(x):\n    return x"],
            [([], ["assert python_function(1) == 1"])],
            max_workers=8,
        )
        # Only 1 sandbox created (min(8, 1) == 1)
        assert MockSandbox.call_count == 1

    @patch("dialecttax.data.graders.algorithm.DockerSandbox")
    def test_sandboxes_closed_on_success(self, MockSandbox):
        mock_sb = MagicMock()
        mock_sb.run.return_value = {
            "correct": True,
            "stdout": "",
            "stderr": "",
            "error": None,
        }
        MockSandbox.return_value = mock_sb

        grade_completions(
            ["def python_function(x):\n    return x"],
            [([], ["assert True"])],
            max_workers=2,
        )
        mock_sb.close.assert_called()

    @patch("dialecttax.data.graders.algorithm.DockerSandbox")
    def test_sandboxes_closed_on_error(self, MockSandbox):
        mock_sb = MagicMock()
        mock_sb.run.side_effect = RuntimeError("boom")
        MockSandbox.return_value = mock_sb

        with pytest.raises(RuntimeError, match="boom"):
            grade_completions(
                ["def python_function(x):\n    return x"],
                [([], ["assert True"])],
            )
        mock_sb.close.assert_called()

    @patch("dialecttax.data.graders.algorithm.DockerSandbox")
    def test_none_extraction_handled(self, MockSandbox):
        mock_sb = MagicMock()
        MockSandbox.return_value = mock_sb

        results = grade_completions(
            ["no function here"],
            [([], ["assert True"])],
        )
        assert len(results) == 1
        assert results[0]["correct"] is False
        assert results[0]["extracted"] is None
        # grade() short-circuits on None, sandbox.run should not be called
        mock_sb.run.assert_not_called()
