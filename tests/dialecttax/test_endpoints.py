"""Tests for dialecttax.endpoints."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dialecttax.endpoints import (
    _generate_async,
    _post,
    generate,
    get_completions,
    get_message,
)


###########
# HELPERS #
###########

def _make_response(content: str) -> dict:
    """Build a canned OpenRouter API response."""
    return {"choices": [{"message": {"content": content}}]}


def _mock_context_manager(resp_mock):
    """Wrap a response mock in an async context manager."""
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=resp_mock)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _make_resp_mock(status: int, json_data: dict | None = None, text: str = ""):
    """Create a mock aiohttp response."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    resp.text = AsyncMock(return_value=text)
    return resp


def _mock_session(post_side_effect):
    """Patch aiohttp.ClientSession to use a given post side effect."""
    session_instance = AsyncMock()
    session_instance.post = MagicMock(side_effect=post_side_effect)
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=session_instance)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


##################
# GET_COMPLETIONS #
##################

class TestGetCompletions:
    def test_extracts_content(self):
        responses = [_make_response("a"), _make_response("b")]
        assert get_completions(responses) == ["a", "b"]

    def test_missing_choices(self):
        assert get_completions([{}]) == [None]

    def test_empty_choices(self):
        assert get_completions([{"choices": []}]) == [None]

    def test_none_response(self):
        assert get_completions([None]) == [None]


###############
# GET_MESSAGE #
###############

class TestGetMessage:
    def test_user_only(self):
        msgs = get_message("hello", instruct=False)
        assert msgs == [{"role": "user", "content": "hello"}]

    def test_with_system(self):
        msgs = get_message("hello", system="you are helpful", instruct=False)
        assert msgs == [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]

    def test_instruct_strips_whitespace(self):
        msgs = get_message("  hello  ", system="  sys  ")
        assert msgs[0]["content"] == "sys"
        assert msgs[1]["content"] == "hello"


########
# POST #
########

class TestPost:
    def test_success(self):
        data = _make_response("hello")
        resp = _make_resp_mock(200, json_data=data)
        session = MagicMock()
        session.post = MagicMock(return_value=_mock_context_manager(resp))
        semaphore = asyncio.Semaphore(1)

        result = asyncio.run(
            _post(session, "http://test", {}, {}, semaphore)
        )
        assert result == data

    def test_retry_on_429(self):
        rate_limited = _make_resp_mock(429)
        success_data = _make_response("ok")
        success = _make_resp_mock(200, json_data=success_data)

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_context_manager(rate_limited)
            return _mock_context_manager(success)

        session = MagicMock()
        session.post = MagicMock(side_effect=side_effect)
        semaphore = asyncio.Semaphore(1)

        with patch("dialecttax.endpoints.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                _post(session, "http://test", {}, {}, semaphore)
            )
        assert result == success_data
        assert call_count == 2

    def test_retry_on_500(self):
        server_error = _make_resp_mock(500)
        success_data = _make_response("recovered")
        success = _make_resp_mock(200, json_data=success_data)

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_context_manager(server_error)
            return _mock_context_manager(success)

        session = MagicMock()
        session.post = MagicMock(side_effect=side_effect)
        semaphore = asyncio.Semaphore(1)

        with patch("dialecttax.endpoints.asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(
                _post(session, "http://test", {}, {}, semaphore)
            )
        assert result == success_data

    def test_non_retryable_error(self):
        resp = _make_resp_mock(400, text="bad request")
        session = MagicMock()
        session.post = MagicMock(return_value=_mock_context_manager(resp))
        semaphore = asyncio.Semaphore(1)

        with pytest.raises(RuntimeError, match="400"):
            asyncio.run(
                _post(session, "http://test", {}, {}, semaphore)
            )

    def test_persistent_failure(self):
        session = MagicMock()
        session.post = MagicMock(
            side_effect=lambda *a, **kw: _mock_context_manager(
                _make_resp_mock(429)
            )
        )
        semaphore = asyncio.Semaphore(1)

        with patch("dialecttax.endpoints.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="after 3 retries"):
                asyncio.run(
                    _post(session, "http://test", {}, {}, semaphore)
                )


############
# GENERATE #
############

def _generate_post_side_effect(responses):
    """Return a post side_effect that maps prompt_N -> responses[N]."""
    def side_effect(*args, **kwargs):
        payload = kwargs.get("json", args[2] if len(args) > 2 else {})
        # Find user message content to determine index
        for msg in payload["messages"]:
            if msg["role"] == "user":
                idx = int(msg["content"].split("_")[1])
                return _mock_context_manager(
                    _make_resp_mock(200, json_data=responses[idx])
                )
    return side_effect


class TestGenerate:
    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_returns_dicts_in_order(self, mock_cls):
        """Returns full response dicts in message order."""
        responses = [_make_response(f"answer_{i}") for i in range(3)]
        mock_cls.return_value = _mock_session(
            _generate_post_side_effect(responses)
        )

        messages = [[{"role": "user", "content": f"prompt_{i}"}] for i in range(3)]
        result = generate(messages, "test-key", max_workers=2)

        assert len(result) == 3
        for i in range(3):
            assert result[i] == responses[i]

    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_single_message(self, mock_cls):
        resp_data = _make_response("42")
        mock_cls.return_value = _mock_session(
            lambda *a, **kw: _mock_context_manager(
                _make_resp_mock(200, json_data=resp_data)
            )
        )

        result = generate(
            [[{"role": "user", "content": "prompt_0"}]], "test-key"
        )
        assert len(result) == 1
        assert result[0] == resp_data

    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_save_to_path(self, mock_cls, tmp_path):
        """All responses saved as JSONL to path_save."""
        n = 5
        responses = [_make_response(f"ans_{i}") for i in range(n)]
        mock_cls.return_value = _mock_session(
            _generate_post_side_effect(responses)
        )

        path = str(tmp_path / "out.jsonl")
        messages = [[{"role": "user", "content": f"prompt_{i}"}] for i in range(n)]
        generate(messages, "test-key", path_save=path)

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == n
        for i, line in enumerate(lines):
            saved = json.loads(line)
            assert saved["choices"][0]["message"]["content"] == f"ans_{i}"

    @patch("dialecttax.endpoints._post")
    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_save_in_index_order_despite_reverse_completion(
        self, mock_session_cls, mock_post, tmp_path
    ):
        """Responses saved in index order even when completing in reverse."""
        n = 5
        responses = [_make_response(f"ans_{i}") for i in range(n)]

        async def delayed_post(session, url, headers, payload, semaphore):
            for msg in payload["messages"]:
                if msg["role"] == "user":
                    idx = int(msg["content"].split("_")[1])
                    await asyncio.sleep((n - idx) * 0.01)
                    return responses[idx]

        mock_post.side_effect = delayed_post
        session_cm = AsyncMock()
        session_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
        session_cm.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = session_cm

        path = str(tmp_path / "out.jsonl")
        messages = [
            [{"role": "user", "content": f"prompt_{i}"}] for i in range(n)
        ]
        result = generate(messages, "test-key", path_save=path, max_workers=n)

        # Returned list in input order
        for i in range(n):
            assert result[i] == responses[i]

        # File in index order
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == n
        for i, line in enumerate(lines):
            saved = json.loads(line)
            assert saved["choices"][0]["message"]["content"] == f"ans_{i}"

    def test_flush_buffers_until_contiguous(self, tmp_path):
        """Flush only writes the contiguous prefix; gaps are buffered."""
        n = 3
        responses = [_make_response(f"ans_{i}") for i in range(n)]
        completion_order = []

        # idx=0 completes last so flush must wait
        delays = {0: 0.04, 1: 0.02, 2: 0.0}

        async def delayed_post(session, url, headers, payload, semaphore):
            for msg in payload["messages"]:
                if msg["role"] == "user":
                    idx = int(msg["content"].split("_")[1])
                    await asyncio.sleep(delays[idx])
                    completion_order.append(idx)
                    return responses[idx]

        headers = {"Authorization": "Bearer test"}
        messages = [
            [{"role": "user", "content": f"prompt_{i}"}] for i in range(n)
        ]
        path = str(tmp_path / "out.jsonl")

        with patch("dialecttax.endpoints._post", side_effect=delayed_post):
            results = asyncio.run(
                _generate_async(
                    messages, "test-model", headers,
                    max_tokens_new=32, max_tokens_reasoning=None,
                    reasoning_effort=None, temperature=0.0,
                    max_workers=n, save_every=1, path_save=path,
                )
            )

        # Verify completion was out of order
        assert completion_order == [2, 1, 0]

        # But file is in index order
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == n
        for i, line in enumerate(lines):
            saved = json.loads(line)
            assert saved["choices"][0]["message"]["content"] == f"ans_{i}"

    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_no_save_without_path(self, mock_cls, tmp_path):
        """No file created when path_save is None."""
        responses = [_make_response("ans")]
        mock_cls.return_value = _mock_session(
            lambda *a, **kw: _mock_context_manager(
                _make_resp_mock(200, json_data=responses[0])
            )
        )

        messages = [[{"role": "user", "content": "prompt_0"}]]
        result = generate(messages, "test-key", path_save=None)

        assert len(result) == 1
        assert len(list(tmp_path.iterdir())) == 0

    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_reasoning_payload(self, mock_cls):
        """max_tokens_reasoning adds reasoning field to payload."""
        payloads = []

        def capture_post(*args, **kwargs):
            payload = kwargs.get("json", args[2] if len(args) > 2 else {})
            payloads.append(payload)
            return _mock_context_manager(
                _make_resp_mock(200, json_data=_make_response("ok"))
            )

        mock_cls.return_value = _mock_session(capture_post)

        messages = [[{"role": "user", "content": "prompt_0"}]]
        generate(messages, "test-key", max_tokens_reasoning=512)

        assert len(payloads) == 1
        assert payloads[0]["reasoning"]["max_tokens"] == 512
        assert payloads[0]["reasoning"]["thinking_budget"] == 512

    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_no_reasoning_payload_when_none(self, mock_cls):
        """No reasoning field when max_tokens_reasoning is None."""
        payloads = []

        def capture_post(*args, **kwargs):
            payload = kwargs.get("json", args[2] if len(args) > 2 else {})
            payloads.append(payload)
            return _mock_context_manager(
                _make_resp_mock(200, json_data=_make_response("ok"))
            )

        mock_cls.return_value = _mock_session(capture_post)

        messages = [[{"role": "user", "content": "prompt_0"}]]
        generate(messages, "test-key", max_tokens_reasoning=None)

        assert "reasoning" not in payloads[0]


###############
# TMP / IDX  #
###############

class TestGenerateTmpIdx:
    """Tests for the .tmp / .idx crash-recovery pattern in generate()."""

    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_tmp_and_idx_removed_on_success(self, mock_cls, tmp_path):
        """Temp files are cleaned up after successful generation."""
        resp = _make_response("ans")
        mock_cls.return_value = _mock_session(
            lambda *a, **kw: _mock_context_manager(
                _make_resp_mock(200, json_data=resp)
            )
        )

        path = str(tmp_path / "out.jsonl")
        generate(
            [[{"role": "user", "content": "prompt_0"}]],
            "test-key",
            path_save=path,
        )

        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        assert not os.path.exists(path + ".idx")

    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_final_file_has_sequential_idx(self, mock_cls, tmp_path):
        """Reconstructed file includes _idx = 0, 1, 2, ... by default."""
        n = 3
        responses = [_make_response(f"ans_{i}") for i in range(n)]
        mock_cls.return_value = _mock_session(
            _generate_post_side_effect(responses)
        )

        path = str(tmp_path / "out.jsonl")
        messages = [
            [{"role": "user", "content": f"prompt_{i}"}] for i in range(n)
        ]
        generate(messages, "test-key", path_save=path)

        with open(path) as f:
            for i, line in enumerate(f):
                assert json.loads(line)["_idx"] == i

    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_custom_save_indices_in_final_file(self, mock_cls, tmp_path):
        """Custom save_indices appear in the final file's _idx fields."""
        n = 3
        indices = [5, 10, 15]
        responses = [_make_response(f"ans_{i}") for i in range(n)]
        mock_cls.return_value = _mock_session(
            _generate_post_side_effect(responses)
        )

        path = str(tmp_path / "out.jsonl")
        messages = [
            [{"role": "user", "content": f"prompt_{i}"}] for i in range(n)
        ]
        generate(messages, "test-key", path_save=path, save_indices=indices)

        with open(path) as f:
            for line, expected_idx in zip(f, indices):
                assert json.loads(line)["_idx"] == expected_idx

        assert not os.path.exists(path + ".tmp")
        assert not os.path.exists(path + ".idx")

    @patch("dialecttax.endpoints.MAX_ERRORS", 1)
    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_idx_survives_generation_failure(self, mock_cls, tmp_path):
        """The .idx sidecar persists when generation aborts."""
        mock_cls.return_value = _mock_session(
            lambda *a, **kw: _mock_context_manager(
                _make_resp_mock(
                    200, json_data={"error": {"message": "fail"}}
                )
            )
        )

        path = str(tmp_path / "out.jsonl")
        with pytest.raises(RuntimeError, match="Aborting"):
            generate(
                [[{"role": "user", "content": "prompt_0"}]],
                "test-key",
                path_save=path,
            )

        assert os.path.exists(path + ".idx")
        with open(path + ".idx") as f:
            assert json.load(f) == [0]
        assert not os.path.exists(path)

    @patch("dialecttax.endpoints.MAX_ERRORS", 1)
    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_custom_save_indices_in_idx_on_failure(
        self, mock_cls, tmp_path
    ):
        """Custom save_indices are written to .idx before generation."""
        mock_cls.return_value = _mock_session(
            lambda *a, **kw: _mock_context_manager(
                _make_resp_mock(
                    200, json_data={"error": {"message": "fail"}}
                )
            )
        )

        path = str(tmp_path / "out.jsonl")
        indices = [42, 99]
        messages = [
            [{"role": "user", "content": f"prompt_{i}"}] for i in range(2)
        ]

        with pytest.raises(RuntimeError, match="Aborting"):
            generate(
                messages, "test-key",
                path_save=path, save_indices=indices,
            )

        with open(path + ".idx") as f:
            assert json.load(f) == indices

    @patch("dialecttax.endpoints.MAX_ERRORS", 1)
    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_partial_results_flushed_to_tmp(self, mock_cls, tmp_path):
        """Partial results are saved to .tmp when generation aborts."""
        def dispatch(*a, **kw):
            payload = kw.get("json", {})
            for msg in payload.get("messages", []):
                if msg["role"] == "user" and "prompt_0" in msg["content"]:
                    return _mock_context_manager(
                        _make_resp_mock(
                            200, json_data=_make_response("ok")
                        )
                    )
            return _mock_context_manager(
                _make_resp_mock(
                    200, json_data={"error": {"message": "fail"}}
                )
            )

        mock_cls.return_value = _mock_session(dispatch)

        path = str(tmp_path / "out.jsonl")
        messages = [
            [{"role": "user", "content": f"prompt_{i}"}] for i in range(2)
        ]

        with pytest.raises(RuntimeError, match="Aborting"):
            generate(
                messages, "test-key",
                path_save=path, max_workers=1,
            )

        tmp_file = path + ".tmp"
        assert os.path.exists(tmp_file)
        with open(tmp_file) as f:
            lines = f.readlines()
        assert len(lines) >= 1

    @patch("dialecttax.endpoints.aiohttp.ClientSession")
    def test_stale_tmp_is_truncated(self, mock_cls, tmp_path):
        """A .tmp file left by a crashed run is replaced, not appended."""
        resp = _make_response("fresh")
        mock_cls.return_value = _mock_session(
            lambda *a, **kw: _mock_context_manager(
                _make_resp_mock(200, json_data=resp)
            )
        )

        path = str(tmp_path / "out.jsonl")
        # Simulate stale .tmp from a previous crash
        with open(path + ".tmp", "w") as f:
            f.write(json.dumps(_make_response("stale")) + "\n")

        generate(
            [[{"role": "user", "content": "prompt_0"}]],
            "test-key",
            path_save=path,
        )

        # Final file has exactly 1 line, not 2
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["choices"][0]["message"]["content"] == "fresh"
