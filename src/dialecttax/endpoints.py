"""OpenRouter API client for chat completions."""

import aiohttp
import asyncio
import json
import os
import sys

URL_OPENROUTER_API = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 3
BACKOFF_BASE = 1  # seconds
REQUEST_TIMEOUT = 300  # seconds
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

SAVE_EVERY = 16
MAX_ERRORS = 32


##########
# MODELS #
##########

MODELS = {
    # Close-sourced
    "gpt": {"id_openrouter": "openai/gpt-5-chat", "family": "gpt", "type": "instruct"},
    "gpt_mini": {"id_openrouter": "openai/gpt-5-mini", "family": "gpt", "type": "instruct"},
    # Open-sourced
    "gemma_12b": {"id_openrouter": "google/gemma-3-12b-it", "family": "gemma", "type": "instruct"},
    "gemma_27b": {"id_openrouter": "google/gemma-3-27b-it", "family": "gemma", "type": "instruct"},
    "llama_8b": {"id_openrouter": "meta-llama/llama-3.1-8b-instruct", "family": "llama", "type": "instruct"},
    "llama_70b": {"id_openrouter": "meta-llama/llama-3.3-70b-instruct", "family": "llama", "type": "instruct"},
    "qwen_8b": {"id_openrouter": "qwen/qwen3-8b", "family": "qwen", "type": "instruct"},
    "qwen_27b": {"id_openrouter": "qwen/qwen3.5-27b", "family": "qwen", "type": "instruct"},
    "qwen_32b": {"id_openrouter": "qwen/qwen3-32b", "family": "qwen", "type": "instruct"},
}


############
# MESSAGES #
############

def is_error(data: dict) -> bool:
    """Check if a response is an error or has empty content.

    Args:
        data: A single OpenRouter response dict.

    Returns:
        True if the response has an ``error`` key, empty/None content,
        or is malformed.
    """
    if "error" in data:
        return True
    try:
        content = data["choices"][0]["message"]["content"]
        return content is None or content == ""
    except (KeyError, IndexError, TypeError):
        return True


def get_completions(responses: list[dict]) -> list[str | None]:
    """Extract completion strings from response dicts.

    Args:
        responses: list of full OpenRouter response dicts.

    Returns:
        List of completion strings, or None for failed responses.
    """
    completions = []
    for r in responses:
        try:
            completions.append(r["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            completions.append(None)
    return completions


def get_message(prompt: str, system: str | None = None, instruct: bool = True) -> list[dict]:
    # Remove trailing spaces for instruction-tuned models
    if instruct:
        system = system.strip()
        prompt = prompt.strip()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


############
# REQUESTS #
############

async def _post(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Single POST to OpenRouter with retry on 429/5xx.

    Args:
        session: aiohttp client session.
        url: API endpoint URL.
        headers: request headers.
        payload: JSON request body.
        semaphore: concurrency limiter.

    Returns:
        Parsed JSON response dict.
    """
    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                async with session.post(
                    url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status == 429 or resp.status >= 500:
                        if attempt == MAX_RETRIES - 1:
                            raise RuntimeError(
                                f"OpenRouter request failed after {MAX_RETRIES} retries"
                            )
                    else:
                        text = await resp.text()
                        raise RuntimeError(
                            f"OpenRouter request failed ({resp.status}): {text}"
                        )
            except asyncio.TimeoutError:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"OpenRouter request timed out after {REQUEST_TIMEOUT}s"
                    )
        # Semaphore released — backoff before retry
        delay = BACKOFF_BASE * (2 ** attempt)
        await asyncio.sleep(delay)
    raise RuntimeError(
        f"OpenRouter request failed after {MAX_RETRIES} retries"
    )


async def _generate_async(
    messages: list[list[dict]],
    model: str,
    headers: dict,
    max_tokens_new: int,
    max_tokens_reasoning: int | None,
    reasoning_effort: str | None,
    temperature: float,
    max_workers: int,
    save_every: int,
    path_save: str | None,
) -> list[dict]:
    """Fan out messages across concurrent requests, saving periodically.

    Args:
        messages: list of list of dicts.
        model: OpenRouter model ID.
        headers: request headers with auth.
        max_tokens_new: max output tokens per completion.
        max_tokens_reasoning: max reasoning/thinking tokens (None to omit).
        temperature: sampling temperature.
        max_workers: concurrent request limit.
        save_every: save to path_save every N completed responses.
        path_save: path to append JSONL responses. None to skip saving.

    Returns:
        List of full response dicts in message order.
    """
    semaphore = asyncio.Semaphore(max_workers)
    results = [None] * len(messages)
    n_done = 0
    n_errors = 0
    next_flush_idx = 0

    def _status():
        sys.stdout.write(f"\rPROCESSED {n_done} OF {len(messages)} [ERRORS: {n_errors}]")
        sys.stdout.flush()

    def _flush():
        nonlocal next_flush_idx
        if not path_save:
            return
        to_write = []
        while next_flush_idx < len(results) and results[next_flush_idx] is not None:
            to_write.append(results[next_flush_idx])
            next_flush_idx += 1
        if to_write:
            with open(path_save, "a") as f:
                for r in to_write:
                    f.write(json.dumps(r) + "\n")

    async def _call(idx: int, message: list[dict]):
        nonlocal n_done, n_errors
        payload = {
            "model": model,
            "messages": message,
            "max_tokens": max_tokens_new,
            "temperature": temperature,
        }
        if max_tokens_reasoning is not None or reasoning_effort is not None:
            reasoning = {}
            # Only one of reasoning_effort or max_tokens can be defined
            # We prioritize reasoning_effort
            if reasoning_effort is not None:
                reasoning["effort"] = reasoning_effort
            elif max_tokens_reasoning is not None:
                reasoning["max_tokens"] = max_tokens_reasoning
                reasoning["thinking_budget"] = max_tokens_reasoning

            payload["reasoning"] = reasoning
        try:
            data = await _post(session, URL_OPENROUTER_API, headers, payload, semaphore)
        except RuntimeError as e:
            n_errors += 1
            _status()
            print(f"\n[{idx}] {e}", file=sys.stderr)
            data = {"error": str(e)}
        else:
            if is_error(data):
                n_errors += 1
                print(f"\n[{idx}] Empty or invalid response", file=sys.stderr)

        results[idx] = data
        n_done += 1
        _status()

        if n_errors >= MAX_ERRORS:
            _flush()
            raise RuntimeError(f"Aborting: {MAX_ERRORS} consecutive errors")
        if n_done % save_every == 0:
            _flush()

    async with aiohttp.ClientSession() as session:
        tasks = [_call(i, m) for i, m in enumerate(messages)]
        try:
            await asyncio.gather(*tasks)
        finally:
            _flush()

    print()  # newline after status
    return results


def generate(
    messages: list[list[dict]],
    api_key: str,
    *,
    model: str = "openai/gpt-5-mini",
    max_tokens_new: int = 2048,
    max_tokens_reasoning: int | None = None,
    reasoning_effort: str | None = None,
    temperature: float = 0.0,
    max_workers: int = 16,
    save_every: int = SAVE_EVERY,
    path_save: str | None = None,
    save_indices: list[int] | None = None,
) -> list[dict]:
    """Call OpenRouter chat completions for a list of messages.

    Writes responses to a temporary ``.tmp`` file during generation for
    crash recovery, then reconstructs the final ``path_save`` on success.
    A sidecar ``.idx`` file maps temporary-file line numbers back to
    caller-provided indices (defaulting to ``0 .. N-1``).

    Args:
        messages: list of list of dicts.
        model: OpenRouter model ID.
        max_tokens_new: max output tokens per completion.
        max_tokens_reasoning: max reasoning/thinking tokens (None to omit).
        temperature: sampling temperature.
        max_workers: concurrent request limit.
        save_every: save to path_save every N responses.
        path_save: path to save responses as JSONL. None to skip saving.
        save_indices: caller-defined indices written to the ``.idx``
            sidecar.  Defaults to ``[0, 1, ..., len(messages)-1]``.

    Returns:
        List of full response dicts (same order as messages).
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    tmp_path = None
    idx_path = None
    if path_save:
        tmp_path = path_save + ".tmp"
        idx_path = path_save + ".idx"
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        indices = (
            save_indices if save_indices is not None
            else list(range(len(messages)))
        )
        with open(idx_path, "w") as f:
            json.dump(indices, f)
        print(f"Saving responses to: {path_save}")

    results = asyncio.run(
        _generate_async(
            messages, model, headers,
            max_tokens_new, max_tokens_reasoning,
            reasoning_effort, temperature,
            max_workers, save_every, tmp_path,
        )
    )

    # Reconstruct final file and clean up temp
    if path_save:
        indices = (
            save_indices if save_indices is not None
            else list(range(len(results)))
        )
        with open(path_save, "w") as f:
            for r, idx in zip(results, indices):
                r["_idx"] = idx
                f.write(json.dumps(r) + "\n")
        for p in (tmp_path, idx_path):
            if p and os.path.exists(p):
                os.remove(p)

    return results
