"""
Grading utilities for algorithm questions.

Datasets: HumanEval, MBPP

Extracts a ``python_function`` definition from the model completion,
runs it against unit tests inside a Docker sandbox, and checks whether
all assertions pass.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import docker

_PASS_BANNER = "####################\n# ALL TESTS PASSED #\n####################"
_SANDBOX_LABEL = "dialecttax.sandbox"


def is_refusal(completion: str | None, extracted: str | None) -> bool:
    """Check if a completion is a model refusal."""
    return extracted is None and completion is not None and "I cannot " in completion


########
# DATA #
########

def extract_answer(text: str | None) -> str | None:
    """Extract imports and the python_function definition from a completion."""
    if text is None:
        return None

    # Collect import lines before the function
    imports = re.findall(r"^(?:import .+|from .+ import .+)$", text, re.MULTILINE)

    match = re.search(r"(def python_function\b.*)", text, re.DOTALL)
    if not match:
        return None

    # Take everything from the def to the end of the function (next non-indented line or EOF)
    lines = match.group(0).split("\n")
    func_lines = [lines[0]]
    for line in lines[1:]:
        if line.strip() == "" or line[0:1] in (" ", "\t"):
            func_lines.append(line)
        else:
            break

    parts = imports + [""] + func_lines if imports else func_lines
    return "\n".join(parts).rstrip()


##############
# FORMATTING #
##############

def normalize_answer(response: str) -> str:
    """Returns code stripped of unnecessary wrappers.

    ```, python, etc.
    """
    response = response.strip()

    if response.startswith("```python"):
        response = response[len("```python"):]
    elif response.startswith("```"):
        response = response[len("```"):]

    if response.endswith("```"):
        response = response[:-3]

    return response.strip()


def make_test_function(test_imports: list[str], tests: list[str]) -> str:
    lines = ["import sys", "", *test_imports, "", "results = []"]
    for assertion in tests:
        lines.append("try:")
        indented = "\n".join("    " + line for line in assertion.split("\n"))
        lines.append(indented)
        lines.append("    results.append(True)")
        lines.append("except Exception as e:")
        assertion_print = assertion
        if "\n" in assertion_print:
            assertion_print = assertion_print.split("\n")[1]
        lines.append("    print('FAILED: ' + " + repr(assertion_print) + " + ' [' + e.__class__.__name__ + ': ' + e.__str__() + ']', file=sys.stderr)")
        lines.append("    results.append(False)")
    lines.append("\nprint(f'{sum(results)}/{len(results)} TESTS PASSED')\n")
    lines.append("if sum(results) == len(results):")
    lines.append("    print('\\n####################')")
    lines.append("    print('# ALL TESTS PASSED #')")
    lines.append("    print('####################')")
    return "\n".join(lines)


###########
# SANDBOX #
###########

def _cleanup_stale_sandboxes() -> None:
    """Remove sandbox containers that are not running (crashed/stale)."""
    try:
        client = docker.from_env(timeout=120)
        stale = client.containers.list(
            all=True,
            filters={"label": _SANDBOX_LABEL, "status": "created"},
        )
        stale += client.containers.list(
            all=True,
            filters={"label": _SANDBOX_LABEL, "status": "exited"},
        )
        for c in stale:
            c.remove(force=True)
    except Exception:
        pass


class DockerSandbox:
    """Long-lived Docker container for running multiple scripts.

    Each script runs as a separate ``python -c`` process via
    ``container.exec_run``, so memory is isolated between runs without
    the overhead of creating/removing a container per script.

    Args:
        timeout: Per-script wall-clock timeout in seconds.
        memory_limit: Docker memory limit (e.g. ``"256m"``).
        python_version: CPython tag for the ``python:*-slim`` image.
    """

    def __init__(
        self,
        timeout: int = 10,
        memory_limit: str = "256m",
        python_version: str = "3.13",
    ):
        self.timeout = timeout
        self._client = docker.from_env(timeout=120)
        self._container = self._client.containers.run(
            f"python:{python_version}-slim",
            ["sleep", "infinity"],
            mem_limit=memory_limit,
            cpu_period=100000,
            cpu_quota=50000,  # 50% of one core
            network_disabled=True,
            read_only=True,
            labels={_SANDBOX_LABEL: "1"},
            detach=True,
        )

    # Context manager

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        try:
            self._container.remove(force=True)
        except Exception:
            pass

    # Public API

    def run(self, script: str) -> dict:
        """Run *script* inside the container.

        Returns dict with keys: correct, stdout, stderr, error.
        """
        try:
            exit_code, (stdout_b, stderr_b) = self._container.exec_run(
                ["timeout", "--kill-after=5", str(self.timeout), "python", "-c", script],
                demux=True,
            )
            stdout = (stdout_b or b"").decode("utf-8")
            stderr = (stderr_b or b"").decode("utf-8")

            if exit_code == 124:
                return {
                    "correct": False,
                    "stdout": stdout,
                    "stderr": stderr,
                    "error": "Timed out",
                }
            return {
                "correct": _PASS_BANNER in stdout,
                "stdout": stdout,
                "stderr": stderr,
                "error": (
                    f"Process exited with code: {exit_code}"
                    if exit_code != 0
                    else None
                ),
            }
        except Exception as e:
            return {
                "correct": False,
                "stdout": "",
                "stderr": "",
                "error": str(e),
            }


def run_in_docker(
    script: str,
    timeout: int = 10,
    memory_limit: str = "256m",
    python_version: str = "3.13",
) -> dict:
    """Execute generated code and tests inside a Docker container.

    Returns dict with:
        passed: bool
        stdout: str
        stderr: str
        error: str | None
    """
    client = docker.from_env()
    container = None
    try:
        container = client.containers.run(
            f"python:{python_version}-slim",
            ["python", "-c", script],
            mem_limit=memory_limit,
            cpu_period=100000,
            cpu_quota=50000,  # 50% of one core
            network_disabled=True,  # no network access
            read_only=True,  # no filesystem writes
            stdout=True,
            stderr=True,
            detach=True,
        )

        result = container.wait(timeout=timeout)
        stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
        stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
        exit_code = result["StatusCode"]

        return {
            "correct": _PASS_BANNER in stdout,
            "stdout": stdout,
            "stderr": stderr,
            "error": f"Container exited with code: {exit_code}" if exit_code != 0 else None,
        }
    except Exception as e:
        stdout, stderr = "", ""
        if container:
            try:
                container.kill()
                stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
                stderr = container.logs(stdout=False, stderr=True).decode("utf-8")
            except Exception:
                pass
        return {
            "correct": False,
            "stdout": stdout,
            "stderr": stderr,
            "error": str(e),
        }
    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception:
                pass


#########
# GRADE #
#########

def grade(
    code: str,
    test_imports: list[str],
    tests: list[str],
    timeout: int = 10,
    memory_limit: str = "256m",
    python_version: str = "3.13",
    sandbox: DockerSandbox | None = None,
) -> dict:
    """Check whether the code matches the gold answer.

    Args:
        sandbox: If provided, run inside this long-lived container
            instead of creating a new one.
    """
    if code is None:
        return {
            "correct": False,
            "stdout": "",
            "stderr": "",
            "error": "No code extracted",
        }

    # Make script for running in container
    code = normalize_answer(code)
    test_function = make_test_function(test_imports, tests)
    script = f"{code}\n\n{test_function}"

    if sandbox is not None:
        return sandbox.run(script)
    return run_in_docker(script, timeout, memory_limit, python_version)


def grade_completions(
    completions: str | list[str],
    gold_answers: tuple | list[tuple],
    timeout: int = 10,
    memory_limit: str = "256m",
    python_version: str = "3.13",
    max_workers: int = 4,
) -> list[dict]:
    """Grade a list of completions against gold answers.

    Spins up multiple Docker containers and grades in parallel using
    a thread pool.

    Args:
        completions: Model output(s) to grade.
        gold_answers: Tuple(s) of (test_imports, tests) for each completion.
        timeout: Per-script wall-clock timeout in seconds.
        memory_limit: Docker memory limit per container.
        python_version: CPython tag for the ``python:*-slim`` image.
        max_workers: Maximum number of parallel sandboxes.

    Returns:
        A list of dicts with keys: completion, extracted, test_imports,
        tests, correct, stdout, stderr, error.
    """
    if isinstance(completions, str):
        assert isinstance(gold_answers, tuple)
        completions = [completions]
        gold_answers = [gold_answers]

    _cleanup_stale_sandboxes()

    n_sandboxes = min(max_workers, len(completions))
    sandboxes = []
    try:
        for _ in range(n_sandboxes):
            sandboxes.append(DockerSandbox(timeout, memory_limit, python_version))
        with ThreadPoolExecutor(max_workers=n_sandboxes) as pool:
            futures = {}
            for i, (completion, gold) in enumerate(zip(completions, gold_answers)):
                sandbox = sandboxes[i % n_sandboxes]
                test_imports, tests = gold
                extracted = extract_answer(completion)
                fut = pool.submit(grade, extracted, test_imports, tests, sandbox=sandbox)
                futures[fut] = (i, completion, extracted, test_imports, tests)

            results: list[dict | None] = [None] * len(completions)
            for fut in as_completed(futures):
                i, completion, extracted, test_imports, tests = futures[fut]
                graded = fut.result()
                results[i] = {
                    "completion": completion,
                    "extracted": extracted,
                    "refusal": is_refusal(completion, extracted),
                    "test_imports": test_imports,
                    "tests": tests,
                    **graded,
                }
    finally:
        for sb in sandboxes:
            sb.close()

    return results
