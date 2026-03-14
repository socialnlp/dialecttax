"""
Extract core questions from all ReDial splits and save as clean JSONL files.
Output: {datasets_dir}/ReDial/{task}_{dialect}.json

Usage:
    python scripts/preprocess_redial.py
    python scripts/preprocess_redial.py --rewrite
    python scripts/preprocess_redial.py --config tucana
"""

import argparse
import datetime
import json
import os
import re

import pandas as pd
from datasets import concatenate_datasets, load_dataset

import dialecttax


def _normalize_unicode(s: str) -> str:
    """Normalize Windows-1252 artifacts from ReDial to ASCII equivalents."""
    return s.replace("\x97", "-").replace("\u2010", "-").replace("\u2014", "-").replace("\u2013", "-").replace("\u2018", "'").replace("\u2019", "'").replace("\u0092", "'")


def _fix_logicbench_nesting(directory: str) -> None:
    """Ensure LogicBench(Aug)/(Eval) sit directly under the logicbench root.

    The upstream LogicBench repo nests these under a `data/` subdirectory, but
    `load_logicbench` expects them at the top level. When only the nested copy is
    present, create relative symlinks pointing into `data/` so the loader resolves
    them. Idempotent and non-destructive (never clobbers an existing entry).

    Args:
        directory: The logicbench root, e.g. `{datasets}/logicbench`.
    """
    for name in ("LogicBench(Aug)", "LogicBench(Eval)"):
        expected = os.path.join(directory, name)
        nested = os.path.join(directory, "data", name)
        if os.path.lexists(expected) or not os.path.isdir(nested):
            continue
        os.symlink(os.path.join("data", name), expected)
        print(f"Linked logicbench: {name} -> data/{name}")


def extract_question(text: str, task: str, subset: str | None = None, dialect: str | None = None) -> str:
    """Extract the core question from a ReDial example.

    - Algorithm: varied structure
    - Math: "{instructions}\nQuestion: {question}\\nAnswer:" structure
    - Comprehensive??: preamble is problem context (steps, constraints) -> keep it.
    - Logic??: "{question}{instructions}" format
    """
    if task == "algorithm":
        if subset == "humaneval":
            if dialect is None or dialect == "sae":
                pattern_humaneval_docstring = r"to realize the following functionality:\n(.*?)\nGenerate a Python function to solve this problem."
            else:
                pattern_humaneval_docstring = r"do this following functionality:\n(.*?)\nYou gotta whip up a Python function to handle this problem."
            match = re.search(pattern_humaneval_docstring, text, re.DOTALL)
            assert match
            question = match.group(1)
        elif subset == "mbpp":
            question = text.split("\n")[0]
        else:
            raise ValueError(f"Subset {subset} is invalid for {task} task.")
    elif task == "logic":
        if subset == "folio":
            if dialect is None or dialect == "sae":
                match = re.search(
                    r'Consider the following premises: "(.*?)"'
                    r"\nAssuming no other commonsense or world knowledge, "
                    r'is the sentence "(.*?)" necessarily',
                    text,
                    re.DOTALL,
                )
            else:
                match = re.search(
                    r'''Aight, check this. You got 'em premises right here: "(.*?)"'''
                    r"\nAin't no using no other commonsense or world knowledge, "
                    r'''you gon' try find out if the sentence "(.*?)" necessarily true, necessarily false, or neither''',
                    text,
                    re.DOTALL,
                )
            premises = match.group(1)
            conclusion = match.group(2)
            return premises, conclusion
        elif subset == "logicbench":
            if "?" not in text:
                return '', ''

            if dialect is None or dialect == "sae":
                redial_instruction = "Encode the final answer"
            else:
                redial_instruction = "Wrap that answer up in"
            body = text[:text.index(redial_instruction)].rstrip()
            # Split at the last ". " that precedes the first "?"
            q_start = body.index("?")
            split_point = body[:q_start].rfind(". ") + 1
            premises = body[:split_point].strip()
            conclusion = body[split_point:].strip().rstrip(".")
            return premises, conclusion
        else:
            raise ValueError(f"Subset {subset} is invalid for {task} task.")
    elif task == "math":
        question = text.split("Question: ")[1].split("\nAnswer:")[0]
    elif task == "planning":
        question = text.split("\n\nQuestion: ")[0].strip()
    else:
        return text
    return question


def _parse_humaneval_tests(raw_test: str, str_replace_entry_point: str) -> tuple[list[str], list[str]]:
    """Parse a HumanEval ``check(candidate)`` body into imports and tests.

    Returns:
        (test_imports, tests) where each test is a single-line assert or
        a multi-line block (for-loop with setup vars).
    """
    # Extract body of def check(candidate):
    idx = raw_test.find("def check(candidate):")
    if idx == -1:
        return [], []
    body = raw_test[idx + len("def check(candidate):"):]

    # De-indent by 4 spaces and replace function names
    lines = []
    for line in body.split("\n"):
        if line.startswith("    "):
            line = line[4:]
        else:
            line = line.lstrip()
        line = line.replace("candidate(", "python_function(")
        line = line.replace(str_replace_entry_point, "python_function(")
        lines.append(line)

    # Split into top-level statements using bracket depth tracking
    statements = []
    current = []
    depth = 0
    for line in lines:
        stripped = line.rstrip()
        is_blank = not stripped
        is_indented = stripped and line[0] in (" ", "\t")

        if is_blank:
            if current and depth > 0:
                current.append(stripped)
            continue

        # New top-level statement starts at depth 0 with non-indented line
        if not is_indented and depth == 0 and current:
            statements.append(current)
            current = []

        current.append(stripped)

        in_string = None  # tracks active quote char (' or ")
        i = 0
        while i < len(stripped):
            ch = stripped[i]
            if in_string:
                if ch == "\\" and i + 1 < len(stripped):
                    i += 2  # skip escaped char
                    continue
                if ch == in_string:
                    in_string = None
            elif ch in ("'", '"'):
                in_string = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth = max(0, depth - 1)
            i += 1

    if current:
        statements.append(current)

    # Classify statements
    test_imports = []
    tests = []
    setup_buf = []  # accumulate setup vars to bundle with next for/assert

    for stmt_lines in statements:
        first = stmt_lines[0]

        # Skip comments and bare print
        if first.startswith("#") or first == "print":
            continue

        # Imports
        if first.startswith("import ") or first.startswith("from "):
            test_imports.append(first)
            continue

        # Assert (possibly multi-line) — condense to single line
        if first.startswith("assert "):
            condensed = " ".join(l.strip() for l in stmt_lines)
            if setup_buf:
                setup_buf.append(condensed)
                tests.append("\n".join(setup_buf))
                setup_buf = []
            else:
                tests.append(condensed)
            continue

        # For loop — bundle with any accumulated setup vars
        if first.startswith("for "):
            block = "\n".join(stmt_lines)
            if setup_buf:
                setup_buf.append(block)
                tests.append("\n".join(setup_buf))
                setup_buf = []
            else:
                tests.append(block)
            continue

        # Setup code (variable assignments, etc.) — buffer it
        setup_buf.append("\n".join(stmt_lines))

    # Flush any remaining setup (shouldn't happen, but be safe)
    if setup_buf:
        tests.extend(setup_buf)

    return test_imports, tests


##############
# PREPROCESS #
##############

def preprocess_task(ds_split, split_name, task: str, **kwargs):
    if task == "algorithm":
        preprocessed_split = preprocess_algorithm(ds_split, split_name, **kwargs)
    elif task == "logic":
        preprocessed_split = preprocess_logic(ds_split, split_name, **kwargs)
    elif task == "math":
        preprocessed_split = preprocess_math(ds_split, split_name, **kwargs)
    elif task == "planning":
        preprocessed_split = preprocess_planning(ds_split, split_name)
    else:
        raise ValueError(f"Cannot preprocess task: {task}")
    return preprocessed_split


def preprocess_task_aave(ds_split, split_name, task, preprocessed_original, **kwargs):
    if task == "algorithm":
        preprocessed_split = preprocess_algorithm_aave(ds_split, split_name, preprocessed_original, **kwargs)
    elif task == "logic":
        preprocessed_split = preprocess_logic_aave(ds_split, split_name, preprocessed_original)
    elif task == "math":
        preprocessed_split = preprocess_math_aave(ds_split, split_name, preprocessed_original)
    elif task == "planning":
        preprocessed_split = preprocess_planning_aave(ds_split, split_name, preprocessed_original)
    else:
        raise ValueError(f"Cannot preprocess task: {task}")
    return preprocessed_split


# ALGORITHM

def preprocess_algorithm(ds_split, split_name, **kwargs):
    """
    HumanEval (Instruction format)
        HumanEval format:
        ```
        from typing import List\n\n
        def function_name(...):
            {docstring}
        ```

        ReDial format:
        ```
        {pattern_humaneval}
        original_docstring.strip()
        {instructions}
        ```

    MBPP
        MBPP format:
        ```
        {task}
        ```

        ReDial format:
        ```
        {question}
        {test_case}
        {instructions}
    """
    def _annotate_context(preprocessed: list, indices_start: int) -> list[str]:
        """Add context for code."""
        system_prompt = "Follow instructions. Be accurate. Return ONLY the typed function header."
        instructions = "Annotate this Python function header with types. Return ONLY the typed function header."
        messages = []
        indices_end = len(preprocessed)
        for i in range(indices_start, indices_end):
            answer = preprocessed[i]["answer"]
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{instructions}\n\n```\n{answer}\n```"},
            ]
            messages.append(message)
        responses = dialecttax.endpoints.generate(messages, **kwargs)
        return dialecttax.endpoints.get_completions(responses)

    config = kwargs.pop("config")
    preprocessed_split = []

    # HumanEval
    # Original: https://huggingface.co/datasets/openai/openai_humaneval
    # We assume the ordering is consistent in ReDial as there were many bespoke errors
    humaneval = load_dataset("codeparrot/instructhumaneval", split="test")
    n_humaneval = len(humaneval)
    humaneval_questions = []
    for i in range(n_humaneval):
        question = humaneval[i]["docstring"].replace(humaneval[i]["entry_point"], "python_function")
        humaneval_questions.append(question)
    humaneval_questions_set = set(humaneval_questions)
    humaneval_questions_dict = {q: i for i, q in enumerate(humaneval_questions)}

    # MBPP
    # Original: https://github.com/google-research/google-research/tree/master/mbpp
    mbpp_path = os.path.join(config["directories"]["datasets"], "mbpp", "sanitized-mbpp.json")
    with open(mbpp_path) as f:
        mbpp = json.load(f)
    mbpp_questions = [re.sub(r"https?://\S+", "", row["prompt"]).strip() for row in mbpp]  # remove websites
    mbpp_questions = [_normalize_unicode(q) for q in mbpp_questions]
    mbpp_questions_set = set(mbpp_questions)
    mbpp_questions_dict = {q: i for i, q in enumerate(mbpp_questions)}
    # Check if ReDial incorrectly applied .replace(function_name, "python_function") to the question
    mbpp_questions_corrupted = {}
    for j, row in enumerate(mbpp):
        # Check if MBPP has inconsistent function naming
        function_name = row["code"].split("def ")[1].split("(")[0].strip()
        assertion_name = row["test_list"][0].split("assert ")[1].split("(")[0].strip()

        corrupted = mbpp_questions[j].replace(function_name, "python_function")
        corrupted_assert = mbpp_questions[j].replace(assertion_name, "python_function")
        mbpp_questions_corrupted[corrupted] = j
        mbpp_questions_corrupted[corrupted_assert] = j

    mbpp_indices_start = None
    mbpp_indices_set = set()
    inconsistencies = []
    for i, row in enumerate(ds_split):
        pattern_humaneval = r"Write a function python_function\(.*?\).* to realize the following functionality:"
        subset = "humaneval" if re.search(pattern_humaneval, row["question"]) else "mbpp"

        question = extract_question(row["question"], "algorithm", subset)
        question = _normalize_unicode(question).strip()

        if question in humaneval_questions_set or i < n_humaneval:
            """
            Indices in ReDial with inconsistencies:
            [10, 32, 38, 50, 57, 68, 89, 93, 106, 110, 115, 130, 134, 137, 149, 159]
            """
            j = humaneval_questions_dict.get(question)
            if j is None:
                j = i
                inconsistencies.append(i)

            entry_point = humaneval[j]["entry_point"]
            str_replace_entry_point = f"{entry_point}("
            problem = humaneval[j]["docstring"].strip()
            problem = problem.replace(str_replace_entry_point, "python_function(")
            context = humaneval[j]["context"].strip().replace(str_replace_entry_point, "python_function(")
            answer = humaneval[j]["canonical_solution"].strip()
            test_imports, tests = _parse_humaneval_tests(
                humaneval[j]["test"], str_replace_entry_point
            )
            # Extract helper function definitions from context
            # (everything before the entry-point stub "def python_function(")
            helper_idx = context.rfind("def python_function(")
            if helper_idx > 0:
                helpers = context[:helper_idx].strip()
                if helpers:
                    test_imports = [helpers] + test_imports
            preprocessed_split.append({
                "problem": problem,
                "context": context,
                "answer": answer,
                "test_imports": test_imports,
                "tests": tests,
                "task": "algorithm",
                "unique_id": f"redial-{split_name}-{i}",
                "original_id": f"humaneval-test-{j}",
                "meta": {"dataset": "codeparrot/instructhumaneval"}
            })
        else:
            """
            Indices in ReDial with inconsistencies:
            [168, 171, 176, 229, 240, 244, 267, 295, 306]
            """
            if question in mbpp_questions_set:
                j = mbpp_questions_dict[question]
            else:
                # Deal with inconsistencies
                inconsistencies.append(i)
                if question in mbpp_questions_corrupted:
                    j = mbpp_questions_corrupted[question]
                else:
                    raise ValueError(f"ReDial index {i} could not be matched to any MBPP question: {question[:80]}")

            # Remove duplicates in ReDial
            if j in mbpp_indices_set:
                continue

            question = mbpp_questions[j]
            if j == 388:  # Mistake in MBPP
                inconsistencies.append(i)
                question = "Write a python function to count the number of pairs whose sum is equal to 'sum'."

            # Set beginning of MBPP since HumanEval questions come before
            if mbpp_indices_start is None:
                mbpp_indices_start = len(preprocessed_split)

            answer = mbpp[j]["code"]
            function_name = answer.split("def ")[1].split("(")[0]
            # Remove trailing spaces from function definition
            if function_name != function_name.strip():
                answer = re.sub(r"(def \w+)\s+\(", r"\1(", answer)
                function_name = function_name.strip()

            tests = mbpp[j]["test_list"]
            assertion_name = tests[0].split("assert ")[1].split("(")[0].strip()  # Check if MBPP has inconsistent function naming
            tests = [t.replace(function_name, "python_function").replace(assertion_name, "python_function") for t in tests]

            example = tests[0][7:].split(" == ")
            if len(example) == 2:
                function_call, function_response = example
            else:  # "assert math.isclose" or "assert function(**args)"
                function_call = example[0]
                function_response = "True"
            problem = (
                f"{question}\n"
                f">>> {function_call}\n{function_response}"
            )

            test_imports = mbpp[j]["test_imports"]
            preprocessed_split.append({
                "problem": problem,
                "context": None,
                "answer": answer,
                "test_imports": test_imports,
                "tests": tests,
                "task": "algorithm",
                "unique_id": f"redial-{split_name}-{i}",
                "original_id": f"mbpp-test-{j}",
                "meta": {"dataset": "google-research-datasets/mbpp"}
            })
            mbpp_indices_set.add(j)

    n = len(preprocessed_split)
    assert len({p["problem"] for p in preprocessed_split}) == n

    mbpp_contexts = _annotate_context(preprocessed_split, mbpp_indices_start)
    for j, i in enumerate(range(mbpp_indices_start, n)):
        result_name = mbpp_contexts[j].split("def ")[1].split("(")[0]
        code_name = preprocessed_split[i]["answer"].split("def ")[1].split("(")[0]
        assert result_name == code_name
    mbpp_contexts = ["def python_function(" + result.split("(", 1)[1] for result in mbpp_contexts]
    for j, i in enumerate(range(mbpp_indices_start, n)):
        preprocessed_split[i]["context"] = mbpp_contexts[j]
    return preprocessed_split


def preprocess_algorithm_aave(ds_split, split_name, preprocessed_original, **kwargs):
    config = kwargs.pop("config")

    # Map original ds_split index to preprocessed index (skips duplicates)
    idx_map = {}
    for j, entry in enumerate(preprocessed_original):
        ds_idx = int(entry["unique_id"].rsplit("-", 1)[1])
        idx_map[ds_idx] = j

    # Fix appropriate inconsistencies
    inconsistencies = [
        # HumanEval
        10, 32, 38, 50, 115, 130,
        # MBPP
        168, 171, 176, 229, 240, 244, 267, 295, 306
    ]

    # HumanEval
    # We assumed the ordering is consistent in ReDial as there were many bespoke errors
    humaneval = load_dataset("codeparrot/instructhumaneval", split="test")
    n_humaneval = len(humaneval)

    # MBPP
    # Original function names that ReDial incorrectly replaced within the question
    mbpp_path = os.path.join(config["directories"]["datasets"], "mbpp", "sanitized-mbpp.json")
    with open(mbpp_path) as f:
        mbpp = json.load(f)
    mbpp_function_names = {}
    for j, row in enumerate(mbpp):
        function_name = row["code"].split("def ")[1].split("(")[0].strip()
        assertion_name = row["test_list"][0].split("assert ")[1].split("(")[0].strip()
        if function_name == assertion_name:
            mbpp_function_names[f"mbpp-test-{j}"] = (function_name,)
        else:
            mbpp_function_names[f"mbpp-test-{j}"] = (function_name, assertion_name)

    preprocessed_split = []
    for i, row in enumerate(ds_split):
        # Skip duplicates in SAE split
        if i not in idx_map:
            continue

        subset = "humaneval" if i < n_humaneval else "mbpp"

        question = extract_question(row["question"], "algorithm", subset=subset, dialect="aave")
        question = _normalize_unicode(question).strip()
        if i < 68:
            # Add space after period, question, or exclamation mark if not followed by newline
            question = re.sub(r"([.?!])\s*(?=[a-zA-Z])", r"\1 ", question)
        elif i >= n_humaneval:
            # Two spaces between two words should be one space for the question
            question = re.sub(r"([a-zA-Z])  ([a-zA-Z])", r"\1 \2", question)
            # Remove randomly lower-cased sentences
            question = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), question)
        problem = question

        # Correspondence with original dataset
        row_original = preprocessed_original[idx_map[i]]
        problem_original = row_original["problem"]
        context = row_original["context"]
        if i in inconsistencies:
            if subset == "humaneval":
                if i == 38:
                    context = question[question.index("def encode_cyclic"):]
                    problem = problem_original
                elif i == 115:  # GPT-5.2 translation
                    problem = (
                        "Aight, peep this.\n"
                        "You got a rectangular grid full of wells. Each row be one well,\n"
                        "and every 1 up in that row mean one unit of water sittin in there.\n"
                        "Each well got its own bucket you can drop down to pull water out,\n"
                        "and all them buckets hold the same amount.\n"
                        "Your job? Use them buckets to empty out all the wells.\n"
                        "Then tell how many times you had to lower them buckets total.\n"
                        "Example 1:\n"
                        "Input:\n"
                        "grid : [[0,0,1,0], [0,1,0,0], [1,1,1,1]]\n"
                        "bucket_capacity : 1\n"
                        "Output:\n"
                        "6\n"
                        "Example 2:\n"
                        "Input:\n"
                        "grid : [[0,0,1,1], [0,0,0,0], [1,1,1,1], [0,1,1,1]]\n"
                        "bucket_capacity : 2\n"
                        "Output:\n"
                        "5\n"
                        "Example 3:\n"
                        "Input:\n"
                        "grid : [[0,0,0], [0,0,0]]\n"
                        "bucket_capacity : 5\n"
                        "Output:\n"
                        "0\n"
                        "Constraints:\n"
                        "* All wells the same length.\n"
                        "* 1 <= number of wells <= 10^2\n"
                        "* 1 <= length of each well <= 10^2\n"
                        "* grid[i][j] always 0 or 1\n"
                        "* 1 <= bucket_capacity <= 10\n"
                    )
                elif i == 130:  # GPT-5.2 translation
                    problem = (
                        "Everybody know about the Fibonacci sequence, mathematicians been studyin it heavy "
                        "for the last couple centuries. But what folks do not really talk about is the "
                        "Tribonacci sequence.\n"
                        "Tribonacci sequence get defined like this:\n"
                        "tribo(1) = 3\n"
                        "tribo(n) = 1 + n / 2, if n even.\n"
                        "tribo(n) = tribo(n - 1) + tribo(n - 2) + tribo(n + 1), if n odd and n > 1.\n"
                        "For example:\n"
                        "tribo(0) = 1\n"
                        "tribo(2) = 1 + (2 / 2) = 2\n"
                        "tribo(4) = 3\n"
                        "tribo(3) = tribo(2) + tribo(1) + tribo(4)\n"
                        "= 2 + 3 + 3 = 8\n"
                        "You get a non-negative integer n, and you gotta return a list of the first n + 1 "
                        "numbers in the Tribonacci sequence, from tribo(0) up to tribo(n).\n"
                        "Example:\n"
                        "python_function(3) = [1, 3, 2, 8]"
                    )
                else:
                    if "\n\n" in question:
                        problem, context = question.split("\n\n", 1)
                    else:
                        index_function_def = question.index("def ")
                        problem = question[:index_function_def].strip()
                        context = question[index_function_def:].strip()
                context = re.sub(r'\\(["\'])', r"\1", context)
            else:
                if row_original["original_id"] == "mbpp-test-388":  # Mistake in MBPP
                    question = "Aight, you gon' write a python function that be countin' how many pairs add up to 'sum'."
                else:
                    function_names = mbpp_function_names[row_original["original_id"]]
                    if len(function_names) == 1:
                        question = question.replace("python_function", function_names[0])
                    else:
                        question = question.replace("python_function", function_names[1])

        if subset == "mbpp":
            example = problem_original[problem_original.index(">>>"):]
            problem = f"{question}\n{example}"

        preprocessed_split.append({
            "problem": problem,
            "context": context,
            "answer": row_original["answer"],
            "test_imports": list(row_original["test_imports"]),
            "tests": list(row_original["tests"]),
            "task": "algorithm",
            "unique_id": f"redial-{split_name}-{i}",
            "original_id": row_original["original_id"],
            "meta": dict(row_original["meta"])
        })

    return preprocessed_split


# LOGIC

def map_to_alphabet(choice):
    """Map choices_{i} to capital alphabet letter.

    answer = map_to_alphabet("choice_2")
    """
    return chr(64 + int(choice.split("_")[1]))


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?<=\. )[a-z]", lambda m: m.group().upper(), text)
    text = re.sub(r"^[a-z]", lambda m: m.group().upper(), text)
    if not text.endswith("."):
        text += "."
    return text


def preprocess_logic(ds_split, split_name, **kwargs):
    config = kwargs.pop("config")

    # FOLIO
    # Original: https://huggingface.co/datasets/yale-nlp/FOLIO
    # Counterfactual: https://huggingface.co/datasets/ZhaofengWu/FOLIO-counterfactual/resolve/main/folio_v2_perturbed.jsonl
    folio = load_dataset("yale-nlp/FOLIO")
    folio = concatenate_datasets([folio["train"], folio["validation"]])
    folio_counterfactual = load_dataset("ZhaofengWu/FOLIO-counterfactual", data_files="folio_v2_perturbed.jsonl", split="train")
    # Checks
    assert len(set(folio["example_id"])) == len(folio)  # "example_id" is unique
    counts_consider_premises = 0
    for row in ds_split:
        counts_consider_premises += int(row["question"].startswith("Consider the following premises: "))
    assert counts_consider_premises == 2 * len(folio_counterfactual)  # Correct way to identify FOLIO from ReDial
    n_folio_counterfactual = len(folio_counterfactual)
    n_folio = 2 * n_folio_counterfactual

    # LogicBench
    # Original: https://github.com/Mihir3009/LogicBench
    logicbench_dir = os.path.join(config["directories"]["datasets"], "logicbench")
    _fix_logicbench_nesting(logicbench_dir)
    logicbench = dialecttax.data.redial.load_logicbench(logicbench_dir)

    preprocessed_split = []
    CHOICES_DEFAULT = {"A": "True", "B": "False", "C": "Uncertain"}
    CHOICES_INVERSE = {v: k for k, v in CHOICES_DEFAULT.items()}
    for i, row in enumerate(ds_split):
        pattern_folio = "Consider the following premises: "
        subset = "folio" if row["question"].startswith(pattern_folio) else "logicbench"
        if subset == "folio" or i < n_folio:
            # ReDial loops over FOLIO-Counterfactual with all originals then all perturbed
            j = i % n_folio_counterfactual
            folio_info = folio_counterfactual[j]
            if i < n_folio_counterfactual:
                origin = "original"
                premises_name = "orig_premises"
                conclusion_name = "orig_conclusion"
            else:
                origin = "counterfactual"
                premises_name = "premises"
                conclusion_name = "conclusion"
            premises = _normalize_unicode(folio_info[premises_name]).strip()
            conclusion = _normalize_unicode(folio_info[conclusion_name]).strip()
            answer = CHOICES_INVERSE[folio_info["label"]]
            preprocessed_split.append({
                "premises": premises,
                "conclusion": conclusion,
                "choices": dict(CHOICES_DEFAULT),
                "answer": answer,
                "task": "logic",
                "unique_id": f"redial-{split_name}-{i}",
                "original_id": f"folio_counterfactual-{j}",
                "meta": {"dataset": "yale-nlp/FOLIO", "origin": origin, "FOLIO_ID": folio_info["example_id"]}
            })
        else:
            premises, conclusion = extract_question(row["question"], "logic", subset)
            premises = _normalize_unicode(premises).strip()
            conclusion = _normalize_unicode(conclusion).strip()

            logicbench_info = logicbench[(logicbench["context"] == premises) & (logicbench["question"] == conclusion)]

            # Fix bespoke mistakes
            # LogicBench context says "Sandra" but question says "Sarah"
            # Keep corrected ReDial premises
            if i == 182:
                logicbench_info = logicbench[logicbench.index == 13048]
            elif i == 183:
                logicbench_info = logicbench[logicbench.index == 13049]
            elif i == 184:
                logicbench_info = logicbench[logicbench.index == 13050]
            elif i == 185:
                logicbench_info = logicbench[logicbench.index == 13051]

            if i == 345:
                # ReDial left out the question
                logicbench_info = logicbench[logicbench.index == 14447]
                premises = _normalize_unicode(logicbench_info["context"].item()).replace("\n\n", " ").strip()
                conclusion = _normalize_unicode(logicbench_info["question"].item()).strip()

            assert len(logicbench_info) == 1

            j = logicbench_info.index.item()
            choices = {"A": "True", "B": "False", "C": "Uncertain"}
            label = logicbench_info["answer"].item()
            if label == "yes":
                answer = "A"
            elif label == "no":
                answer = "B"
            else:
                # MQA
                # Update choices to reflect that of ReDial
                # ReDial shuffles the choices, so answer != row["answer"]
                choices_logicbench = logicbench_info["choices"].item()
                choices_redial = dict(re.findall(r"\n([A-Z])\. (.+)", row["question"]))
                assert choices_redial[row["answer"]] == choices_logicbench[label]
                choices = {k: clean_text(v) for k, v in choices_redial.items()}
                # LogicBench: choices = {map_to_alphabet(k): clean_text(v) for k, v in logicbench_info["choices"].item().items()}
                answer = row["answer"]

                # We fix a few bespoke capitalization issues here
                if i == 263:
                    premises = premises.replace("harry", "Harry")
                elif i == 264:
                    premises = premises.replace("jack", "Jack")
                elif i == 265:
                    premises = premises.replace("sophie", "Sophie")
                elif i == 266:
                    premises = premises.replace("meera", "Meera")

            # Fix sentence capitalization issues
            premises = re.sub(r"(?<=\. )[a-z]", lambda m: m.group().upper(), premises)

            logicbench_metadata = logicbench_info[["task", "pattern", "rule"]].to_dict(orient="records")[0]
            logicbench_metadata["origin"] = logicbench_metadata.pop("task")
            logicbench_split = logicbench_info["split"].item()
            preprocessed_split.append({
                "premises": premises,
                "conclusion": conclusion,
                "choices": choices,
                "answer": answer,
                "task": "logic",
                "unique_id": f"redial-{split_name}-{i}",
                "original_id": f"logicbench-{logicbench_split}-{j}",
                "meta": {
                    "dataset": "Mihir3009/LogicBench",
                    **logicbench_metadata
                }
            })

    assert len({(p["premises"], p["conclusion"]) for p in preprocessed_split}) == len(preprocessed_split)
    return preprocessed_split


def preprocess_logic_aave(ds_split, split_name, preprocessed_original):
    # FOLIO
    folio_counterfactual = load_dataset("ZhaofengWu/FOLIO-counterfactual", data_files="folio_v2_perturbed.jsonl", split="train")
    n_folio = 2 * len(folio_counterfactual)
    logic_answer_mapping = {
        "necessarily true": "True",
        "yes": "True",
        "necessarily false": "False",
        "no": "False",
        "neither": "Uncertain",
    }
    # SAE canonicalizes logic answers as letters (CHOICES_DEFAULT in preprocess_logic),
    # so map AAVE word answers True/False/Uncertain -> A/B/C to match row_original["answer"].
    CHOICES_INVERSE = {"True": "A", "False": "B", "Uncertain": "C"}

    preprocessed_split = []
    for i, row in enumerate(ds_split):
        # Correspondence with original dataset
        row_original = preprocessed_original[i]

        subset = "folio" if i < n_folio else "logicbench"
        premises, conclusion = extract_question(row["question"], "logic", subset, dialect="aave")
        premises = _normalize_unicode(premises).strip()
        conclusion = _normalize_unicode(conclusion).strip()
        mapped_answer = logic_answer_mapping.get(row["answer"], row["answer"])
        assert CHOICES_INVERSE.get(mapped_answer, mapped_answer) == row_original["answer"]
        if subset == "folio":
            assert CHOICES_INVERSE[logic_answer_mapping[row["answer"]]] == row_original["answer"]
            preprocessed_split.append({
                "premises": premises,
                "conclusion": conclusion,
                "choices": dict(row_original["choices"]),
                "answer": row_original["answer"],
                "task": "logic",
                "unique_id": f"redial-{split_name}-{i}",
                "original_id": row_original["original_id"],
                "meta": dict(row_original["meta"])
            })
        else:
            # We fix a few bespoke issues fixed in the SAE version
            # A couple (i = [264, 265]) already fixed
            if i == 263:
                premises = premises.replace("harry", "Harry")
            elif i == 266:
                premises = premises.replace("meera", "Meera")

            # ReDial left out the question in SAE but seems fine in AAVE
            if i == 345:
                # Fix space issues
                premises = premises.replace("or(2)", "or (2)").replace("only(2)", "only (2)")

            choices = dict(row_original["choices"])
            if row_original["meta"]["origin"] == "MCQA":
                choices = dict(re.findall(r"\n([A-Z])\. (.+)", row["question"]))
                choices = {k: clean_text(v) for k, v in choices.items()}

            # Fix sentence capitalization issues
            premises = re.sub(r"(?<=\. )[a-z]", lambda m: m.group().upper(), premises)

            preprocessed_split.append({
                "premises": premises,
                "conclusion": conclusion,
                "choices": choices,
                "answer": row_original["answer"],
                "task": "logic",
                "unique_id": f"redial-{split_name}-{i}",
                "original_id": row_original["original_id"],
                "meta": dict(row_original["meta"])
            })
    return preprocessed_split


# MATH

def preprocess_math(ds_split, split_name, **kwargs):
    kwargs.pop("config", None)

    def _edit_grammar(preprocessed: list, indices: list) -> list[str]:
        """Edit problem for grammar and punctuation."""
        system_prompt = "Follow instructions. Return ONLY the corrected text."
        instructions = "Minimally edit the grammar and punctuation. Return ONLY the corrected text:"
        messages = []
        for j in indices:
            problem = preprocessed[j]["problem"]
            message = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{instructions}\n\n{problem}"},
            ]
            messages.append(message)
        responses = dialecttax.endpoints.generate(messages, **kwargs)
        return dialecttax.endpoints.get_completions(responses)

    preprocessed_split = []

    # GSM8K
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    gsm8k_questions = [_normalize_unicode(q) for q in gsm8k["question"]]
    gsm8k_questions_set = set(gsm8k_questions)
    gsm8k_questions_dict = {q: i for i, q in enumerate(gsm8k_questions)}

    # SVAMP
    svamp = load_dataset("ChilleD/SVAMP", split="test")
    svamp_questions = svamp["question_concat"]
    svamp_questions_set = set(svamp_questions)
    svamp_questions_dict = {q: i for i, q in enumerate(svamp_questions)}

    svamp_edit_indices = []
    for i, row in enumerate(ds_split):
        question = extract_question(row["question"], "math")
        question = _normalize_unicode(question)
        answer = int(dialecttax.data.graders.math.normalize_answer(row["answer"]))

        if question in gsm8k_questions_set:
            j = gsm8k_questions_dict[question]
            solution, check = gsm8k[j]["answer"].split("\n#### ")
            solution = _normalize_unicode(solution)
            check = int(check.replace(",", "").replace("_", ""))
            assert answer == check
            preprocessed_split.append({
                "problem": question.strip(),
                "solution": solution.strip(),
                "answer": answer,
                "task": "math",
                "unique_id": f"redial-{split_name}-{i}",
                "original_id": f"gsm8k-test-{j}",
                "meta": {"dataset": "openai/gsm8k"}
            })
        else:
            assert question in svamp_questions_set
            j = svamp_questions_dict[question]
            problem = _normalize_unicode(svamp[j]["question_concat"])
            solution = _normalize_unicode(svamp[j]["Equation"])
            check = svamp[j]["Answer"]
            check = int(dialecttax.data.graders.math.normalize_answer(check))
            assert answer == check
            preprocessed_split.append({
                "problem": problem.strip(),
                "solution": solution.strip(),
                "answer": answer,
                "task": "math",
                "unique_id": f"redial-{split_name}-{i}",
                "original_id": f"svamp-test-{j}",
                "meta": {"dataset": "ChilleD/SVAMP", "SVAMP_ID": svamp[j]["ID"]}
            })
            svamp_edit_indices.append(i)

    svamp_probs_edited = _edit_grammar(preprocessed_split, svamp_edit_indices)
    for j, i in enumerate(svamp_edit_indices):
        preprocessed_split[i]["problem"] = svamp_probs_edited[j]
    return preprocessed_split


def preprocess_math_aave(ds_split, split_name, preprocessed_original):
    preprocessed_split = []
    for i, row in enumerate(ds_split):
        question = extract_question(row["question"], "math")
        answer = int(dialecttax.data.graders.math.normalize_answer(row["answer"]))

        # Correspondence with original dataset
        row_original = preprocessed_original[i]
        check = row_original["answer"]
        assert answer == check
        preprocessed_split.append({
            "problem": question,
            "solution": row_original["solution"],
            "answer": answer,
            "task": "math",
            "unique_id": f"redial-{split_name}-{i}",
            "original_id": row_original["original_id"],
            "meta": dict(row_original["meta"])
        })
    return preprocessed_split


# PLANNING

def preprocess_planning(ds_split, split_name):
    """Preprocess planning tasks.

    Each sample has the following information:
    ```
    {
        "problem": problem,
        "answer": [],
        "task": "planning",
        "unique_id": f"redial-{split_name}-{i}",
        "original_id": f"asynchow-{asynchow_split}-{j}",
        "meta": {"dataset": "fangrulin/asynchow"}
    }
    ```
    """
    def _match_key(q: str) -> str:
        """First two lines, alphanumeric only, for matching.

        Matching on two lines should be unique, as verified by the code below.
        ```
        from collections import Counter
        asynchow = load_dataset("fangrulin/asynchow")
        asynchow = concatenate_datasets([asynchow["train"], asynchow["test"]])
        counts = Counter(["\n".join(q["question"].strip().split("\n")[:2]) for q in asynchow])
        assert sum([int(c > 1) for c in counts.values()]) == 0
        ```

        We also verify that the total number of items matches with the number of unique items in ReDial.
        ```
        redial = load_dataset("fangrulin/redial")["comprehensive_vanilla_original"]
        assert len(set(redial["question"])) == 225
        ```
        """
        text = _normalize_unicode("\n".join(q.strip().split("\n")[:2]))
        return re.sub(r"[^a-z0-9]", "", text.lower())

    # AsyncHow
    # Original: https://huggingface.co/datasets/fangrulin/asynchow
    asynchow = load_dataset("fangrulin/asynchow")
    asynchow_questions_dict = {}
    for j, q in enumerate(asynchow["train"]["question"]):
        asynchow_questions_dict[_match_key(q)] = (j, "train")
    for j, q in enumerate(asynchow["test"]["question"]):
        asynchow_questions_dict[_match_key(q)] = (j, "test")

    # Remove weird capitalizations throughout
    exceptions_capitalization = {
        1: "Nerf",
        27: "Springsteen",
        40: "Salata Balati",
        43: "Pokemon",
        44: "Indian",
        47: "Russian",
        48: "Swedish",
        57: "ADP",
        61: "Uromastyx",
        66: "Gouda",
        91: "Cinderella",
        93: "Finnish",
        97: "Persian New Year",
        103: "Norwegian",
        107: "Chingling",
        116: "Five Great Lakes",
        119: "Jordans",
        125: "Lent",
        131: "Tiffany's",
        138: "E6000",
        142: "Peruvian",
        163: "Hindbeh Bi Zeit",
        174: "Día de los Muertos",
        179: "Thin Mints",
        202: "CPM",
    }
    exceptions_indices = {e for e in exceptions_capitalization.keys()}

    preprocessed_split = []
    seen = set()
    for i, row in enumerate(ds_split):
        key = _match_key(row["question"])
        asynchow_info = asynchow_questions_dict[key]
        if asynchow_info in seen:
            continue
        j, asynchow_split = asynchow_info

        # Items in ReDial does not exactly match those in AsyncHow
        # There are more bespoke errors in row["question"] so we use the original AsyncHow
        question = extract_question(asynchow[asynchow_split][j]["question"], "planning")
        question = _normalize_unicode(question).replace("\n\n", "\n").strip()

        # Change description of each step to have a colon to match with instructions
        description_old = "here are the steps and the times needed for each step."
        description_new = "here are the steps and the times needed for each step:"
        question = question.replace(description_old, description_new)

        # Change ungrammatical instructions
        instructions_old = "These ordering constraints need to be obeyed when executing above steps"
        instructions_new = "These ordering constraints must be followed when executing the above steps"
        question = question.replace(instructions_old, instructions_new)

        # Correct capitalization of the first line
        lines = question.split("\n", 1)
        lines[0] = lines[0].lower().capitalize().replace(" i ", " I ")
        index_current = len(preprocessed_split)
        if index_current in exceptions_indices:
            replacement = exceptions_capitalization[index_current]
            lines[0] = lines[0].replace(replacement.lower(), replacement)
        question = "\n".join(lines)

        # All upper case after "Step {i}."
        question = re.sub(r"(Step \d+\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), question)

        # Make "Step {i}." into "Step {i}:" because it functions as a label
        question = re.sub(r"(Step \d+)\. (\w)", r"\1: \2", question)

        # Upper case "Step {i}" since it refers to a specific step name
        question = re.sub(r"(?<=\b)step(?= \d)", "Step", question)

        answer = [t.total_seconds() for t in eval(row["answer"])]
        preprocessed_split.append({
            "problem": question,
            "answer": answer,
            "task": "planning",
            "unique_id": f"redial-{split_name}-{i}",
            "original_id": f"asynchow-{asynchow_split}-{j}",
            "meta": {"dataset": "fangrulin/asynchow"}
        })
        seen.add(asynchow_info)

    assert len({p["problem"] for p in preprocessed_split}) == len(preprocessed_split)
    return preprocessed_split


def preprocess_planning_aave(ds_split, split_name, preprocessed_original, **kwargs):
    preprocessed_split = []
    j = 0
    for i, row in enumerate(ds_split):
        row_original = preprocessed_original[j]

        # Skip duplicates
        if str(i) != row_original["unique_id"].split("-")[-1]:
            continue

        question = extract_question(row["question"], "planning")

        # All upper case after "Step {i}."
        question = re.sub(r"(Step \d+\. )([a-z])", lambda m: m.group(1) + m.group(2).upper(), question)

        # Make "Step {i}." into "Step {i}:" because it functions as a label
        question = re.sub(r"(Step \d+)\. (\w)", r"\1: \2", question)

        # Change descriptions to be a colon to match with SAE
        lines = question.split("\n", 1)
        lines[0] = lines[0].rstrip(".") + ":"
        question = "\n".join(lines)

        answer = list(row_original["answer"])
        check = [t.total_seconds() for t in eval(row["answer"])]
        assert answer[0] == check[0] and answer[1] == check[1]
        preprocessed_split.append({
            "problem": question,
            "answer": answer,
            "task": "planning",
            "unique_id": f"redial-{split_name}-{i}",
            "original_id": row_original["original_id"],
            "meta": dict(row_original["meta"])
        })
        j += 1
    return preprocessed_split


##############
# PREPROCESS #
##############

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess ReDial dataset")
    parser.add_argument("--config", default="default", help="Config file name")
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite existing files")
    return parser.parse_args()


def preprocess():
    args = parse_args()
    config = dialecttax.utils.load_config(args.config)

    dir_redial = os.path.join(config["directories"]["preprocessed"], "ReDial")
    os.makedirs(dir_redial, exist_ok=True)

    api_key_openrouter = dialecttax.utils.get_api_key(config["keys"]["openrouter"])
    kwargs_preprocess_task = {
        "config": config,
        "api_key": api_key_openrouter,
        "model": "openai/gpt-5-mini",
        "temperature": 0.0,
        "max_workers": 16,
    }

    print("Loading ReDial dataset...")
    ds = load_dataset("fangrulin/redial")
    split_names = sorted(ds.keys(), reverse=True)
    tasks = ["algorithm", "comprehensive", "logic", "math"]
    for task in tasks:
        task_name = "planning" if task == "comprehensive" else task

        # SAE
        path_sae = os.path.join(dir_redial, f"{task_name}_sae.jsonl")
        if not args.rewrite and os.path.exists(path_sae):
            print(f"Skipping {task_name}_sae (already exists): {path_sae}")
            with open(path_sae) as f:
                preprocessed_split = [json.loads(line) for line in f]
        else:
            split_name = f"{task}_vanilla_original"
            ds_split = ds[split_name]
            preprocessed_split = preprocess_task(ds_split, split_name, task_name, **kwargs_preprocess_task)
            with open(path_sae, "w") as f:
                for item in preprocessed_split:
                    f.write(json.dumps(item) + "\n")
            print(f"Preprocessed {split_name} ({len(preprocessed_split)} examples) -> {path_sae}")

        # AAVE
        path_aave = os.path.join(dir_redial, f"{task_name}_aave.jsonl")
        if not args.rewrite and os.path.exists(path_aave):
            print(f"Skipping {task_name}_aave (already exists): {path_aave}")
        else:
            split_name = f"{task}_vanilla_aave"
            ds_split = ds[split_name]
            preprocessed_split_aave = preprocess_task_aave(ds_split, split_name, task_name, preprocessed_split, **kwargs_preprocess_task)
            with open(path_aave, "w") as f:
                for item in preprocessed_split_aave:
                    f.write(json.dumps(item) + "\n")
            print(f"Preprocessed {split_name} ({len(preprocessed_split_aave)} examples) -> {path_aave}")

    print(f"\nDone! Saved to: {dir_redial}")


if __name__ == "__main__":
    preprocess()
