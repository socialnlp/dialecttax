import json
import os

import pandas as pd

# Options
TASKS = ["algorithm", "logic", "math", "planning"]
DIALECTS = ["sae", "aave"]
REASONING_OPTIONS = ["naive", "cot"]

# File system
DIRECTORY_NAME = "ReDial"
FILE_NAME_FORMAT = "{task}_{dialect}.jsonl"
FILE_NAME_QA_FORMAT = "{task}_{dialect}_qa.jsonl"


def load_dataset(dir_preprocessed: str, path_file: str, **kwargs) -> list[dict]:
    """Load a preprocessed ReDial split from JSONL."""
    path = os.path.join(dir_preprocessed, path_file)
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data


##############
# COMPONENTS #
##############

def load_logicbench(directory: str) -> pd.DataFrame:
    """Load the full LogicBench dataset into a flat DataFrame.

    Walks LogicBench(Aug) (train) and LogicBench(Eval)/{BQA,MCQA} (test)
    directories, flattening QA pairs so each row is one question.

    Args:
        directory: Root path containing LogicBench(Aug)/ and LogicBench(Eval)/.

    Returns:
        DataFrame with columns: split, task, pattern, rule, context,
        question, answer, choices.
    """
    rows: list[dict] = []

    # Aug (train) — structured as {pattern}/{rule}/data_instances.json
    aug_dir = os.path.join(directory, "LogicBench(Aug)")
    for pattern in sorted(os.listdir(aug_dir)):
        pattern_dir = os.path.join(aug_dir, pattern)
        if not os.path.isdir(pattern_dir):
            continue
        for rule in sorted(os.listdir(pattern_dir)):
            path = os.path.join(pattern_dir, rule, "data_instances.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                data = json.load(f)
            for sample in data["data_samples"]:
                for qa in sample["qa_pairs"]:
                    rows.append({
                        "split": "train",
                        "task": "BQA",
                        "pattern": pattern,
                        "rule": rule,
                        "context": sample["context"],
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "choices": None,
                    })

    # Eval (test) — structured as {task}/{pattern}/{rule}/data_instances.json
    eval_dir = os.path.join(directory, "LogicBench(Eval)")
    for task in ("BQA", "MCQA"):
        task_dir = os.path.join(eval_dir, task)
        if not os.path.isdir(task_dir):
            continue
        for pattern in sorted(os.listdir(task_dir)):
            pattern_dir = os.path.join(task_dir, pattern)
            if not os.path.isdir(pattern_dir):
                continue
            for rule in sorted(os.listdir(pattern_dir)):
                path = os.path.join(pattern_dir, rule, "data_instances.json")
                if not os.path.isfile(path):
                    continue
                with open(path) as f:
                    data = json.load(f)
                for sample in data["samples"]:
                    if task == "BQA":
                        for qa in sample["qa_pairs"]:
                            rows.append({
                                "split": "test",
                                "task": "BQA",
                                "pattern": pattern,
                                "rule": rule,
                                "context": sample["context"],
                                "question": qa["question"],
                                "answer": qa["answer"],
                                "choices": None,
                            })
                    else:  # MCQA
                        rows.append({
                            "split": "test",
                            "task": "MCQA",
                            "pattern": pattern,
                            "rule": rule,
                            "context": sample["context"],
                            "question": sample["question"],
                            "answer": sample["answer"],
                            "choices": sample["choices"],
                        })

    return pd.DataFrame(rows)
