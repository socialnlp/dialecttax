import os


# Options
DIALECTS = ["sae", "aave"]

# File system
DIRECTORY_NAME = "parallelaave"
FILE_NAME_FORMAT = "{dialect}_samples.txt"


def load_dataset(dir_datasets: str, path_file: str, return_id: bool = True) -> list[dict] | list[str]:
    """Load a ParallelAAVE text file.

    Args:
        dir_datasets: Root directory containing the parallelaave folder.
        path_file: Relative path to the text file within dir_datasets.
        return_id: If True, return list of dicts with keys text, unique_id.
            If False, return list of raw text strings.

    Returns:
        List of dicts (return_id=True) or list of strings (return_id=False).
    """
    path = os.path.join(dir_datasets, path_file)
    with open(path, "r") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not return_id:
        return lines
    return [
        {
            "text": line,
            "unique_id": f"parallelaave-{i}",
        }
        for i, line in enumerate(lines)
    ]
