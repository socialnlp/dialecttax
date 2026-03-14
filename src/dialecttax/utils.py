import socket
import os

import numpy as np
import yaml


##################
# CONFIGURATIONS #
##################

# Repository root: this file is src/dialecttax/utils.py, so two levels up.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def load_config(filename: str = "default") -> dict:
    """Load a YAML config file, expanding its path placeholders.

    Two placeholders are substituted before the YAML is parsed:
        {hostname}  the short hostname of this machine
        {repo}      the repository root, so paths that live inside the checkout
                    (secrets/, data/) stay correct wherever it is cloned to

    Args:
        filename: Config file name without '.yaml', e.g. "default" or "server".

    Returns:
        The parsed config dict.
    """
    path = os.path.join(REPO_ROOT, "configs", f"{filename}.yaml")
    if not os.path.exists(path):
        raise ValueError("`filename` not found! It should be the name of the file without '.yaml'.")

    print(f"Configuration: {path}")
    with open(path) as f:
        raw = f.read()
    raw = raw.replace("{hostname}", socket.gethostname().split(".")[0])
    raw = raw.replace("{repo}", REPO_ROOT)
    return yaml.safe_load(raw)


def get_api_key(path: str) -> str:
    """Return API key from a file."""
    try:
        with open(path, 'r') as f:
            return f.readline().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"API key path not found: {path}")


########
# MATH #
########

def divide_nan(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator
