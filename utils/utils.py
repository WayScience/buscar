"""
Module: utils.py

A collection of common utility functions for data processing,
as well as for saving, loading, and writing files.
"""

import pathlib


def create_results_dir() -> pathlib.Path:
    """Creates a results directory in the current path

    Returns
    -------
    pathlib.Path
        Path to results directory
    """

    # create results directory. if it does exist, do not raise error
    results_dir = pathlib.Path("./results").resolve()
    results_dir.mkdir(exist_ok=True)

    return results_dir
