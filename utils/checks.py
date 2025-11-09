import numpy as np
import polars as pl
from beartype import beartype


@beartype
def check_for_nans(profiles: pl.DataFrame, columns: list[str]) -> None:
    """
    Check if the specified columns in the DataFrame contain any NaN values.

    Parameters
    ----------
    profiles : pl.DataFrame
        The DataFrame to check.
    columns : list of str
        List of column names to check for NaN values.

    Raises
    ------
    ValueError
        If any NaN values are found in the specified columns.
    """
    if (
        profiles.select(columns).null_count().sum_horizontal().sum() > 0
        or np.isinf(profiles.select(columns).to_numpy()).any()
    ):
        raise ValueError("Profiles contain NaN or Inf values.")
