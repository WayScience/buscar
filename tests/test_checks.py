import numpy as np
import polars as pl
import pytest

from buscar.checks import check_for_nans


def test_check_for_nans_clean():
    df = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    check_for_nans(df, ["A", "B"])


def test_check_for_nans_with_null():
    df = pl.DataFrame({"A": [1.0, None, 3.0], "B": [4.0, 5.0, 6.0]})
    with pytest.raises(ValueError, match="Profiles contain NaN or Inf values."):
        check_for_nans(df, ["A", "B"])


def test_check_for_nans_with_inf():
    df = pl.DataFrame({"A": [1.0, np.inf, 3.0], "B": [4.0, 5.0, 6.0]})
    with pytest.raises(ValueError, match="Profiles contain NaN or Inf values."):
        check_for_nans(df, ["A", "B"])


def test_check_for_nans_ignore_other_columns():
    df = pl.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, None, 6.0]})
    check_for_nans(df, ["A"])
