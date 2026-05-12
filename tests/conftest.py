"""Pytest configuration for local src-layout imports."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

# Add src directory to Python path for local test execution.
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))


@pytest.fixture
def synthetic_profiles():
    """
    Generates simulated morphological profiling data for testing.

    Characteristics:
    - 1000 rows for control.
    - 300 features.
    - ~30 non-significant features between control and disease, rest are significant.
    """
    seed = 0
    np.random.seed(seed)
    n_features = 300
    n_ctrl = 1000
    n_others = 300

    feature_names = [f"Feature_{i}" for i in range(n_features)]
    sig_indices = list(range(30, n_features))

    # helper internal function
    def create_profiles(n, mean=0, shift_indices=None, shift_val=2.0):
        data = np.random.normal(mean, 1, (n, n_features))
        if shift_indices:
            data[:, shift_indices] += shift_val
        return data

    # Create states
    ctrl = create_profiles(n_ctrl)
    disease = create_profiles(n_others, shift_indices=sig_indices)
    t_ctrl_like = create_profiles(n_others, shift_val=0.1, shift_indices=sig_indices)
    t_disease_like = create_profiles(n_others, shift_val=1.9, shift_indices=sig_indices)

    # t_different should be unlike control and disease in BOTH on and off features.
    # by shifting ALL features with a large value, it'll trigger high off_buscar_score
    # and a high on_buscar_score
    all_indices = list(range(n_features))
    t_different = create_profiles(n_others, shift_val=-5.0, shift_indices=all_indices)

    all_data = np.vstack([ctrl, disease, t_ctrl_like, t_disease_like, t_different])
    df = pl.DataFrame(all_data, schema=feature_names)

    treatments = (
        ["control"] * n_ctrl
        + ["disease"] * n_others
        + ["treatment_control_like"] * n_others
        + ["treatment_disease_like"] * n_others
        + ["treatment_different"] * n_others
    )

    df = df.with_columns(pl.Series("Metadata_treatment", treatments))

    return df, feature_names


@pytest.fixture
def not_significant_data():
    """
    Generates simulated morphological profiling data where no features are significant.
    """
    seed = 42
    np.random.seed(seed)
    n_features = 50
    n_ctrl = 100
    n_disease = 100

    feature_names = [f"Feature_{i}" for i in range(n_features)]

    # Both sets from same distribution (no shift)
    ctrl_data = np.random.normal(0, 1, (n_ctrl, n_features))
    disease_data = np.random.normal(0, 1, (n_disease, n_features))

    all_data = np.vstack([ctrl_data, disease_data])
    df = pl.DataFrame(all_data, schema=feature_names)

    treatments = ["control"] * n_ctrl + ["disease"] * n_disease
    df = df.with_columns(pl.Series("Metadata_treatment", treatments))

    return df, feature_names
