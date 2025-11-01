"""
Tests for utils.preprocess module
"""

import contextlib

import numpy as np
import polars as pl
import pytest

from utils.preprocess import apply_pca


@contextlib.contextmanager
def temporary_seed(seed: int):
    """Context manager to temporarily set numpy random seed"""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


# Fixtures at module level (shared across all test classes)
@pytest.fixture
def dummy_profiles():
    """Create dummy profiles DataFrame for testing"""
    with temporary_seed(0):
        n_samples = 100
        n_features = 50

        # Create dummy data
        morph_data = np.random.randn(n_samples, n_features)
        feature_cols = [f"Feature_{i}" for i in range(n_features)]

        # Create metadata
        metadata = {
            "Metadata_Well": [f"A{i:02d}" for i in range(n_samples)],
            "Metadata_Plate": ["Plate1"] * 50 + ["Plate2"] * 50,
            "Metadata_Treatment": np.random.choice(
                ["DMSO", "Drug_A", "Drug_B"], n_samples
            ),
        }

        # Combine into DataFrame
        df_dict = {**metadata}
        for i, col in enumerate(feature_cols):
            df_dict[col] = morph_data[:, i]

        return pl.DataFrame(df_dict)


@pytest.fixture
def meta_features():
    """Metadata feature names"""
    return ["Metadata_Well", "Metadata_Plate", "Metadata_Treatment"]


@pytest.fixture
def morph_features():
    """Morphological feature names"""
    return [f"Feature_{i}" for i in range(50)]


class TestApplyPCA:
    """Test suite for apply_pca function"""

    def test_basic_pca_application(self, dummy_profiles, meta_features, morph_features):
        """Test basic PCA application returns correct structure"""
        result = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.95,
        )

        # Check result is a DataFrame
        assert isinstance(result, pl.DataFrame)

        # Check number of rows matches input
        assert len(result) == len(dummy_profiles)

        # Check metadata columns are present
        for col in meta_features:
            assert col in result.columns

        # Check PC columns are present
        pc_cols = [col for col in result.columns if col.startswith("PC")]
        assert len(pc_cols) > 0

    def test_pca_variance_explained(
        self, dummy_profiles, meta_features, morph_features
    ):
        """Test that PCA captures requested variance"""
        result = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.90,
        )

        # Check that we get fewer components with lower variance threshold
        pc_cols = [col for col in result.columns if col.startswith("PC")]
        assert len(pc_cols) > 0
        assert len(pc_cols) < len(morph_features)  # Should reduce dimensions

    def test_pca_with_standardization(
        self, dummy_profiles, meta_features, morph_features
    ):
        """Test PCA with standardization enabled"""
        result_no_std = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.95,
            standardize=False,
        )

        result_with_std = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.95,
            standardize=True,
        )

        # Both should return valid DataFrames
        assert isinstance(result_no_std, pl.DataFrame)
        assert isinstance(result_with_std, pl.DataFrame)

        # Results should differ (standardization changes the transformation)
        pc1_no_std = result_no_std["PC1"].to_numpy()
        pc1_with_std = result_with_std["PC1"].to_numpy()
        assert not np.allclose(pc1_no_std, pc1_with_std)

    def test_pca_column_names(self, dummy_profiles, meta_features, morph_features):
        """Test that PC columns are named correctly"""
        result = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.80,
        )

        pc_cols = [col for col in result.columns if col.startswith("PC")]

        # Check PC naming is sequential
        for i, col in enumerate(pc_cols, start=1):
            assert col == f"PC{i}"

    def test_pca_preserves_metadata(
        self, dummy_profiles, meta_features, morph_features
    ):
        """Test that metadata values are preserved"""
        result = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.95,
        )

        # Check metadata values match original
        for col in meta_features:
            assert result[col].to_list() == dummy_profiles[col].to_list()

    def test_pca_with_nan_raises_error(
        self, dummy_profiles, meta_features, morph_features
    ):
        """Test that NaN values raise ValueError"""
        # Add NaN to one feature
        dummy_with_nan = dummy_profiles.with_columns(
            pl.when(pl.col("Feature_0").abs() > 2)
            .then(None)
            .otherwise(pl.col("Feature_0"))
            .alias("Feature_0")
        )

        with pytest.raises(ValueError, match="Input data contains NaNs"):
            apply_pca(
                profiles=dummy_with_nan,
                meta_features=meta_features,
                morph_features=morph_features,
            )

    def test_pca_random_state(self, dummy_profiles, meta_features, morph_features):
        """Test that random state produces reproducible results"""
        result1 = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.95,
            random_state=42,
        )

        result2 = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.95,
            random_state=42,
        )

        # Results should be identical with same random state
        pc1_result1 = result1["PC1"].to_numpy()
        pc1_result2 = result2["PC1"].to_numpy()
        assert np.allclose(pc1_result1, pc1_result2)

    def test_pca_high_variance_threshold(
        self, dummy_profiles, meta_features, morph_features
    ):
        """Test with high variance threshold (captures most variance)"""
        result = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.99,
        )

        pc_cols = [col for col in result.columns if col.startswith("PC")]
        # Should need more components for 99% variance
        assert len(pc_cols) > 0

    def test_pca_low_variance_threshold(
        self, dummy_profiles, meta_features, morph_features
    ):
        """Test with low variance threshold (fewer components)"""
        result = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.50,
        )

        pc_cols = [col for col in result.columns if col.startswith("PC")]
        # Should need fewer components for 50% variance
        assert len(pc_cols) > 0
        assert len(pc_cols) < len(morph_features)

    def test_pca_output_shape(self, dummy_profiles, meta_features, morph_features):
        """Test that output shape is correct"""
        result = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.95,
        )

        # Number of rows should match
        assert len(result) == len(dummy_profiles)

        # Number of columns = metadata + PCs
        pc_cols = [col for col in result.columns if col.startswith("PC")]
        expected_cols = len(meta_features) + len(pc_cols)
        assert len(result.columns) == expected_cols

    def test_pca_with_small_dataset(self, meta_features, morph_features):
        """Test PCA with very small dataset"""
        # Create minimal dataset
        with temporary_seed(42):
            small_data = {
                "Metadata_Well": ["A01", "A02", "A03"],
                "Metadata_Plate": ["P1", "P1", "P1"],
                "Metadata_Treatment": ["DMSO", "Drug", "DMSO"],
            }
            for feat in morph_features[:10]:  # Use only 10 features
                small_data[feat] = np.random.randn(3)

            small_df = pl.DataFrame(small_data)

        result = apply_pca(
            profiles=small_df,
            meta_features=meta_features,
            morph_features=morph_features[:10],
            var_explained=0.95,
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_pca_kwargs_passed_to_sklearn(
        self, dummy_profiles, meta_features, morph_features
    ):
        """Test that additional kwargs are passed to sklearn PCA"""
        # Using whiten parameter as example
        result = apply_pca(
            profiles=dummy_profiles,
            meta_features=meta_features,
            morph_features=morph_features,
            var_explained=0.95,
            whiten=True,  # Additional sklearn PCA parameter
        )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(dummy_profiles)
