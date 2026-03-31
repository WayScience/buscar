import polars as pl
import pytest

from buscar.signatures import identify_signatures


def test_identify_signatures(synthetic_profiles):
    df, features = synthetic_profiles

    # Filter for control and disease
    ctrl_df = df.filter(pl.col("Metadata_treatment") == "control")
    disease_df = df.filter(pl.col("Metadata_treatment") == "disease")

    # Test KS-test
    sig, non_sig, ambig = identify_signatures(
        ref_profiles=ctrl_df,
        exp_profiles=disease_df,
        morph_feats=features,
        test_method="ks_test",
        p_threshold=0.05,
    )

    # Assertions based on data characteristics (30 non-sig, 270 sig)
    assert len(sig) > 200  # Should catch most sig ones
    assert len(non_sig) > 0  # Should catch some non-sig ones
    assert isinstance(sig, list)
    assert isinstance(non_sig, list)
    assert isinstance(ambig, list)


def test_identify_signatures_methods(synthetic_profiles):
    df, features = synthetic_profiles
    ctrl_df = df.filter(pl.col("Metadata_treatment") == "control")
    disease_df = df.filter(pl.col("Metadata_treatment") == "disease")

    # Quick check for other methods (only checking it runs)
    for method in ["welchs_ttest", "rank_test"]:
        sig, _, _ = identify_signatures(
            ref_profiles=ctrl_df,
            exp_profiles=disease_df,
            morph_feats=features[:50],  # Subset for speed
            test_method=method,
        )
        assert len(sig) >= 0


def test_identify_signatures_no_significance(not_significant_data):
    """
    Tests that identify_signatures raises a ValueError when no features are significant.
    """
    df, features = not_significant_data

    ctrl_df = df.filter(pl.col("Metadata_treatment") == "control")
    disease_df = df.filter(pl.col("Metadata_treatment") == "disease")

    with pytest.raises(
        ValueError,
        match=(
            "No significant features found. "
            "Consider adjusting the p-value threshold or padding."
        ),
    ):
        identify_signatures(
            ref_profiles=ctrl_df,
            exp_profiles=disease_df,
            morph_feats=features,
            test_method="ks_test",
            p_threshold=0.05,
        )
