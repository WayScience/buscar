import numpy as np
import polars as pl
import pytest

from buscar.metrics import (
    calculate_buscar_scores,
    calculate_score,
    compute_earth_movers_distance,
)


def test_calculate_buscar_scores(synthetic_profiles):
    df, features = synthetic_profiles

    # Identify signatures first (simple way, subsets for speed)
    on_sig = features[30:80]  # Part of significantly different ones
    off_sig = features[:30]  # Non-significant ones

    # Run the main scoring function
    with pytest.warns(
        UserWarning,
        match="No features were assigned to the following signature categories",
    ):
        scored_df = calculate_buscar_scores(
            profiles=df,
            meta_cols=["Metadata_treatment"],
            on_morphology_signature=on_sig,
            off_morphology_signature=off_sig,
            ref_state="disease",
            target="control",
            perturbation_col="Metadata_treatment",
            on_method="emd",
            off_method="affected_ratio",
            raw_emd_scores=False,
        )

    # Column assertions
    expected_cols = [
        "target",
        "perturbation",
        "on_buscar_scores",
        "off_buscar_scores",
        "is_reference_distance",
    ]
    assert all(col in scored_df.columns for col in expected_cols)

    # Ranking/Logic assertions base on data characteristics
    # Disease-like should have higher on_buscar_score (far from control)
    # Control-like should have lower on_buscar_score (near control)
    # However, normalization makes target state (disease) score 1.0.

    # res = scored_df.to_pandas().set_index("treatment")

    # on_buscar_score for disease should be 1.0 (normalization target)
    disease_score = scored_df.filter(pl.col("perturbation") == "disease")[
        "on_buscar_scores"
    ][0]
    assert disease_score == pytest.approx(1.0)

    # treatment_control_like should have lower on_buscar_score than disease (closer to
    # reference)
    ctrl_like_score = scored_df.filter(
        pl.col("perturbation") == "treatment_control_like"
    )["on_buscar_scores"][0]
    assert ctrl_like_score < 1.0

    # treatment_disease_like should have similar score to disease (~1.0)
    disease_like_score = scored_df.filter(
        pl.col("perturbation") == "treatment_disease_like"
    )["on_buscar_scores"][0]
    assert 0.8 < disease_like_score < 1.2

    # t_different should have the highest on_buscar_score and off_buscar_score
    t_diff_row = scored_df.filter(pl.col("perturbation") == "treatment_different")
    t_diff_on = t_diff_row["on_buscar_scores"][0]
    t_diff_off = t_diff_row["off_buscar_scores"][0]

    # compare t_diff on/off scores to all other treatment scores
    all_other_on_scores = scored_df.filter(
        pl.col("perturbation") != "treatment_different"
    )["on_buscar_scores"]
    all_other_off_scores = scored_df.filter(
        pl.col("perturbation") != "treatment_different"
    )["off_buscar_scores"]

    # Assert t_different is greater than all other treatments in on_buscar_score and
    # off_buscar_score
    assert all(t_diff_on > score for score in all_other_on_scores)
    assert all(t_diff_off > score for score in all_other_off_scores)


def test_emd_direct(synthetic_profiles):
    df, features = synthetic_profiles
    ctrl_df = df.filter(pl.col("Metadata_treatment") == "control").select(features[:10])
    disease_df = df.filter(pl.col("Metadata_treatment") == "disease").select(
        features[:10]
    )

    emd = compute_earth_movers_distance(ctrl_df, disease_df, subsample_size=50)
    assert emd > 0.0


def test_calculate_score_small_group_returns_nan_not_zero():
    """A too-small treated group must not silently report a 0.0 off-score.

    With only 1-2 rows in the treated group, the significance test has no power
    to detect anything -- even a real, moderate shift in every feature (one that
    IS picked up once the group is large enough, see below) comes back as "not
    significant", producing the same 0.0 as a genuinely unaffected group.
    calculate_score should refuse to compute a score below a minimum group
    size, warn, and return NaN so the two situations aren't conflated.
    """
    rng = np.random.default_rng(0)
    n_features = 10
    feature_names = [f"Feature_{i}" for i in range(n_features)]

    target_profile = pl.DataFrame(
        rng.normal(0, 1, (200, n_features)), schema=feature_names
    )

    # A real, moderate shift in every feature. With a large enough treated
    # group this is detected (score > 0, see the comparison below); the bug is
    # that a too-small group hides that real effect behind the same 0.0 a
    # genuinely unaffected group would produce.
    shift = 0.5
    tiny_treated = pl.DataFrame(
        rng.normal(shift, 1, (2, n_features)), schema=feature_names
    )

    with pytest.warns(UserWarning, match="minimum group size"):
        score = calculate_score(
            target_profile,
            tiny_treated,
            feature_names,
            signature_type="off",
        )

    assert score != score  # NaN check (NaN != NaN), never the misleading 0.0

    # Sanity check: the same real effect, given enough rows, is NOT 0.0 --
    # proving 0.0-from-too-few-rows and 0.0-from-no-effect really were
    # indistinguishable before this guard existed.
    large_treated = pl.DataFrame(
        rng.normal(shift, 1, (500, n_features)), schema=feature_names
    )
    large_score = calculate_score(
        target_profile,
        large_treated,
        feature_names,
        signature_type="off",
    )
    assert large_score > 0.0
