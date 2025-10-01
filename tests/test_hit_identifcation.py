import sys
from pathlib import Path

import polars as pl
import pytest

# Add the parent directory to the path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.identify_hits import calculate_weighted_sum, identify_compound_hit


@pytest.fixture
def test_data_path():
    """Return the path to the test parquet file."""
    return Path(__file__).parent / "data" / "test_compound_cluster_scores.parquet"


@pytest.fixture
def distance_df(test_data_path):
    """Load the test distance DataFrame from parquet."""
    return pl.read_parquet(test_data_path)


def test_distance_df_structure(distance_df):
    """Test that the distance DataFrame has the expected structure."""
    expected_columns = [
        "control_cluster_id",
        "treatment_cluster_id",
        "treatment",
        "on_score",
        "off_score",
        "ratio",
    ]

    assert distance_df.columns == expected_columns
    assert distance_df.shape == (40, 6)


def test_distance_df_has_three_treatments(distance_df):
    """Test that there are exactly 3 treatments in the dataset."""
    treatments = distance_df.select("treatment").unique()
    assert treatments.height == 3
    assert set(treatments["treatment"].to_list()) == {"drug_x", "drug_y", "drug_z"}


def test_ratios_sum_to_one_per_treatment(distance_df):
    """Test that ratios sum to approximately 1.0 for each treatment."""
    ratio_sums = distance_df.group_by("treatment").agg(
        pl.col("ratio").sum().alias("ratio_sum")
    )

    for ratio_sum in ratio_sums["ratio_sum"]:
        assert abs(ratio_sum - 1.0) < 0.01, f"Ratio sum {ratio_sum} is not close to 1.0"


def test_calculate_weighted_sum_returns_correct_columns(distance_df):
    """Test that calculate_weighted_sum returns the expected columns."""
    # First create paired scores (simulating the pairing step)
    paired_df = (
        distance_df.sort(["treatment", "treatment_cluster_id", "on_score", "off_score"])
        .group_by(["treatment", "treatment_cluster_id"])
        .first()
    )

    result = calculate_weighted_sum(paired_df)

    assert "treatment" in result.columns
    assert "compound_score" in result.columns


def test_calculate_weighted_sum_correct_number_of_rows(distance_df):
    """Test that one row per treatment is returned."""
    paired_df = (
        distance_df.sort(["treatment", "treatment_cluster_id", "on_score", "off_score"])
        .group_by(["treatment", "treatment_cluster_id"])
        .first()
    )

    result = calculate_weighted_sum(paired_df)

    # 3 unique treatments in test data
    assert result.height == 3


def test_calculate_weighted_sum_sorted_ascending(distance_df):
    """Test that results are sorted by compound_score in ascending order."""
    paired_df = (
        distance_df.sort(["treatment", "treatment_cluster_id", "on_score", "off_score"])
        .group_by(["treatment", "treatment_cluster_id"])
        .first()
    )

    result = calculate_weighted_sum(paired_df)

    scores = result["compound_score"].to_list()
    assert scores == sorted(scores), "Compound scores are not sorted in ascending order"


def test_identify_compound_hit_returns_correct_columns(distance_df):
    """Test that identify_compound_hit returns expected columns."""
    result = identify_compound_hit(distance_df, method="weighted_sum")

    assert "treatment" in result.columns
    assert "compound_score" in result.columns
    assert "rank" in result.columns


def test_identify_compound_hit_ranks_correctly(distance_df):
    """Test that rank column is assigned correctly (1 = best/lowest score)."""
    result = identify_compound_hit(distance_df, method="weighted_sum")

    # Check that rank 1 has the lowest compound_score
    rank_1_row = result.filter(pl.col("rank") == 1)
    min_score = result["compound_score"].min()

    assert rank_1_row["compound_score"][0] == min_score


def test_identify_compound_hit_one_row_per_treatment(distance_df):
    """Test that one row per treatment is returned."""
    result = identify_compound_hit(distance_df, method="weighted_sum")

    # 3 unique treatments in test data
    assert result.height == 3


def test_identify_compound_hit_all_ranks_unique(distance_df):
    """Test that all ranks are unique (no ties)."""
    result = identify_compound_hit(distance_df, method="weighted_sum")

    ranks = result["rank"].to_list()
    assert len(ranks) == len(set(ranks)), "Ranks are not unique"
    assert set(ranks) == {1, 2, 3}, "Ranks should be 1, 2, 3"


def test_identify_compound_hit_drug_z_is_best(distance_df):
    """Test that drug_z has the best (lowest) compound score based on test data."""
    result = identify_compound_hit(distance_df, method="weighted_sum")

    # drug_z should have rank 1 (best performance - lowest scores)
    best_drug = result.filter(pl.col("rank") == 1)["treatment"][0]

    # Based on the test data, drug_z has the lowest on/off scores
    assert best_drug == "drug_z", f"Expected drug_z to be best, but got {best_drug}"


def test_pairing_logic_each_treatment_cluster_paired_once(distance_df):
    """Test that each treatment cluster is paired with exactly one control cluster."""
    # Replicate the pairing step from identify_compound_hit
    paired = (
        distance_df.lazy()
        .sort(["treatment", "treatment_cluster_id", "on_score", "off_score"])
        .group_by(["treatment", "treatment_cluster_id"])
        .agg([pl.all().first()])
        .collect()
    )

    # Each (treatment, treatment_cluster_id) should appear exactly once
    unique_pairs = paired.select(["treatment", "treatment_cluster_id"]).unique()
    assert paired.height == unique_pairs.height


def test_control_clusters_can_be_reused(distance_df):
    """Test that control clusters can be paired with multiple treatment clusters."""
    # Replicate the pairing step
    paired = (
        distance_df.sort(["treatment", "treatment_cluster_id", "on_score", "off_score"])
        .group_by(["treatment", "treatment_cluster_id"])
        .first()
    )

    # Check if any control cluster appears more than once
    control_cluster_counts = paired.group_by("control_cluster_id").agg(
        pl.len().alias("count")
    )

    # At least one control cluster should be reused
    max_count = control_cluster_counts["count"].max()
    assert max_count > 1, "Control clusters should be reusable"


def test_compound_score_is_weighted_sum(distance_df):
    """Test that compound_score is calculated as weighted sum of on and off scores."""
    paired = (
        distance_df.sort(["treatment", "treatment_cluster_id", "on_score", "off_score"])
        .group_by(["treatment", "treatment_cluster_id"])
        .first()
    )

    result = calculate_weighted_sum(paired)

    # Manually calculate compound score for one treatment
    drug_x_data = paired.filter(pl.col("treatment") == "drug_x")
    expected_score = (drug_x_data["on_score"] * drug_x_data["ratio"]).sum() + (
        drug_x_data["off_score"] * drug_x_data["ratio"]
    ).sum()

    actual_score = result.filter(pl.col("treatment") == "drug_x")["compound_score"][0]

    assert abs(actual_score - expected_score) < 0.01


def test_empty_dataframe_handling():
    """Test that functions handle empty DataFrames gracefully."""
    empty_df = pl.DataFrame(
        {
            "control_cluster_id": [],
            "treatment_cluster_id": [],
            "treatment": [],
            "on_score": [],
            "off_score": [],
            "ratio": [],
        }
    ).cast(
        {
            "control_cluster_id": pl.Int64,
            "treatment_cluster_id": pl.Int64,
            "treatment": pl.String,
            "on_score": pl.Float64,
            "off_score": pl.Float64,
            "ratio": pl.Float64,
        }
    )

    result = identify_compound_hit(empty_df, method="weighted_sum")
    assert result.height == 0


def test_all_scores_are_positive(distance_df):
    """Test that all on_score and off_score values are non-negative."""
    assert (distance_df["on_score"] >= 0).all()
    assert (distance_df["off_score"] >= 0).all()


def test_all_ratios_are_valid_probabilities(distance_df):
    """Test that all ratio values are between 0 and 1."""
    assert (distance_df["ratio"] >= 0).all()
    assert (distance_df["ratio"] <= 1).all()
