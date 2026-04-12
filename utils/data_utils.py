"""
Module: data_utils.py

Utility functions for processing image-based single-cell profiles, including
feature/metadata splitting, hash-based cell ID generation, data shuffling,
consensus signature generation, and profile loading/concatenation.
"""

import hashlib
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from pycytominer.cyto_utils import infer_cp_features


def split_meta_and_features(
    profile: pd.DataFrame | pl.DataFrame,
    compartments: list[str] = ["Nuclei", "Cells", "Cytoplasm"],
    metadata_tag: bool | None = False,
) -> tuple[list[str], list[str]]:
    """Split column names of an image-based profile into metadata and feature lists.

    Uses pycytominer's `infer_cp_features` to identify CellProfiler feature columns
    based on the specified compartments. Metadata columns are identified as all
    remaining columns not in the feature set (when `metadata_tag=False`), or via
    `infer_cp_features(metadata=True)` when `metadata_tag=True`.

    Parameters
    ----------
    profile : pd.DataFrame | pl.DataFrame
        Dataframe containing image-based profile
    compartments : list, optional
        compartments used to generated image-based profiles, by default
        ["Nuclei", "Cells", "Cytoplasm"]
    metadata_tag : Optional[bool], optional
        indicating if the profiles have metadata columns tagged with 'Metadata_'
        , by default False

    Returns
    -------
    tuple[List[str], List[str]]
        Tuple containing metadata and feature column names

    Notes
    -----
    - If a polars DataFrame is provided, it will be converted to a pandas DataFrame in
      order to maintain compatibility with the `infer_cp_features` function.
    """

    # type checking
    if not isinstance(profile, (pd.DataFrame, pl.DataFrame)):
        raise TypeError("profile must be a pandas or polars DataFrame")
    if isinstance(profile, pl.DataFrame):
        # convert Polars DataFrame to Pandas DataFrame for compatibility
        profile = profile.to_pandas()
    if not isinstance(compartments, list):
        raise TypeError("compartments must be a list of strings")

    # identify features names
    features_cols = infer_cp_features(profile, compartments=compartments)

    # iteratively search metadata features and retain order if the Metadata tag is not
    # added
    if metadata_tag is False:
        meta_cols = [
            colname
            for colname in profile.columns.tolist()
            if colname not in features_cols
        ]
    else:
        meta_cols = infer_cp_features(profile, metadata=metadata_tag)

    return (meta_cols, features_cols)


def generate_consensus_signatures(
    signatures_dict, features: list[str], min_consensus_threshold=0.5
) -> dict:
    """
    Generate consensus on/off morphological signatures across multiple comparisons.

    For each positive control, aggregates on-morphology features across all
    comparisons and retains only those features that appear in at least
    `min_consensus_threshold` fraction of comparisons. Off-morphology features are
    defined as all features NOT in the consensus on-set.

    Parameters
    ----------
    signatures_dict : dict
        Dictionary containing signature results with structure:
        {comparison_id: {"controls": {"positive": label, "negative": seed},
                        "signatures": {"on": [...], "off": [...]}}}
    features : list[str]
        Complete list of all available morphological features
    min_consensus_threshold : float, default 0.5
        Minimum fraction of comparisons a feature must appear in to be included
        in consensus (0.0 to 1.0). Use 1.0 for strict intersection (default behavior)

    Returns
    -------
    dict
        Dictionary with structure:
        {label: {"on": [feature1, feature2, ...], "off": [feature1, feature2, ...]}}
        where "off" features are the complement of "on" features from the full feature
        set

    Raises
    ------
    ValueError
        If min_consensus_threshold is not between 0.0 and 1.0
    KeyError
        If required keys are missing from signatures_dict

    """
    # Input validation
    if not 0.0 <= min_consensus_threshold <= 1.0:
        raise ValueError(
            "min_consensus_threshold must be between 0.0 and 1.0, "
            f"got {min_consensus_threshold}"
        )

    if not signatures_dict:
        return {}

    # Group on-morphology signatures by positive control label
    on_signatures_by_label = defaultdict(list)

    try:
        for _, sig_results in signatures_dict.items():
            positive_control = sig_results["controls"]["positive"]
            on_signature_features = sig_results["signatures"]["on"]
            on_signatures_by_label[positive_control].append(on_signature_features)

    except KeyError as e:
        raise KeyError(f"Missing required key in signatures_dict: {e}")

    # Generate consensus signatures for each label
    consensus_signatures = {}
    full_features_set = set(features)

    for label, feature_lists in on_signatures_by_label.items():
        # Calculate consensus on-features
        if not feature_lists:
            consensus_on_features = []
        elif len(feature_lists) == 1:
            consensus_on_features = sorted(feature_lists[0])
        else:
            # Count feature occurrences and apply threshold
            feature_counts = defaultdict(int)
            for feature_list in feature_lists:
                for feature in set(feature_list):  # Remove duplicates within list
                    feature_counts[feature] += 1

            # Determine minimum count threshold
            total_lists = len(feature_lists)
            min_count = (
                total_lists
                if min_consensus_threshold == 1.0
                else max(1, int(total_lists * min_consensus_threshold))
            )

            # Select features meeting threshold
            consensus_on_features = sorted(
                [
                    feature
                    for feature, count in feature_counts.items()
                    if count >= min_count
                ]
            )

        # Generate off-features as complement of on-features
        consensus_off_features = sorted(
            list(full_features_set - set(consensus_on_features))
        )

        # Store results
        consensus_signatures[label] = {
            "on": consensus_on_features,
            "off": consensus_off_features,
        }

    return consensus_signatures


def _hash_string(s: str) -> str:
    """Apply MD5 hash to a string."""
    return hashlib.md5(s.encode()).hexdigest()


def add_cell_id_hash(
    profiles: pl.DataFrame,
    seed: int = 0,
    force: bool = False,
) -> pl.DataFrame:
    """Add a unique hash column to a DataFrame of image-based profiles.

    This function generates a unique hash identifier for each row in the DataFrame
    based on all column values and an optional seed for reproducibility. The hash
    is added as a new column named 'Metadata_cell_id' placed as the first column.

    Null values are temporarily replaced with a string representation for hashing
    purposes only - the original data remains unchanged.

    Parameters
    ----------
    profiles : pl.DataFrame
        DataFrame containing single-cell profiles or image-based data.
    seed : int, optional
        Seed value for reproducible hash generation, by default 0.
    force : bool, optional
        If True, overwrites existing 'Metadata_cell_id' column. If False and the
        column exists, returns the DataFrame unchanged with a warning message,
        by default False.
    Returns
    -------
    pl.DataFrame
        Original DataFrame with 'Metadata_cell_id' column added as the first column.
        Original data including nulls remains unchanged.

    Raises
    ------
    TypeError
        If profiles is not a Polars DataFrame.

    Notes
    -----
    - The hash is deterministic: same data and seed always produce the same hash
    - Uses MD5 hashing for stable, reproducible results across platforms and versions
    - The 'Metadata_cell_id' column is positioned as the first column in the output
    - Null values are converted to strings ONLY for hash generation; original nulls
      are preserved in the returned DataFrame
    """
    if not isinstance(profiles, pl.DataFrame):
        raise TypeError("profiles must be a Polars DataFrame")

    # Handle existing column
    if "Metadata_cell_id" in profiles.columns:
        if not force:
            print(
                "'Metadata_cell_id' column already exists in the DataFrame. "
                "Set force=True to overwrite the existing column."
            )
            return profiles
        else:
            profiles = profiles.drop("Metadata_cell_id")

    # Create hash column using a null-filled version of each column (nulls → "NULL")
    hash_column = (
        pl.concat_str(
            [pl.col(col).cast(pl.Utf8).fill_null("NULL") for col in profiles.columns]
            + [pl.lit(f"|{seed}")],
            separator="|",
        )
        .map_elements(_hash_string, return_dtype=pl.Utf8)
        .alias("Metadata_cell_id")
    )

    # Add the hash column to the ORIGINAL profiles (with nulls intact)
    return profiles.with_columns(hash_column).select(
        ["Metadata_cell_id"] + profiles.columns
    )


def shuffle_feature_profiles(
    profiles: pl.DataFrame,
    feature_cols: list[str],
    method: Literal["row", "column", "label"] = "row",
    label_col: str | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Create a shuffled version of the dataset where each morphological feature
    column is independently shuffled (values permuted within each column).

    This breaks the correlation structure between features while preserving
    the marginal distributions, creating a null baseline for comparison.

    Parameters
    ----------
    profiles : pl.DataFrame
        Original dataframe with features and metadata
    feature_cols : list[str]
        List of morphological feature column names to shuffle
    method : str, optional
        Method of shuffling: "row" (shuffle rows) or "column" (shuffle each column
        independently), by default "row"
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pl.DataFrame
        Shuffled dataframe with same structure but permuted feature values
    """
    np.random.seed(seed)

    # Get metadata columns (everything not in feature_cols)
    meta_cols = [c for c in profiles.columns if c not in feature_cols]
    metadata_df = profiles.select(meta_cols)

    # Create shuffled feature columns
    if method == "row":
        return pl.concat(
            [
                metadata_df,
                profiles.select(feature_cols).sample(
                    fraction=1.0, seed=seed, shuffle=True
                ),
            ],
            how="horizontal",
        )

    # column-wise shuffling
    elif method == "column":
        # Use a stable hash-derived seed per column so shuffling is reproducible
        return profiles.with_columns(
            [
                pl.col(col).shuffle(
                    seed=seed
                    + int(hashlib.md5(col.encode("utf-8")).hexdigest()[:8], 16)
                )
                for col in feature_cols
            ]
        )
    elif method == "label":
        if label_col is None:
            raise ValueError(
                "label_col must be specified when using 'label' shuffle method."
            )

        # return the profiels with
        return profiles.with_columns(pl.col(label_col).shuffle(seed=seed))
    else:
        raise ValueError(f"Unknown shuffle method: {method}")


def shuffle_signatures(
    on_sig: list[str], off_sig: list[str], all_features: list[str], seed: int = 0
) -> tuple[list[str], list[str]]:
    """
    Breaks biological meaning of on/off signatures by randomly sampling
    features from the full feature space, while preserving the original
    on/off size ratio.

    Preserves:
      - len(on_sig) and len(off_sig)  ← ratio intact
      - Features drawn from same pool as real signatures

    Breaks:
      - Which specific features are "on" vs "off"
      - Any biological grouping derived from KS test
    """
    rng = np.random.default_rng(seed)

    n_on = len(on_sig)
    n_off = len(off_sig)

    # guard: need enough features to fill both without overlap
    if n_on + n_off > len(all_features):
        raise ValueError(
            f"Not enough features ({len(all_features)}) to fill "
            f"on ({n_on}) + off ({n_off}) without replacement"
        )

    # sample without replacement so on and off don't overlap
    sampled = rng.choice(all_features, size=n_on + n_off, replace=False)

    shuffled_on = sampled[:n_on].tolist()
    shuffled_off = sampled[n_on:].tolist()

    return shuffled_on, shuffled_off


def split_data(
    pycytominer_output: pl.DataFrame, dataset: str = "CP_and_DP"
) -> pl.DataFrame:
    """
    Filter a pycytominer output DataFrame to retain only metadata and the
    selected feature modality columns.

    Parameters
    ----------
    pycytominer_output : pl.DataFrame
        Polars DataFrame from pycytominer containing both metadata and feature columns.
    dataset : str, optional
        Feature modality to retain. One of:
        - ``"CP"`` — CellProfiler features only (columns containing ``"CP__"``)
        - ``"DP"`` — DeepProfiler features only (columns containing ``"DP__"``)
        - ``"CP_and_DP"`` — both modalities (default)

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with metadata and selected features
    """
    all_cols = pycytominer_output.columns

    # Get DP, CP, or both features from all columns depending on desired dataset
    if dataset == "CP":
        feature_cols = [col for col in all_cols if "CP__" in col]
    elif dataset == "DP":
        feature_cols = [col for col in all_cols if "DP__" in col]
    elif dataset == "CP_and_DP":
        feature_cols = [col for col in all_cols if "P__" in col]
    else:
        raise ValueError(
            f"Invalid dataset '{dataset}'. Choose from 'CP', 'DP', or 'CP_and_DP'."
        )

    # Metadata columns is all columns except feature columns
    metadata_cols = [col for col in all_cols if "P__" not in col]

    # Select metadata and feature columns
    selected_cols = metadata_cols + feature_cols

    return pycytominer_output.select(selected_cols)


def remove_feature_prefixes(df: pl.DataFrame, prefix: str = "CP__") -> pl.DataFrame:
    """
    Strip a feature-modality prefix from all matching column names.

    For example, ``"CP__Cells_AreaShape_Area"`` becomes ``"Cells_AreaShape_Area"``
    when ``prefix="CP__"``.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame whose column names may contain the prefix.
    prefix : str, default ``"CP__"``
        Prefix string to strip from matching column names.

    Returns
    -------
    pl.DataFrame
        DataFrame with the prefix removed from all matching column names.
    """
    return df.rename(lambda x: x.replace(prefix, "") if prefix in x else x)
