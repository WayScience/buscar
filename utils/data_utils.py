import pathlib

import pandas as pd
import polars as pl
from beartype import beartype
from pycytominer.cyto_utils import infer_cp_features


@beartype
def split_meta_and_features(
    profile: pd.DataFrame | pl.DataFrame,
    compartments: list[str] = ["Nuclei", "Cells", "Cytoplasm"],
    metadata_tag: bool | None = False,
) -> tuple[list[str], list[str]]:
    """Splits metadata and feature column names

    This function takes a DataFrame containing image-based profiles and splits
    the column names into metadata and feature columns. It uses the Pycytominer's
    `infer_cp_features` function to identify feature columns based on the specified compartments.
    If the `metadata_tag` is set to False, it assumes that metadata columns do not have a specific tag
    and identifies them by excluding feature columns. If `metadata_tag` is True, it uses
    the `infer_cp_features` function with the `metadata` argument set to True.


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
    - If a polars DataFrame is provided, it will be converted to a pandas DataFrame in order
    to maintain compatibility with the `infer_cp_features` function.
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

    # iteratively search metadata features and retain order if the Metadata tag is not added
    if metadata_tag is False:
        meta_cols = [
            colname
            for colname in profile.columns.tolist()
            if colname not in features_cols
        ]
    else:
        meta_cols = infer_cp_features(profile, metadata=metadata_tag)

    return (meta_cols, features_cols)


@beartype
def load_group_stratified_data(
    dataset_path: str | pathlib.Path,
    group_columns: list[str] = ["Metadata_Plate", "Metadata_Well"],
    sample_percentage: float = 0.2,
) -> pl.DataFrame:
    """Memory-efficiently sample a percentage of rows from each group in a dataset.

    This function performs stratified sampling by loading only the grouping columns first
    to dtermine group memberships and sizes, then samples indices from each group, and
    finally loads the full dataset filtered to only the sampled rows. This approach
    minimizes memory usage compared to loading the entire dataset upfront.

    Parameters
    ----------
    dataset_path : str or pathlib.Path
        Path to the parquet dataset file to sample from
    group_columns : list[str], default ["Metadata_Plate", "Metadata_Well"]
        Column names to use for grouping. Sampling will be performed independently
        within each unique combination of these columns
    sample_percentage : float, default 0.2
        Fraction of rows to sample from each group (must be between 0.0 and 1.0)

    Returns
    -------
    pl.DataFrame
        Subsampled dataframe containing the sampled rows from each group,
        preserving all original columns

    Raises
    ------
    ValueError
        If sample_percentage is not between 0 and 1
    FileNotFoundError
        If dataset_path does not exist
    """
    # validate inputs
    if not 0 <= sample_percentage <= 1:
        raise ValueError("sample_percentage must be between 0 and 1")

    # convert str types to pathlib types
    if isinstance(dataset_path, str):
        dataset_path = pathlib.Path(dataset_path)

    dataset_path = dataset_path.resolve(strict=True)

    # load only the grouping columns to determine groups
    metadata_df = pl.read_parquet(dataset_path, columns=group_columns).with_row_index(
        "original_idx"
    )

    # sample indices for each group based on the group_columns
    sampled_indices = (
        metadata_df
        # group rows by the specified columns (e.g., Plate and Well combinations)
        .group_by(group_columns)
        # for each group, randomly sample a fraction of the original row indices
        .agg(
            pl.col("original_idx")
            .sample(fraction=sample_percentage)  # sample specified percentage from each group
            .alias("sampled_idx")  # rename the sampled indices column
        )
        # extract only the sampled indices column, discarding group identifiers
        .select("sampled_idx")
        # convert list of indices per group into individual rows (flatten the structure)
        .explode("sampled_idx")
        # extract the sampled indices as a single column series
        .get_column("sampled_idx")
        .sort()
    )

    # load the entire dataset and filter to sampled indices
    sampled_df = (
        pl.read_parquet(dataset_path)
        .with_row_index("idx")
        .filter(pl.col("idx").is_in(sampled_indices))
        .drop("idx")
    )

    return sampled_df
