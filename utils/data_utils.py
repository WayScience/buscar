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
