"""
Module: utils.py

A collection of common utility functions for data processing,
as well as for saving, loading, and writing files.
"""

from collections import defaultdict

import pandas as pd
import polars as pl
from pycytominer.cyto_utils import infer_cp_features


def _sort_features_by_compartment_organelles(
    features: list[str],
    compartment_pos: int = 0,
    organelle_pos: int = 3,
    organelles: list[str] = ["DNA", "RNA", "ER", "Mito", "AGP"],
) -> dict:
    """Sort features by compartment and organelle.

    This function takes a list of feature names and organizes them into a nested dictionary
    structure where the first level is compartments and the second level is organelles.
    It filters out features that do not match the specified organelle list.

    Parameters
    ----------
    features : list[str]
        list of morpholgy features
    compartment_pos : int, optional
        position where the compartment name resides with the feature name
        , by default 0
    organelle_pos : int, optional
        position where the organelle name resides within the feature name
        , by default 3
    organelles : list[str], optional
        List of organelles that are measured in the feature space,
        by default ["DNA", "RNA", "ER", "Mito", "AGP"]

    Returns
    -------
    dict
        Nested dictionary: compartment -> organelle -> features
    """

    result = defaultdict(list)
    for feature in features:
        # Skip AreaShape features as they don't contain organelle information
        if "AreaShape" in feature:
            continue

        # Split feature name and validate structure
        split_feature = feature.split("_")
        if len(split_feature) < 4:
            continue

        # Extract compartment and organelle from feature name
        compartment = split_feature[compartment_pos]
        organelle = split_feature[organelle_pos]

        # Only include features with valid organelles
        if organelle in organelles:
            result[compartment].append(feature)

    # Create nested dictionary: compartment -> organelle -> features
    compartment_organelle_dict = defaultdict(dict)
    for compartment, features_list in result.items():
        organelle_dict = defaultdict(list)

        # Group features by organelle within each compartment
        for feature in features_list:
            organelle = feature.split("_")[organelle_pos]
            organelle_dict[organelle].append(feature)

        compartment_organelle_dict[compartment] = organelle_dict

    return compartment_organelle_dict


def _generate_organelle_counts(compartment_organelle_dict: dict) -> dict:
    """Generate a count of organelles per compartment for each gene.

    This function processes a nested dictionary containing gene signatures organized
    by compartment and organelle, and returns the count of features for each
    organelle within each compartment for every gene.

    Parameters
    ----------
    compartment_organelle_dict : dict
        Nested dictionary structure:
        gene -> signature_type -> compartment -> organelle -> list of features
        Where signature_type is 'on_morph_sig' or 'off_morph_sig'

    Returns
    -------
    dict
        Dictionary structure: gene -> signature_type -> compartment -> organelle -> count
        Where count is the number of features for each organelle in each compartment

    Raises
    ------
    TypeError
        If the organelle_dict for any gene is not a dictionary
    """
    # Initialize a nested dictionary to hold the counts
    # This will be structured as: gene -> signature_type -> compartment -> organelle -> count
    feature_count_per_organelle = defaultdict(lambda: defaultdict(dict))

    # Iterate through every gene's on and off morphology signatures that are sorted by
    # compartment and organelle
    for gene, signature_dict in compartment_organelle_dict.items():
        if not isinstance(signature_dict, dict):
            raise TypeError(
                f"Expected signature_dict to be a dict for gene {gene}, got {type(signature_dict)}"
            )

        # Process each signature type (on_morph_sig, off_morph_sig)
        counted_organelle_per_signature = defaultdict(dict)
        for sig_type, compartment_dict in signature_dict.items():
            # For each compartment-organelle combination, count the number of features
            counted_organelle_dict = defaultdict(dict)
            for compartment, organelle_dict in compartment_dict.items():
                for organelle, features in organelle_dict.items():
                    counted_organelle_dict[compartment][organelle] = len(features)
            counted_organelle_per_signature[sig_type] = counted_organelle_dict

        # Store the counted organelle dictionary per gene and signature type
        feature_count_per_organelle[gene] = counted_organelle_per_signature

    return feature_count_per_organelle


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


def group_signature_by_compartment(signatures: dict, compartment_pos: int = 0):
    """Group gene features in each signature by their compartment.

    This function takes a dictionary of gene signatures and groups the features
    by their compartment. The compartment is determined by the position in the
    feature string, which is specified by the `compartment_pos` parameter.

    Parameters
    ----------
    signatures : dict
        A dictionary containing gene signatures.
    compartment_pos : int, optional
        The position of the compartment in the feature string, by default 0

    Returns
    -------
    dict
        A dictionary with genes as keys and their grouped features as values.
        The structure is: gene --> signature_type -> compartment -> features
    """
    # Type validation
    if not isinstance(signatures, dict):
        raise TypeError("signatures must be a dictionary")
    if not isinstance(compartment_pos, int):
        raise TypeError("compartment_pos must be an integer")

    # Initialize the result dictionary
    gene_signature_grouped_by_compartment = defaultdict(lambda: defaultdict(dict))

    # Process each gene and its signatures
    for gene, signature_dict in signatures.items():
        # get features from each signature type
        for sig_type, features in signature_dict.items():
            # Group features by compartment for this signature type
            compartment_groups = defaultdict(list)
            for feature in features:
                try:
                    compartment = feature.split("_")[compartment_pos]
                    compartment_groups[compartment].append(feature)

                # Handle features that don't have enough parts when split
                except IndexError:
                    continue

            # Store the grouped features
            gene_signature_grouped_by_compartment[gene][sig_type] = dict(
                compartment_groups
            )

    return gene_signature_grouped_by_compartment


def group_features_by_compartment_organelle(
    signatures: dict,
    compartments: list[str] = ["Nuclei", "Cytoplasm", "Cells"],
    organelles: list[str] = ["DNA", "RNA", "ER", "Mito", "AGP"],
    compartment_pos: int = 0,
    organelle_pos: int = 3,
) -> dict:
    """Group features by compartment and organelle from gene on- and off-morphology
    signatures.

    This function processes on- off- signatures of each gene to organize morphological
    features into nested dictionaries based on compartment and organelle groupings.
    It applies validation checks and uses the helper function `_sort_compartment_organelles`
    to structure the data.

    Keep note that some features are removed since this function is solely looking
    for features that contain organelle information. For example, features that have AreaShape
    measurements do not contain organelle information and therefore are excluded.

    Parameters
    ----------
    signatures : dict
        Dictionary where keys are gene names and values are dictionaries containing
        'on_morph_sig' and 'off_morph_sig' lists of morphological features
    compartments : list[str], optional
        List of valid compartment names, by default ["Nuclei", "Cytoplasm", "Cells"]
    organelles : list[str], optional
        List of valid organelle names, by default ["DNA", "RNA", "ER", "Mito", "AGP"]
    compartment_pos : int, optional
        Position index for compartment name in feature string, by default 0
    organelle_pos : int, optional
        Position index for organelle name in feature string, by default 3

    Returns
    -------
    dict
        Nested dictionary structure:
        gene -> {'on_morph_sig': {compartment: {organelle: [features]}},
                'off_morph_sig': {compartment: {organelle: [features]}}}

    Raises
    ------
    TypeError
        If signatures is not a dict with proper structure, or if compartments/organelles
        are not lists of strings, or if position parameters are not integers
    ValueError
        If position parameters are negative or equal to each other
    """

    # type checking for compartments and organelles
    if not isinstance(signatures, dict):
        raise TypeError("Signatures must be a dictionary.")
    if not isinstance(compartments, list) or not isinstance(organelles, list):
        raise TypeError("Compartments and organelles must be lists.")
    if not all(isinstance(compartment, str) for compartment in compartments):
        raise TypeError("All compartments must be strings.")
    if not all(isinstance(organelle, str) for organelle in organelles):
        raise TypeError("All organelles must be strings.")
    if not isinstance(compartment_pos, int) or not isinstance(organelle_pos, int):
        raise TypeError("Compartment and organelle positions must be integers.")
    if compartment_pos < 0 or organelle_pos < 0:
        raise ValueError("Compartment and organelle positions must be non-negative.")
    if compartment_pos == organelle_pos:
        raise ValueError("Compartment and organelle positions must be different.")

    # Group features by compartment that contain organelle information
    sorted_compartment_and_organelle_per_gene = defaultdict(dict)
    for gene, signature_dict in signatures.items():
        # extracting features from signatures
        on_sig_features = _sort_features_by_compartment_organelles(
            signature_dict["on_morph_sig"]
        )
        off_sig_features = _sort_features_by_compartment_organelles(
            signature_dict["off_morph_sig"]
        )

        # Combine the sorted features for the gene
        sorted_compartment_and_organelle_per_gene[gene] = {
            "on_morph_sig": on_sig_features,
            "off_morph_sig": off_sig_features,
        }

    return sorted_compartment_and_organelle_per_gene


def organelle_count_table_per_gene(
    sorted_signatures: dict, stratify_by_compartment: bool = False
) -> pd.DataFrame:
    """Generate a count table of organelles per gene from morphological signatures.

    This function processes gene signatures that have been organized by compartment
    and organelle to create a summary table showing the count of features for each
    organelle within each gene's on- and off-morphology signatures.

    Parameters
    ----------
    sorted_signatures : dict
        Nested dictionary structure containing gene signatures organized by compartment
        and organelle. Expected format:
        gene -> signature_type -> compartment -> organelle -> list of features
        where signature_type is 'on_morph_sig' or 'off_morph_sig'
    stratify_by_compartment : bool, optional
        If True, creates separate columns for each compartment-organelle combination
        (e.g., "Cyto_DNA", "Nuc_RNA"). If False, sums counts across all compartments
        for each organelle, by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with organelle counts per gene and signature type. Structure depends
        on stratify_by_compartment parameter:
        - If True: columns are compartment_organelle combinations (e.g., "Cyto_DNA")
        - If False: columns are organelle names with counts summed across compartments
        Index contains gene names, with 'sig_type' column indicating 'on' or 'off'

    Notes
    -----
    - Each gene will have two rows in the output: one for 'on' signatures and one for 'off'
    - Compartment names are abbreviated: "Cytoplasm" -> "Cyto", "Nuclei" -> "Nuc"
    - Missing organelle counts are filled with 0
    - The function uses the helper function `_generate_organelle_counts` to process
      the input data structure


    """
    # count organelles per compartment
    organelle_counts = _generate_organelle_counts(sorted_signatures)

    # initialize an empty DataFrame to hold the counts
    organelle_counted_per_gene = pd.DataFrame()

    # iterate through each gene and its morphological signatures
    for gene, morph_signatures in organelle_counts.items():
        # iterate through each signature type (on_morph_sig, off_morph_sig)
        for sig_type, compartment_organelle_counts in morph_signatures.items():
            # convert nested dict to DataFrame with compartments as index and organelles as columns
            count_table = (
                pd.DataFrame.from_dict(compartment_organelle_counts, orient="index")
                .fillna(0)
                .astype(int)
            )

            if stratify_by_compartment:
                # create compartment-organelle combinations as columns
                flattened_data = []
                column_names = []

                for compartment in count_table.index:
                    # abbreviate compartment names
                    compartment_abbrev = (
                        "Cyto"
                        if compartment == "Cytoplasm"
                        else "Nuc"
                        if compartment == "Nuclei"
                        else compartment
                    )

                    # add compartment-organelle combinations
                    for organelle in count_table.columns:
                        column_names.append(f"{compartment_abbrev}_{organelle}")
                        flattened_data.append(count_table.loc[compartment, organelle])

                # create DataFrame with flattened structure
                gene_row = pd.DataFrame(
                    [flattened_data], columns=column_names, index=[gene]
                )
            else:
                # sum counts across all compartments for each organelle
                gene_row = count_table.sum().to_frame().T
                gene_row.index = [gene]

            # add signature type column
            gene_row.insert(0, "sig_type", sig_type.split("_")[0])

            # concatenate to main DataFrame
            organelle_counted_per_gene = pd.concat(
                [organelle_counted_per_gene, gene_row]
            ).fillna(0)

    return organelle_counted_per_gene


def generate_consensus_signatures(
    signatures_dict, features: list[str], min_consensus_threshold=0.5
):
    """
    Generate consensus morphological signatures from multiple comparisons.

    This function aggregates on-morphology signatures across different negative control samples
    for each positive control, finding features that consistently appear across multiple comparisons.
    The off-morphology signatures are then defined as the complement of on-morphology features
    from the full feature set.

    Parameters
    ----------
    signatures_dict : dict
        Dictionary containing signature results with structure:
        {comparison_id: {"controls": {"positive": gene, "negative": seed},
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
        {gene: {"on": [feature1, feature2, ...], "off": [feature1, feature2, ...]}}
        where "off" features are the complement of "on" features from the full feature set

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
            f"min_consensus_threshold must be between 0.0 and 1.0, got {min_consensus_threshold}"
        )

    if not signatures_dict:
        return {}

    # Group on-morphology signatures by positive control gene
    on_signatures_by_gene = defaultdict(list)

    try:
        for _, sig_results in signatures_dict.items():
            positive_control = sig_results["controls"]["positive"]
            on_signature_features = sig_results["signatures"]["on"]
            on_signatures_by_gene[positive_control].append(on_signature_features)

    except KeyError as e:
        raise KeyError(f"Missing required key in signatures_dict: {e}")

    # Generate consensus signatures for each gene
    consensus_signatures = {}
    full_features_set = set(features)

    for gene, feature_lists in on_signatures_by_gene.items():
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
            min_count = total_lists if min_consensus_threshold == 1.0 else max(1, int(total_lists * min_consensus_threshold))

            # Select features meeting threshold
            consensus_on_features = sorted([
                feature for feature, count in feature_counts.items()
                if count >= min_count
            ])

        # Generate off-features as complement of on-features
        consensus_off_features = sorted(list(full_features_set - set(consensus_on_features)))

        # Store results
        consensus_signatures[gene] = {
            "on": consensus_on_features,
            "off": consensus_off_features,
        }

    return consensus_signatures
