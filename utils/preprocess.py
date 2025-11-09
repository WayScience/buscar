"""
This module contains utility functions for data preprocessing
"""

import polars as pl
from beartype import beartype
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@beartype
def apply_pca(
    profiles: pl.DataFrame,
    meta_features: list[str],
    morph_features: list[str],
    var_explained: float | int = 0.95,
    svd_solver="full",
    standardize=False,
    seed=0,
    **kwargs,
) -> pl.DataFrame:
    """Apply PCA to the morphological features of the profiles DataFrame.

    Parameters
    ----------
    profiles : pl.DataFrame
        Input DataFrame containing morphological and metadata features.
    meta_features : list[str]
        List of column names corresponding to metadata features.
    morph_features : list[str]
        List of column names corresponding to morphological features.
    var_explained : float or int, optional
        If a float between 0 and 1 is provided, it specifies the fraction of variance
        to be explained by the selected components (default is 0.95). If an integer is
        provided, it specifies the exact number of principal components to keep.
    svd_solver : str, optional
        The SVD solver to use for PCA. Default is 'full'. Options include 'auto', 'full',
        'arpack', and 'randomized'.  Note: The behavior of the 'n_components' parameter
        depends on its type:
        - If a float between 0 and 1 is provided (e.g., var_explained), it specifies
        the fraction of variance to preserve, and the 'full' solver must be used.
        - If an integer is provided, it specifies the exact number of principal
        components to keep, and other solvers such as 'arpack' or 'randomized' may be
        appropriate.
        See scikit-learn's PCA documentation for details.
    standardize : bool, optional
        Whether to standardize the morphological features before applying PCA.
        Default is False.
    random_state : int, optional
        Random state for reproducibility. Default is 0.
    **kwargs
        Additional keyword arguments for PCA that can be found here:
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

    Returns
    -------
    pl.DataFrame
        DataFrame containing the metadata features and the principal components.

    """

    # check if there are nans in the feature space
    if profiles.select(morph_features).null_count().sum_horizontal().sum() > 0:
        raise ValueError(
            "Input data contains NaNs. Please handle them before applying PCA."
        )

    # if the var explained is between 0 and 1 and is not "full" raise an error
    if isinstance(var_explained, float) and not (0 < var_explained < 1):
        raise ValueError(
            "'var_explained' as a float must be between 0 and 1 (exclusive)."
            "Provide an integer for number of components instead."
        )
    # check if the solver is compatible with var_explained
    if isinstance(var_explained, float) and svd_solver != "full":
        raise ValueError(
            "When 'var_explained' is a float between 0 and 1, 'svd_solver' must be set "
            "to 'full' because PCA needs to determine the number of components that "
            "explain the specified variance."
        )

    # standardize data if specified
    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(profiles.select(morph_features).to_numpy())
    else:
        data_scaled = profiles.select(morph_features).to_numpy()

    # apply PCA
    pca = PCA(
        n_components=var_explained,
        random_state=seed,
        svd_solver=svd_solver,
        **kwargs,
    )
    principal_components = pca.fit_transform(data_scaled)

    # concat metadata infromation with principal components
    pca_colnames = [f"PC{i + 1}" for i in range(principal_components.shape[1])]
    return pl.concat(
        [
            profiles.select(meta_features),  # metadata df
            pl.DataFrame(
                principal_components, schema=pca_colnames
            ),  # PCA components df
        ],
        how="horizontal",
    )
