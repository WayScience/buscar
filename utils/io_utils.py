import json
import pathlib
import pickle

import polars as pl
import yaml


def load_profiles(fpath: str | pathlib.Path, convert_to_f32: bool = False, verbose: bool | None = False) -> pl.DataFrame:
    """Load single-cell profiles from given file path.

    Loads single-cell profiles and returns them into a Polars DataFrame. The supported
    file formats are Parquet (.parquet, .pq, .arrow). If the file does not exist or
    the format is not supported, an error is raised.

    Parameters
    ----------
    fpath : str | pathlib.Path
        Path to the file containing single-cell profiles.
    convert_to_f32 : bool, optional
        If True, converts all Float64 columns to Float32 to save memory. Default is False
    verbose : bool, optional
        If True, prints information about the loaded profiles. Default is False.

    Returns
    -------
    pl.DataFrame
        DataFrame containing the loaded single-cell profiles.

    Raises
    ------
    TypeError
        If `fpath` is not a string or pathlib.Path.
    FileNotFoundError
        If the file at `fpath` does not exist.
    ValueError
        If the file format is not supported. Supported formats are: .parquet, .pq, .arrow.
    """

    # type checking
    if not isinstance(fpath, (str, pathlib.Path)):
        raise TypeError(f"Expected str or pathlib.Path, got {type(fpath)}")
    if isinstance(fpath, str):
        fpath = pathlib.Path(fpath)
    if not fpath.is_file():
        raise FileNotFoundError(f"File not found: {fpath}")
    # check for supported file format
    if fpath.suffix.lower() not in [".parquet", ".pq", ".arrow"]:
        raise ValueError(f"Unsupported file format: {fpath.suffix}. Supported formats are: .parquet, .pq, .arrow")

    # load profiles
    loaded_profiles = pl.read_parquet(fpath)

    # convert all Float64 columns to Float32 if convert_to_f32 is True
    if convert_to_f32:
        loaded_profiles = loaded_profiles.with_columns(
            [pl.col(col).cast(pl.Float32) for col in loaded_profiles.columns if loaded_profiles.schema[col] == pl.Float64]
        )

    # if verbose is True, print information about the loaded profiles
    if verbose:
        print(f"Loading profiles from {fpath}...")
        print(f"Loaded profiles shape: rows: {loaded_profiles.shape[0]}, columns: {loaded_profiles.shape[1]}")
        print(f"Estimated loaded dataframe size: {round(loaded_profiles.estimated_size("mb"), 2)} MB")

    return loaded_profiles


def load_configs(fpath: str | pathlib.Path) -> dict:
    """Load a configuration file and return its contents as a dictionary.
    Parameters
    ----------
    fpath : str or pathlib.Path
        Path to the YAML, JSON, or pickle configuration file.
    Returns
    -------
    dict
        Dictionary containing the configuration loaded from the file.
    Raises
    ------
    TypeError
        If `fpath` is not a string or pathlib.Path.
    FileNotFoundError
        If the file at `fpath` does not exist.
    ValueError
        Not a valid config file or unsupported file format.
    """
    # type check
    if not isinstance(fpath, (str, pathlib.Path)):
        raise TypeError(f"Expected str or pathlib.Path, got {type(fpath)}")
    if isinstance(fpath, str):
        fpath = pathlib.Path(fpath)
    if not fpath.is_file():
        raise FileNotFoundError(f"File not found: {fpath}")

    # Load file based on extension
    if fpath.suffix.lower() == ".yaml":
        yaml_content = fpath.read_text(encoding="utf-8")
        try:
            config = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {fpath}: {e}")
    elif fpath.suffix.lower() == ".json":
        json_content = fpath.read_text(encoding="utf-8")
        try:
            config = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file {fpath}: {e}")
    elif fpath.suffix.lower() in [".pkl", ".pickle"]:
        try:
            with open(fpath, "rb") as f:
                config = pickle.load(f)
        except (pickle.PickleError, EOFError) as e:
            raise ValueError(f"Error parsing pickle file {fpath}: {e}")
    else:
        raise ValueError(
            f"Unsupported file format: {fpath.suffix}. Expected .yaml, .json, .pkl, or .pickle"
        )
    return config
