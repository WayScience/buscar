import gzip
import json
import pathlib
import pickle
import shutil

import polars as pl
import requests
import yaml
from tqdm import tqdm

from .data_utils import split_meta_and_features


def load_profiles(
    fpath: str | pathlib.Path,
    convert_to_f32: bool = False,
    verbose: bool | None = False,
    shared_features: list[str] | None = None,
) -> pl.DataFrame:
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
    shared_features : list[str] | None, optional
        If provided, only loads metadata columns and these specific feature columns.
        Default is None (loads all columns).

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
        raise ValueError(
            f"Unsupported file format: {fpath.suffix}. Supported formats are: .parquet, .pq, .arrow"
        )

    # load profiles
    loaded_profiles = pl.read_parquet(fpath)

    # filter to shared features if provided
    if shared_features is not None:
        meta_cols, _ = split_meta_and_features(loaded_profiles)
        loaded_profiles = loaded_profiles.select(meta_cols + shared_features)

    # convert all Float64 columns to Float32 if convert_to_f32 is True
    if convert_to_f32:
        loaded_profiles = loaded_profiles.with_columns(
            [
                pl.col(col).cast(pl.Float32)
                for col in loaded_profiles.columns
                if loaded_profiles.schema[col] == pl.Float64
            ]
        )

    # if verbose is True, print information about the loaded profiles
    if verbose:
        print(f"Loading profiles from {fpath}...")
        print(
            f"Loaded profiles shape: rows: {loaded_profiles.shape[0]}, columns: {loaded_profiles.shape[1]}"
        )
        print(
            f"Estimated loaded dataframe size: {round(loaded_profiles.estimated_size('mb'), 2)} MB"
        )

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


def download_file(
    source_url: str,
    output_path: pathlib.Path | str,
    chunk_size: int = 8192,
) -> pathlib.Path:
    """Downloads a file from a URL with progress tracking.

    Downloads a file from the specified URL and saves it to the given output path.
    The download is performed in chunks to handle large files efficiently, and the progress is displayed using
    the `tqdm` library.

    Parameters
    ----------
    source_url : str
        URL to download the file from.
    output_path : pathlib.Path | str
        Full path where the file should be saved.
    chunk_size : int, optional
        Size of chunks to download in bytes. Defaults to 8192.

    Returns
    -------
    pathlib.Path
        The path where the file was downloaded.

    Raises
    ------
    requests.exceptions.RequestException
        If there is an error during the download request.
    TypeError
        If input types are invalid.
    FileNotFoundError
        If the output directory does not exist.
    """
    # type checking
    if not isinstance(source_url, str):
        raise TypeError(f"source_url must be a string, got {type(source_url)}")
    if not isinstance(output_path, (pathlib.Path, str)):
        raise TypeError(
            f"output_path must be a pathlib.Path or str, got {type(output_path)}"
        )
    if isinstance(output_path, str):
        output_path = pathlib.Path(output_path)
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory {output_path.parent} does not exist."
        )
    if output_path.exists() and not output_path.is_file():
        raise FileExistsError(f"Output path {output_path} exists and is not a file.")

    # starting downloading process
    try:
        # sending GET request to the source URL
        with requests.get(source_url, stream=True) as response:
            # raise an error if the request was unsuccessful
            response.raise_for_status()

            # get the total size of the file from the response headers
            total_size = int(response.headers.get("content-length", 0))

            # using tqdm to track the download progress
            with (
                open(output_path, "wb") as file,
                tqdm(
                    desc="Downloading",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
                # iterating over the response content in chunks
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)

                        # this updates the progress bar
                        pbar.update(len(chunk))
        return output_path

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Error downloading file: {e}")
    except Exception as e:
        raise Exception(f"Unexpected error during download: {e}")


def extract_file(
    file_path: pathlib.Path | str,
    extract_dir: pathlib.Path | str | None = None,
) -> None:
    """Extract a compressed file using native Python libraries.

    Supports zip, tar, tar.gz, tar.bz2, tar.xz, and standalone gz files.

    Parameters
    ----------
    file_path : pathlib.Path | str
        Path to the compressed file.
    extract_dir : pathlib.Path | str, optional
        Directory where the file should be extracted. If None, extracts to the same directory as the file.

    Returns
    -------
    None
        Extracted files are saved in the specified extract_dir or in the same
        directory if the extract_dir option is None.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file format is unsupported.
    """
    # type checking
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")

    if extract_dir is None:
        extract_dir = file_path.parent
    elif isinstance(extract_dir, str):
        extract_dir = pathlib.Path(extract_dir)

    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Handle single .gz files (not .tar.gz)
        if file_path.suffix == ".gz" and not file_path.name.endswith(".tar.gz"):
            extracted_path = extract_dir / file_path.stem
            with gzip.open(file_path, "rb") as f_in:
                with open(extracted_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Extracted to: {extracted_path}")
        else:
            # Use shutil.unpack_archive for zip, tar, tar.gz, tar.bz2, tar.xz
            shutil.unpack_archive(file_path, extract_dir)
            print(f"Extracted to: {extract_dir}")

    except shutil.ReadError:
        raise ValueError(f"Unsupported file format for extraction: {file_path.suffix}")
    except Exception as e:
        raise Exception(f"Unexpected error during extraction: {e}")


def download_compressed_file(
    source_url: str,
    output_path: pathlib.Path | str,
    chunk_size: int = 8192,
    extract: bool = True,
) -> None:
    """
    Download and optionally extract a compressed file from a URL.

    Parameters
    ----------
    source_url : str
        The URL of the compressed file to download.
    output_path : pathlib.Path | str
        The local path where the downloaded file should be saved.
    chunk_size : int, optional
        The size of chunks to download in bytes, by default 8192.
    extract : bool, optional
        Whether to extract the file after downloading, by default True.

    Returns
    -------
    pathlib.Path
        The path to the downloaded (and possibly extracted) file.
    """
    downloaded_path = download_file(source_url, output_path, chunk_size)
    if extract:
        extract_file(downloaded_path)

    return downloaded_path
