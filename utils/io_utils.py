import pathlib
import pickle

import yaml


def load_configs(fpath: str | pathlib.Path) -> dict:
    """Load a configuration file and return its contents as a dictionary.
    Parameters
    ----------
    fpath : str or pathlib.Path
        Path to the YAML or pickle configuration file.
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
    elif fpath.suffix.lower() in [".pkl", ".pickle"]:
        try:
            with open(fpath, 'rb') as f:
                config = pickle.load(f)
        except (pickle.PickleError, EOFError) as e:
            raise ValueError(f"Error parsing pickle file {fpath}: {e}")
    else:
        raise ValueError(f"Unsupported file format: {fpath.suffix}. Expected .yaml, .pkl, or .pickle")
    return config
