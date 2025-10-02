"""
Utility functions for validating parameter grids for clustering optimization.
"""

from typing import Any


def _get_valid_cluster_params() -> set[str]:
    """Get the set of valid parameter names for cluster_profiles function.

    Returns
    -------
    set[str]
        Set of valid parameter names that can be used in param_grid.
    """
    return {
        "cluster_method",
        "cluster_resolution",
        "dim_reduction",
        "n_neighbors",
        "neighbor_distance_metric",
        "pca_variance_explained",
        "pca_n_components_to_capture_variance",
        "pca_svd_solver",
    }


def _get_valid_param_types() -> set[str]:
    """Get the set of valid parameter types for param_grid configuration.

    Returns
    -------
    set[str]
        Set of valid parameter types ('float', 'int', 'categorical').
    """
    return {"float", "int", "categorical"}


def _validate_param_grid(param_grid: dict[str, Any]) -> None:
    """Validate the parameter grid for optimized_clustering function.

    This function checks that the provided param_grid contains valid parameter names
    and types for the cluster_profiles function. It raises a ValueError if any invalid
    parameters are found.

    Parameters
    ----------
    param_grid : dict[str, Any]
        Dictionary defining the parameter search space. Each key should be a parameter
        name from cluster_profiles, and each value should be a dictionary with 'type'
        and range info.

    Raises
    ------
    ValueError
        If param_grid contains unsupported parameter types or invalid parameter names.
    TypeError
        If param_grid structure is invalid (missing required keys, wrong value types).
    """
    valid_params = _get_valid_cluster_params()
    valid_types = _get_valid_param_types()

    for param_name, param_config in param_grid.items():
        # 1. Check if parameter name is valid
        if param_name not in valid_params:
            raise ValueError(
                f"Invalid parameter name: '{param_name}'. "
                f"Valid parameters are: {sorted(valid_params)}"
            )

        # 2. Check if param_config is a dictionary
        if not isinstance(param_config, dict):
            raise TypeError(
                f"Parameter config for '{param_name}' must be a dictionary, "
                f"got {type(param_config).__name__}"
            )

        # 3. Check if 'type' key exists
        if "type" not in param_config:
            raise TypeError(
                f"Parameter config for '{param_name}' must contain a 'type' key"
            )

        param_type = param_config["type"]

        # 4. Check if type is valid
        if param_type not in valid_types:
            raise ValueError(
                f"Invalid parameter type '{param_type}' for '{param_name}'. "
                f"Valid types are: {sorted(valid_types)}"
            )

        # 5. Validate type-specific requirements
        if param_type in ["float", "int"]:
            # Check for 'low' and 'high' keys
            if "low" not in param_config:
                raise TypeError(
                    f"Parameter config for '{param_name}' with type '{param_type}' "
                    f"must contain a 'low' key"
                )
            if "high" not in param_config:
                raise TypeError(
                    f"Parameter config for '{param_name}' with type '{param_type}' "
                    f"must contain a 'high' key"
                )

            # Check that low and high are numbers
            if not isinstance(param_config["low"], (int, float)):
                raise TypeError(
                    f"'low' value for '{param_name}' must be a number, "
                    f"got {type(param_config['low']).__name__}"
                )
            if not isinstance(param_config["high"], (int, float)):
                raise TypeError(
                    f"'high' value for '{param_name}' must be a number, "
                    f"got {type(param_config['high']).__name__}"
                )

            # Check that low < high
            if param_config["low"] >= param_config["high"]:
                raise ValueError(
                    f"'low' must be less than 'high' for '{param_name}'. "
                    f"Got low={param_config['low']}, high={param_config['high']}"
                )

            # If 'log' is present, check it's a boolean
            if "log" in param_config and not isinstance(param_config["log"], bool):
                raise TypeError(
                    f"'log' value for '{param_name}' must be a boolean, "
                    f"got {type(param_config['log']).__name__}"
                )

        elif param_type == "categorical":
            # Check for 'choices' key
            if "choices" not in param_config:
                raise TypeError(
                    f"Parameter config for '{param_name}' with type 'categorical' "
                    f"must contain a 'choices' key"
                )

            # Check that choices is a list or tuple
            if not isinstance(param_config["choices"], (list, tuple)):
                raise TypeError(
                    f"'choices' for '{param_name}' must be a list or tuple, "
                    f"got {type(param_config['choices']).__name__}"
                )

            # Check that choices is not empty
            if len(param_config["choices"]) == 0:
                raise ValueError(
                    f"'choices' for '{param_name}' must contain at least one option"
                )
