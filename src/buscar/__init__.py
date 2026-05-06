"""Buscar: Bioactive Unbiased Single-cell Compound Assessment and Ranking.

A Python framework for prioritizing compounds in high-content imaging drug screening
using single-cell profiles.
"""

from importlib.metadata import PackageNotFoundError, version

from buscar._data_utils import add_cell_id_hash as add_cell_id_hash
from buscar.metrics import calculate_buscar_scores as calculate_buscar_scores
from buscar.metrics import (
    compute_earth_movers_distance as compute_earth_movers_distance,
)
from buscar.signatures import identify_signatures as identify_signatures

try:
    __version__ = version("buscar")
except PackageNotFoundError:
    __version__ = "0.0.0"
