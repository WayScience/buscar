"""Buscar: Bioactive Unbiased Single-cell Compound Assessment and Ranking.

A Python framework for prioritizing compounds in high-content imaging drug screening
using single-cell profiles.
"""

from buscar.data_utils import add_cell_id_hash
from buscar.metrics import compute_earth_movers_distance, score_compounds
from buscar.signatures import identify_signatures

__version__ = "0.1.0"

__all__ = [
    "add_cell_id_hash",
    "compute_earth_movers_distance",
    "identify_signatures",
    "score_compounds",
]
