"""Cytorete — cell-type-resolved inference of gene regulatory networks and their dynamics.

*Cytorete* (pronounced "sigh-toe-reet"; *cyto-* + Latin *rete*, "cell network")
is a GRN method built on the PIASO single-cell stack. It reuses PIASO's public
API (scoring/INFOG, GDR via ``piaso.tl.runGDR``, co-specificity, peak selection,
``piaso.pp.scan_motifs``, genome loaders), the Cytome streaming backend, and
COSG — a one-directional dependency (``cytorete → piaso-tools``), so there is no
packaging cycle.

Namespaces mirror PIASO for muscle-memory (``cytorete.tl is cytorete.tools``):

- :mod:`cytorete.tl`   — GRN inference (``infer_grn`` / ``infer_regulon`` / ``infer_tf_activity`` / ``regulon_activity``).
- :mod:`cytorete.pp`   — GRN preprocessing (cistrome / promoter builders).
- :mod:`cytorete.pl`   — regulon plotting.
- :mod:`cytorete.data` — motif DBs, PWMs, .2bit sequence access.

The most-used entry points are re-exported at the top level (both ``snake_case``
and legacy ``camelCase``).
"""
from __future__ import annotations

__version__ = "0.1.0"

from . import tools as tl
from . import preprocessing as pp
from . import plotting as pl
from . import data

# Top-level convenience re-exports (the primary user surface).
from .tools import (
    infer_grn, infer_grn_consensus, infer_regulon, infer_tf_activity,
    regulon_activity, regulon_specificity,
    inferGRN, inferGRN_consensus, inferRegulon, inferTFActivity,
    regulonActivity, regulonSpecificity,
)

__all__ = [
    "__version__",
    "tl", "tools", "pp", "preprocessing", "pl", "plotting", "data",
    "infer_grn", "infer_grn_consensus", "infer_regulon", "infer_tf_activity",
    "regulon_activity", "regulon_specificity",
    "inferGRN", "inferGRN_consensus", "inferRegulon", "inferTFActivity",
    "regulonActivity", "regulonSpecificity",
]
