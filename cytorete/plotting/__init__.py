"""``cytorete.plotting`` (aliased ``cytorete.pl``) — regulon/GRN plotting."""
from __future__ import annotations

from ._plotRegulon import regulonActivity, regulonNetwork, regulonEmbedding
from ._plotRegulonScatter import regulonSpecificityScatter

# snake_case aliases
regulon_activity = regulonActivity
regulon_network = regulonNetwork
regulon_embedding = regulonEmbedding
regulon_specificity_scatter = regulonSpecificityScatter

__all__ = [
    "regulonActivity", "regulonNetwork", "regulonEmbedding",
    "regulonSpecificityScatter",
    "regulon_activity", "regulon_network", "regulon_embedding",
    "regulon_specificity_scatter",
]
