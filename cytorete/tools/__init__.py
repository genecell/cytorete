"""``cytorete.tools`` (aliased ``cytorete.tl``) — GRN inference tools."""
from __future__ import annotations

try:
    from ._inferGRN import inferGRN
except ImportError:                      # public distribution
    from .._held import inferGRN
try:
    from ._consensus import inferGRN_consensus
except ImportError:
    from .._held import inferGRN_consensus
try:
    from ._inferTFActivity import inferTFActivity
except ImportError:
    from .._held import inferTFActivity
from ._grn import inferRegulon
from ._activity import regulonActivity, regulonSpecificity

# snake_case aliases (preferred Cytorete surface)
infer_grn = inferGRN
infer_grn_consensus = inferGRN_consensus
infer_tf_activity = inferTFActivity
infer_regulon = inferRegulon
regulon_activity = regulonActivity
regulon_specificity = regulonSpecificity

__all__ = [
    "inferGRN", "inferGRN_consensus", "inferRegulon", "inferTFActivity",
    "regulonActivity", "regulonSpecificity",
    "infer_grn", "infer_grn_consensus", "infer_regulon", "infer_tf_activity",
    "regulon_activity", "regulon_specificity",
]
