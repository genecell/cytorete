"""``cytorete.preprocessing`` (aliased ``cytorete.pp``) — GRN preprocessing.

The low-level motif scanner stays in PIASO (``piaso.pp.scan_motifs``,
Rust-accelerated). These are the GRN-specific cistrome / promoter builders that
call it.
"""
from __future__ import annotations

from ._promoters import extract_promoter_sequences
from ._cistrome import build_cistrome
try:
    from ._peak_cistrome import build_peak_cistrome, bulk_base_cistrome
except ImportError:                      # public distribution
    from .._held import build_peak_cistrome
    bulk_base_cistrome = build_peak_cistrome

# camelCase aliases (piaso.pp continuity)
extractPromoterSequences = extract_promoter_sequences
buildCistrome = build_cistrome
buildPeakCistrome = build_peak_cistrome
bulkBaseCistrome = bulk_base_cistrome

__all__ = [
    "extract_promoter_sequences", "build_cistrome",
    "build_peak_cistrome", "bulk_base_cistrome",
    "extractPromoterSequences", "buildCistrome",
    "buildPeakCistrome", "bulkBaseCistrome",
]
