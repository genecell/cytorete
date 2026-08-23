"""Forwarders for capabilities that are not in the public distribution.

`import cytorete` must succeed either way: in the internal tree the real
modules exist and are imported; in the public package the names below exist
but raise an actionable ImportError at CALL time, naming what they need.
Mirrors PIASO's `_internal_shim` pattern.
"""
from __future__ import annotations


def _held(name: str, needs: str):
    def _raiser(*_a, **_k):
        raise ImportError(
            f"cytorete.{name} is not part of this distribution: it requires "
            f"{needs}, which is not yet released. The RNA regulon workflow "
            f"(build_promoter_cistrome -> inferRegulon -> regulonActivity) "
            f"is fully available."
        )
    _raiser.__name__ = name
    _raiser.__qualname__ = name
    return _raiser


inferGRN = _held("inferGRN", "the multiome (RNA+ATAC) GRN chain")
inferGRN_consensus = _held("inferGRN_consensus", "the multiome (RNA+ATAC) GRN chain")
inferTFActivity = _held("inferTFActivity", "the ATAC TF-activity chain")
build_peak_cistrome = _held("build_peak_cistrome", "the peak/ATAC cistrome chain")
