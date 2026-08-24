"""Vendored small helpers cytorete owns rather than importing from PIASO privates.

Phase 2 copies the real implementations here:

- ``interval_overlap`` — pure-Python interval overlap (from PIASO's
  ``preprocessing/_interval_overlap.py``; *copied*, not moved, since PICCO also
  uses PIASO's copy).
- ``is_cytome`` — 3-line "is this a cytome Dataset / path?" check (replaces
  PIASO's private ``_cytome_compat.is_cytome_input`` / ``_cospecificity._is_cytome``).

Keeping these vendored means cytorete depends on PIASO **only** through its
public API (`piaso.tl.*`, `piaso.pp.*`, `piaso.data.*`).
"""
from __future__ import annotations

__all__ = ["is_cytome"]


def is_cytome(obj) -> bool:
    """True if ``obj`` is a cytome ``Dataset`` or a path to a ``.cytome`` file.

    Lightweight, dependency-optional check (does not import cytome unless a
    Dataset-like object is passed).
    """
    if isinstance(obj, str):
        return obj.endswith(".cytome")
    # Duck-type: a cytome Dataset exposes iter_chunks + a cells table.
    return hasattr(obj, "iter_chunks") and hasattr(obj, "cells")
