"""Layer resolution shared by the regulon and GRN entry points.

Lives in its own module (rather than inside the GRN implementation) because
``inferRegulon`` — part of the public distribution — needs it too, and a
published module must not import a withheld one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def _ensure_norm_layer(ds, modality, layer, *, fallback="counts", recompute=False,
                       verbose=1):
    """Return the cytome layer to read for ``modality``, computing + persisting the
    normalized layer if it's absent.

    ``layer='infog'`` → INFOG (variance-stabilising, RNA); ``'tfidf'`` → TF-IDF
    (down-weights ubiquitous peaks, ATAC); ``'counts'``/``None`` → raw (no-op).
    The normalized matrix is stored as ``{modality}_{layer}`` and reused on
    re-runs (``recompute=True`` forces recompute). Unknown layer names that
    already exist as a stored matrix are passed through; otherwise we fall back to
    ``fallback`` so a misconfigured layer never silently scores the wrong matrix.

    ``None`` resolves to ``'counts'`` (the raw matrix is always ``{modality}_counts``)
    so the cytome score path reads ``{modality}_counts`` rather than a non-existent
    ``{modality}_None`` (e.g. ``score_layer_rna=None`` → scores raw ``RNA_counts``).
    """
    if layer in (None, "counts"):
        return "counts"
    name = f"{modality}_{layer}"
    try:
        existing = {r[0] for r in ds._conn.execute("SELECT matrix_name FROM matrix_meta")}
    except Exception:
        existing = set()
    if name in existing and not recompute:
        return layer
    if layer == "infog":
        if verbose:
            print(f"[inferGRN] computing {name} (INFOG normalization)…", flush=True)
        from piaso.tools import infog as _infog
        _infog(ds, modality=modality, key_added="infog", save_layer=True, inplace=True)
        return layer
    if layer == "tfidf":
        if verbose:
            print(f"[inferGRN] computing {name} (TF-IDF normalization)…", flush=True)
        from piaso.tools import run_TFIDF
        run_TFIDF(ds, modality=modality, output_layer="tfidf", inplace=True)
        return layer
    if name in existing:
        return layer
    if verbose:
        print(f"[inferGRN] layer {name!r} absent and not auto-computable; "
              f"falling back to {fallback!r}.", flush=True)
    return fallback


# ---------------------------------------------------------------- cis linkage
