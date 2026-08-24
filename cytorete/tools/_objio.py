"""Backend-agnostic accessors so the regulon `tl`/`pl` functions work on an
AnnData OR a cytome Dataset identically (scanpy-style: write to the object, read
from the object).

AnnData: embeddings → ``adata.obsm``; the regulon struct → ``adata.uns``.
cytome:  embeddings → ``ds.add_embedding`` / ``ds.embeddings``; struct →
``ds.metadata``.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np


def open_if_path(data):
    """Open a ``.cytome`` path, or pass an already-open object through.

    Returns ``(obj, opened_here)``. PIASO's plotting and preprocessing accept a
    path string, so the regulon entry points would otherwise be the one place
    in the stack where a path fails -- and it failed with an AttributeError
    about ``var_names``, which names nothing the caller did wrong.
    """
    if not isinstance(data, str):
        return data, False
    try:
        import cytome
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            f"{data!r} looks like a path, but cytome is not installed") from exc
    return cytome.open(data), True


def is_cytome(data) -> bool:
    try:
        from cytome import CytomeDataset
    except Exception:
        return False
    return isinstance(data, CytomeDataset)


def n_obs(data) -> int:
    return int(data.n_cells) if is_cytome(data) else int(data.n_obs)


def get_celltypes(data, groupby: str) -> np.ndarray:
    col = data.cells[groupby] if is_cytome(data) else data.obs[groupby]
    return np.asarray(col).astype(str)


def get_var_names(data, modality: str = "RNA") -> List[str]:
    if not is_cytome(data):
        return [str(x) for x in data.var_names]
    try:
        from cytome.utils.modality import modality_feature_table_info
        feat_tbl, _idx, name_col = modality_feature_table_info(data, modality)
        return [str(x) for x in np.asarray(getattr(data, feat_tbl)[name_col])]
    except Exception:
        return [str(x) for x in np.asarray(data.genes["gene_name"])]


def _flush(data) -> None:
    try:
        data.flush()
    except Exception:
        pass


def set_embedding(data, key: str, X: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float32)
    if is_cytome(data):
        data.add_embedding(key, X)
        _flush(data)            # persist so reads/plots see it immediately
    else:
        data.obsm[key] = X


def _resolve_embedding_key(data, key: str):
    """Match an embedding name the way plotting's ``basis=`` does, or None.

    Conversion stores obsm arrays under a modality prefix -- ``RNA_umap``
    since cytome 0.2.6, ``RNA_obsm_X_umap`` before it -- so an exact lookup
    of ``'X_umap'`` fails on every cytome ever written. Try exact, then a
    prefix-insensitive match on the short name.
    """
    if not is_cytome(data):
        return key if key in data.obsm else None
    try:
        names = list(data.list_embeddings())
    except Exception:
        return None
    if key in names:
        return key
    short = key[2:] if key.startswith("X_") else key
    hits = [n for n in names if short.lower() in n.lower()]
    return hits[-1] if hits else None


def get_embedding(data, key: str) -> np.ndarray:
    resolved = _resolve_embedding_key(data, key)
    if resolved is None:
        available = (list(data.list_embeddings()) if is_cytome(data)
                     else list(data.obsm))
        raise KeyError(
            f"Embedding not found: {key!r}. Available: {available}")
    if is_cytome(data):
        return np.asarray(data.embeddings[resolved])
    return np.asarray(data.obsm[resolved])


def has_embedding(data, key: str) -> bool:
    try:
        return _resolve_embedding_key(data, key) is not None
    except Exception:
        return False


def set_struct(data, key: str, value: Any) -> None:
    """Store the regulon result struct (dict) on the object."""
    if is_cytome(data):
        data.metadata[key] = value
        _flush(data)
    else:
        data.uns[key] = value


def get_struct(data, key: str, default: Any = None) -> Any:
    if is_cytome(data):
        return data.metadata.get(key, default)
    return data.uns.get(key, default)


# --- regulon struct helpers (JSON-safe so AnnData uns AND cytome metadata work) ---

def _jsonable(v):
    """Recursively coerce numpy scalars/arrays → plain python for JSON storage."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    return v


def update_regulon_struct(data, uns_key: str, **fields) -> None:
    """Merge ``fields`` into the object's regulon struct (creating it if absent).
    DataFrame fields are stored as records (JSON-safe)."""
    import pandas as pd
    struct = dict(get_struct(data, uns_key, default={}) or {})
    for k, v in fields.items():
        if isinstance(v, pd.DataFrame):
            v = v.to_dict("records")
        struct[k] = _jsonable(v)
    set_struct(data, uns_key, struct)


def regulon_table(data, uns_key: str, field: str):
    """Return a struct field that was stored as records back as a DataFrame
    (e.g. ``field='specificity'`` or ``'edges'``)."""
    import pandas as pd
    struct = get_struct(data, uns_key, default=None)
    if struct is None or field not in struct:
        raise KeyError(
            f"no '{field}' in {uns_key!r} on this object — run "
            f"piaso.tl.inferRegulon / regulonSpecificity first.")
    return pd.DataFrame(struct[field])
