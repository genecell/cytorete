"""Regulon activity (AUCell analog via PIASO ``score()``) and cell-type regulon
specificity (via COSG λ on the activity matrix). Both are **object-centric**:
they write results onto the AnnData/cytome object and return ``None`` by default
(``copy=True`` to also return the value), and read inputs back from the object.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import _objio as io


def tf_detection_fraction(data, tfs, groupby, *, modality="RNA", layer=None):
    """Fraction of cells in each cell type in which each TF is **detected**
    (expression > 0) — a robust, group-level TF-presence signal (immune to
    single-cell dropout; ambient gives a low ~uniform fraction).

    Returns a DataFrame ``[TF × cell_type]`` of detection fractions (TFs absent
    from the expression matrix are skipped). AnnData uses ``layer``/``X``; cytome
    streams the relevant gene columns in cell order.
    """
    import pandas as pd
    import scipy.sparse as sp
    universe = io.get_var_names(data, modality)
    gidx = {g: i for i, g in enumerate(universe)}
    tf_in = [t for t in tfs if t in gidx]
    cols = np.array([gidx[t] for t in tf_in], dtype=int)
    ct = io.get_celltypes(data, groupby)
    cts = sorted(set(ct))
    if cols.size == 0:
        return pd.DataFrame(0.0, index=[], columns=cts)
    det = np.zeros((len(tf_in), len(cts)), dtype=np.float64)
    cnt = np.zeros(len(cts), dtype=np.int64)
    ct_to_j = {c: j for j, c in enumerate(cts)}
    if io.is_cytome(data):
        ct = np.asarray(ct)
        start = 0
        for item in data.iter_chunks(modality=modality, layer=(layer or "counts"),
                                     batch_size=2048):
            # iter_chunks yields (chunk, row_idx); tolerate a bare chunk too
            if isinstance(item, tuple):
                X, ridx = item
            else:
                X, ridx = item, None
            n = X.shape[0]
            sub = X[:, cols]
            b = (sub > 0)
            b = b.toarray() if sp.issparse(b) else np.asarray(b)
            cct = ct[ridx] if ridx is not None else ct[start:start + n]
            for j, c in enumerate(cts):
                m = cct == c
                if m.any():
                    det[:, j] += b[m].sum(axis=0)
                    cnt[j] += int(m.sum())
            start += n
    else:
        L = data.layers[layer] if (layer and layer in data.layers) else data.X
        sub = L[:, cols]
        b = (sub > 0)
        for j, c in enumerate(cts):
            m = ct == c
            bm = b[m]
            det[:, j] = (np.asarray(bm.sum(axis=0)).ravel() if sp.issparse(bm)
                         else np.asarray(bm).sum(axis=0))
            cnt[j] = int(m.sum())
    frac = det / np.maximum(cnt, 1)[None, :]
    return pd.DataFrame(frac, index=tf_in, columns=cts)


def regulonActivity(
    data,
    regulons: Optional[Dict[str, List[str]]] = None,
    weights: Optional[Dict[str, np.ndarray]] = None,
    *,
    score_layer: str = "infog",
    modality: str = "RNA",
    cytome_layer: Optional[str] = None,        # DEPRECATED → use score_layer (unified for both backends)
    batch_size: int = 1024,
    n_nearest_neighbors: int = 30,
    n_ctrl_set: int = 1000,
    use_rust: bool = True,
    compute_pvalues: bool = True,
    max_workers: int = 8,
    key_added: str = "X_regulon",
    uns_key: str = "regulon",
    copy: bool = False,
    verbose: int = 1,
):
    """Score each TF regulon per cell (the AUCell analog), via :func:`piaso.tl.score`.

    For every regulon (a TF and its target genes) a KNN-controlled, standardized
    gene-set score is computed per cell — :func:`piaso.tl.score` samples control
    gene sets from each gene's expression-KNN and subtracts their mean, so the
    activity is comparable across regulons of different sizes. Streaming +
    Rust-accelerated, so it scales to millions of cells.

    Parameters
    ----------
    data : AnnData | cytome.Dataset
        RNA object. Activity is written back onto it.
    regulons : dict[str, list[str]], optional
        ``{TF: [target genes]}``. If ``None``, read from
        ``data.uns[uns_key]['regulons']`` (AnnData) / ``data.metadata[...]``
        (cytome) — e.g. as produced by :func:`piaso.tl.inferRegulon`.
    weights : dict[str, ndarray], optional
        Per-target weights aligned to ``regulons`` (e.g. coSpecificity weights);
        ``None`` → uniform.
    score_layer : str, default ``'infog'``
        Layer scored (AnnData ``layers[...]``; the INFOG layer is recommended).
    modality, cytome_layer, batch_size
        cytome streaming knobs (modality, layer, rows per chunk).
    n_nearest_neighbors : int, default 30
        Gene-KNN size for control-set sampling in ``score()``.
    n_ctrl_set : int, default 1000
        Number of control gene sets per regulon (1000 → finer empirical p-values,
        i.e. a higher ``-log10(p)`` ceiling for the ``values='pval'`` view; the
        Rust ``score`` backend makes this cheap).
    use_rust : bool, default True
        Use the Rust ``score`` backend when available (else numpy/streaming).
    max_workers : int, default 8
        Worker threads passed to :func:`piaso.tl.score` (the Rust-accelerated
        scorer). ``8`` parallelizes the per-chunk control-set scoring on large
        datasets (notably the cytome-streaming path).
    compute_pvalues : bool, default True
        Also compute the per-cell empirical p-value of each regulon; stored at
        ``obsm[key_added + '_pval']`` and usable as a ``-log10(p)`` activity view.
    key_added : str, default ``'X_regulon'``
        Embedding key for the ``cells × n_TF`` activity matrix
        (AnnData ``obsm``; cytome embedding).
    uns_key : str, default ``'regulon'``
        Struct key on the object whose ``'names'`` field is set to the TF order.
    copy : bool, default False
        If True, also return ``(activity, names, pvals)``; otherwise return ``None``
        (results are on the object).

    Returns
    -------
    None, or ``(activity [cells × n_TF], names, pvals)`` if ``copy=True``.

    Examples
    --------
    >>> piaso.tl.regulonActivity(adata)            # uses adata.uns['regulon']['regulons']
    >>> adata.obsm['X_regulon'].shape              # cells × n_TF
    """
    from piaso.tools import score as piaso_score

    if regulons is None:
        struct = io.get_struct(data, uns_key, default=None)
        if not struct or "regulons" not in struct:
            raise ValueError(
                f"no regulons on the object ({uns_key!r}); pass `regulons=` or run "
                f"piaso.tl.inferRegulon first.")
        regulons = struct["regulons"]
        if weights is None:
            weights = struct.get("weights")

    tfs = list(regulons.keys())
    gene_list = {tf: list(regulons[tf]) for tf in tfs}
    gene_weights = ([np.asarray(weights[tf], dtype=float) for tf in tfs]
                    if weights is not None else None)
    # score_layer is the single, canonical layer for BOTH AnnData and cytome (matches score()/runGDR).
    # `cytome_layer` is a deprecated alias kept for back-compat: honour it only if explicitly set.
    if cytome_layer is not None:
        import warnings as _w
        _w.warn("regulonActivity: `cytome_layer=` is deprecated; use `score_layer=` "
                "(applies to both AnnData and cytome).", FutureWarning, stacklevel=2)
        _score_layer = cytome_layer
    else:
        _score_layer = score_layer
    if verbose:
        print(f"[regulonActivity] scoring {len(tfs)} regulons "
              f"(layer={_score_layer}, pvalues={compute_pvalues})")
    activity, names, pvals = piaso_score(
        data, gene_list, gene_weights=gene_weights,
        layer=_score_layer, modality=modality,
        batch_size=batch_size, n_nearest_neighbors=n_nearest_neighbors,
        n_ctrl_set=n_ctrl_set, use_rust=use_rust, compute_pvalues=compute_pvalues,
        max_workers=max_workers)
    names = list(names)
    activity = np.asarray(activity, dtype=np.float32)

    io.set_embedding(data, key_added, activity)
    if pvals is not None:
        io.set_embedding(data, key_added + "_pval", np.asarray(pvals, np.float32))
    io.update_regulon_struct(data, uns_key, names=names, activity_key=key_added)

    return (activity, names, pvals) if copy else None


def regulonSpecificity(
    data,
    *,
    groupby: str,
    groups: Optional[Sequence[str]] = None,
    activity_key: str = "X_regulon",
    use_neglog10p: bool = False,
    iqr_normalize: bool = True,
    mu: float = 1.0,
    n_top: Optional[int] = None,
    uns_key: str = "regulon",
    copy: bool = False,
    verbose: int = 1,
):
    """Cell-type regulon specificity (the RSS analog), via **COSG λ** on the
    ``cells × regulon`` activity matrix grouped by ``groupby``.

    COSG's μ-penalized cosine specificity ranks, for each cell type, the regulons
    whose activity is most specific to it. (This is computed with COSG, not
    SCENIC's Jensen–Shannon RSS — the score column is named ``cosg_score`` to be
    honest about the method.)

    Parameters
    ----------
    data : AnnData | cytome.Dataset
        Object carrying the activity embedding (from :func:`regulonActivity` /
        :func:`piaso.tl.inferRegulon`).
    groupby : str
        Cell-type column (``obs``/``cells``).
    activity_key : str, default ``'X_regulon'``
        Embedding key holding the activity matrix. Its ``'_pval'`` sibling is used
        when ``use_neglog10p=True``.
    use_neglog10p : bool, default False
        Run COSG on ``-log10(p)`` (always non-negative, bounded) instead of the raw
        activity — sometimes a cleaner cell-type-specificity signal.
    iqr_normalize : bool, default True
        Report the IQR-log1p-normalized COSG score (comparable across regulons).
    mu : float, default 1.0
        COSG cross-cluster penalty.
    n_top : int, optional
        Keep only the top ``n_top`` regulons per cell type (default: all).
    uns_key : str, default ``'regulon'``
        Struct key on the object; the table is written to its ``'specificity'`` field.
    copy : bool, default False
        If True, also return the DataFrame; else ``None`` (it's on the object,
        read via ``piaso.pl.regulonSpecificity`` or the struct's ``'specificity'``).

    Returns
    -------
    None, or a tidy DataFrame ``[cell_type, regulon, cosg_score, rank]`` if ``copy``.
    """
    import anndata as ad
    import pandas as pd
    import cosg as _cosg

    names = (io.get_struct(data, uns_key, default={}) or {}).get("names")
    X = io.get_embedding(data, activity_key)
    if use_neglog10p:
        if not io.has_embedding(data, activity_key + "_pval"):
            raise ValueError("use_neglog10p=True needs the '_pval' embedding "
                             "(run regulonActivity with compute_pvalues=True).")
        p = io.get_embedding(data, activity_key + "_pval")
        X = -np.log10(np.clip(p.astype(np.float64), 1e-300, 1.0))
    X = np.asarray(X, dtype=np.float32)
    if names is None:
        names = [f"R{i}" for i in range(X.shape[1])]
    celltypes = np.asarray(io.get_celltypes(data, groupby))
    if groups is not None:                       # exclude cells outside the subset
        m = np.isin(celltypes, list(groups))
        X, celltypes = X[m], celltypes[m]

    a = ad.AnnData(
        X=X, obs=pd.DataFrame({"cell_type": celltypes},
                              index=[f"c{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=list(names)))
    # n_genes_user = all regulons → a complete regulon × cell-type matrix
    _cosg.cosg(a, groupby="cell_type", mu=mu, n_genes_user=len(names), key_added="cosg")
    res = a.uns["cosg"]
    groups = list(res["names"].dtype.names)
    mat = pd.DataFrame(0.0, index=list(names), columns=groups)   # dense, no dropping
    rows = []
    for grp in groups:
        nm = np.asarray(res["names"][grp])
        sc = np.asarray(res["scores"][grp], dtype=float)
        if iqr_normalize:
            pos = sc[sc > 0]
            if pos.size:
                iqr = np.subtract(*np.percentile(pos, [75, 25])) or np.median(pos)
                norm = np.log1p(np.maximum(sc, 0.0) / (iqr + 1e-12))
                sc = np.where(sc > 0, norm, sc)
        mat.loc[nm, grp] = sc                                    # full matrix (incl. ≤0)
        for rank, (r, s) in enumerate(zip(nm, sc)):
            if s > 0:                                            # tidy table = top dotplot
                rows.append((grp, r, float(s), rank))
    df = pd.DataFrame(rows, columns=["cell_type", "regulon", "cosg_score", "rank"])
    if n_top:
        df = (df.sort_values("cosg_score", ascending=False)
                .groupby("cell_type").head(int(n_top)).reset_index(drop=True))
    spec_matrix = {"regulons": list(mat.index), "celltypes": groups,
                   "matrix": mat.to_numpy().astype(np.float32).tolist()}
    io.update_regulon_struct(data, uns_key, specificity=df, specificity_matrix=spec_matrix)
    if verbose:
        print(f"[regulonSpecificity] COSG on "
              f"{'-log10(p)' if use_neglog10p else 'activity'} → "
              f"{len(names)} regulons × {len(groups)} cell types (dense matrix stored)")
    return df if copy else None
