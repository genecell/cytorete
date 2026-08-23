"""Regulon-activity vs TF-expression specificity scatter for inferGRN eGRNs.

:func:`regulonSpecificityScatter` — one panel per cell type, one dot per TF:
  x = **IQR-normalized COSG specificity of the TF's expression**,
  y = **IQR-normalized COSG specificity of its regulon activity** (``X_grn``).
The two axes are scaled independently (each on its own range — they measure different
things), so the panels stay clean; the labelled dots are the TFs whose regulation and
expression specificity disagree most (scale-invariant z-scored discordance): a TF whose
*regulation* is cell-type-specific even though its *mRNA* is broad, or the rarer reverse.

The y-axis (regulon specificity) reuses :func:`piaso.tl.regulonSpecificity` (the canonical
IQR-log1p COSG specificity, written to the ``grn`` struct); the x-axis applies the same
IQR-log1p transform to COSG specificity of expression, so the two are computed identically.

Works on a cytome ``Dataset`` or ``AnnData`` carrying inferGRN's ``X_grn`` embedding + the
``grn``/``regulon`` struct. COSG specificity is computed in-memory (a visualisation helper
on an already-computed eGRN, not a streaming step).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..tools import _objio as io
from ..tools._activity import regulonSpecificity
from piaso.settings import _savefig


# ───────────────────────── helpers ─────────────────────────

def _iqr_log1p(sc):
    """COSG → IQR-log1p normalization (matches piaso.tl.regulonSpecificity), per column."""
    sc = np.asarray(sc, dtype=float)
    pos = sc[sc > 0]
    if pos.size:
        iqr = np.subtract(*np.percentile(pos, [75, 25])) or np.median(pos)
        norm = np.log1p(np.maximum(sc, 0.0) / (iqr + 1e-12))
        sc = np.where(sc > 0, norm, sc)
    return sc


def _annotate_no_overlap(ax, xs, ys, labels, fontsize=5, color="black"):
    """Label a few points with a greedy stagger so the text doesn't collide.

    Dependency-free de-overlap (no adjustText): labels are tethered to their point
    with a thin connector and offset in *display points* — horizontal side chosen by
    the point's x-quadrant, vertical offset fanned by rank so adjacent labels separate.
    """
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(labels) == 0:
        return
    xmid = np.nanmedian(xs)
    order = np.argsort(ys)                       # place bottom→top for a stable fan
    for rank, j in enumerate(order):
        right = xs[j] >= xmid
        dx = 6 if right else -6
        dy = 5 + 4 * (rank % 3)                  # fan vertical offset
        if rank % 2:
            dy = -dy
        ax.annotate(
            labels[j], (xs[j], ys[j]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, ha="left" if right else "right", va="center",
            color=color,
            arrowprops=dict(arrowstyle="-", lw=0.3, color="0.6", shrinkA=0, shrinkB=1),
        )


def _cosg_spec_matrix(X, labels, feature_names, mu: float = 1.0, iqr_normalize: bool = True):
    """Full per-(feature, group) COSG specificity → DataFrame [features × groups].

    ``n_genes_user = n_features`` + no expression filter so every feature scores in every
    group; ``iqr_normalize`` applies the same IQR-log1p as ``regulonSpecificity``.
    """
    import anndata as ad
    import pandas as pd
    import cosg as _cosg

    feats = [str(n) for n in feature_names]
    obs = pd.DataFrame({"_grp": pd.Categorical([str(l) for l in labels])},
                       index=[str(i) for i in range(np.asarray(X).shape[0])])
    a = ad.AnnData(np.asarray(X, dtype=np.float32), obs=obs, var=pd.DataFrame(index=feats))
    _cosg.cosg(a, groupby="_grp", n_genes_user=a.n_vars, remove_lowly_expressed=False,
               expressed_pct=0.0, mu=mu, key_added="cosg", verbosity=0)
    nm, sc = a.uns["cosg"]["names"], a.uns["cosg"]["scores"]
    out = pd.DataFrame(0.0, index=feats, columns=list(nm.dtype.names), dtype=float)
    for g in nm.dtype.names:
        vals = np.asarray(sc[g], dtype=float)
        if iqr_normalize:
            vals = _iqr_log1p(vals)
        out.loc[[str(n) for n in nm[g]], g] = vals
    return out


def _resolve_layer(data, modality, layer):
    """Pick the modality's preferred layer when ``layer`` is None (infog > counts)."""
    if layer is not None or not io.is_cytome(data):
        return layer
    try:
        names = [r[0] for r in data._conn.execute("SELECT matrix_name FROM matrix_meta").fetchall()]
    except Exception:
        return "counts"
    suff = [n[len(modality) + 1:] for n in names if n.startswith(modality + "_")]
    for pref in ("infog", "lognorm", "log1p", "counts"):
        if pref in suff:
            return pref
    return suff[0] if suff else "counts"


def _expr_matrix(data, genes, modality="RNA", layer=None):
    """(n_cells × len(genes)) dense expression aligned to ``genes`` (NaN for absent)."""
    import scipy.sparse as sp
    genes = [str(g) for g in genes]
    layer = _resolve_layer(data, modality, layer)
    var = list(io.get_var_names(data, modality))
    pos = {g: i for i, g in enumerate(var)}
    cols = [pos.get(g, -1) for g in genes]
    present = [c for c in cols if c >= 0]
    where = [k for k, c in enumerate(cols) if c >= 0]
    if io.is_cytome(data):
        rows = []
        for chunk, _ri in data.iter_chunks(modality=modality, layer=layer, batch_size=4096):
            m = chunk.tocsc()[:, present] if sp.issparse(chunk) else np.asarray(chunk)[:, present]
            rows.append(m.toarray() if sp.issparse(m) else np.asarray(m))
        sub = np.vstack(rows) if rows else np.empty((io.n_obs(data), len(present)))
    else:
        X = data[:, [var[c] for c in present]].X
        sub = X.toarray() if sp.issparse(X) else np.asarray(X)
    full = np.full((sub.shape[0], len(genes)), np.nan, dtype=float)
    if where:
        full[:, where] = sub
    return full


def _regulon_spec_matrix(data, groupby, key, uns_key, groups, mu):
    """IQR-normalized regulon specificity [regulon × group] via piaso.tl.regulonSpecificity."""
    import pandas as pd
    regulonSpecificity(data, groupby=groupby, groups=groups, activity_key=key,
                       iqr_normalize=True, mu=mu, uns_key=uns_key, verbose=0)
    st = io.get_struct(data, uns_key, default={}) or {}
    sm = st.get("specificity_matrix")
    if sm is None:
        raise ValueError("regulonSpecificity did not write specificity_matrix.")
    return pd.DataFrame(np.asarray(sm["matrix"], dtype=float),
                        index=[str(r) for r in sm["regulons"]],
                        columns=[str(c) for c in sm["celltypes"]])


def _z(v):
    v = np.asarray(v, dtype=float)
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / s if s > 0 else np.zeros_like(v)


# ───────────────────────── plot ─────────────────────────

def regulonSpecificityScatter(
    data, *, groupby: str, key: str = "X_grn", uns_key: str = "regulon",
    expr_modality: str = "RNA", expr_layer: Optional[str] = None,
    groups: Optional[Sequence[str]] = None, mu: float = 1.0,
    n_label: int = 8, ncols: int = 4, panel_size: float = 2.6,
    point_size: float = 14, highlight_color: str = "#D55E00",
    save=None, show: Optional[bool] = None, return_fig: bool = False,
):
    """Per-cell-type scatter of regulon-activity vs TF-expression specificity (IQR-COSG).

    One panel per cell type; one dot per TF. ``x`` = IQR-normalized COSG specificity of the
    TF's expression, ``y`` = IQR-normalized COSG specificity of its regulon activity. The two
    axes autoscale **independently** — they are different quantities — so off-diagonal TFs are
    flagged by scale-invariant z-scored discordance (|z(y) − z(x)| largest within the panel),
    not by a shared-range diagonal.

    Parameters
    ----------
    data : cytome.Dataset | AnnData
        Carries the ``key`` regulon-activity embedding + the regulon ``names`` struct
        (``uns_key``, falls back to ``'grn'``), and the ``expr_modality`` matrix.
    groupby : str
        Cell-type column.
    key : str, default ``'X_grn'``
        Regulon-activity embedding (passed to :func:`piaso.tl.regulonSpecificity`).
    uns_key : str, default ``'regulon'``
        Struct holding regulon ``names`` / written ``specificity_matrix`` (use ``'grn'`` for inferGRN).
    groups : sequence of str, optional
        Restrict to these cell types (e.g. drop an 'Unassigned' label).
    n_label : int, default 8
        Per panel, label this many most-discordant TFs.
    save, show, return_fig
        Standard PIASO plotting controls.

    Returns
    -------
    matplotlib Figure if ``return_fig`` else None.

    Notes
    -----
    Calls :func:`piaso.tl.regulonSpecificity`, which writes the IQR ``specificity_matrix``
    into the ``uns_key`` struct as a side effect.
    """
    import matplotlib.pyplot as plt

    glist = list(groups) if groups is not None else None
    S_reg = _regulon_spec_matrix(data, groupby, key, uns_key, glist, mu)   # regulon × group
    names = list(S_reg.index)
    var = set(str(g) for g in io.get_var_names(data, expr_modality))
    tfs = [n for n in names if n in var]
    if not tfs:
        raise ValueError("No regulon name matches a gene in modality "
                         f"'{expr_modality}' — cannot pair regulation with expression.")
    labels = np.asarray(io.get_celltypes(data, groupby)).astype(str)
    E = _expr_matrix(data, tfs, modality=expr_modality, layer=expr_layer)
    S_exp = _cosg_spec_matrix(np.nan_to_num(E), labels, tfs, mu=mu, iqr_normalize=True)

    cts = [c for c in S_reg.columns if c in S_exp.columns and (glist is None or c in set(glist))]
    n = len(cts)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, min(ncols, n),
                            figsize=(panel_size * min(ncols, n), panel_size * nrows),
                            squeeze=False)
    for ax in axs.flat:
        ax.set_visible(False)
    for i, ct in enumerate(cts):
        ax = axs[i // ncols][i % ncols]
        ax.set_visible(True)
        x = S_exp[ct].reindex(tfs).values
        y = S_reg[ct].reindex(tfs).values
        ax.scatter(x, y, s=point_size, c="#4E79A7", alpha=0.6, edgecolors="none")
        # scale-invariant discordance (axes are on their own ranges)
        d = np.abs(_z(y) - _z(x))
        sel = [j for j in np.argsort(-d)[:n_label] if np.isfinite(d[j])]
        if sel:
            ax.scatter(x[sel], y[sel], s=point_size + 6, c=highlight_color, edgecolors="none", zorder=3)
            _annotate_no_overlap(ax, x[sel], y[sel], [tfs[j] for j in sel], fontsize=5)
        r = np.corrcoef(np.nan_to_num(x), np.nan_to_num(y))[0, 1]
        ax.set_title(f"{ct}\n(r={r:.2f})", fontsize=7)
        ax.set_xlabel("TF expression specificity", fontsize=6)
        ax.set_ylabel("Regulon activity specificity", fontsize=6)
        ax.tick_params(labelsize=5, length=2)
        ax.margins(0.08)
    fig.suptitle("Regulon-activity vs TF-expression specificity (IQR-COSG; dot = TF)",
                 fontsize=9, y=1.0)
    fig.tight_layout()
    _savefig(fig, save, "regulonSpecificityScatter")
    if show is False or return_fig:
        return fig if return_fig else None
    plt.show()
    return None
