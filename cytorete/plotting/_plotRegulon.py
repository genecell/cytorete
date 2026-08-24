"""Plotting for PIASO-GRN regulons — **object-centric** (AnnData or cytome): each
function reads the regulon results that ``piaso.tl.inferRegulon`` /
``regulonActivity`` / ``regulonSpecificity`` wrote onto the object.

- :func:`regulonActivity` — regulon × cell-type heatmap OR dotplot
  (``style=``); ``values='specificity'`` shows the COSG specificity matrix,
  ``values='significance'`` / dotplot dot-size encode ACAT-combined p-values.
- :func:`regulonNetwork` — TF→target network (igraph layout + matplotlib)
- :func:`regulonEmbedding` — embedding coloured by per-regulon activity
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..tools import _objio as io


def _significance_per_celltype(data, groupby, key, alpha=0.05, uns_key="regulon"):
    """Per-(cell type, regulon) significance from the per-cell ``_pval`` embedding.

    Cells within a cell type are correlated, which violates the independence
    assumption of Fisher/Stouffer — so per-cell p-values are combined with the
    **Cauchy combination test (ACAT)**, which is robust to arbitrary dependence.
    The combined p-values are then BH-FDR corrected across the whole
    cell-type × regulon grid. Also returns the fraction of cells with raw
    ``p < alpha`` per block (a dependence-free alternative size metric).

    Returns ``{neglog10p, reject, pct_significant [n_ct × n_reg], cts, names}``.
    """
    if not io.has_embedding(data, key + "_pval"):
        raise ValueError("significance needs the '_pval' embedding "
                         "(run inferRegulon/regulonActivity with compute_pvalues=True).")
    names = (io.get_struct(data, uns_key, default={}) or {}).get("names")
    P = io.get_embedding(data, key + "_pval").astype(np.float64)   # cells × n_reg
    if names is None:
        names = [f"R{i}" for i in range(P.shape[1])]
    ct = io.get_celltypes(data, groupby)
    cts = sorted(set(ct))
    n_ct, n_reg = len(cts), P.shape[1]
    # ACAT (Cauchy) combination, equal weights: T = tan((0.5 - p)·π); the
    # combined p is 0.5 - arctan(mean(T))/π. Dominated by the smallest p, robust
    # to correlated cells.
    Pc = np.clip(P, 1e-300, 1.0 - 1e-16)
    T = np.tan((0.5 - Pc) * np.pi)
    comb = np.empty((n_ct, n_reg), dtype=np.float64)
    pct = np.zeros((n_ct, n_reg), dtype=np.float64)
    for j, c in enumerate(cts):
        m = ct == c
        comb[j] = 0.5 - np.arctan(np.mean(T[m], axis=0)) / np.pi
        pct[j] = (P[m] < alpha).mean(axis=0)
    comb = np.clip(comb, 1e-300, 1.0)
    neglog10 = -np.log10(comb)
    # BH-FDR across the full grid
    flat = comb.ravel()
    order = np.argsort(flat)
    m_tests = flat.size
    bh = flat[order] * m_tests / (np.arange(m_tests) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    fdr = np.empty(m_tests); fdr[order] = np.clip(bh, 0, 1)
    reject = fdr.reshape(n_ct, n_reg) < alpha
    return {"neglog10p": neglog10, "reject": reject, "pct_significant": pct,
            "cts": cts, "names": list(names)}


def _activity_per_celltype(data, groupby, key, values, uns_key="regulon", groups=None):
    """Return (matrix [n_celltype × n_regulon], celltypes, names) of the chosen
    per-cell value aggregated (mean) by cell type.
    ``values`` ∈ zscore|mean|pval|cosg|specificity|significance. 'cosg'/'specificity'
    read the dense regulon×cell-type IQR-COSG specificity matrix; 'significance'
    reads the ACAT-combined −log10(p) per cell type.
    ``groups`` restricts the displayed cell types to that subset (drops the rest,
    e.g. an 'Unassigned' label)."""
    def _sub(M, cts, names):
        if groups is None:
            return M, cts, names
        gset = set(groups)
        keep = [i for i, c in enumerate(cts) if c in gset]
        return M[keep], [cts[i] for i in keep], names

    if values in ("cosg", "specificity"):
        sm = (io.get_struct(data, uns_key, default={}) or {}).get("specificity_matrix")
        if sm is None:
            raise ValueError("values='%s' needs the specificity matrix — run "
                             "piaso.tl.regulonSpecificity / inferRegulon first." % values)
        mat = np.asarray(sm["matrix"], dtype=np.float64)          # [n_reg, n_ct]
        return _sub(mat.T, list(sm["celltypes"]), list(sm["regulons"]))
    if values == "significance":
        sig = _significance_per_celltype(data, groupby, key, uns_key=uns_key)
        return _sub(sig["neglog10p"], sig["cts"], sig["names"])
    names = (io.get_struct(data, uns_key, default={}) or {}).get("names")
    if values == "pval":
        if not io.has_embedding(data, key + "_pval"):
            raise ValueError("values='pval' needs the '_pval' embedding "
                             "(run inferRegulon/regulonActivity with compute_pvalues=True).")
        p = io.get_embedding(data, key + "_pval").astype(np.float64)
        act = -np.log10(np.clip(p, 1e-300, 1.0))
    else:
        act = io.get_embedding(data, key).astype(np.float64)
    if names is None:
        names = [f"R{i}" for i in range(act.shape[1])]
    ct = io.get_celltypes(data, groupby)
    cts = sorted(set(ct)) if groups is None else [c for c in sorted(set(ct)) if c in set(groups)]
    M = np.vstack([act[ct == c].mean(axis=0) for c in cts])   # [n_ct, n_reg]
    return M, cts, list(names)


def regulonActivity(
    data, *, groupby: str, key: str = "X_regulon", values: str = "zscore",
    style: str = "heatmap", groups: Optional[Sequence[str]] = None,
    regulons: Optional[Sequence[str]] = None, top_per_celltype: int = 5,
    square: bool = True, tf_min_pct: Optional[float] = None,
    cmap=None, vmin=None, vmax=None, vmax_pct: float = 98.0,
    significance: bool = False, alpha: float = 0.05, size_by: str = "significance",
    uns_key: str = "regulon",
    figsize=None, save=None, show=True, return_fig=False,
):
    """Regulon activity / specificity per cell type (regulon × cell type).

    A single entry point for the cell-type-level regulon view, as a **heatmap**
    (default) or a **dotplot**. Every selected regulon is shown across **all**
    cell types (no per-cell-type masking) — read from the object.

    Parameters
    ----------
    data : AnnData | cytome.Dataset
        Carries ``obsm[key]`` (activity) + ``uns['regulon']['names']`` from
        :func:`piaso.tl.inferRegulon` / :func:`piaso.tl.regulonActivity`.
    groupby : str
        Cell-type column.
    key : str, default 'X_regulon'
        Activity embedding key.
    values : {'zscore','mean','pval','cosg','specificity','significance'}, default 'zscore'
        Quantity displayed (heatmap colour, or dot colour in dotplot style).
        ``'zscore'`` = each regulon's mean activity z-scored across cell types
        (diverging). ``'mean'`` = raw mean activity. ``'pval'`` = mean
        ``-log10(p)`` per cell type (reads ``key + '_pval'``). ``'cosg'`` /
        ``'specificity'`` = the dense regulon × cell-type IQR-COSG specificity
        matrix. ``'significance'`` = ACAT-combined ``-log10(p)`` per cell type.
    style : {'heatmap', 'dotplot'}, default 'heatmap'
        ``'heatmap'`` keeps the classic block view (optionally starred at
        FDR-significant blocks via ``significance=True``). ``'dotplot'`` colours
        each dot by ``values`` and scales its **size** by ``size_by`` (a
        significance metric) — meaningful because regulon activity is non-zero in
        ~all cells, so "fraction expressing" would be uninformative.
    regulons : sequence[str], optional
        Rows to show. Default = union of the top-``top_per_celltype`` per cell type.
    top_per_celltype : int, default 5
        Per-cell-type row selection (the rows are the union; each row is then
        rendered across all cell types).
    square : bool, default True
        Square blocks (``aspect='equal'``).
    tf_min_pct : float, optional
        Grey out (heatmap) / drop (dotplot) blocks where the regulon's TF is
        detected in fewer than this fraction of a cell type's cells
        (uses ``uns['regulon']['tf_pct']``).
    vmin, vmax : float, optional
        Colour limits. If ``None``, data-driven from ``vmax_pct`` (symmetric for
        zscore).
    vmax_pct : float, default 98
        Percentile (0-100) for the auto colour limit.
    significance : bool, default False
        Heatmap: overlay a ``*`` on blocks passing BH-FDR < ``alpha`` (ACAT-combined
        per-cell p-values). Ignored in dotplot style (size already encodes it).
    alpha : float, default 0.05
        FDR threshold for the significance overlay / reporting.
    size_by : {'significance', 'pct_significant'}, default 'significance'
        Dotplot dot-size metric: ACAT-combined ``-log10(p)`` (default), or the
        fraction of a cell type's cells with raw ``p < alpha`` (dependence-free).
    cmap : str, optional
        Override colormap (default 'RdBu_r' for zscore, 'magma' otherwise).
    save, show, return_fig
        Standard plotting controls.
    """
    import matplotlib.pyplot as plt
    import copy as _copy
    if values not in ("zscore", "mean", "pval", "cosg", "specificity", "significance"):
        raise ValueError("values must be 'zscore','mean','pval','cosg',"
                         "'specificity' or 'significance'.")
    if style not in ("heatmap", "dotplot"):
        raise ValueError("style must be 'heatmap' or 'dotplot'.")
    if size_by not in ("significance", "pct_significant"):
        raise ValueError("size_by must be 'significance' or 'pct_significant'.")

    M, cts, names = _activity_per_celltype(data, groupby, key, values, uns_key=uns_key,
                                           groups=groups)
    if regulons is None:
        sel = set()
        for i in range(len(cts)):
            sel.update(names[j] for j in np.argsort(-M[i])[:top_per_celltype])
        regulons = [r for r in names if r in sel]
    ridx = [names.index(r) for r in regulons]
    H = M[:, ridx].T                                          # [n_reg, n_ct]

    if values == "zscore":
        H = (H - H.mean(axis=1, keepdims=True)) / (H.std(axis=1, keepdims=True) + 1e-9)
        if vmin is None and vmax is None:
            lim = float(np.percentile(np.abs(H), vmax_pct)) or 1.0
            vmin, vmax = -lim, lim
        cmap = cmap or "RdBu_r"
    else:                                                    # sequential
        if vmax is None:
            vmax = float(np.percentile(H, vmax_pct))
        if vmin is None:
            vmin = float(H.min()) if values == "mean" else 0.0
        cmap = cmap or "magma"

    # TF-presence mask (regulon × cell type): TF detected in < tf_min_pct of cells.
    mask = None
    if tf_min_pct is not None:
        tp = (io.get_struct(data, uns_key, default={}) or {}).get("tf_pct")
        if tp is not None:
            tpm = np.asarray(tp["matrix"], dtype=np.float64)  # [tf, ct]
            tr = {r: i for i, r in enumerate(tp["regulons"])}
            tc = {c: i for i, c in enumerate(tp["celltypes"])}
            mask = np.zeros_like(H, dtype=bool)
            for ri, r in enumerate(regulons):
                for ci, c in enumerate(cts):
                    if r in tr and c in tc and tpm[tr[r], tc[c]] < tf_min_pct:
                        mask[ri, ci] = True

    # Significance (ACAT + BH-FDR), aligned to (regulons, cts), for the FDR
    # overlay (heatmap) or the dot size (dotplot).
    sig_neglog = sig_reject = sig_pct = None
    if significance or style == "dotplot":
        sig = _significance_per_celltype(data, groupby, key, alpha=alpha, uns_key=uns_key)
        sni = {n: i for i, n in enumerate(sig["names"])}
        sci = {c: i for i, c in enumerate(sig["cts"])}

        def _align(mat, fill=np.nan):
            out = np.full((len(regulons), len(cts)), fill, dtype=float)
            for ri, r in enumerate(regulons):
                if r not in sni:
                    continue
                for ci, c in enumerate(cts):
                    if c in sci:
                        out[ri, ci] = mat[sci[c], sni[r]]
            return out
        sig_neglog = _align(sig["neglog10p"])
        sig_reject = _align(sig["reject"].astype(float), fill=0.0) > 0.5
        sig_pct = _align(sig["pct_significant"], fill=0.0)

    title = {"zscore": "Regulon activity (z per regulon)",
             "mean": "Mean regulon activity",
             "pval": "Regulon activity  −log10(p)",
             "cosg": "Regulon specificity (IQR-COSG)",
             "specificity": "Regulon specificity (IQR-COSG)",
             "significance": "Regulon significance  −log10(ACAT p)"}[values]

    if style == "heatmap":
        Hm = np.ma.masked_array(H, mask=mask) if mask is not None else H
        figsize = figsize or (max(4, 0.42 * len(cts) + 2), max(3, 0.30 * len(regulons) + 1))
        fig, ax = plt.subplots(figsize=figsize)
        cmap_obj = _copy.copy(plt.get_cmap(cmap)); cmap_obj.set_bad("#dddddd")
        im = ax.imshow(Hm, aspect=("equal" if square else "auto"), cmap=cmap_obj,
                       vmin=vmin, vmax=vmax)
        if significance and sig_reject is not None:
            yy, xx = np.where(sig_reject)
            ax.scatter(xx, yy, marker="*", s=16, c="k", linewidths=0, zorder=3)
        ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=90, fontsize=7)
        ax.set_yticks(range(len(regulons))); ax.set_yticklabels(regulons, fontsize=6)
        ax.set_title(title + ("  (* FDR<%.2g)" % alpha if significance else ""))
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    else:                                                    # dotplot
        size_src = sig_neglog if size_by == "significance" else sig_pct
        smax = float(np.nanmax(size_src)) if np.isfinite(np.nanmax(size_src)) else 0.0
        sizes_mat = 30 + 250 * (np.nan_to_num(size_src) / (smax + 1e-9))
        if mask is not None:
            sizes_mat = np.where(mask, 0.0, sizes_mat)
        xs, ys, cc, ss = [], [], [], []
        for ri in range(len(regulons)):
            for ci in range(len(cts)):
                xs.append(ci); ys.append(ri); cc.append(H[ri, ci]); ss.append(sizes_mat[ri, ci])
        figsize = figsize or (max(4, 0.45 * len(cts) + 2.5), max(3, 0.30 * len(regulons) + 1.5))
        fig, ax = plt.subplots(figsize=figsize)
        scat = ax.scatter(xs, ys, s=ss, c=cc, cmap=cmap, vmin=vmin, vmax=vmax,
                          edgecolor="k", linewidth=0.3, zorder=3)
        if square:
            ax.set_aspect("equal")
        ax.set_xticks(range(len(cts))); ax.set_xticklabels(cts, rotation=90, fontsize=7)
        ax.set_yticks(range(len(regulons))); ax.set_yticklabels(regulons, fontsize=6)
        ax.set_xlim(-0.5, len(cts) - 0.5); ax.set_ylim(-0.5, len(regulons) - 0.5)
        ax.invert_yaxis()
        ax.set_title(title)
        fig.colorbar(scat, ax=ax, fraction=0.025, pad=0.02, label=title.split("  ")[-1])
        # dot-size legend (3 ticks of the size metric)
        lbl = "−log10(ACAT p)" if size_by == "significance" else f"% cells p<{alpha:g}"
        handles = []
        for frac in (0.33, 0.66, 1.0):
            sv = smax * frac
            handles.append(ax.scatter([], [], s=30 + 250 * frac, c="grey",
                                      edgecolors="k", linewidths=0.3,
                                      label=(f"{sv:.0%}" if size_by == "pct_significant"
                                             else f"{sv:.1f}")))
        ax.legend(handles=handles, title=lbl, loc="center left",
                  bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=6,
                  title_fontsize=6, labelspacing=1.0)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if return_fig:
        return fig


def regulonNetwork(
    data=None, tf: Optional[str] = None, *, regulons=None, uns_key: str = "regulon",
    max_targets: int = 25, layout: str = "fr", seed: int = 0, figsize=(7, 7),
    node_size_tf=600, node_size_target=120, label_targets: bool = True,
    target_fontsize: float = 7.0,
    save=None, show=True, return_fig=False,
):
    """TF→target network via igraph layout + matplotlib (no networkx).

    Reads ``regulons`` from the object (``uns[uns_key]['regulons']`` / cytome
    metadata) unless a ``{TF: [targets]}`` dict is passed via ``regulons=``.

    Parameters
    ----------
    data : AnnData | cytome.Dataset, optional
        Object carrying the regulons (from :func:`piaso.tl.inferRegulon`).
    tf : str, optional
        Plot one TF's regulon; otherwise the largest few (or those in ``regulons``).
    max_targets : int, default 25
        Targets shown per TF.
    """
    import igraph as ig
    import matplotlib.pyplot as plt

    if regulons is None:
        if data is None:
            raise ValueError("pass `data` (object) or `regulons` (dict).")
        struct = io.get_struct(data, uns_key, default=None)
        if not struct or "regulons" not in struct:
            raise KeyError(f"no regulons in {uns_key!r} on this object — run inferRegulon.")
        regulons = struct["regulons"]
    reg = regulons
    if tf is not None:
        tfs = [tf]
    else:
        tfs = sorted(reg, key=lambda t: -len(reg[t]))[:6]

    nodes, is_tf, edges, nidx = [], {}, [], {}
    def _add(n, tf_flag):
        if n not in nidx:
            nidx[n] = len(nodes); nodes.append(n); is_tf[n] = tf_flag
        elif tf_flag:
            is_tf[n] = True
    for t in tfs:
        _add(t, True)
        for g in list(reg.get(t, []))[:max_targets]:
            _add(g, False); edges.append((nidx[t], nidx[g]))
    g = ig.Graph(n=len(nodes), edges=edges, directed=True)
    try:
        ig.set_random_number_generator(np.random.RandomState(seed))
    except Exception:
        pass
    lay = np.asarray(g.layout(layout).coords) if g.vcount() else np.zeros((0, 2))

    fig, ax = plt.subplots(figsize=figsize)
    for s, d in edges:
        ax.plot([lay[s, 0], lay[d, 0]], [lay[s, 1], lay[d, 1]],
                color="#bbbbbb", lw=0.5, zorder=1)
    tf_mask = np.array([is_tf[n] for n in nodes])
    ax.scatter(lay[~tf_mask, 0], lay[~tf_mask, 1], s=node_size_target,
               c="#7D80DA", edgecolor="k", linewidth=0.3, zorder=2, label="target")
    ax.scatter(lay[tf_mask, 0], lay[tf_mask, 1], s=node_size_tf,
               c="#D55E00", edgecolor="k", linewidth=0.5, zorder=3, label="TF")
    # Target labels are the point of a single-TF network: an unlabelled ring
    # of dots says only "this TF has some targets", which the target count
    # already said. Off is available for dense multi-TF layouts.
    for i, n in enumerate(nodes):
        if is_tf[n]:
            ax.text(lay[i, 0], lay[i, 1], n, fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=4)
        elif label_targets:
            ax.text(lay[i, 0], lay[i, 1], n, fontsize=target_fontsize,
                    ha="center", va="bottom", zorder=4,
                    color="#333333",
                    path_effects=[__import__("matplotlib.patheffects",
                                             fromlist=["withStroke"])
                                  .withStroke(linewidth=2, foreground="white")])
    ax.set_axis_off()
    ax.set_title("Regulon network" + (f": {tf}" if tf else ""))
    ax.legend(loc="upper right", fontsize=8, markerscale=0.6)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if return_fig:
        return fig


def regulonEmbedding(
    data, regulons, *, basis: str = "X_umap", key: str = "X_regulon",
    uns_key: str = "regulon", groupby: Optional[str] = None,
    groups: Optional[Sequence[str]] = None,
    use_pval: bool = False, ncols: int = 4, point_size=None, cmap="magma",
    vmax_pct: float = 99.0, figsize=None, save=None, show=True, return_fig=False,
):
    """Scatter a cell embedding (e.g. UMAP) colored by each regulon's activity.

    The regulon activity matrix ``X_regulon`` (cells × n_TF) is itself a cell
    embedding on the object, so this reads column ``regulon`` from it and the base
    ``basis`` coordinates — for AnnData and cytome alike.

    Parameters
    ----------
    data : AnnData | cytome.Dataset
        Object with ``obsm[key]`` (+ ``key + '_pval'``) and ``obsm[basis]``.
    regulons : str | sequence[str]
        Regulon TF name(s) to plot (one panel each).
    basis : str, default 'X_umap'
        Base embedding key for the x/y coordinates.
    key : str, default 'X_regulon'
        Activity embedding key.
    use_pval : bool, default False
        Color by ``-log10(p)`` (from ``key + '_pval'``) instead of raw activity.
    ncols : int, default 4
        Panels per row.
    """
    import matplotlib.pyplot as plt
    if isinstance(regulons, str):
        regulons = [regulons]
    names = (io.get_struct(data, uns_key, default={}) or {}).get("names")
    if names is None:
        raise ValueError("no regulon names on the object — run inferRegulon first.")
    if use_pval:
        if not io.has_embedding(data, key + "_pval"):
            raise ValueError("use_pval=True needs the '_pval' embedding.")
        act = -np.log10(np.clip(io.get_embedding(data, key + "_pval").astype(np.float64), 1e-300, 1.0))
    else:
        act = io.get_embedding(data, key).astype(np.float64)
    emb = io.get_embedding(data, basis)
    if groups is not None:                       # restrict to a cell-type subset
        if groupby is None:
            raise ValueError("regulonEmbedding: groups= requires groupby=.")
        m = np.isin(np.asarray(io.get_celltypes(data, groupby)), list(groups))
        act, emb = act[m], emb[m]
    n = len(regulons)
    nrows = int(np.ceil(n / ncols)); ncols = min(ncols, n)
    if point_size is None:
        point_size = max(1.0, min(12.0, 50000.0 / emb.shape[0]))
    figsize = figsize or (4 * ncols, 3.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for k, tf in enumerate(regulons):
        ax = axes[k // ncols][k % ncols]
        if tf not in names:
            ax.set_axis_off(); ax.set_title(f"{tf} (absent)"); continue
        v = act[:, names.index(tf)]
        order = np.argsort(v)                      # plot high-activity cells on top
        s = ax.scatter(emb[order, 0], emb[order, 1], c=v[order], s=point_size,
                       cmap=cmap, vmin=0, vmax=float(np.percentile(v, vmax_pct)))
        ax.set_title(f"{tf}" + ("  −log10(p)" if use_pval else "  activity"))
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(s, ax=ax, fraction=0.045, pad=0.02)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if return_fig:
        return fig
