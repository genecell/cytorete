"""``inferRegulon`` — the PIASO-GRN orchestrator (AnnData + cytome), object-centric.

COSG specificity → target-gene selection → promoter motif scan (cistrome) AND
trans co-specificity → positive regulons → score() activity → COSG-λ cell-type
regulon specificity. Reuses existing PIASO primitives (COSG, score,
_pairwise_metric, _genome, _interval_overlap); the only new heavy lifting is the
PWM scan (Rust/numpy) and motif/2bit I/O. Results are written onto the object
(``adata.obsm``/``uns`` or cytome embeddings/metadata).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..data import load_jaspar_meme, load_cisbp, load_tf_list, build_tf_motif_map
from ..preprocessing._promoters import extract_promoter_sequences
from ..preprocessing._cistrome import build_cistrome
from ._regulons import build_regulons
from ._activity import regulonActivity, regulonSpecificity
from . import _objio as io


def _select_target_genes(data, S, universe, tf_genes, *, target_genes,
                         n_markers_per_celltype, n_top_genes, score_layer, verbose):
    """Pick the candidate target-gene universe for the promoter scan.

    'cosg' = union of top COSG markers per cell type (free, from ``S``);
    'hvg'  = top-N by INFOG-layer variance; 'all' = every gene; or an explicit list.
    TF genes are always included.
    """
    if isinstance(target_genes, (list, tuple, set)):
        sel = set(map(str, target_genes))
    elif target_genes == "all":
        sel = set(universe)
    elif target_genes == "cosg":
        sel = set()
        for ct in S.columns:
            sel.update(S[ct].nlargest(int(n_markers_per_celltype)).index.tolist())
    elif target_genes == "hvg":
        if io.is_cytome(data):
            raise ValueError("target_genes='hvg' is AnnData-only; use 'cosg' for cytome.")
        import scipy.sparse as sp
        L = data.layers[score_layer] if score_layer in data.layers else data.X
        mean = np.asarray(L.mean(0)).ravel()
        sq = (np.asarray(L.multiply(L).mean(0)).ravel() if sp.issparse(L)
              else (np.asarray(L) ** 2).mean(0))
        order = np.argsort(-(sq - mean ** 2))
        sel = {universe[i] for i in order[:int(n_top_genes)]}
    else:
        raise ValueError(f"target_genes {target_genes!r} invalid "
                         "('cosg' | 'hvg' | 'all' | list).")
    scan_genes = sorted(sel | set(tf_genes))
    if verbose:
        print(f"[inferRegulon] target_genes={target_genes!r} → "
              f"{len(scan_genes)} genes to scan ({len(tf_genes)} TFs always in)")
    return scan_genes


def _is_cytome(data) -> bool:
    try:
        from cytome import CytomeDataset
    except Exception:
        return False
    return isinstance(data, CytomeDataset)


def _gene_universe(data, modality: str = "RNA") -> List[str]:
    if not _is_cytome(data):
        return list(map(str, data.var_names))
    try:
        from cytome.utils.modality import modality_feature_table_info
        feat_tbl, _idx, name_col = modality_feature_table_info(data, modality)
        names = np.asarray(getattr(data, feat_tbl)[name_col])
        return [str(x) for x in names]
    except Exception:
        return [str(x) for x in np.asarray(data.genes["gene_name"])]


def _resolve_rust_scanner(use_rust: bool):
    """Return the Rust scan_motifs wrapper if available + requested, else None
    (caller falls back to the numpy scanner)."""
    if not use_rust:
        return None
    try:
        from piaso.preprocessing import scan_motifs_rust
        return scan_motifs_rust
    except Exception:
        return None


def _ensure_score_layer(data, modality, layer, *, verbose=1):
    """Ensure a normalized `layer` is available for COSG/scoring on **AnnData or cytome**.

    Parity with ``inferGRN._ensure_norm_layer``. ``inferRegulon`` scores on ``'infog'`` (variance-
    stabilising) by default, but the object may hold only raw counts — so we compute + persist the
    normalized layer on demand instead of crashing or silently scoring the wrong matrix:
      * cytome  → ``{modality}_infog`` matrix (delegates to ``_ensure_norm_layer``);
      * AnnData → ``adata.layers['infog']`` (INFOG from raw counts).
    ``None``/``'counts'`` pass through unchanged. Returns the layer name to use.
    """
    if layer is None or layer == "counts":
        return layer
    if io.is_cytome(data):
        from ._layers import _ensure_norm_layer
        return _ensure_norm_layer(data, modality, layer, verbose=verbose)
    # AnnData
    if layer in data.layers:
        return layer
    if layer == "infog":
        # INFOG needs RAW counts: prefer adata.layers['counts'], else adata.X if it is integer-valued.
        src = "counts" if "counts" in data.layers else None
        if src is None:
            X = data.X
            samp = X[:100] if X.shape[0] > 100 else X
            samp = samp.toarray() if hasattr(samp, "toarray") else np.asarray(samp)
            if not np.allclose(samp, np.round(samp)):
                raise KeyError(
                    "inferRegulon: layer 'infog' is absent and cannot be computed — no raw counts "
                    "found (need adata.layers['counts'] or integer adata.X). Run piaso normalization "
                    "(piaso.tl.infog / piaso.tl.score) first, or pass an existing layer.")
        if verbose:
            _from = "adata.layers['counts']" if src else "adata.X"
            print(f"[inferRegulon] computing adata.layers['infog'] (INFOG from {_from})…", flush=True)
        from piaso.tools import infog as _infog
        _infog(data, key_added="infog", inplace=False, layer=src,
               n_top_genes=min(3000, data.shape[1]),   # robust to small gene panels
               verbosity=(1 if verbose else 0))
        return "infog"
    raise KeyError(
        f"inferRegulon: layer '{layer}' not found in adata.layers ({list(data.layers)}). "
        f"Create it (piaso normalization) or pass an existing layer.")


def inferRegulon(
    data,
    genome: str,
    groupby: str,
    *,
    # --- motif DB ---
    motif_db: str = "jaspar",
    jaspar_path: Optional[str] = None,
    cisbp_dir: Optional[str] = None,
    cisbp_tf_info: Optional[str] = None,
    # --- TF list ---
    tf_list: Optional[Sequence[str]] = None,
    tf_list_path: Optional[str] = None,
    species: Optional[str] = None,
    # --- genome / promoter ---
    twobit_path: Optional[str] = None,
    tss_bed: Optional[str] = None,
    data_dir: Optional[str] = None,
    upstream: int = 1000,
    downstream: int = 500,
    biotypes: Optional[Sequence[str]] = ("protein_coding",),
    regulatory_regions: str = "promoter",
    screen_bed: Optional[str] = None,
    cre_window: int = 100_000,
    # --- scan / cistrome ---
    cistrome_method: str = "motif_bg",
    pvalue: float = 1e-4,
    motif_bg_quantile: float = 0.05,
    motif_bg_combine_pvalue: bool = True,
    flat_motif_filter: bool = True,
    nes_threshold: float = 3.0,
    both_strands: bool = True,
    use_rust: bool = True,
    # --- target-gene selection (the candidate target universe, in-function) ---
    target_genes="cosg",
    n_markers_per_celltype: int = 200,
    n_top_genes: int = 5000,
    # --- cospecificity ---
    metric: str = "weighted_cosine",
    modality: str = "RNA",
    cosg_mu: float = 1.0,
    cosg_expressed_pct: float = 0.1,
    cosg_layer: str = "infog",                 # aligned with inferGRN (cosg_layer_rna='infog'); was 'counts'
    # --- regulons ---
    min_targets: int = 10,
    weight_mode: str = "binary",
    top_targets_per_tf: Optional[int] = None,
    per_celltype: bool = True,
    # --- activity / specificity ---
    compute_activity: bool = True,
    score_layer: str = "infog",
    n_ctrl_set: int = 1000,
    compute_pvalues: bool = True,
    max_workers: int = 8,
    specificity: bool = True,
    specificity_use_neglog10p: bool = False,
    # --- output ---
    key_added: str = "X_regulon",
    uns_key: str = "regulon",
    copy: bool = False,
    verbose: int = 1,
):
    """Infer a TF→target-gene regulatory network (regulons) from scRNA-seq.

    A SCENIC-analog built on PIASO primitives: **COSG** cell-type specificity
    replaces GENIE3 co-expression, a **promoter motif scan** (cistrome) replaces
    RcisTarget cis-pruning, and **score()** replaces AUCell for regulon activity.
    coSpecificity is cell-type-resolved, so regulons are cell-type-aware by
    construction. Works on **AnnData and cytome**; results are written onto the
    object (scanpy-style) and returned only when ``copy=True``.

    Pipeline: COSG λ matrix ``S`` → select candidate target genes (``target_genes``)
    → scan only those promoters → cistrome ``M`` (TF×gene) → keep motif-supported,
    co-specific, positive edges → regulons → ``score()`` activity → COSG-λ
    cell-type specificity.

    Parameters
    ----------
    data : AnnData | cytome.Dataset
        RNA object with raw/normalized counts and a cell-type column.
    genome : {'hg38', 'mm10'}
        Genome for TSS/promoter resolution. Needs a local ``.2bit``
        (``piaso.data.fetch_2bit``) and the PIASO-data TSS BED.
    groupby : str
        Cell-type column (``obs``/``cells``) defining the specificity groups.
    motif_db : {'jaspar', 'cisbp', 'both'}, default 'jaspar'
        Motif source. ``jaspar_path`` / ``cisbp_dir`` (+ ``cisbp_tf_info``) supply files.
    tf_list, tf_list_path, species
        Restrict TFs to a curated list (e.g. AnimalTFDB); ``None`` uses the motif
        DB's TFs ∩ the expression genes.
    twobit_path, tss_bed, data_dir
        Override the auto-resolved genome files / cache dir.
    upstream, downstream : int, default 1000 / 500
        Promoter window around each TSS (strand-aware; alt promoters kept separate).
    biotypes : sequence | None, default ('protein_coding',)
        Restrict TSS to these biotypes.
    regulatory_regions : {'promoter', 'promoter+cre'}, default 'promoter'
        ``'promoter+cre'`` also scans SCREEN cCREs within ``cre_window`` of the TSS
        (needs ``piaso.data.fetch_screen`` / ``screen_bed``).
    cistrome_method : {'motif_bg', 'hits', 'nes'}, default 'motif_bg'
        How a motif→gene edge is called. ``'motif_bg'`` keeps only edges in a
        motif's upper score tail across all promoters (caps promiscuity; see
        ``motif_bg_quantile`` / ``flat_motif_filter``). ``'hits'`` = absolute
        p-value. ``'nes'`` = RcisTarget-style recovery-AUC pruning.
    pvalue, motif_bg_quantile, motif_bg_combine_pvalue, flat_motif_filter, nes_threshold
        Cistrome-method knobs (see :func:`piaso.pp.buildCistrome`).
    both_strands, use_rust : bool
        Scan both strands; use the Rust scanner when built (numpy fallback).
    target_genes : {'cosg', 'hvg', 'all'} | list, default 'cosg'
        Candidate target universe for the scan. ``'cosg'`` = union of the top
        ``n_markers_per_celltype`` COSG markers per cell type (free, from ``S``;
        biologically targeted). ``'hvg'`` = top ``n_top_genes`` by INFOG-layer
        variance (AnnData only). ``'all'`` = every gene (slow). TFs always included.
    metric : str, default 'weighted_cosine'
        coSpecificity pairwise metric ('weighted_cosine'|'geomean'|'cosine'|'outer').
    modality, cosg_mu, cosg_expressed_pct, cosg_layer
        COSG knobs (modality, μ penalty, expression filter, layer).
    min_targets : int, default 10
        Minimum regulon size.
    weight_mode : {'binary', 'cospec', 'motif_cospec'}, default 'binary'
        Target weighting for activity scoring.
    top_targets_per_tf : int | None
        Cap targets per TF (by coSpecificity).
    per_celltype : bool, default True
        Also emit per-cell-type regulons (COSG-style).
    compute_activity : bool, default True
        Score regulon activity into ``obsm[key_added]``.
    score_layer : str, default 'infog'
        Layer for ``score()`` activity.
    compute_pvalues : bool, default True
        Also store per-cell regulon p-values at ``obsm[key_added + '_pval']``.
    max_workers : int, default 8
        Worker threads for the ``score()`` regulon-activity step (Rust-accelerated).
        parallelizes the activity scoring (set ``1`` for fully reproducible single-thread).
    specificity : bool, default True
        Compute cell-type regulon specificity (COSG λ on activity).
    specificity_use_neglog10p : bool, default False
        Run the specificity COSG on ``-log10(p)`` instead of the raw activity.
    key_added : str, default 'X_regulon'
        Embedding key for the ``cells × n_TF`` activity matrix.
    uns_key : str, default 'regulon'
        Struct key on the object (``adata.uns`` / cytome ``metadata``) holding
        ``regulons``, ``per_celltype``, ``edges``, ``specificity``, ``names``,
        ``cistrome_density``, ``celltypes``, ``params``.
    copy : bool, default False
        If True, also return the result dict; otherwise return ``None`` (everything
        is on the object; read via ``piaso.pl.regulon*`` or ``data.uns[uns_key]``).
    verbose : int, default 1

    Returns
    -------
    None, or the result dict if ``copy=True``.

    Examples
    --------
    >>> piaso.tl.inferRegulon(adata, genome='mm10', groupby='Subclass',
    ...                       jaspar_path=JASPAR, twobit_path=TWOBIT)
    >>> adata.obsm['X_regulon'].shape           # cells × n_TF
    >>> piaso.pl.regulonActivity(adata, groupby='Subclass')
    >>> piaso.pl.regulonActivity(adata, groupby='Subclass',     # specificity dotplot
    ...                          values='specificity', style='dotplot')
    """
    from piaso.tools import cospecificity_trans
    from piaso.tools import specificity_matrix as _specificity_matrix

    universe = io.get_var_names(data, modality=modality)
    if verbose:
        print(f"[inferRegulon] genome={genome} groupby={groupby!r} "
              f"gene universe = {len(universe)} features")

    # 1) motifs + TF list → TF→PWM map restricted to the expression universe
    pwms = []
    if motif_db in ("jaspar", "both"):
        if jaspar_path is None:
            raise ValueError("motif_db='jaspar' needs jaspar_path=<.meme file>.")
        pwms += load_jaspar_meme(jaspar_path)
    if motif_db in ("cisbp", "both"):
        if cisbp_dir is None:
            raise ValueError("motif_db includes cisbp but cisbp_dir is None.")
        pwms += load_cisbp(cisbp_dir, cisbp_tf_info)
    if not pwms:
        raise ValueError(f"no motifs loaded (motif_db={motif_db!r}).")
    tfset = set(tf_list) if tf_list is not None else None
    if tfset is None and tf_list_path is not None:
        tfset = load_tf_list(species or "human", path=tf_list_path)
    tf_motif_map = build_tf_motif_map(pwms, tf_list=tfset, gene_universe=universe)
    if not tf_motif_map:
        raise ValueError("no TFs left after intersecting motif DB ∩ TF list ∩ universe.")
    if verbose:
        print(f"[inferRegulon] {len(tf_motif_map)} TFs with motifs in the data")

    # 2) COSG λ specificity S (computed once; drives target selection + coSpec).
    #    Ensure the normalized COSG + score layers exist up front (both backends), so a counts-only
    #    object is normalized on demand rather than crashing / silently scoring the wrong layer.
    cosg_layer = _ensure_score_layer(data, modality, cosg_layer, verbose=verbose)
    score_layer = _ensure_score_layer(data, modality, score_layer, verbose=verbose)
    S = _specificity_matrix(
        data, groupby=groupby, modality=modality, cosg_mu=cosg_mu,
        cosg_expressed_pct=cosg_expressed_pct, cosg_layer=cosg_layer, verbose=verbose)

    # 3) candidate target genes → scan only those promoters
    scan_genes = _select_target_genes(
        data, S, universe, set(tf_motif_map), target_genes=target_genes,
        n_markers_per_celltype=n_markers_per_celltype, n_top_genes=n_top_genes,
        score_layer=score_layer, verbose=verbose)
    promoter_data = extract_promoter_sequences(
        genome, tss_bed=tss_bed, twobit_path=twobit_path, genes=scan_genes,
        upstream=upstream, downstream=downstream, biotypes=biotypes,
        data_dir=data_dir, regulatory_regions=regulatory_regions,
        screen_bed=screen_bed, cre_window=cre_window)
    if verbose:
        print(f"[inferRegulon] {len(set(promoter_data['seq_genes']))} genes with "
              f"promoters ({len(promoter_data['sequences'])} intervals)")

    # 4) cistrome M [TF × gene]
    scan_fn = _resolve_rust_scanner(use_rust)
    _cm = "motif_bg" if cistrome_method == "nes" else cistrome_method
    cistrome = build_cistrome(
        promoter_data, tf_motif_map, cistrome_method=_cm, pvalue=pvalue,
        motif_bg_quantile=(0.20 if cistrome_method == "nes" else motif_bg_quantile),
        motif_bg_combine_pvalue=motif_bg_combine_pvalue,
        flat_motif_filter=flat_motif_filter,
        both_strands=both_strands, scan_fn=scan_fn, verbose=bool(verbose))
    M = cistrome["M"].tocsr()
    tfs, genes = cistrome["tfs"], np.asarray(cistrome["genes"])
    pairs = {tf: list(genes[M.indices[M.indptr[i]:M.indptr[i + 1]]])
             for i, tf in enumerate(tfs) if M.indptr[i + 1] > M.indptr[i]}

    # 5) trans co-specificity over motif-supported pairs (REUSE S)
    cospec = cospecificity_trans(
        data, groupby=groupby, pairs=pairs, metric=metric, modality=modality,
        specificity=S, verbose=verbose)

    # 6) regulons (optionally RcisTarget-style NES pruning)
    reg = build_regulons(
        cistrome, cospec, min_targets=min_targets, weight_mode=weight_mode,
        top_targets_per_tf=top_targets_per_tf, per_celltype=per_celltype,
        nes=(cistrome_method == "nes"), nes_threshold=nes_threshold,
        verbose=bool(verbose))

    density = float(M.nnz) / max(1, M.shape[0] * M.shape[1])
    params = {"genome": genome, "groupby": groupby, "cistrome_method": cistrome_method,
              "target_genes": (target_genes if isinstance(target_genes, str) else "list"),
              "metric": metric, "weight_mode": weight_mode, "min_targets": min_targets}
    # 7) write the regulon struct onto the object (JSON-safe)
    io.update_regulon_struct(
        data, uns_key, regulons=reg["regulons"],
        weights={t: list(map(float, w)) for t, w in reg["weights"].items()},
        per_celltype=reg["per_celltype"], edges=reg["edges"],
        cistrome_density=density, celltypes=cospec["celltypes"], params=params)

    result = {"regulons": reg["regulons"], "weights": reg["weights"],
              "per_celltype": reg["per_celltype"], "edges": reg["edges"],
              "cistrome_density": density, "celltypes": cospec["celltypes"]}

    # 8) activity + specificity (written onto the object)
    if compute_activity and reg["regulons"]:
        activity, names, pvals = regulonActivity(
            data, reg["regulons"],
            weights=(reg["weights"] if weight_mode != "binary" else None),
            score_layer=score_layer, modality=modality, n_ctrl_set=n_ctrl_set,
            compute_pvalues=compute_pvalues, max_workers=max_workers,
            key_added=key_added, uns_key=uns_key, copy=True, verbose=verbose)
        result.update(activity=activity, names=names, activity_pval=pvals)
        # TF detection fraction per cell type (optional display mask in the plots)
        from ._activity import tf_detection_fraction
        tf_pct = tf_detection_fraction(data, names, groupby, modality=modality,
                                       layer=(score_layer if not io.is_cytome(data) else None))
        io.update_regulon_struct(data, uns_key, tf_pct={
            "regulons": list(tf_pct.index), "celltypes": list(tf_pct.columns),
            "matrix": tf_pct.to_numpy().astype(np.float32).tolist()})
        result["tf_pct"] = tf_pct
        if specificity:
            df = regulonSpecificity(
                data, groupby=groupby, activity_key=key_added,
                use_neglog10p=specificity_use_neglog10p, mu=cosg_mu,
                uns_key=uns_key, copy=True, verbose=verbose)
            result["specificity"] = df

    return result if copy else None
