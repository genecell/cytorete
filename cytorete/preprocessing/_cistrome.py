"""Cistrome prior `M` [TF × gene] — which TF can bind which gene's promoter.

Scans all promoter sequences with every TF's PWM(s) (one pass), then aggregates
(union over a TF's motifs = max score; max over a gene's alternative promoters)
into a sparse TF×gene matrix.

The motif→edge call (`cistrome_method`) is the lever that controls cistrome
density / promiscuity:

- ``'hits'``      — absolute PWM significance (per-motif p-value → score cutoff).
                    Simple, but short/degenerate motifs hit ~everywhere.
- ``'motif_bg'``  — **per-motif background model** (recommended): a `(motif, gene)`
                    edge is called only if its score is in the **upper tail of that
                    motif's score distribution across all promoters** (top
                    ``motif_bg_quantile``), optionally combined with the absolute
                    p-value. A **flat-motif filter** drops motifs whose top tail
                    isn't separated from their bulk (promiscuous / uninformative).
                    Normalises each motif against its own "background binding",
                    auto-caps density, and is distribution-free.

`'nes'` (RcisTarget-style set enrichment) is a *gene-set* test and is applied in
`build_regulons` over the coSpecificity candidate modules, using the continuous
``motif_gene_score`` this function also returns.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from piaso.preprocessing import scan_motifs, estimate_background
from piaso.preprocessing import pvalue_to_threshold


def build_cistrome(
    promoter_data: dict,
    tf_motif_map: Dict[str, list],
    *,
    background: Optional[np.ndarray] = None,
    cistrome_method: str = "motif_bg",
    pvalue: float = 1e-4,
    motif_bg_quantile: float = 0.05,
    motif_bg_combine_pvalue: bool = True,
    flat_motif_filter: bool = True,
    flat_motif_min_sep: float = 1.0,
    both_strands: bool = True,
    pseudocount: float = 0.01,
    scan_fn=None,
    verbose: bool = True,
) -> dict:
    """Build the TF×gene cistrome from promoter sequences + a TF→PWM map.

    Parameters
    ----------
    cistrome_method
        ``'hits'`` (absolute p-value) or ``'motif_bg'`` (per-motif tail, default).
    motif_bg_quantile
        Upper-tail fraction kept per motif in ``'motif_bg'`` (default 0.05 = top 5%).
    motif_bg_combine_pvalue
        In ``'motif_bg'``, also require the absolute p-value cutoff (AND).
    flat_motif_filter / flat_motif_min_sep
        Drop a motif whose ``(top-quantile − median)/MAD`` separation is below
        ``flat_motif_min_sep`` (promiscuous / flat motif carries no signal).
    scan_fn
        Optional scanner override (e.g. the Rust backend).

    Returns
    -------
    dict
        ``{"tfs", "genes", "M" (csr bool), "scores" (csr float32, best edge score),
        "motif_gene_score" (ndarray [n_motif, n_gene], continuous, for NES),
        "motif_tf_idx", "motif_ids", "background"}``.
    """
    if cistrome_method not in ("hits", "motif_bg"):
        raise ValueError(f"cistrome_method {cistrome_method!r} must be 'hits' or 'motif_bg' "
                         "('nes' is applied in build_regulons).")
    seq_genes = list(promoter_data["seq_genes"])
    sequences = list(promoter_data["sequences"])
    if len(sequences) == 0:
        raise ValueError("promoter_data has no sequences.")
    if background is None:
        background = estimate_background(sequences)
    background = np.asarray(background)

    # Flatten PWMs across TFs, remember each motif's TF.
    tfs = sorted(tf_motif_map)
    tf_index = {tf: i for i, tf in enumerate(tfs)}
    all_pwms: List = []
    motif_tf_idx: List[int] = []
    for tf in tfs:
        for pwm in tf_motif_map[tf]:
            all_pwms.append(pwm)
            motif_tf_idx.append(tf_index[tf])
    if not all_pwms:
        raise ValueError("tf_motif_map has no PWMs.")
    motif_tf_idx = np.asarray(motif_tf_idx)

    scan = scan_fn if scan_fn is not None else scan_motifs
    if verbose:
        print(f"[cistrome] scanning {len(all_pwms)} PWMs ({len(tfs)} TFs) × "
              f"{len(sequences)} promoter sequences (method={cistrome_method})")
    # Scan continuously (relative_frac=0 → best_score is the max window score for
    # EVERY (motif, seq), giving each motif's full score distribution).
    res = scan(all_pwms, sequences, background=background, relative_frac=0.0,
               both_strands=both_strands, pseudocount=pseudocount)
    best = np.asarray(res["best_score"], dtype=np.float64)        # (n_motif, n_seq)
    best = np.where(np.isfinite(best), best, -np.inf)

    genes = sorted(set(seq_genes))
    gene_index = {g: i for i, g in enumerate(genes)}
    seq_gene_idx = np.asarray([gene_index[g] for g in seq_genes])
    n_motif, n_gene, n_tf = best.shape[0], len(genes), len(tfs)

    # seq→gene reduction, per motif (max over a gene's alternative promoters).
    mg = np.full((n_motif, n_gene), -np.inf, dtype=np.float64)
    for m in range(n_motif):
        np.maximum.at(mg[m], seq_gene_idx, best[m])
    valid_gene = np.isfinite(mg).any(axis=0)                       # gene has a promoter

    # per-motif absolute p-value cutoff (used by 'hits' and the combine option)
    abs_thr = np.array([pvalue_to_threshold(p.pssm(background, pseudocount), background, pvalue)
                        for p in all_pwms])

    hit = np.zeros((n_motif, n_gene), dtype=bool)
    n_flat = 0
    for m in range(n_motif):
        sc = mg[m]
        fin = np.isfinite(sc)
        if not fin.any():
            continue
        if cistrome_method == "hits":
            hit[m] = sc >= abs_thr[m]
        else:  # motif_bg
            vals = sc[fin]
            thr_q = np.quantile(vals, 1.0 - motif_bg_quantile)
            if flat_motif_filter:
                med = np.median(vals)
                mad = np.median(np.abs(vals - med)) + 1e-9
                if (thr_q - med) / mad < flat_motif_min_sep:
                    n_flat += 1
                    continue                                       # flat motif → no edges
            h = fin & (sc >= thr_q)
            if motif_bg_combine_pvalue:
                h &= (sc >= abs_thr[m])
            hit[m] = h

    # motif→TF reduction (union; best score where a hit).
    M = np.zeros((n_tf, n_gene), dtype=bool)
    scores = np.zeros((n_tf, n_gene), dtype=np.float32)
    for m in range(n_motif):
        ti = motif_tf_idx[m]
        h = hit[m]
        if not h.any():
            continue
        M[ti] |= h
        sc = np.where(h, mg[m], -np.inf)
        scores[ti] = np.maximum(scores[ti], np.where(np.isfinite(sc), sc, 0.0).astype(np.float32))

    Msp = sp.csr_matrix(M)
    if verbose:
        dens = 100.0 * Msp.nnz / max(1, n_tf * n_gene)
        flat_msg = f", {n_flat} flat motifs dropped" if cistrome_method == "motif_bg" else ""
        print(f"[cistrome] M = {n_tf} TFs × {n_gene} genes, {int(Msp.nnz)} edges "
              f"({dens:.2f}% density){flat_msg}")
    return {"tfs": tfs, "genes": genes, "M": Msp,
            "scores": sp.csr_matrix(scores),
            "motif_gene_score": mg, "motif_tf_idx": motif_tf_idx,
            "motif_ids": [p.motif_id for p in all_pwms],
            "valid_gene": valid_gene, "background": background}
