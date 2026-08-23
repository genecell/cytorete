"""Regulon assembly: motif-supported (cistrome) AND co-specific (trans) edges →
positive TF regulons. Emits a **global** regulon per TF (for compact activity
scoring) plus **per-cell-type** regulons (COSG-style, the richer PIASO output).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _motif_score_lookup(cistrome) -> Dict[tuple, float]:
    M = cistrome["scores"].tocoo()
    tfs, genes = cistrome["tfs"], cistrome["genes"]
    return {(tfs[i], genes[j]): float(v)
            for i, j, v in zip(M.row, M.col, M.data)}


def _recovery_auc(ranks_T: np.ndarray, R: int) -> float:
    """AUCell-style recovery AUC of a target set within a motif's gene ranking.

    ``ranks_T`` = 0-based ranks of the target genes (0 = top motif score); ``R`` =
    rank cutoff (top fraction). Area under the cumulative-recovery curve over the
    top ``R``, normalized to [0, 1]; higher = targets concentrated at the top.
    """
    if len(ranks_T) == 0 or R <= 0:
        return 0.0
    within = ranks_T[ranks_T < R]
    return float(np.sum(R - within) / (R * len(ranks_T)))


def _nes_prune(cistrome, regulons, weights, *, nes_threshold, rank_frac,
               min_targets, verbose):
    """RcisTarget-analog: keep a TF only if its motif's genome-wide gene-ranking
    recovers the TF's candidate targets with a normalized enrichment (NES) ≥
    ``nes_threshold`` (null = the recovery AUC of the same target set across ALL
    motifs); restrict surviving regulons to the leading edge (rank < R)."""
    mg = np.asarray(cistrome["motif_gene_score"], dtype=np.float64)   # [n_motif, n_gene]
    mg = np.where(np.isfinite(mg), mg, -1e18)
    n_motif, n_gene = mg.shape
    genes = list(cistrome["genes"])
    gene_index = {g: i for i, g in enumerate(genes)}
    motif_tf_idx = np.asarray(cistrome["motif_tf_idx"])
    tfs = list(cistrome["tfs"])
    tf_to_motifs = {}
    for m, ti in enumerate(motif_tf_idx):
        tf_to_motifs.setdefault(tfs[ti], []).append(m)

    # per-motif rank of each gene (0 = highest motif score)
    order = np.argsort(-mg, axis=1)
    rank = np.empty_like(order)
    rows = np.arange(n_motif)[:, None]
    rank[rows, order] = np.arange(n_gene)[None, :]
    R = max(1, int(rank_frac * n_gene))

    kept, kept_w, nes_scores = {}, {}, {}
    for tf, tgts in regulons.items():
        motifs = tf_to_motifs.get(tf, [])
        t_idx = np.array([gene_index[g] for g in tgts if g in gene_index])
        if t_idx.size == 0 or not motifs:
            continue
        auc_all = np.array([_recovery_auc(rank[m][t_idx], R) for m in range(n_motif)])
        mu, sd = auc_all.mean(), auc_all.std() + 1e-12
        nes_all = (auc_all - mu) / sd
        best_m = motifs[int(np.argmax(nes_all[motifs]))]
        tf_nes = float(nes_all[best_m])
        if tf_nes < nes_threshold:
            continue
        # leading edge in the best motif
        keep_mask = np.array([rank[best_m][gene_index[g]] < R if g in gene_index else False
                              for g in tgts])
        le = [g for g, k in zip(tgts, keep_mask) if k]
        if len(le) < min_targets:
            continue
        kept[tf] = le
        kept_w[tf] = np.asarray(weights[tf])[keep_mask]
        nes_scores[tf] = tf_nes
    if verbose:
        print(f"[regulons] NES pruning: {len(kept)}/{len(regulons)} TFs pass "
              f"NES≥{nes_threshold} (leading-edge targets)")
    return kept, kept_w, nes_scores


def build_regulons(
    cistrome: dict,
    cospec: dict,
    *,
    min_targets: int = 10,
    weight_mode: str = "binary",
    cospec_min: float = 0.0,
    top_targets_per_tf: Optional[int] = None,
    per_celltype: bool = True,
    per_celltype_frac: float = 0.5,
    per_celltype_min_targets: int = 5,
    nes: bool = False,
    nes_threshold: float = 3.0,
    nes_rank_frac: float = 0.03,
    verbose: bool = True,
) -> dict:
    """Assemble regulons from the cistrome ``M`` and trans co-specificity edges.

    ``cospec["edges"]`` is already motif-supported (we scored only M pairs) and
    positive-sign. Here we additionally threshold on ``cospec_max`` (and optional
    per-TF ``top_targets_per_tf``), apply ``min_targets``, and split out
    per-cell-type regulons.

    weight_mode : ``"binary"`` (default; gene_weights = 1, the AND-gate only),
    ``"cospec"`` (gene_weights = cospec_max), or ``"motif_cospec"``
    (cospec_max × motif score). Weights feed :func:`regulon_activity`.

    Returns ``{"regulons": {tf: [targets]}, "weights": {tf: ndarray},
    "edges": DataFrame, "per_celltype": {ct: {tf: [targets]}}, "tfs": [...]}``.
    """
    edges = cospec["edges"].copy()
    if len(edges) == 0:
        return {"regulons": {}, "weights": {}, "edges": edges,
                "per_celltype": {}, "tfs": []}

    edges = edges[edges["cospec_max"] > cospec_min]
    if weight_mode in ("motif_cospec",):
        ms = _motif_score_lookup(cistrome)
        edges = edges.assign(motif_score=[
            ms.get((s, t), 0.0) for s, t in zip(edges["source"], edges["target"])])

    regulons: Dict[str, List[str]] = {}
    weights: Dict[str, np.ndarray] = {}
    for tf, grp in edges.groupby("source"):
        grp = grp.sort_values("cospec_max", ascending=False)
        if top_targets_per_tf is not None:
            grp = grp.head(top_targets_per_tf)
        if len(grp) < min_targets:
            continue
        tgts = list(grp["target"])
        if weight_mode == "binary":
            w = np.ones(len(tgts), dtype=np.float32)
        elif weight_mode == "cospec":
            w = grp["cospec_max"].to_numpy(np.float32)
        elif weight_mode == "motif_cospec":
            w = (grp["cospec_max"].to_numpy(np.float32)
                 * np.maximum(grp["motif_score"].to_numpy(np.float32), 1e-6))
        else:
            raise ValueError(f"weight_mode {weight_mode!r} invalid")
        regulons[tf] = tgts
        weights[tf] = w

    # RcisTarget-style NES pruning (optional): keep a TF only if its motif's
    # genome-wide gene-ranking recovers the candidate target set with a normalized
    # enrichment ≥ nes_threshold, and restrict targets to the leading edge.
    nes_scores: Dict[str, float] = {}
    if nes and cistrome.get("motif_gene_score") is not None:
        regulons, weights, nes_scores = _nes_prune(
            cistrome, regulons, weights, nes_threshold=nes_threshold,
            rank_frac=nes_rank_frac, min_targets=min_targets, verbose=verbose)

    # per-cell-type regulons: an edge is "on" in cell type t if its per-cell-type
    # metric >= per_celltype_frac * its own max across cell types.
    per_ct: Dict[str, Dict[str, List[str]]] = {}
    if per_celltype and cospec.get("per_celltype") is not None and len(cospec["per_celltype"]):
        celltypes = cospec["celltypes"]
        pcm = np.asarray(cospec["per_celltype"])          # [n_all_edges, n_ct]
        # align pcm rows to the *filtered* edges via the original index
        keep_idx = edges.index.to_numpy()
        pcm = pcm[keep_idx]
        emax = np.maximum(pcm.max(axis=1, keepdims=True), 1e-12)
        on = pcm >= (per_celltype_frac * emax)
        src = edges["source"].to_numpy()
        tgt = edges["target"].to_numpy()
        for ci, ct in enumerate(celltypes):
            d: Dict[str, List[str]] = {}
            mask = on[:, ci]
            for s, t in zip(src[mask], tgt[mask]):
                d.setdefault(s, []).append(t)
            d = {tf: ts for tf, ts in d.items() if len(ts) >= per_celltype_min_targets}
            if d:
                per_ct[ct] = d

    if verbose:
        sizes = [len(v) for v in regulons.values()]
        print(f"[regulons] {len(regulons)} global regulons "
              f"(median {int(np.median(sizes)) if sizes else 0} targets); "
              f"per-cell-type regulons for {len(per_ct)} cell types")
    return {"regulons": regulons, "weights": weights, "edges": edges,
            "per_celltype": per_ct, "tfs": list(regulons),
            "nes_scores": nes_scores}
