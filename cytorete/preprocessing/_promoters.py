"""Promoter-interval + sequence extraction for PIASO-GRN.

Source of truth is the PIASO-data ``*_transcript_tss.bed`` (9 columns:
``chrom, start, end, gene_symbol, ., strand, biotype, gene_id, transcript_id``).
The 4th column is the gene **symbol**, so transcript TSS group to a gene directly
— no GTF parse needed.

Promoter model (locked in design Q3): per TSS a strand-aware window
``[-upstream, +downstream]``; per gene, **merge only overlapping windows** so
clustered TSS collapse to one interval while genuine alternative promoters stay
separate. Sequences are reverse-complemented on the ``-`` strand so the scanner
always sees 5'→3' relative to the gene.
"""
from __future__ import annotations

import gzip
import os
from typing import Dict, List, Optional, Sequence, Tuple

from ..data import extract_sequences, resolve_2bit_path


def _open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_tss(
    tss_bed: str,
    genes: Optional[Sequence[str]] = None,
    biotypes: Optional[Sequence[str]] = None,
) -> Dict[str, List[Tuple[str, int, str]]]:
    """Parse a ``*_transcript_tss.bed`` into ``{gene_symbol: [(chrom, tss, strand)]}``.

    ``genes`` (e.g. the RNA ``var_names``) restricts parsing to those symbols
    (case-insensitive) for speed/RAM. ``biotypes`` (e.g. ``["protein_coding"]``)
    filters on the 7th column when present.
    """
    gene_filter = {g.upper() for g in genes} if genes is not None else None
    biotype_filter = {b.lower() for b in biotypes} if biotypes is not None else None
    out: Dict[str, List[Tuple[str, int, str]]] = {}
    exact_case: Dict[str, str] = {}  # UPPER -> first-seen exact symbol
    with _open_text(tss_bed) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            chrom, start, sym, strand = f[0], f[1], f[3], f[5]
            up = sym.upper()
            if gene_filter is not None and up not in gene_filter:
                continue
            if biotype_filter is not None and len(f) >= 7 and f[6].lower() not in biotype_filter:
                continue
            try:
                tss = int(start)
            except ValueError:
                continue
            key = exact_case.setdefault(up, sym)
            out.setdefault(key, []).append((chrom, tss, strand))
    return out


def promoter_intervals(
    tss_dict: Dict[str, List[Tuple[str, int, str]]],
    upstream: int = 1000,
    downstream: int = 500,
) -> Dict[str, List[Tuple[str, int, int, str]]]:
    """Turn TSS into merged promoter intervals per gene.

    Window per TSS is strand-aware: ``+`` → ``[tss-upstream, tss+downstream]``,
    ``-`` → ``[tss-downstream, tss+upstream]``. Windows are merged only when they
    overlap (within the same chrom+strand); non-overlapping windows (alternative
    promoters) are kept as separate intervals. Returns
    ``{gene: [(chrom, start, end, strand), ...]}``.
    """
    out: Dict[str, List[Tuple[str, int, int, str]]] = {}
    for gene, tss_list in tss_dict.items():
        # group by (chrom, strand); build windows
        groups: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for chrom, tss, strand in tss_list:
            if strand == "-":
                s, e = tss - downstream, tss + upstream
            else:
                s, e = tss - upstream, tss + downstream
            groups.setdefault((chrom, strand), []).append((max(0, s), e))
        merged: List[Tuple[str, int, int, str]] = []
        for (chrom, strand), wins in groups.items():
            wins.sort()
            cs, ce = wins[0]
            for s, e in wins[1:]:
                if s <= ce:  # overlap (book-ended counts as overlap)
                    ce = max(ce, e)
                else:
                    merged.append((chrom, cs, ce, strand))
                    cs, ce = s, e
            merged.append((chrom, cs, ce, strand))
        out[gene] = merged
    return out


def extract_promoter_sequences(
    genome: Optional[str] = None,
    *,
    tss_bed: Optional[str] = None,
    twobit_path: Optional[str] = None,
    genes: Optional[Sequence[str]] = None,
    upstream: int = 1000,
    downstream: int = 500,
    biotypes: Optional[Sequence[str]] = None,
    data_dir: Optional[str] = None,
    regulatory_regions: str = "promoter",
    screen_bed: Optional[str] = None,
    cre_window: int = 100_000,
    cre_classes: Optional[Sequence[str]] = ("PLS", "pELS", "dELS"),
    min_len: int = 10,
) -> Dict[str, object]:
    """Resolve genome files → promoter intervals → sequences.

    Returns a dict with parallel lists (one entry per promoter interval):
    ``{"seq_genes": [gene,...], "sequences": [str,...],
       "intervals": [(chrom,start,end,strand),...],
       "promoters_per_gene": {gene: n}}``.
    A gene with multiple non-overlapping promoters appears multiple times in
    ``seq_genes`` — the cistrome step aggregates the best motif hit per gene.

    ``tss_bed`` / ``twobit_path`` override the auto-resolved genome files. The
    ``.2bit`` must already exist locally (use ``piaso.grn.fetch_2bit`` /
    ``fetch_genome(..., download_fasta=True)`` first — it is never auto-fetched).
    """
    if tss_bed is None:
        if genome is None:
            raise ValueError("provide either `genome` or `tss_bed`.")
        from piaso.data import resolve_genome_files
        tss_bed = resolve_genome_files(genome)["tss_bed"]
    if twobit_path is None:
        if genome is None:
            raise ValueError("provide either `genome` or `twobit_path`.")
        twobit_path = resolve_2bit_path(genome, data_dir=data_dir)
        if twobit_path is None:
            raise FileNotFoundError(
                f"No local .2bit for genome {genome!r}. Download it first with "
                f"piaso.grn.fetch_2bit({genome!r}) or "
                f"fetch_genome({genome!r}, download_fasta=True), or pass "
                f"twobit_path=."
            )

    tss = load_tss(tss_bed, genes=genes, biotypes=biotypes)
    intervals_by_gene = promoter_intervals(tss, upstream=upstream, downstream=downstream)

    # optional: add SCREEN cCREs within ±cre_window of each gene's TSS
    if regulatory_regions == "promoter+cre":
        from piaso.data import resolve_screen_path, load_screen_ccres
        from piaso.data import ccres_near_tss
        sp = screen_bed or resolve_screen_path(genome, data_dir=data_dir)
        if sp is None:
            raise FileNotFoundError(
                f"regulatory_regions='promoter+cre' needs the SCREEN cCRE BED for "
                f"{genome!r}. Fetch it with piaso.data.fetch_screen({genome!r}) or "
                f"pass screen_bed=.")
        ccres = load_screen_ccres(sp, classes=cre_classes)
        for gene, tss_list in tss.items():
            seen = set(intervals_by_gene.get(gene, []))
            for chrom, pos, strand in tss_list:
                for s, e in ccres_near_tss(ccres, chrom, pos, cre_window):
                    iv = (chrom, s, e, "+")
                    if iv not in seen:
                        seen.add(iv)
                        intervals_by_gene.setdefault(gene, []).append(iv)
    elif regulatory_regions != "promoter":
        raise ValueError("regulatory_regions must be 'promoter' or 'promoter+cre'.")

    seq_genes: List[str] = []
    flat_intervals: List[Tuple[str, int, int, str]] = []
    for gene, ivs in intervals_by_gene.items():
        for iv in ivs:
            seq_genes.append(gene)
            flat_intervals.append(iv)

    sequences = extract_sequences(twobit_path, flat_intervals)

    # drop empties / too-short
    keep_genes, keep_seqs, keep_ivs = [], [], []
    ppg: Dict[str, int] = {}
    for g, s, iv in zip(seq_genes, sequences, flat_intervals):
        if len(s) >= min_len:
            keep_genes.append(g)
            keep_seqs.append(s)
            keep_ivs.append(iv)
            ppg[g] = ppg.get(g, 0) + 1
    return {
        "seq_genes": keep_genes,
        "sequences": keep_seqs,
        "intervals": keep_ivs,
        "promoters_per_gene": ppg,
    }
