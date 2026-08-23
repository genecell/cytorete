# cytorete

**Cell-type-resolved inference of gene regulatory networks and their dynamics.**

## The name

**cytorete** = *cyto-* + *rete*, "the cell's network" — Ancient Greek
**κύτος** (*kýtos*), the combining form for **cell**, and Latin **rēte**,
"**net**", the word anatomy already uses in *rete mirabile* and *rete testis*.

Pronounced **sy-toh-REE-tee** (/ˌsaɪtoʊˈriːtiː/) — *rete* keeps its
two-syllable English anatomical sound, not a one-syllable "reet".

## What this package does

cytorete infers cell-type-resolved gene regulatory networks (GRNs) from
single-cell data, combining COSG-derived co-specificity, marker-gene
dimensionality reduction (GDR), and motif-cistrome evidence into TF→gene
regulons with per-cell-type activity.

It is built on the **PIASO** single-cell stack (a one-directional dependency,
`cytorete → piaso-tools`): it reuses PIASO's public API for scoring, GDR,
co-specificity, motif scanning (Rust-accelerated), and the cytome streaming
backend, so it scales from small AnnData objects to atlas-scale on-disk
cytomes.

**This release ships the RNA regulon workflow**, end to end:

> promoter cistrome → `inferRegulon` → `regulonActivity` /
> `regulonSpecificity` → plots

The multiome (RNA+ATAC) GRN chain, the ATAC TF-activity chain and the peak
cistrome are not part of this distribution. Their names exist in the package
and raise an `ImportError` at call time saying so, rather than failing at
import — so `import cytorete` behaves the same either way.

## Installation

```bash
pip install cytorete          # pulls piaso-tools, cosg, cytome
pip install "cytorete[motif]" # + py2bit for .2bit genome sequence extraction
```

## Documentation

Tutorials live with the rest of the stack on **[piaso.org](https://piaso.org)**:

- [RNA regulon inference](https://piaso.org/tutorials/cytorete-regulons/) — the end-to-end workflow
- [Motif analysis](https://piaso.org/tutorials/motif-analysis/) — scanning and motif databases

cytorete shares PIASO's scoring, GDR and co-specificity, so its tutorials sit
beside theirs rather than on a site of their own.

## Quickstart

```python
import cytorete as cr

# 1. Promoter cistrome: which TF motifs occur in each gene's promoter
cistrome = cr.pp.build_cistrome(promoter_seqs, tf_motif_map)

# 2. Regulons: motif evidence x trans co-specificity across cell types
regulons = cr.tl.inferRegulon(adata, groupby="cell_type", copy=True)

# 3. Per-cell-type activity and specificity
cr.tl.regulonActivity(adata, regulons)
spec = cr.tl.regulonSpecificity(adata, groupby="cell_type", copy=True)

# 4. Plots
cr.pl.plotRegulon(adata, regulon="SOX2")
```

`inferRegulon` and `regulonSpecificity` follow the scanpy convention: they
write in place and return `None` unless `copy=True`. `regulonSpecificity`
returns **long-form** results — pivot before passing them to a heatmap.

Both `snake_case` (`infer_regulon`) and `camelCase` (`inferRegulon`) names are
provided; `camelCase` matches `piaso.tl` for continuity.

Calling a name from a withheld chain tells you so at the call site:

```python
>>> cr.inferGRN(ds)
ImportError: cytorete.inferGRN is not part of this distribution: it requires
the multiome (RNA+ATAC) GRN chain, which is not yet released. The RNA regulon
workflow (build_promoter_cistrome -> inferRegulon -> regulonActivity) is
fully available.
```

## Relationship to PIASO

The dependency runs one way — `cytorete → piaso-tools` — and never back.
cytorete is deliberately *not* a dependency of PIASO, which would be a
packaging cycle.

| Concern | Lives in |
|---|---|
| Regulons, promoter cistrome, regulon activity & specificity, regulon plots | **cytorete** (this package) |
| Scoring, INFOG normalization, GDR, co-specificity, motif scanning (`pp.scan_motifs`), motif/genome loaders | PIASO (`piaso-tools`) |
| Streaming on-disk backend | Cytome |
| Marker specificity scoring | COSG |

The GRN entry points that used to live in `piaso.tl` remain there as thin
forwarders: each resolves cytorete at call time and, if it is not installed,
raises an `ImportError` pointing at `pip install cytorete`. They exist for
existing notebooks — new code should `import cytorete` directly.

## License

BSD 3-Clause. Copyright (c) 2025, Min Dai.
