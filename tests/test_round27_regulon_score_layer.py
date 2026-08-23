"""Round 27 — score()/regulonActivity layer precedence + inferRegulon ensure-layer.

Regression for `KeyError: 'counts'` in inferRegulon → regulonActivity → score(): a deprecated
`cytome_layer='counts'` forwarded by regulonActivity clobbered the canonical `layer='infog'`.
Fix: (A) regulonActivity forwards only `layer=score_layer`; (C′) score() lets the canonical `layer`
beat the deprecated `cytome_layer` alias; (D) inferRegulon ensures the score layer exists on both
backends (computes INFOG from raw counts if absent).
"""
import cytorete  # noqa: F401
import warnings
import numpy as np
import anndata as ad
import scipy.sparse as sp
import pytest

from piaso.tools._normalization import score
from cytorete.tools._activity import regulonActivity
from cytorete.tools._grn import _ensure_score_layer


def _adata_infog_no_counts(n=200, g=80, seed=0):
    """AnnData with X = normalized, layers['infog'] present, NO 'counts' layer (the user's shape)."""
    rng = np.random.RandomState(seed)
    counts = rng.poisson(1.0, (n, g)).astype(np.float32)
    A = ad.AnnData(X=sp.csr_matrix(np.log1p(counts)))
    A.var_names = [f"G{i}" for i in range(g)]
    A.obs["ct"] = ["a"] * (n // 2) + ["b"] * (n - n // 2)
    A.layers["infog"] = sp.csr_matrix(counts / (counts.sum(1, keepdims=True) + 1) * 1e4)
    assert "counts" not in A.layers
    return A


def test_score_reads_infog_not_counts():
    A = _adata_infog_no_counts()
    sc, _n, _p = score(A, {"s": ["G1", "G2", "G3", "G4", "G5"]},
                       layer="infog", compute_pvalues=False, n_ctrl_set=5)
    assert np.asarray(sc).shape[0] == A.n_obs           # no KeyError: 'counts'


def test_canonical_layer_beats_deprecated_cytome_layer():
    """PIASO 1.2.0 removed score()'s `cytome_layer=` alias outright (it never
    shipped publicly, and an alias that could clobber the canonical `layer`
    was the KeyError-'counts' bug). The current contract is a TypeError; the
    alias lives on only at the regulonActivity level, tested below."""
    A = _adata_infog_no_counts()
    with pytest.raises(TypeError, match="cytome_layer"):
        score(A, {"s": ["G1", "G2", "G3"]}, layer="infog", cytome_layer="counts",
              compute_pvalues=False, n_ctrl_set=5)


def test_regulonActivity_no_cytome_layer_clobber():
    A = _adata_infog_no_counts()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        act, _n, _p = regulonActivity(A, {"TF1": ["G1", "G2", "G3", "G4", "G5", "G6"]},
                                      score_layer="infog", n_ctrl_set=5,
                                      compute_pvalues=False, copy=True, verbose=0)
    assert np.asarray(act).shape[0] == A.n_obs
    assert not any("cytome_layer" in str(x.message) for x in w)    # no deprecated alias forwarded


def test_ensure_score_layer_computes_infog_from_counts():
    rng = np.random.RandomState(1)
    counts = rng.poisson(1.0, (150, 70)).astype(np.float32)
    B = ad.AnnData(X=sp.csr_matrix(counts)); B.var_names = [f"G{i}" for i in range(70)]  # integer X, no infog
    out = _ensure_score_layer(B, "RNA", "infog", verbose=0)
    assert out == "infog" and "infog" in B.layers
    assert _ensure_score_layer(B, "RNA", "infog", verbose=0) == "infog"   # idempotent


def test_ensure_score_layer_clear_error_when_uncomputable():
    rng = np.random.RandomState(2)
    norm = np.log1p(rng.poisson(1.0, (100, 50))).astype(np.float32)      # X normalized, no counts, no infog
    C = ad.AnnData(X=sp.csr_matrix(norm)); C.var_names = [f"G{i}" for i in range(50)]
    with pytest.raises(KeyError, match="cannot be computed"):
        _ensure_score_layer(C, "RNA", "infog", verbose=0)


def test_ensure_score_layer_passthrough():
    A = _adata_infog_no_counts()
    assert _ensure_score_layer(A, "RNA", None, verbose=0) is None
    assert _ensure_score_layer(A, "RNA", "counts", verbose=0) == "counts"
    assert _ensure_score_layer(A, "RNA", "infog", verbose=0) == "infog"   # already present


def test_anndata_layer_none_uses_X():
    """AnnData: layer=None reads adata.X (the base/raw matrix), no crash."""
    A = _adata_infog_no_counts()
    sc, _n, _p = score(A, {"s": ["G1", "G2", "G3", "G4"]}, layer=None,
                       compute_pvalues=False, n_ctrl_set=5)
    assert np.asarray(sc).shape[0] == A.n_obs


def test_cytome_layer_none_maps_to_counts(tmp_path):
    """E: a cytome has no `.X`; layer=None must map to {modality}_counts, not crash on RNA_None."""
    cytome = pytest.importorskip("cytome")
    rng = np.random.RandomState(3)
    counts = rng.poisson(1.0, (120, 40)).astype(np.float32)
    A = ad.AnnData(X=sp.csr_matrix(counts))
    A.var_names = [f"g{i}" for i in range(40)]
    A.obs["ct"] = ["A"] * 60 + ["B"] * 60
    ds = cytome.from_anndata(A, modality="RNA", output=str(tmp_path / "rna.cytome"))
    ds.flush()
    sc, _n, _p = score(ds, {"s": [f"g{i}" for i in range(6)]}, layer=None, modality="RNA",
                       compute_pvalues=False, n_ctrl_set=5)          # was KeyError: 'RNA_None'
    assert np.asarray(sc).shape[0] == ds.n_cells
    ds.close()
