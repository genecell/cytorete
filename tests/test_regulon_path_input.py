"""`inferRegulon` accepts a `.cytome` path, like the rest of the stack.

PIASO's plotting and preprocessing take a path string, so the regulon entry
points were the one place where passing one failed -- and it failed with
`AttributeError: 'str' object has no attribute 'var_names'`, which names
nothing the caller did wrong.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from cytorete.tools import _objio as io


def test_open_if_path_passes_objects_through():
    obj = object()
    out, opened = io.open_if_path(obj)
    assert out is obj and opened is False


def test_open_if_path_opens_a_cytome(tmp_path):
    cytome = pytest.importorskip("cytome")
    anndata = pytest.importorskip("anndata")
    a = anndata.AnnData(X=sp.csr_matrix(np.eye(4, dtype=np.float32)))
    p = str(tmp_path / "t.cytome")
    cytome.from_anndata(a, output=p).close()

    ds, opened = io.open_if_path(p)
    try:
        assert opened is True
        assert io.n_obs(ds) == 4
    finally:
        ds.close()


def test_embedding_lookup_resolves_the_stored_name(tmp_path):
    """`basis='X_umap'` must find the embedding a cytome actually stores.

    Conversion writes obsm arrays under a modality prefix -- `RNA_umap` since
    cytome 0.2.6, `RNA_obsm_X_umap` before it -- so an exact lookup of
    'X_umap' raised `Embedding not found` on every cytome ever written, which
    made cytorete.pl.regulonEmbedding unusable on a file.
    """
    cytome = pytest.importorskip("cytome")
    anndata = pytest.importorskip("anndata")
    a = anndata.AnnData(X=sp.csr_matrix(np.eye(6, dtype=np.float32)))
    a.obsm["X_umap"] = np.random.RandomState(0).rand(6, 2)
    p = str(tmp_path / "e.cytome")
    cytome.from_anndata(a, output=p).close()

    ds = cytome.open(p)
    try:
        assert io.has_embedding(ds, "X_umap")
        assert io.get_embedding(ds, "X_umap").shape == (6, 2)
        with pytest.raises(KeyError, match="Available"):
            io.get_embedding(ds, "tsne")
    finally:
        ds.close()
