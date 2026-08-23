"""cytorete.pl.regulonSpecificityScatter — synthetic AnnData with a planted discordance.

One TF (TF0) whose mRNA is broad but whose regulon activity is cell-type-specific (B);
checks the IQR-COSG specificity pairing recovers the discordance and the plot runs + saves.
"""
import cytorete  # noqa: F401
import numpy as np
import pytest

import piaso
from cytorete.plotting._plotRegulonScatter import _cosg_spec_matrix


def _synthetic():
    from anndata import AnnData
    rng = np.random.RandomState(0)
    n, g = 300, 12
    groups = np.array(["A", "B", "C"])[rng.randint(0, 3, n)]
    TFS = [f"TF{i}" for i in range(g)]
    E = rng.gamma(1.0, 1.0, size=(n, g)).astype(np.float32)
    E[:, 0] += 3.0                       # TF0 broadly high (expression not specific)
    E[groups == "A", 1] += 4.0           # TF1 expression high only in A
    ad = AnnData(E)
    ad.obs_names = [f"c{i}" for i in range(n)]
    ad.var_names = TFS
    ad.obs["ct"] = groups
    A = rng.gamma(1.0, 1.0, size=(n, g)).astype(np.float32)
    A[groups == "B", 0] += 5.0           # TF0 regulon specific to B (discordant!)
    A[groups == "A", 1] += 5.0           # TF1 regulon specific to A (concordant)
    ad.obsm["X_grn"] = A
    ad.uns["grn"] = {"names": list(TFS)}
    return ad


def test_cosg_spec_matrix_iqr_shape_and_specificity():
    ad = _synthetic()
    S = _cosg_spec_matrix(ad.X, ad.obs["ct"].values, ad.var_names, iqr_normalize=True)
    assert set(S.columns) == {"A", "B", "C"}
    assert S.shape[0] == ad.n_vars
    assert (S.values >= 0).all()                 # IQR-log1p of positive COSG is ≥ 0
    assert S.loc["TF1"].idxmax() == "A"          # TF1 expression is A-specific


def test_regulon_specificity_scatter_runs(tmp_path):
    ad = _synthetic()
    fig = cytorete.pl.regulonSpecificityScatter(
        ad, groupby="ct", key="X_grn", uns_key="grn", ncols=3, n_label=3,
        save=str(tmp_path / "scatter.pdf"), show=False, return_fig=True)
    assert sum(a.get_visible() for a in fig.axes) == 3      # one panel per cell type
    assert (tmp_path / "scatter.pdf").exists()
    # regulonSpecificity wrote its IQR specificity_matrix into the grn struct
    assert "specificity_matrix" in ad.uns["grn"]


def test_discordance_signal_real():
    ad = _synthetic()
    S_exp = _cosg_spec_matrix(ad.X, ad.obs["ct"].values, ad.uns["grn"]["names"], iqr_normalize=True)
    S_reg = _cosg_spec_matrix(ad.obsm["X_grn"], ad.obs["ct"].values, ad.uns["grn"]["names"], iqr_normalize=True)
    # TF0: regulon B-specific, expression broad → regulon B-spec ≫ expression B-spec
    assert S_reg.loc["TF0"].idxmax() == "B"                          # regulation B-specific
    assert S_reg.loc["TF0", "B"] > S_exp.loc["TF0", "B"] + 0.2       # ≫ expression B-specificity
    assert S_exp.loc["TF0"].idxmax() != "B"                          # expression NOT B-specific


def test_no_matching_gene_errors():
    ad = _synthetic()
    ad.var_names = [f"X{i}" for i in range(ad.n_vars)]   # regulon names no longer match genes
    with pytest.raises(ValueError, match="No regulon name matches"):
        cytorete.pl.regulonSpecificityScatter(ad, groupby="ct", key="X_grn", uns_key="grn", show=False)
