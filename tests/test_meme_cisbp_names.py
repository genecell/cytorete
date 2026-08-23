"""load_meme must clean CIS-BP homology-inferred TF names.

CIS-BP names homology-inferred motifs as ``(TF)_(species)_(DBD_x)`` — e.g.
``(Ascl1)_(Homo_sapiens)_(DBD_1.00)``. Taking the token verbatim as ``tf_name`` hid ~40 % of the
mouse CIS-BP motifs (320/790, incl. Ascl1/Neurog2/Neurod2/Bcl11b/Olig2/Mef2c) from the GRN because
the garbled name never matched a gene symbol. This locks the fix.
"""
import cytorete  # noqa: F401
import io as _io

from cytorete.data import load_meme
from piaso.data._motifs import _clean_meme_tf


def test_clean_meme_tf_unit():
    assert _clean_meme_tf("(Ascl1)_(Homo_sapiens)_(DBD_1.00)") == "Ascl1"
    assert _clean_meme_tf("(TFAP2D)_(Mus_musculus)_(DBD_0.80)") == "TFAP2D"
    assert _clean_meme_tf("Tfap2a") == "Tfap2a"        # plain CIS-BP — unchanged
    assert _clean_meme_tf("PAX6") == "PAX6"            # plain JASPAR — unchanged
    assert _clean_meme_tf("Nkx2-2") == "Nkx2-2"        # hyphen preserved


_MINI_MEME = """MEME version 5.5

ALPHABET= ACGT

MOTIF M1 Tfap2a
letter-probability matrix: alength= 4 w= 2 nsites= 1 E= 0
 0.25 0.25 0.25 0.25
 0.10 0.70 0.10 0.10

MOTIF M2 (Ascl1)_(Homo_sapiens)_(DBD_1.00)
letter-probability matrix: alength= 4 w= 2 nsites= 1 E= 0
 0.40 0.10 0.40 0.10
 0.25 0.25 0.25 0.25
"""


def test_load_meme_cleans_parenthesized(tmp_path):
    p = tmp_path / "mini.meme"
    p.write_text(_MINI_MEME)
    pwms = load_meme(str(p))
    names = {pw.tf_name for pw in pwms}
    assert names == {"Tfap2a", "Ascl1"}, names   # NOT the garbled parenthesized string
