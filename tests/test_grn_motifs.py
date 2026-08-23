"""Tests for cytorete.data — motif-DB and TF-list loaders.

Covers:
- load_jaspar_meme: real JASPAR 2024 CORE vertebrates file
- load_cisbp: synthetic fixture
- load_tf_list: symbol-per-line and TSV-with-header
- build_tf_motif_map: tf_list filter + gene_universe remap
- harmonize_symbol helper
- Robustness: 2-token MOTIF line → tf_name == motif_id
"""
from __future__ import annotations

import cytorete  # noqa: F401  (Phase-2 migration target)
import os
import pathlib
import textwrap

import numpy as np
import pytest

from cytorete.data import (
    build_tf_motif_map,
    fetch_animaltfdb_tf_list,
    load_cisbp,
    load_jaspar_meme,
    load_tf_list,
)
from piaso.data._motifs import harmonize_symbol
from piaso.data._pwm import PWM

# ---------------------------------------------------------------------------
# Path to the real JASPAR MEME file
# ---------------------------------------------------------------------------
# Real-file tests need a local JASPAR MEME; point the env var at yours or
# these tests skip. (Fetchable with piaso.data.fetch_jaspar().)
import os

JASPAR_MEME = pathlib.Path(
    os.environ.get("CYTORETE_TEST_JASPAR_MEME", "/nonexistent/jaspar.meme")
)


# ===========================================================================
# 1. load_jaspar_meme — real JASPAR file
# ===========================================================================

@pytest.fixture(scope="module")
def jaspar_pwms() -> list[PWM]:
    """Parse the real JASPAR MEME file once for all module-scope tests."""
    if not JASPAR_MEME.exists():
        pytest.skip(f"set CYTORETE_TEST_JASPAR_MEME to a JASPAR MEME file "
                    f"(got {JASPAR_MEME})")
    return load_jaspar_meme(str(JASPAR_MEME))


class TestLoadJasparMeme:
    def test_count_roughly_700(self, jaspar_pwms):
        """The JASPAR 2024 CORE vertebrates file should have ~700-1000 motifs."""
        n = len(jaspar_pwms)
        assert n >= 700, f"Expected >=700 motifs, got {n}"
        assert n <= 1200, f"Suspiciously many motifs: {n}"

    def test_all_are_pwm_instances(self, jaspar_pwms):
        for pwm in jaspar_pwms:
            assert isinstance(pwm, PWM), f"Expected PWM, got {type(pwm)}"

    def test_non_empty_tf_names(self, jaspar_pwms):
        for pwm in jaspar_pwms:
            assert pwm.tf_name, f"Empty tf_name for {pwm.motif_id!r}"

    def test_probs_shape_and_dtype(self, jaspar_pwms):
        for pwm in jaspar_pwms:
            assert pwm.probs.ndim == 2, f"{pwm.motif_id}: probs not 2-D"
            assert pwm.probs.shape[0] == 4, (
                f"{pwm.motif_id}: probs row count {pwm.probs.shape[0]} != 4"
            )
            assert pwm.probs.shape[1] >= 4, (
                f"{pwm.motif_id}: motif width {pwm.probs.shape[1]} < 4"
            )
            assert pwm.probs.dtype == np.float32, (
                f"{pwm.motif_id}: dtype {pwm.probs.dtype} != float32"
            )

    def test_columns_approximately_sum_to_one(self, jaspar_pwms):
        """Each column (position) of the probability matrix should sum to ~1."""
        for pwm in jaspar_pwms:
            col_sums = pwm.probs.sum(axis=0)  # shape (w,)
            bad = np.abs(col_sums - 1.0) > 0.05
            assert not bad.any(), (
                f"{pwm.motif_id}: columns {np.where(bad)[0].tolist()} "
                f"deviate from 1 (sums={col_sums[bad].tolist()})"
            )

    def test_source_is_jaspar(self, jaspar_pwms):
        for pwm in jaspar_pwms:
            assert pwm.source == "jaspar"

    def test_known_motif_arnt_ma0004(self, jaspar_pwms):
        """MA0004.1 (Arnt) must be present with width 6."""
        arnt = next(
            (p for p in jaspar_pwms if p.motif_id == "MA0004.1"), None
        )
        assert arnt is not None, "MA0004.1 (Arnt) not found"
        assert arnt.tf_name == "Arnt"
        assert arnt.width == 6

    def test_ctcf_present(self, jaspar_pwms):
        """At least one PWM with 'CTCF' in tf_name should be present."""
        ctcf_hits = [p for p in jaspar_pwms if "CTCF" in p.tf_name.upper()]
        assert ctcf_hits, "No CTCF motif found in JASPAR file"

    def test_print_actual_count(self, jaspar_pwms):
        """Print the actual motif count for the report."""
        print(f"\n[JASPAR] Parsed {len(jaspar_pwms)} motifs")


# ===========================================================================
# 2. Robustness: 2-token MOTIF line (tf_name falls back to motif_id)
# ===========================================================================

def test_two_token_motif_line(tmp_path):
    """A MOTIF line with only 2 tokens must set tf_name == motif_id."""
    content = textwrap.dedent("""\
        MEME version 4

        ALPHABET= ACGT

        MOTIF MYID_NO_TF
        letter-probability matrix: alength= 4 w= 4 nsites= 10 E= 0
         0.25  0.25  0.25  0.25
         0.50  0.00  0.25  0.25
         0.00  0.00  1.00  0.00
         0.10  0.40  0.40  0.10

    """)
    meme_file = tmp_path / "two_token.meme"
    meme_file.write_text(content)
    pwms = load_jaspar_meme(str(meme_file))
    assert len(pwms) == 1
    assert pwms[0].motif_id == "MYID_NO_TF"
    assert pwms[0].tf_name == "MYID_NO_TF", (
        f"Expected tf_name == motif_id, got {pwms[0].tf_name!r}"
    )
    assert pwms[0].width == 4


# ===========================================================================
# 3. load_cisbp — synthetic fixture
# ===========================================================================

@pytest.fixture
def cisbp_fixture(tmp_path) -> pathlib.Path:
    """Create a minimal CIS-BP directory with 2 PWM files + TF_Information.txt."""
    pwm_dir = tmp_path / "cisbp_pwms"
    pwm_dir.mkdir()

    # PWM file 1: M00001.txt
    (pwm_dir / "M00001.txt").write_text(
        "Pos\tA\tC\tG\tT\n"
        "1\t0.90\t0.05\t0.03\t0.02\n"
        "2\t0.10\t0.10\t0.70\t0.10\n"
        "3\t0.25\t0.25\t0.25\t0.25\n"
        "4\t0.00\t0.00\t0.00\t1.00\n"
    )

    # PWM file 2: M00002.txt (no Pos column)
    (pwm_dir / "M00002.txt").write_text(
        "A\tC\tG\tT\n"
        "0.50\t0.20\t0.20\t0.10\n"
        "0.00\t0.00\t1.00\t0.00\n"
        "0.30\t0.30\t0.20\t0.20\n"
        "0.25\t0.25\t0.25\t0.25\n"
        "0.10\t0.80\t0.05\t0.05\n"
    )

    # TF_Information.txt
    (pwm_dir.parent / "TF_Information.txt").write_text(
        "Motif_ID\tTF_Name\tFamily\n"
        "M00001\tFOXA2\tForkhead\n"
        "M00002\tGATA3\tGATA\n"
    )

    return pwm_dir


class TestLoadCisbp:
    def test_loads_two_pwms(self, cisbp_fixture):
        tf_info = str(cisbp_fixture.parent / "TF_Information.txt")
        pwms = load_cisbp(str(cisbp_fixture), tf_info_path=tf_info)
        assert len(pwms) == 2, f"Expected 2 PWMs, got {len(pwms)}"

    def test_tf_names_from_info(self, cisbp_fixture):
        tf_info = str(cisbp_fixture.parent / "TF_Information.txt")
        pwms = load_cisbp(str(cisbp_fixture), tf_info_path=tf_info)
        tf_names = {p.tf_name for p in pwms}
        assert "FOXA2" in tf_names
        assert "GATA3" in tf_names

    def test_tf_name_fallback_to_motif_id(self, cisbp_fixture):
        """Without tf_info, tf_name should be the file stem (motif_id)."""
        pwms = load_cisbp(str(cisbp_fixture), tf_info_path=None)
        assert len(pwms) == 2
        tf_names = {p.tf_name for p in pwms}
        assert "M00001" in tf_names
        assert "M00002" in tf_names

    def test_probs_shape(self, cisbp_fixture):
        tf_info = str(cisbp_fixture.parent / "TF_Information.txt")
        pwms = load_cisbp(str(cisbp_fixture), tf_info_path=tf_info)
        by_id = {p.motif_id: p for p in pwms}
        assert by_id["M00001"].probs.shape == (4, 4)
        assert by_id["M00002"].probs.shape == (4, 5)

    def test_source_is_cisbp(self, cisbp_fixture):
        pwms = load_cisbp(str(cisbp_fixture))
        for p in pwms:
            assert p.source == "cisbp"

    def test_missing_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="CIS-BP PWM directory not found"):
            load_cisbp(str(tmp_path / "does_not_exist"))


# ===========================================================================
# 4. load_tf_list — symbol-per-line and TSV-with-header
# ===========================================================================

class TestLoadTfList:
    def test_symbol_per_line(self, tmp_path):
        tf_file = tmp_path / "tfs.txt"
        tf_file.write_text("CTCF\nGATA1\nTP53\n# comment\n\nFOXA2\n")
        result = load_tf_list(path=str(tf_file))
        assert isinstance(result, set)
        assert result == {"CTCF", "GATA1", "TP53", "FOXA2"}

    def test_tsv_with_symbol_header(self, tmp_path):
        tsv_file = tmp_path / "tfs.tsv"
        tsv_file.write_text("Symbol\tFamily\nCTCF\tZinc finger\nGATA1\tGATA\n")
        result = load_tf_list(path=str(tsv_file))
        assert "CTCF" in result
        assert "GATA1" in result
        assert len(result) == 2

    def test_tsv_with_tf_header(self, tmp_path):
        """TSV with a 'TF' column should also be parsed correctly."""
        tsv_file = tmp_path / "tfs2.tsv"
        tsv_file.write_text("TF\tSpecies\nSOX2\thuman\nOCT4\thuman\n")
        result = load_tf_list(path=str(tsv_file))
        assert result == {"SOX2", "OCT4"}

    def test_motifdb_source_returns_none(self):
        result = load_tf_list(source="motifdb")
        assert result is None

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_tf_list(path=str(tmp_path / "nonexistent.txt"))

    def test_animaltfdb_no_cache_raises_runtime_error(self, tmp_path, monkeypatch):
        """When no cache file exists, load_tf_list should raise RuntimeError (not download)."""
        import piaso.data._motifs as _m
        monkeypatch.setattr(_m, "_DEFAULT_CACHE_DIR", tmp_path / "empty_cache")
        with pytest.raises(RuntimeError, match="fetch_animaltfdb_tf_list"):
            load_tf_list(species="human", source="animaltfdb")

    def test_animaltfdb_reads_cache_when_present(self, tmp_path, monkeypatch):
        """When a cached file exists, load_tf_list should read it without downloading."""
        import piaso.data._motifs as _m
        # Set cache dir to tmp_path and place a fake TF list there
        monkeypatch.setattr(_m, "_DEFAULT_CACHE_DIR", tmp_path)
        cached = tmp_path / "Homo_sapiens_TF"
        cached.write_text("CTCF\nTP53\nMYC\n")
        result = load_tf_list(species="human", source="animaltfdb")
        assert result == {"CTCF", "TP53", "MYC"}

    def test_unsupported_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            load_tf_list(source="unknown_db")


# ===========================================================================
# 5. build_tf_motif_map
# ===========================================================================

@pytest.fixture(scope="module")
def small_pwm_set(jaspar_pwms) -> list[PWM]:
    """Return a small slice of JASPAR PWMs for fast map tests."""
    return jaspar_pwms[:50]


class TestBuildTfMotifMap:
    def test_no_filter_groups_all_tfs(self, small_pwm_set):
        result = build_tf_motif_map(small_pwm_set)
        all_tfs = {p.tf_name for p in small_pwm_set}
        assert set(result.keys()) == all_tfs

    def test_tf_list_filter_case_insensitive(self, jaspar_pwms):
        """Filtering by {'CTCF', 'GATA1'} should keep only those TFs."""
        tf_list = {"CTCF", "GATA1"}
        result = build_tf_motif_map(jaspar_pwms, tf_list=tf_list)
        for tf in result:
            assert tf.upper() in {"CTCF", "GATA1"}, f"Unexpected TF: {tf!r}"
        # At least one CTCF motif expected
        ctcf_matches = [tf for tf in result if "CTCF" in tf.upper()]
        assert ctcf_matches, "CTCF not found after tf_list filter"

    def test_gene_universe_remap_mouse_case(self, jaspar_pwms):
        """gene_universe uses mouse Title-case; TF symbols from JASPAR are human ALL-CAPS.
        After remapping, keys must use the mouse-case from gene_universe.
        """
        # JASPAR has CTCF and GATA1 in all-caps; we pretend they're mouse genes
        gene_universe = ["Ctcf", "Gata1", "Actb", "Sox2"]
        tf_list = {"CTCF", "GATA1"}
        result = build_tf_motif_map(jaspar_pwms, tf_list=tf_list, gene_universe=gene_universe)

        # Keys should be remapped to mouse-case
        for key in result:
            assert key in gene_universe, (
                f"Key {key!r} not in gene_universe={gene_universe}"
            )
        # Actual TF names inside PWMs must also be remapped
        for key, pwm_list in result.items():
            for pwm in pwm_list:
                assert pwm.tf_name == key, (
                    f"PWM tf_name {pwm.tf_name!r} not remapped to {key!r}"
                )

    def test_gene_universe_excludes_absent_tfs(self, jaspar_pwms):
        """TFs not in gene_universe should be excluded even if in tf_list."""
        gene_universe = ["Actb", "Gapdh"]  # no TFs present
        tf_list = {"CTCF", "GATA1"}
        result = build_tf_motif_map(jaspar_pwms, tf_list=tf_list, gene_universe=gene_universe)
        assert result == {}, f"Expected empty map, got {result}"

    def test_empty_pwm_list(self):
        result = build_tf_motif_map([])
        assert result == {}

    def test_values_are_lists_of_pwm(self, small_pwm_set):
        result = build_tf_motif_map(small_pwm_set)
        for tf, lst in result.items():
            assert isinstance(lst, list)
            for p in lst:
                assert isinstance(p, PWM)


# ===========================================================================
# 6. harmonize_symbol
# ===========================================================================

class TestHarmonizeSymbol:
    @pytest.fixture
    def lu(self):
        return {s.upper(): s for s in ["Ctcf", "Gata1", "Actb"]}

    def test_exact_match(self, lu):
        assert harmonize_symbol("Ctcf", lu) == "Ctcf"

    def test_upper_input(self, lu):
        assert harmonize_symbol("CTCF", lu) == "Ctcf"

    def test_lower_input(self, lu):
        assert harmonize_symbol("ctcf", lu) == "Ctcf"

    def test_mixed_case(self, lu):
        assert harmonize_symbol("gAtA1", lu) == "Gata1"

    def test_absent_returns_none(self, lu):
        assert harmonize_symbol("TP53", lu) is None

    def test_empty_universe(self):
        assert harmonize_symbol("CTCF", {}) is None
