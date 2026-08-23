"""Tests for the motif-database FETCHERS (opt-in downloads) + their loaders.

- fetch_jaspar  → load_jaspar_meme   (JASPAR2024 CORE vertebrates MEME)
- fetch_cisbp   → load_cisbp_meme    (CIS-BP 2.00 from the MEME Suite bundle, MEME format)
- load_meme source param, resolve_* cache hit, genome→species convenience

These are LIVE smoke tests (small real downloads) that SKIP gracefully when offline
(URLError / timeout) rather than failing CI. No new package dependency — the parser is
pure-Python and the fetchers use only urllib + tarfile + gzip (stdlib).
"""
from __future__ import annotations

import cytorete  # noqa: F401
import socket
import urllib.error

import numpy as np
import pytest

from cytorete.data import (
    fetch_jaspar,
    fetch_cisbp,
    resolve_jaspar_path,
    resolve_cisbp_meme_path,
    load_meme,
    load_jaspar_meme,
    load_cisbp_meme,
)
from piaso.data._pwm import PWM

_NET_ERRORS = (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError)


def _skip_if_offline(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except _NET_ERRORS as e:  # pragma: no cover - network dependent
        pytest.skip(f"network unavailable for {fn.__name__}: {e}")


def _check_pwms(pwms, source):
    assert isinstance(pwms, list) and len(pwms) > 0
    for p in pwms[:25]:
        assert isinstance(p, PWM)
        assert p.source == source
        assert p.probs.shape[0] == 4 and p.probs.shape[1] >= 1
        col_sums = p.probs.sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-2)  # probability columns


def test_fetch_jaspar_and_load(tmp_path):
    path = _skip_if_offline(fetch_jaspar, dest_dir=str(tmp_path))
    assert path.endswith("JASPAR2024_CORE_vertebrates.meme")
    pwms = load_jaspar_meme(path)
    _check_pwms(pwms, "jaspar")
    assert len(pwms) > 500  # JASPAR2024 CORE vertebrates has ~879 motifs
    # cache hit via resolve
    assert resolve_jaspar_path(dest_dir=str(tmp_path)) == path
    # second fetch (no force) returns the cached path without re-download
    assert fetch_jaspar(dest_dir=str(tmp_path)) == path


def test_fetch_cisbp_human_and_load(tmp_path):
    path = _skip_if_offline(fetch_cisbp, species="Homo_sapiens", dest_dir=str(tmp_path))
    assert path.endswith("CIS-BP_2.00_Homo_sapiens.meme")
    pwms = load_cisbp_meme(path)
    _check_pwms(pwms, "cisbp")
    assert len(pwms) > 800  # ~1065 human CIS-BP motifs
    assert resolve_cisbp_meme_path(species="Homo_sapiens", dest_dir=str(tmp_path)) == path


def test_fetch_cisbp_genome_convenience(tmp_path):
    # genome='mm10' must map to Mus_musculus
    path = _skip_if_offline(fetch_cisbp, genome="mm10", dest_dir=str(tmp_path))
    assert path.endswith("CIS-BP_2.00_Mus_musculus.meme")
    pwms = load_cisbp_meme(path)
    _check_pwms(pwms, "cisbp")
    assert len(pwms) > 500  # ~790 mouse CIS-BP motifs


def test_load_meme_source_param(tmp_path):
    path = _skip_if_offline(fetch_jaspar, dest_dir=str(tmp_path))
    pwms = load_meme(path, source="custom_src")
    assert pwms and all(p.source == "custom_src" for p in pwms[:10])


def test_resolve_returns_none_when_absent(tmp_path):
    assert resolve_jaspar_path(dest_dir=str(tmp_path)) is None
    assert resolve_cisbp_meme_path(species="Homo_sapiens", dest_dir=str(tmp_path)) is None
    # explicit non-existent path → None
    assert resolve_jaspar_path(jaspar_path=str(tmp_path / "nope.meme")) is None
