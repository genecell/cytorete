"""``cytorete.data`` — re-exports of PIASO's motif and sequence loaders.

Every name here now lives in PIASO and is imported from there. The motif-DB and
.2bit loaders moved back in PIASO 1.2.0: they are the inputs to
``piaso.pp.scan_motifs``, which PIASO publishes, and keeping them downstream
left that scanner with no supported way to obtain a sequence or a PWM.

This module stays so that ``cytorete.data.load_meme`` keeps working; it is a
re-export, not a second implementation.
"""
from __future__ import annotations

from piaso.data._pwm import PWM
from piaso.data import (
    fetch_2bit,
    resolve_2bit_path,
    extract_sequences,
    revcomp,
)
from piaso.data import (
    load_meme,
    load_jaspar_meme,
    load_cisbp_meme,
    load_cisbp,
    load_tf_list,
    fetch_jaspar,
    resolve_jaspar_path,
    fetch_cisbp,
    resolve_cisbp_meme_path,
    fetch_cistarget_motifs,
    load_cistarget_motifs,
    resolve_cistarget_paths,
    write_meme,
    fetch_animaltfdb_tf_list,
    build_tf_motif_map,
)

# camelCase aliases (piaso.data continuity)
fetchGenomeFasta = fetch_2bit
loadMotifs = load_jaspar_meme
loadTFList = load_tf_list
buildTFMotifMap = build_tf_motif_map
fetchJASPAR = fetch_jaspar
fetchCISBP = fetch_cisbp

__all__ = [
    "PWM",
    "fetch_2bit", "resolve_2bit_path", "extract_sequences", "revcomp",
    "load_meme", "load_jaspar_meme", "load_cisbp_meme", "load_cisbp",
    "load_tf_list", "fetch_jaspar", "resolve_jaspar_path", "fetch_cisbp",
    "resolve_cisbp_meme_path", "fetch_cistarget_motifs", "load_cistarget_motifs",
    "resolve_cistarget_paths", "write_meme", "fetch_animaltfdb_tf_list",
    "build_tf_motif_map",
    "fetchGenomeFasta", "loadMotifs", "loadTFList", "buildTFMotifMap",
    "fetchJASPAR", "fetchCISBP",
]
