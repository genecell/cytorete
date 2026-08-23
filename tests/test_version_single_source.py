"""The package version must have exactly one source of truth.

Two literals -- one in ``pyproject.toml``, one in the package -- agree right
up until a release is prepared and only one of them is bumped. The failure is
quiet in a development checkout, because a stale editable install can make
both reads return the same old number, and loud on a clean CI install, which
is the worst possible order to discover it in.

Hatch reads ``cytorete.__version__``, so the package holds the only literal.
"""
import pathlib
import re

import pytest

import cytorete


def _pyproject() -> str:
    root = pathlib.Path(cytorete.__file__).resolve().parent.parent
    pp = root / "pyproject.toml"
    if not pp.exists():                    # installed wheel, no source tree
        pytest.skip("no pyproject.toml beside the package")
    return pp.read_text()


def test_pyproject_carries_no_version_literal():
    text = _pyproject()
    assert re.search(r'^version\s*=\s*"', text, re.M) is None, (
        "pyproject.toml has its own version literal -- it drifts from "
        "cytorete/__init__.py the moment a release is prepared")
    assert re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', text, re.M), (
        'pyproject.toml must declare dynamic = ["version"]')
    assert re.search(r'^path\s*=\s*"cytorete/__init__\.py"', text, re.M), (
        "[tool.hatch.version] must read cytorete/__init__.py")


def test_the_single_literal_is_a_plain_module_level_string():
    """Hatch parses the attribute statically; a computed value breaks the
    build in a way no test here would otherwise see."""
    src = pathlib.Path(cytorete.__file__).read_text()
    m = re.search(r'^__version__ = "([^"]+)"$', src, re.M)
    assert m, 'cytorete/__init__.py needs one plain __version__ = "..." literal'
    assert m.group(1) == cytorete.__version__
    assert len(re.findall(r'^__version__\s*=', src, re.M)) == 1
