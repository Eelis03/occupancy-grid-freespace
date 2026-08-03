"""The package ships the things an installer needs, not only the things a test imports.

A repository that passes mypy in strict mode and then ships no PEP 561 marker has type
checked itself and delivered nothing: an application that installs it gets ``Any`` for
every symbol, because a type checker is required to ignore inline annotations in an
installed package that does not declare itself typed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import freespace_grid

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(freespace_grid.__file__).resolve().parent


def test_py_typed_marker_sits_inside_the_package() -> None:
    """PEP 561 requires the marker beside ``__init__.py``, not beside ``pyproject.toml``."""
    marker = PACKAGE_ROOT / "py.typed"
    assert marker.is_file()
    assert marker.parent == PACKAGE_ROOT
    assert (marker.parent / "__init__.py").is_file()


def test_py_typed_marker_is_empty() -> None:
    """The file is a flag. Content in it is at best ignored and at worst a partial marker."""
    assert (PACKAGE_ROOT / "py.typed").read_bytes() == b""


def test_source_tree_carries_the_marker_too() -> None:
    """The wheel is built from ``src``, so the marker has to be in version control there."""
    assert (REPO_ROOT / "src" / "freespace_grid" / "py.typed").is_file()


def test_coverage_is_available_and_configured_in_the_dev_group() -> None:
    """The measured coverage figure in the README needs the plugin to be a declared dependency."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = pyproject["dependency-groups"]["dev"]
    assert any(entry.startswith("pytest-cov") for entry in dev), dev
