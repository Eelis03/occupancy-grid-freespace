"""Tier three: every script in examples/ runs to completion under a reduced step count.

The scripts are run as subprocesses so that the test exercises the same entry point a
reader would use, including argument parsing and the module level matplotlib backend
choice. Figures are written into a temporary directory, so the suite leaves nothing
behind.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"

# Script, reduced arguments, and the figure it is expected to write, if any.
INVOCATIONS: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    ("map_static_scene.py", ("--steps", "6"), "static_scene_map.png"),
    ("sweep_agreement.py", ("--steps", "6"), "threshold_sweep.png"),
    ("compare_grid_frames.py", ("--steps", "6"), None),
    ("dynamic_smear.py", ("--steps", "5"), "dynamic_smear.png"),
)
IDS = [name for name, _, _ in INVOCATIONS]


def run_example(script: str, arguments: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *arguments],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
        check=False,
    )


def test_every_example_script_is_covered() -> None:
    """A new script in examples/ must be added here, or this test fails."""
    scripts = {path.name for path in EXAMPLES.glob("*.py")}
    assert scripts == {name for name, _, _ in INVOCATIONS}


@pytest.mark.parametrize(("script", "arguments", "figure"), INVOCATIONS, ids=IDS)
def test_example_runs_to_completion(
    script: str, arguments: tuple[str, ...], figure: str | None, tmp_path: Path
) -> None:
    extra = () if figure is None else ("--outdir", str(tmp_path))
    completed = run_example(script, (*arguments, *extra))
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
    assert not completed.stderr.strip()
    if figure is None:
        assert list(tmp_path.iterdir()) == []
    else:
        assert (tmp_path / figure).is_file()
        assert (tmp_path / figure).stat().st_size > 0


@pytest.mark.parametrize(
    ("script", "arguments", "figure"),
    [entry for entry in INVOCATIONS if entry[2] is not None],
    ids=[name for name, _, figure in INVOCATIONS if figure is not None],
)
def test_no_figure_switch_suppresses_every_write(
    script: str, arguments: tuple[str, ...], figure: str | None, tmp_path: Path
) -> None:
    completed = run_example(script, (*arguments, "--outdir", str(tmp_path), "--no-figure"))
    assert completed.returncode == 0, completed.stderr
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
