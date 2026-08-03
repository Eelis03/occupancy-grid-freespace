"""Tier three: every script in examples/ runs to completion under a reduced step count.

The scripts are run as subprocesses so that the test exercises the same entry point a
reader would use, including argument parsing and the module level matplotlib backend
choice. Figures are written into a temporary directory, so the suite leaves nothing
behind.

The figures tracked in ``docs/figures`` are the exception to that. They are produced by
``examples/publish_figures.py``, they are committed, and the README embeds them, so this
module also asserts that they are present and inside the size budget. It does not
compare them byte for byte against a fresh run: matplotlib renders text through whatever
font stack the machine provides and its PNG output is not reproducible across platforms,
so a byte comparison would fail on one of the two runners for a reason that has nothing
to do with this code.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
PUBLISHED = REPO_ROOT / "docs" / "figures"
PUBLISHED_FIGURES = ("dynamic_obstacle.png", "urban_block_map.png", "pose_drift.png")
FIGURE_BUDGET_BYTES = 250 * 1024


@dataclass(frozen=True, slots=True)
class Invocation:
    """One example script, the arguments that shorten it, and what it should write."""

    script: str
    arguments: tuple[str, ...]
    figures: tuple[str, ...] = ()
    suppressible: bool = True


INVOCATIONS: tuple[Invocation, ...] = (
    Invocation("map_static_scene.py", ("--steps", "6"), ("static_scene_map.png",)),
    Invocation("sweep_agreement.py", ("--steps", "6"), ("threshold_sweep.png",)),
    Invocation("compare_grid_frames.py", ("--steps", "6")),
    Invocation("dynamic_smear.py", ("--steps", "5"), ("dynamic_smear.png",)),
    Invocation("pose_drift.py", ("--steps", "6"), ("pose_drift.png",)),
    # Writing figures is the whole purpose of this one, so it has no switch to turn
    # them off and is excluded from the suppression test rather than given a useless
    # flag to satisfy it.
    Invocation(
        "publish_figures.py",
        ("--steps", "6"),
        PUBLISHED_FIGURES,
        suppressible=False,
    ),
)
IDS = [item.script for item in INVOCATIONS]


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
    assert scripts == {item.script for item in INVOCATIONS}


# Lines a dependency writes to stderr about its own housekeeping. They say nothing
# about the script that triggered them, and on a fresh runner matplotlib emits the
# first of these the very first time any process imports pyplot.
_BENIGN_STDERR: tuple[str, ...] = ("Matplotlib is building the font cache",)


def unexpected_stderr(text: str) -> list[str]:
    """Return the stderr lines that indicate a real problem.

    An earlier version of this test required stderr to be empty. That is stricter
    than the property it was written to check, which is that the script completed
    without complaining. A dependency is entitled to write an informational line
    to stderr, and matplotlib does exactly that on a runner with no font cache,
    which passed on Linux and failed on Windows for a reason unrelated to this
    repository.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    return [line for line in lines if not any(token in line for token in _BENIGN_STDERR)]


@pytest.mark.parametrize("case", INVOCATIONS, ids=IDS)
def test_example_runs_to_completion(case: Invocation, tmp_path: Path) -> None:
    extra = () if not case.figures else ("--outdir", str(tmp_path))
    completed = run_example(case.script, (*case.arguments, *extra))
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip()
    assert unexpected_stderr(completed.stderr) == []
    if not case.figures:
        assert list(tmp_path.iterdir()) == []
        return
    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == sorted(case.figures)
    for name in case.figures:
        assert (tmp_path / name).stat().st_size > 0


@pytest.mark.parametrize(
    "case",
    [item for item in INVOCATIONS if item.figures and item.suppressible],
    ids=[item.script for item in INVOCATIONS if item.figures and item.suppressible],
)
def test_no_figure_switch_suppresses_every_write(case: Invocation, tmp_path: Path) -> None:
    completed = run_example(
        case.script, (*case.arguments, "--outdir", str(tmp_path), "--no-figure")
    )
    assert completed.returncode == 0, completed.stderr
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_published_figures_are_present_and_within_budget() -> None:
    """The tracked figures exist and fit the quarter megabyte the README promises."""
    sizes = {}
    for name in PUBLISHED_FIGURES:
        path = PUBLISHED / name
        assert path.is_file(), f"{path} is missing; run examples/publish_figures.py"
        sizes[name] = path.stat().st_size
        assert sizes[name] > 0
    assert sum(sizes.values()) <= FIGURE_BUDGET_BYTES, sizes


def test_published_figures_are_the_whole_directory() -> None:
    """Nothing else has crept into the tracked figure directory."""
    present = sorted(path.name for path in PUBLISHED.iterdir() if path.is_file())
    assert present == sorted(PUBLISHED_FIGURES)
