# Occupancy Grid Freespace

Bird's-eye-view free space estimation from simulated lidar using an inverse sensor model.

[![CI](https://github.com/Eelis03/occupancy-grid-freespace/actions/workflows/ci.yml/badge.svg)](https://github.com/Eelis03/occupancy-grid-freespace/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

A planar occupancy grid mapper: it turns lidar sweeps taken at known poses into a
bird's-eye-view map that labels every cell free, occupied, or unknown, and it scores
that map against the ground truth of the scene it was built from. The intended reader
is someone deciding how to feed free space to a planner, who needs to know not only
how the inverse sensor model is defined but where it breaks, by how much, and which
parameter controls the trade. The simulator, the mapper, the metrics, and the failure
measurements are all in this repository, so every number below can be reproduced by
one command.

## Problem

A vehicle carries a planar lidar. At each of a sequence of known poses the sensor
returns one range per beam. From that stream, decide for every cell of a grid fixed to
the ground whether the cell is free, occupied, or not yet determined.

Four things make this harder than accumulating hit counts.

1. A beam carries two kinds of evidence at once. It says the cells it passed through
   were empty, and it says the cell it stopped in was not. Those are different claims
   about different cells and they have to be applied separately.
2. A beam that returns nothing carries only the first kind. It certifies free space
   out to the range limit and says nothing whatever about the cell at the limit.
   Treating that cell as occupied paints a wall on the sensor horizon; discarding the
   beam throws away the free space evidence that matters most, because unreturned
   beams are the ones that crossed open ground. In the run reported below, 11699 of
   35976 beams reach the range limit, so this is a third of the evidence.
3. Occupied cells are not observable in the same way as free cells. A beam terminates
   on the near surface of an obstacle and cannot see through it, so the interior of a
   solid object is never measured and can only ever be reported unknown. Any score
   that treats unknown as a mistake will therefore be dominated by the scene rather
   than by the mapper.
4. The world moves and the filter assumes it does not. The evidence a moving object
   deposits stays where the object was, and the evidence it removes stays removed.

The output is judged on three numbers, not one. What fraction of the scored cells the
map is prepared to decide, what fraction of the truly free decided cells it calls
free, and what fraction of the truly occupied decided cells it calls occupied. Any one
of them can be driven to an arbitrary value by moving the decision threshold, so the
Results section reports a sweep rather than a point.

## Approach

The map stores one number per cell, the log odds of occupancy. This is the
construction of Moravec and Elfes as presented by Thrun, Burgard and Fox: rather than
inverting the forward sensor model, the inverse model is written down directly, with
the cells a beam passed through assigned a probability below the prior and the cell in
which the beam terminated assigned one above it. In log odds each observation becomes
an addition, so a whole sweep reduces to a sparse array of increments, and the
posterior is recovered by a logistic transform when it is needed.

Three decisions fix the behaviour of the filter.

The occupied increment is applied only where the beam carries a range return.
`is_hit` is a per-beam flag from the simulator, and a beam at the range limit
contributes the free increment along its whole traversal, including the terminal cell,
and no occupied increment anywhere.

Cells are traversed by the digital differential analyser of Amanatides and Woo, which
tracks the parameter at which the next vertical and the next horizontal grid line are
met and repeatedly steps whichever comes first. It visits every cell the segment
enters, which a Bresenham line does not: Bresenham skips cells at shallow crossings and
would leave holes in the free space. The implementation loops over steps rather than
over rays, so a 720 beam sweep costs a few hundred array operations rather than 720
Python iterations. Where a ray passes exactly through a grid corner the traversal steps
x and then y, visiting the horizontally adjacent cell, because cutting the corner would
let free space leak diagonally through a wall one cell thick.

The accumulated log odds are clamped, asymmetrically, at probabilities 0.12 and 0.97,
the defaults published with octomap by Hornung and colleagues. Without a clamp a cell
observed free a thousand times needs a thousand contrary observations before it can be
believed occupied. With this clamp, one occupied observation is undone by 2.09 free
observations and a fully saturated free cell needs 3.08 occupied observations before it
is called occupied. Those two constants are not incidental: they are what the dynamic
obstacle study measures, and the second of them turns out to decide whether a moving
car appears in the map at all.

A grid that follows the vehicle has to be re-anchored every frame. Three policies are
implemented and compared: snapping the window origin to whole cells, which makes every
shift an exact array translation, and centring the window exactly on the vehicle with
either bilinear or nearest neighbour resampling. Interpolation is applied to log odds
rather than to probability, because log odds is the additive coordinate of the filter,
so a linear blend of log odds is a geometric mean of odds and leaves the prior fixed.

`docs/design-notes.md` records the alternatives that were rejected and the conditions
under which this implementation gives poor results.

## Installation

Requires Python 3.12 or later.

```bash
git clone https://github.com/Eelis03/occupancy-grid-freespace.git
cd occupancy-grid-freespace
uv sync
```

Using pip instead of uv:

```bash
python -m venv .venv
.venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Usage

Build a map of the bundled street block scenario and score it against ground truth:

```python
from freespace_grid import run_mapping, score_grid, urban_block

scenario = urban_block()
trace = run_mapping(scenario)
agreement = score_grid(trace.grid, trace.truth, scenario.model, region=trace.observed_mask)

print(len(trace.steps), trace.totals()["max_range_beams"])
# 51 11699
print(round(agreement.decided_fraction, 4), round(agreement.free_agreement, 4),
      round(agreement.occupied_agreement, 4))
# 0.9831 0.9865 0.9923
```

The maximum range case, which is the part most often got wrong, is visible directly in
the cell sets one beam produces. The same beam is reduced twice, once as a range return
and once as a beam that reached the range limit:

```python
import numpy as np

from freespace_grid import GridSpec, scan_update

spec = GridSpec(resolution=1.0, rows=21, cols=21, origin_x=-10.5, origin_y=-10.5)
endpoint = np.array([[6.2, 0.0]])

with_return = scan_update(spec, np.zeros(2), endpoint, np.array([True]))
at_limit = scan_update(spec, np.zeros(2), endpoint, np.array([False]))

print(with_return.free_cells.shape[0], with_return.occupied_cells.shape[0])
# 6 1
print(at_limit.free_cells.shape[0], at_limit.occupied_cells.shape[0])
# 7 0
```

The traversal is the same in both cases. Only the terminal cell moves, from the
occupied set to the free set.

Runnable examples live in `examples/`:

```bash
uv run python examples/map_static_scene.py
uv run python examples/sweep_agreement.py
uv run python examples/compare_grid_frames.py
uv run python examples/dynamic_smear.py
```

The first builds and scores the static map and writes a figure. The second produces the
two sweeps. The third compares the four ways of maintaining the grid under vehicle
motion. The fourth measures what a moving obstacle does to a filter that assumes a
static world. Each accepts `--steps` to shorten the run, and the three that write
figures accept `--outdir` and `--no-figure`.

## Results

All numbers below are the output of the commands shown, on Python 3.12.10 with numpy
2.5.1, scipy 1.18.0 and matplotlib 3.11.1, on one core of an AMD64 desktop under
Windows 11. Every run is seeded, and the four scripts together take about ten seconds.

### Configuration

The `urban_block` scenario is a 60 by 40 metre street block mapped at 0.2 metres per
cell, so the grid is 200 by 300 cells. Five building footprints bound a twelve metre
corridor, with two parked vehicles, two poles, a bin and a planter near its edges. The
vehicle drives the corridor end to end at 4.7 metres per second over 51 sweeps, along a
straight section, a right hand arc, a left hand arc and a second straight section. The
sensor is a 360 degree planar lidar with 720 beams, a 30 metre range limit, 0.03 metre
Gaussian range noise and a 2 percent beam dropout rate. The filter uses free and
occupied probabilities of 0.4 and 0.7, clamps at 0.12 and 0.97, and a decision
threshold of 0.65.

### Static scene

From `uv run python examples/map_static_scene.py`:

```
quantity             value
-------------------  --------------
sweeps               51
beams retained       35976
range returns        24277
range limit returns  11699
beams dropped        744
cell visits          3135805
cells observed       35482 of 60000

region      cells  decided  free agr  occ agr  balanced
----------  -----  -------  --------  -------  --------
observed    35482  0.9831   0.9865    0.9923   0.9894
whole grid  60000  0.5814   0.9865    0.9923   0.9894
```

The two regions answer two different questions. Over the 35482 cells at least one beam
reached, the map decides 98.31 percent and is right about 98.65 percent of the free
ones and 99.23 percent of the occupied ones. Over the whole grid the decided fraction
falls to 58.14 percent, and that difference is coverage, not error: the missing 24518
cells are building interiors and the ground behind them, which no beam can reach from
the corridor. Reporting only the second number would understate the mapper and
reporting only the first would hide how much of the scene it never saw.

### Decision threshold sweep

From `uv run python examples/sweep_agreement.py`, scored over the observed region:

```
threshold  decided  free agr  occ agr  balanced  free as occ  occ as free
---------  -------  --------  -------  --------  -----------  -----------
0.55       0.9986   0.9853    0.9814   0.9833    510          15
0.62       0.9841   0.9861    0.9911   0.9886    475          7
0.65       0.9831   0.9865    0.9923   0.9894    459          6
0.68       0.9819   0.9871    0.9923   0.9897    438          6
0.72       0.9679   0.9872    0.9943   0.9907    432          4
0.78       0.9534   0.9883    0.9957   0.9920    386          3
0.82       0.9511   0.9896    0.9957   0.9926    344          3
0.86       0.9380   0.9898    0.9953   0.9925    334          3
```

The trade is visible and it is shallow: raising the threshold from 0.55 to 0.86 buys
1.4 percentage points of occupied agreement and costs 6.1 points of coverage. That
shallowness is itself the result, and it comes from the clamp. Most observed cells are
pinned at one bound or the other, 92.71 percent of them, so moving the threshold inside
the interval reclassifies only the small population in between. The sweep stops at 0.86
because the free clamp at 0.12 puts a hard ceiling of 0.88 on any threshold that could
ever call a cell free, and the model raises an error rather than accepting a threshold
it can never meet.

### Spatial tolerance on the occupied class

Same command, threshold held at 0.65. A tolerance of `k` cells counts a truly occupied
cell as agreeing when the map called some cell within `k` of it occupied:

```
tolerance  decided  free agr  occ agr  balanced  free as occ  occ as free
---------  -------  --------  -------  --------  -----------  -----------
0          0.9831   0.9865    0.9923   0.9894    459          6
1          0.9831   0.9865    0.9949   0.9907    459          6
2          0.9831   0.9865    0.9974   0.9920    459          6
3          0.9831   0.9865    0.9987   0.9926    459          6
```

Only 0.26 percentage points of occupied disagreement are recovered by one cell of
slack, so the surfaces this map places are placed in the right cell rather than one
cell out. That is worth measuring because a one cell offset is the expected failure
here: a range return is attributed to the cell containing the measured point while the
ground truth labels a cell by whether its centre lies inside an obstacle, and those two
conventions disagree whenever a surface falls near a cell boundary.

The 459 free cells called occupied, 1.35 percent of the free ones, are dominated by the
same effect at building corners, where the beam grazes the facade and the return lands
in the first cell outside it.

### Maintaining the grid under vehicle motion

From `uv run python examples/compare_grid_frames.py`. The ego window is 200 by 200
cells. The vehicle advances 4.7 cells per sweep, chosen so that a window centred
exactly on it never lands on a whole cell offset:

```
policy        shifts  lossless  decided  free agr  occ agr  at clamp  edge contrast
------------  ------  --------  -------  --------  -------  --------  -------------
world fixed   0       0         0.9831   0.9865    0.9923   0.9271    1.481
ego snap      50      50        0.9516   0.9907    0.9883   0.8224    1.614
ego bilinear  50      0         0.9448   0.9960    0.9226   0.8036    1.180
ego nearest   50      0         0.9464   0.9954    0.6038   0.8092    1.174
```

Snapping the window origin to whole cells makes all 50 shifts lossless copies, and
costs 0.40 points of occupied agreement against the world fixed grid, all of it from
cells that fell off the trailing edge of the window. Centring the window exactly on the
vehicle costs far more. Bilinear resampling drops occupied agreement to 0.9226 and edge
contrast from 1.614 to 1.180: each frame averages every obstacle surface with its free
surroundings, the filters compose over 50 frames, and a wall one cell thick is spread
until it no longer clears the decision threshold. Nearest neighbour introduces no blur,
which its almost identical edge contrast of 1.174 does not distinguish, but it displaces
every cell by up to half a cell every frame, and after 50 frames of that jitter only
0.6038 of the occupied cells survive, the worst figure in the table.

The lesson is not that interpolation is bad but that centring the window exactly is not
worth what it costs. The most a snapped window is ever off centre is half a cell,
0.1 metres here, which no consumer of the map can detect, and it buys exact arithmetic.
Both fractional policies also raise the free agreement, to 0.9960 and 0.9954, because
smoothing removes isolated spurious occupied cells along with the real thin ones; an
occupancy figure that improves while the map gets worse is a good reason to report both
classes separately.

### Moving obstacles under a static world assumption

From `uv run python examples/dynamic_smear.py`. A stationary sensor watches a disc of
radius 1 metre translate through a corridor for 8 seconds over 41 sweeps, and each run
is compared with a control in which the same disc is parked at its final position for
the whole run. Everything else is identical, including the seed. The region of interest
is a capsule of radius 2 metres around the path the disc travels. Holding the sensor
still separates the effect of the obstacle's motion from the effect of the sensor's own
motion, which the previous table already covers.

```
case         swept (m)  parked (m)  moving (m)  smear (m)  smear/swept  stale cells  stale (m2)  footprint  found  unknown  called free  peak returns
-----------  ---------  ----------  ----------  ---------  -----------  -----------  ----------  ---------  -----  -------  -----------  ------------
approaching  9.60       1.00        1.20        0.20       0.021        15           0.60        80         0      20       60           2
receding     9.60       0.80        1.20        0.40       0.042        7            0.28        80         8      72       0            3
crossing     12.80      1.80        1.00        -0.80      -0.063       4            0.16        80         0      1        79           5
```

`smear` is the extra extent along the direction of motion that the moving run produces
over its parked control. `stale cells` are cells the map calls occupied that the
obstacle no longer occupies. The last four columns split the 80 cells of the obstacle's
true final footprint into found, unknown and wrongly cleared, and give the largest
number of range returns any single cell in the region received over the whole run.

Under the standard parameters the dominant failure is not the smear but the miss. The
smear is 0.20 to 0.40 metres, 2 to 4 percent of the distance travelled, while in the
approaching and crossing cases the map places no occupied cell at all on the obstacle's
final position and calls 60 and 79 of its 80 cells free space. A planner reading this
map would drive into the obstacle.

The arithmetic explains it. Overturning a cell already saturated at the free clamp takes
3.08 consecutive occupied observations, and the peak returns column shows that no cell in
the approaching case ever received more than 2. The crossing case does reach 5, above
what is needed, so cells along its path were called occupied while the obstacle stood
over them. They do not survive: lateral motion carries the trail out of the obstacle's
own shadow, later sweeps see straight through it, 2.09 free observations undo each
occupied one, and 4 stale cells are left at the end. The map is not so much smearing the
obstacle as running behind it.

The receding case looks better than it is. It calls nothing free, but only because its
final footprint spent the whole run inside the obstacle's own shadow and therefore
accumulated no free evidence to be wrong about: 72 of the 80 cells are reported unknown.
Unknown is the safe failure, since a planner that treats unknown as impassable is not
endangered by it, but it is not a detection either.

The smear and the miss are two ends of one parameter. Sweeping the free clamp on the
approaching case, with everything else held fixed:

```
clamp  occ obs needed  smear (m)  stale (m2)  found  unknown  called free
-----  --------------  ---------  ----------  -----  -------  -----------
0.05   4.21            0.20       0.60        0      0        80
0.12   3.08            0.20       0.60        0      20       60
0.20   2.37            0.20       0.60        0      80       0
0.28   1.85            9.00       3.24        20     60       0
0.34   1.51            9.00       3.44        22     58       0
```

At a clamp of 0.05 the filter is so certain the corridor is free that all 80 footprint
cells are still called free with the obstacle standing in them. Loosening the clamp
converts those cells first to unknown and then, once one or two occupied observations
suffice, to found. The cost arrives at the same point: at 0.28 the map grows a 9.00
metre streak of stale occupancy behind the obstacle, 94 percent of the 9.60 metres it
travelled, together with 3.24 square metres of ground wrongly held occupied. The streak
survives because an approaching obstacle occludes the cells it has just left, so no
later beam ever contradicts them, and it is absent from the receding case at every
clamp because there the trail lies between the sensor and the obstacle and is corrected
on the next sweep.

Two limits bound what any setting can buy. The 20 to 22 cells found at the loose
settings match the 20 the parked control recovers, so that is the ceiling imposed by the
sensor: a lidar sees the near surface of a disc and nothing behind it, whatever the
filter does. And no setting in the sweep gives both a found obstacle and no streak, so
the trade is not an artefact of poor tuning. Removing it needs a filter that models
motion rather than assuming its absence, which this one deliberately does not;
`docs/design-notes.md` names what that would cost.

The scripts also write `outputs/static_scene_map.png`, `outputs/threshold_sweep.png` and
`outputs/dynamic_smear.png`, which show the same results as figures. The last of these
puts the parked control, the moving obstacle at the published clamp, the moving obstacle
at a clamp of 0.28, and the crossing case side by side, so the trail and its absence can
be seen rather than only counted.

## Architecture

| Module | Responsibility |
| --- | --- |
| `src/freespace_grid/model/grid.py` | `GridSpec` and the world to cell mapping, its inverse, bulk cell centres, and bounds tests. |
| `src/freespace_grid/model/logodds.py` | `LogOddsModel`: increments, asymmetric clamps, decision band, and the closed form observation counts. |
| `src/freespace_grid/model/transform.py` | `Pose2D` and the SE(2) operations, written elementwise so no BLAS kernel is involved. |
| `src/freespace_grid/model/occupancy.py` | The `OccupancyGrid` container and the three way `CellState` decision rule. |
| `src/freespace_grid/model/typing.py` | Array type aliases shared by every layer. |
| `src/freespace_grid/algorithm/raycast.py` | Vectorised Amanatides and Woo grid traversal for a bundle of rays. |
| `src/freespace_grid/algorithm/inverse_sensor.py` | One sweep reduced to disjoint free and occupied cell sets, including the range limit case. |
| `src/freespace_grid/algorithm/accumulation.py` | `Accumulator`, whole cell translation, and the three window re-anchoring policies. |
| `src/freespace_grid/pipeline/scene.py` | Circle and polygon obstacles, closed form ray intersection, and ground truth rasterisation. |
| `src/freespace_grid/pipeline/lidar.py` | `LidarSpec` and `Scan`: noise, dropout, minimum range, and range limit returns. |
| `src/freespace_grid/pipeline/trajectory.py` | Exact unicycle integration and even subsampling of a trajectory. |
| `src/freespace_grid/pipeline/scenarios.py` | The named scenarios, static and dynamic, and the parameters that define them. |
| `src/freespace_grid/pipeline/runner.py` | The trajectory loop and the `MappingTrace` it records. |
| `src/freespace_grid/analysis/metrics.py` | `Agreement`, the threshold sweep, and the spatial tolerance sweep. |
| `src/freespace_grid/analysis/smear.py` | Region of interest construction, smear measurement, and the moving against parked comparison. |
| `src/freespace_grid/analysis/sharpness.py` | Clamp saturation and edge contrast, used to measure resampling loss. |
| `src/freespace_grid/analysis/report.py` | Rendering of measurement records as the tables above. |
| `src/freespace_grid/analysis/figures.py` | The figures. The only module that imports matplotlib. |
| `examples/` | Wiring scripts, no logic. |

Each layer depends only on the ones above it. The model layer performs no input or
output and knows nothing about sensors; the algorithm layer draws nothing and simulates
nothing; the pipeline layer scores nothing.

## Testing

```bash
uv run pytest
uv run ruff check .
uv run mypy
```

The suite has three tiers: property and invariant tests covering the mathematics,
regression tests pinning recorded behaviour, and integration tests running each
example script under a reduced iteration count.

147 tests run in about 11 seconds. The first tier checks that the world to cell mapping
round trips exactly at every cell centre of four different grids, that the probability
and log odds conversions invert each other, that repeated observations reach the clamp
and stop, that an untouched cell holds the prior bit for bit, that a single beam visits
exactly the cells a hand computed crossing sequence names and exactly the cells a dense
sampling of the segment falls in, that a range limit beam marks free space along its
whole traversal and nothing occupied, that a whole cell window shift is a bit exact copy
under all three interpolation settings, that a closed room converges to free interior
and occupied walls with zero error, and that scoring a map against its own decision
gives agreement of one on both classes.

The second tier compares `tests/data/reference_run.json` against a fresh run: five
mapping runs, two sweeps, and three dynamic obstacle cases. Its module docstring states
which quantities are pinned exactly,
which are pinned to a tolerance, and why: beam and dropout counts come from a bit exact
seeded stream and are compared exactly, while counts derived from trigonometry are
allowed two cells or one part in five hundred, and the orderings that the experiment
exists to demonstrate are asserted as inequalities rather than pinned as numbers.
Nothing in the baseline comes from an iterative solve.

The third tier runs each script in `examples/` as a subprocess under a reduced step
count, checks that it exits cleanly and writes nothing to standard error, and checks
that it puts its figure in the requested directory and writes none when asked not to.
A test also fails if a script is added to `examples/` without being listed.

## References

Algorithms:

- H. P. Moravec and A. Elfes, "High resolution maps from wide angle sonar",
  Proceedings of the IEEE International Conference on Robotics and Automation, 1985,
  pages 116 to 121. DOI [10.1109/ROBOT.1985.1087316](https://doi.org/10.1109/ROBOT.1985.1087316)
- A. Elfes, "Using occupancy grids for mobile robot perception and navigation",
  Computer 22(6), 1989, pages 46 to 57.
  DOI [10.1109/2.30720](https://doi.org/10.1109/2.30720)
- S. Thrun, W. Burgard and D. Fox, "Probabilistic Robotics", MIT Press, 2005,
  chapter 9, "Occupancy grid mapping". <https://mitpress.mit.edu/9780262201629/probabilistic-robotics/>
- J. Amanatides and A. Woo, "A fast voxel traversal algorithm for ray tracing",
  Proceedings of Eurographics 1987, pages 3 to 10.
  <https://www.cse.yorku.ca/~amana/research/grid.pdf>
- J. E. Bresenham, "Algorithm for computer control of a digital plotter",
  IBM Systems Journal 4(1), 1965, pages 25 to 30.
  DOI [10.1147/sj.41.0025](https://doi.org/10.1147/sj.41.0025)
- A. Hornung, K. M. Wurm, M. Bennewitz, C. Stachniss and W. Burgard, "OctoMap: an
  efficient probabilistic 3D mapping framework based on octrees", Autonomous Robots
  34(3), 2013, pages 189 to 206.
  DOI [10.1007/s10514-012-9321-0](https://doi.org/10.1007/s10514-012-9321-0)
- D. Hahnel, R. Triebel, W. Burgard and S. Thrun, "Map building with mobile robots in
  dynamic environments", Proceedings of the IEEE International Conference on Robotics
  and Automation, 2003, pages 1557 to 1563.
  DOI [10.1109/ROBOT.2003.1241816](https://doi.org/10.1109/ROBOT.2003.1241816)
- R. Danescu, F. Oniga and S. Nedevschi, "Modeling and tracking the driving environment
  with a particle-based occupancy grid", IEEE Transactions on Intelligent
  Transportation Systems 12(4), 2011, pages 1331 to 1342.
  DOI [10.1109/TITS.2011.2158097](https://doi.org/10.1109/TITS.2011.2158097)
- E. Haines and T. Akenine-Moller, editors, "Ray Tracing Gems", Apress, 2019,
  chapter 6, on ray and disc intersection in floating point.
  DOI [10.1007/978-1-4842-4427-2](https://doi.org/10.1007/978-1-4842-4427-2)

Dependencies:

- [numpy](https://numpy.org/) (BSD 3-Clause). All array arithmetic, the vectorised ray
  traversal, the geometric predicates, and the seeded PCG64 generator that makes every
  run reproducible.
- [scipy](https://scipy.org/) (BSD 3-Clause). `scipy.ndimage.map_coordinates` for grid
  resampling under vehicle motion, and `scipy.ndimage.binary_dilation` and
  `binary_erosion` for the spatial tolerance and the boundary band.
- [matplotlib](https://matplotlib.org/) (matplotlib license, a BSD-style permissive
  license). The figures in the analysis layer, used with the Agg backend so the
  examples need no display.
- [pytest](https://docs.pytest.org/) (MIT), [ruff](https://docs.astral.sh/ruff/) (MIT),
  and [mypy](https://mypy-lang.org/) (MIT). Development only: test running, linting, and
  type checking.

## License

Released under the MIT license. See [LICENSE](LICENSE).
