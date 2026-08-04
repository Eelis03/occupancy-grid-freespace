# Design notes for Occupancy Grid Freespace

## Method selection

### The representation: clamped log odds on a fixed grid

The map is a regular grid of independent Bernoulli cells, each holding the log odds of
occupancy. This is the occupancy grid of Moravec and Elfes, in the form presented by
Thrun, Burgard and Fox in chapter 9 of "Probabilistic Robotics". The reason to store log
odds rather than probability is arithmetic: the Bayes update becomes an addition, so a
sweep of 720 beams reduces to a sparse array of constant increments, no renormalisation
is needed, and the order in which observations arrive cannot change the result.

The method depends on three assumptions and each one costs something.

**Cells are independent.** The posterior of the map is taken to be the product of the
posteriors of its cells, so the fact that a wall is a connected object never enters. The
cost is visible in the results: at building corners the beam grazes the facade, the
return lands in the first cell outside it, and 459 free cells, 1.35 percent of the free
ones, are called occupied. Only a model that knows a facade is straight can remove them,
because they sit against the real surface rather than out on their own, which is measured
under the rejected alternatives below. The independence assumption is what makes the
update a per-cell addition, so giving it up means giving up the whole computational
argument for the representation, which is why almost every deployed occupancy grid keeps
it.

**The world is static.** The filter has no notion of time beyond accumulation, so it
cannot distinguish a cell that changed from a cell that was mismeasured. The cost is
quantified in the Results section of the README: with the standard parameters, a disc of
radius 1 metre crossing the sensor's line of sight has 79 of the 80 cells of its
footprint reported as free space. Loosening the free clamp until the obstacle is found
buys a 9.00 metre trail of stale occupancy behind it. The two failures are opposite ends
of one parameter and the sweep in `examples/dynamic_smear.py` shows there is no setting
that avoids both.

**The pose is known.** Every sweep is placed in the map using the pose the mapper is
handed, so any error in that pose is written into the map as if it were sensor error.
This assumption was originally left standing and is now measured and, within one run,
removed: `pipeline/odometry.py` corrupts the motion between consecutive poses so the
estimate drifts the way a real one does, and `algorithm/scan_match.py` corrects it
against the map built so far. The cost of both is recorded under "Closed limitations"
below. What the assumption still buys is that every other result in this repository is
produced with exact poses, so those numbers measure the sensor model rather than a
localiser.

### The inverse sensor model

The inverse model assigns probability 0.4 to a cell a beam passed through and 0.7 to the
cell in which the beam terminated, against a prior of 0.5. It is a piecewise constant
approximation to the beam model: the true likelihood falls off smoothly around the
measured range with a width set by the range noise, and a smooth model would spread the
occupied evidence over a few cells rather than concentrating it in one. At 0.2 metre
cells and 0.03 metre range noise the noise is a seventh of a cell, so the smooth model
would put essentially all its mass in the same cell, and the piecewise constant form
gives up almost nothing while being an order of magnitude cheaper.

A cell receives at most one increment per sweep. Many beams cross the cells nearest the
sensor, and adding one increment per beam would treat a single sweep as dozens of
independent observations of those cells and saturate them in one frame. When a cell is
both crossed by one beam and terminates another, the occupied increment wins, because a
surface detected in that cell is the more specific claim.

The maximum range case is handled explicitly and has its own tests. A beam that returns
no echo says the sensor saw no surface anywhere between its origin and the range limit,
which is free space evidence along the whole beam, and says nothing about the cell at
the limit. In the reported run 11699 of 35976 beams reach the limit, so this is a third
of the evidence in the map.

### The clamp

Accumulated log odds are clamped at probabilities 0.12 and 0.97, the defaults published
with octomap by Hornung and colleagues. The bounds are asymmetric because the two errors
are not equally expensive: a cell wrongly held free is a cell a planner will drive into,
so free evidence is made cheaper to overturn than occupied evidence. The two derived
constants govern most of what the map does. One occupied observation is undone by 2.09
free observations, and a fully saturated free cell needs 3.08 occupied observations
before it is called occupied. The second constant is why a moving obstacle is missed,
and `LogOddsModel` exposes both as methods so the explanation in the README is computed
rather than asserted.

### Ray traversal

Cells along a beam are enumerated with the digital differential analyser of Amanatides
and Woo. Where a ray passes exactly through a grid corner the traversal steps x and then
y, visiting the horizontally adjacent cell rather than cutting across, so free space
cannot leak diagonally through a wall one cell thick. The implementation loops over
steps and vectorises over rays, which turns a 720 beam sweep into a few hundred array
operations.

### Scoring

The map is scored on three numbers rather than one: the fraction of cells it is prepared
to decide, the agreement on the truly free decided cells, and the agreement on the truly
occupied ones. Overall accuracy is not reported, because free cells outnumber occupied
cells by more than an order of magnitude in an open scene and an overall figure would be
almost exactly the free agreement.

The scoring region is an argument with no default. Scored over the whole grid the run
reported in the README decides 58.14 percent of cells; scored over the cells at least one
beam reached it decides 98.31 percent. The difference is sensor coverage, not mapping
error, and reporting either number alone would be misleading.

None of the three is a planner's question, and `analysis/reachability.py` answers that one
apart from them rather than folding it in. Agreement weights every cell alike, while a
vehicle can only use free space joined to where it stands and can only be hurt by an error
inside it. That module floods the free cells outward from the vehicle with occupied and
unknown cells alike impassable, which is the rule a planner applies to this map and the
sense in which unknown is the safe failure, and it splits the cells wrongly called free
into the ones shut behind a surface the map did find and the ones a vehicle could reach.
The flood is four connected rather than eight, for the reason the ray traversal will not
cut a corner: an eight connected fill passes between two occupied cells meeting
diagonally, and a vehicle does not.

### Odometry error, and correcting it

The error is applied to the motion between consecutive poses in the body frame, not to
each absolute pose. That is the whole point: independent noise on an absolute pose gives
a jitter that stays near the truth, while noise on each increment gives a drift that
compounds, and only the second is indistinguishable from the world having changed. The
three coefficients are the terms of the odometry motion model of Thrun, Burgard and Fox
that matter at this scale, and the same variates are drawn at every noise level, so the
rows of the drift table in the README are one accident of the seed at four amplitudes
rather than four separate accidents.

The correction is a correlative scan to map matcher: score a grid of candidate poses by
how well the sweep's range returns land on the occupied evidence already in the map, and
keep the best. Three details decide whether it works.

The objective is a likelihood field rather than the log odds array. Scoring log odds
directly would reward a candidate that pushes the sweep off the map, because an
unobserved cell holds the prior while a correctly matched free cell holds a large
negative number. The field is therefore zero wherever the map is free or unknown and
positive only where it holds occupied evidence, so a candidate is paid for agreement and
never for absence.

The field is blurred by a Gaussian one cell wide. Unblurred it is a sum of indicator
functions and is flat almost everywhere, so a candidate half a cell from the optimum
scores the same as one ten cells away and the search has nothing to follow. The blur
also absorbs the one cell disagreement between where a range return is attributed and
where the surface actually is.

The search is coarse to fine, three passes each halving the radius and the step. A
single pass at the same final resolution would cost tens of thousands of candidates
instead of a few hundred. The refinement can settle into a local optimum a full search
would have rejected, and the blur is what keeps the first pass wide enough for that to
be rare.

## Rejected alternatives

### Bresenham line drawing instead of a voxel traversal

Bresenham's line algorithm is the obvious way to walk a grid and it is what many
tutorials use. It draws a line of connected cells, but it does not draw every cell the
line passes through: at shallow gradients it picks one cell per column and skips the
other. In a free space map those skipped cells stay unknown, so a swept region acquires
a stipple of undecided cells that gets worse the further a beam travels. The Amanatides
and Woo traversal costs one extra comparison per step and visits every cell the segment
enters. Bresenham is cited in the README as the rejected alternative rather than as a
method used.

### Counting hits and misses instead of a Bayes filter

The counting map stores the number of times each cell terminated a beam and the number
of times a beam passed through, and reports the ratio. It is simpler, needs no
probability, and is the maximum likelihood estimator under the same independence
assumption. It was rejected because it cannot be clamped in a principled way. The ratio
after ten thousand observations is as confident as the ratio after ten, so a cell that
has been free all day cannot become occupied within any useful time, and the response
of the map to a change would depend on how long the vehicle had been standing there.
The log odds form with an explicit clamp makes that behaviour a parameter rather than an
accident, and the clamp sweep in `examples/dynamic_smear.py` measures what the parameter
buys.

### A smooth beam likelihood instead of a piecewise constant one

The full beam model of Thrun, Burgard and Fox mixes a Gaussian around the true range, an
exponential for unexpected obstacles, a uniform term for random readings, and a point
mass at the range limit. Inverting it per cell is more faithful and considerably more
expensive, and at this cell size and range noise the Gaussian is narrower than one cell,
so the extra faithfulness would be discarded by the discretisation. It would matter at a
finer resolution or with a noisier sensor, and the piecewise constant model in
`inverse_sensor.py` is the piece that would have to change.

### Centring the ego window exactly on the vehicle

The natural way to maintain a map that follows the vehicle is to keep the vehicle at the
centre of the window and resample the contents each frame. This was implemented, in both
bilinear and nearest neighbour form, and measured rather than argued about. Against a
window snapped to whole cells, bilinear resampling costs 6.6 points of occupied
agreement and drops edge contrast from 1.614 to 1.180, because each frame averages every
obstacle surface with its free surroundings and the filters compose over 50 frames.
Nearest neighbour introduces no blur but displaces every cell by up to half a cell each
frame, and after 50 frames only 0.6038 of the occupied cells survive. Snapping the window
origin instead leaves the vehicle at most half a cell, 0.1 metres, from the window
centre, which no consumer of the map can detect, and makes every shift an exact copy. All
three policies are kept in `accumulation.py` and compared by
`examples/compare_grid_frames.py`, since the measurement is more useful than the
conclusion.

### Modelling dynamic objects

Three established options were considered for the moving obstacle problem and none was
implemented.

Hahnel and colleagues classify each measurement as belonging to a static or a dynamic
object with an expectation maximisation procedure and build the map from the static
measurements only. It removes the trail cleanly, but it needs the whole sequence in
advance, so it is a post-processing method rather than something a vehicle can run
online.

A Bayesian occupancy filter, or the particle based occupancy grid of Danescu and
colleagues, carries a velocity distribution per cell and propagates occupancy through
it. This is the right answer for a driving stack and it is what a production system would
use. It also replaces one scalar per cell with a particle set per cell, changes the
update from an addition to a resampling step, and turns this repository into a different
project with a different subject.

The cheapest option, decaying the log odds towards the prior with a time constant, was
the closest call. It is two lines and it would bound how long stale evidence survives.
It was rejected because it degrades the static map everywhere in order to fix the map in
the few cells where something moved: a wall observed once and then occluded would fade to
unknown at the same rate as a car that drove away. The clamp sweep already exposes the
same trade with one parameter and without that side effect.

### Gradient scan matching, and full SLAM, instead of a correlative search

The Hector SLAM matcher optimises the same objective by Gauss-Newton on the bilinearly
interpolated field, which is faster and resolves below the search step. It also needs a
starting point inside the basin of the correct optimum, and the whole reason a pose
correction is needed here is that the starting point is wrong by an unknown amount. The
correlative search is slower and has no such requirement, and at 51 sweeps of a few
hundred points the difference is under a second.

Full SLAM, meaning a pose graph with loop closure that re-optimises earlier poses when a
place is revisited, was not implemented. It is the correct answer to the part of the
problem that remains, and it is a different project: it needs a graph, a closure
detector, a solver, and a way to rebuild the map from corrected poses, none of which
this repository has a use for elsewhere. The matcher here localises against the map it
is building and never revises a sweep it has already integrated, which is stated in the
limitations rather than glossed over.

### A morphological opening to remove the spurious occupied cells

This is why the independence limitation is still open. An opening on the decided map
looked like the cheap fix, and it is not, for a reason specific to this map. Only the
near surface of an obstacle is ever observed, so the occupied set of the map is a shell
one cell thick, and a three by three opening deletes all of it rather than the spurious
part of it. A connected component filter does not separate the two either: the
free cells wrongly called occupied are the ones where a beam grazed a facade and the
return landed in the first cell outside it, so they are adjacent to the real surface and
belong to the same component. Suppressing them needs a model that knows a wall is
straight, which is a spatial prior and is the assumption the representation gives up on
purpose.

### Ground truth by rasterising the scene and then ray tracing the raster

Scoring would have been simpler if the scene were rasterised once and the simulated
ranges taken from the raster. It was rejected because it scores the mapper against its
own discretisation: any error caused by the cell size would appear in both the map and
the truth and cancel. Ranges therefore come from closed form ray intersections against
the analytic circles and polygons, and the ground truth comes from a point containment
test on the same shapes. The one cell disagreement this creates between the two
conventions is real, and the spatial tolerance sweep is there to measure it, which is
also why `enclosed_room` places its wall faces away from cell boundaries: a convergence
test that landed on that boundary would be measuring the offset rather than convergence.

## Known limitations

**A moving obstacle is missed or smeared, and there is no setting that avoids both.**
Quantified in the README. With the published clamp of 0.12, 60 of the 80 cells of an
approaching obstacle's footprint and 79 of 80 of a crossing obstacle's are reported free.
Loosening the clamp to 0.28 finds the obstacle and produces a 9.00 metre trail behind it.
This one is structural for this formulation and not a tuning failure. The filter has no
state in which to keep a velocity and no update in which to apply one, so removing the
limitation means replacing the representation, which is the subject of the rejected
alternatives above rather than a change that could be made to what is here.

**Only the near surface of an obstacle is ever observed.** A beam stops at the first
surface, so the interior and far side of a solid object are never measured and can only
be reported unknown. Of the 80 cells of the disc's footprint the parked control recovers
12 at every clamp setting, and the moving runs at the loosest clamps reach 22, so 58 of
the 80 are unobservable in principle rather than missed by the filter. Any figure that
counts unknown interior cells as mapping errors will be dominated by the geometry of the
scene.

**The pose correction has no memory.** The matcher aligns each sweep against the map
built from every earlier sweep and then integrates it. Nothing revisits a sweep once it
is in, so an error the matcher accepts is permanent, and returning to a place after a
long excursion cannot pull the earlier part of the map back into agreement. That is loop
closure and it needs a pose graph, which this repository does not have. The corridor
here is driven once end to end, so the case never arises in the measurements, which is
exactly why it is written down rather than demonstrated.

**The map is two dimensional and the ground is flat.** Every obstacle is a vertical
prism and the sensor scans one plane. A real bird's-eye-view free space estimator has to
decide which returns are ground and which are obstacles, on a surface that is not flat,
and that decision dominates its error budget. Nothing here addresses it.

**Cells are independent, so spurious occupied cells are not suppressed.** The 459 free
cells called occupied are single cells at building corners. The obvious cheap remedy, a
morphological opening on the decided map, was examined and rejected: the occupied set of
this map is a shell one cell thick, so an opening deletes all of it, and the spurious
cells touch the real facade so a connected component filter does not separate them
either. Removing them needs a spatial prior, which is the assumption the representation
gives up on purpose, and applying one would make the reported agreement a property of the
prior rather than of the sensor model.

**The decision threshold is bounded by the clamp.** With a free clamp at 0.12 no
threshold above 0.88 can ever call a cell free, and `LogOddsModel` raises an error rather
than silently accepting one. The threshold sweep therefore stops at 0.86. Reporting
agreement at a higher threshold would need a tighter clamp, which would change every
other number in the results.

**Resampling the ego window is not free even when it is lossless.** A snapped window
loses whatever falls off its trailing edge, which is why its decided fraction is 0.9516
against 0.9831 for a world fixed grid. That is the price of a bounded memory footprint
and it is a property of the window size, 200 by 200 cells here, rather than of the
policy.

**The simulator is not a lidar.** It has a single return per beam, no beam divergence, no
intensity, no multipath and no weather. Each of those changes the inverse sensor model,
and beam divergence in particular would make the occupied evidence span several cells at
range, which is the assumption the piecewise constant model makes least well.

## Closed limitations

### Poses are exact

**What it said.** There was no odometry model, no pose noise and no pose correction.
Evidence from different sweeps was guaranteed to land in the right cells, and the note
recorded that with a drifting pose the map would blur in a way none of the measurements
captured.

**What was closed.** All three. `pipeline/odometry.py` corrupts the body frame motion
between consecutive poses, so the estimate handed to the mapper drifts and compounds;
`algorithm/scan_match.py` corrects it by correlative scan to map matching; and
`examples/pose_drift.py` measures both against the exact pose run at four amplitudes.
The damage is now a number rather than a prediction. At a final drift of 1.76 metres the
occupied agreement falls from 0.9923 to 0.4168 while the free agreement is unchanged at
0.9867, which is the shape of the failure as well as its size: free space is a thick
region and survives being moved, a surface is one cell thick and does not. Correcting
the same run brings the final pose error to 0.123 metres and the occupied agreement back
to 0.9577.

**What it cost.** Three things, in descending order of importance.

The correction is not free even when there is nothing to correct. Run against an
odometry with every coefficient set to zero, the matcher still moves the final pose 0.141
metres off the truth and costs 4.1 points of occupied agreement, 0.9923 down to 0.9517.
It aligns each sweep with the discretised map rather than with the world, and the map's
occupied cells sit on the near surface of every obstacle, up to a cell away from where
the truth says the surface is. That bias is a floor rather than a residue of the input
error: the matcher finishes between 0.12 and 0.17 metres from the truth whether the
drift it started from was zero or 3.5 metres. Because of it, exact poses stay the default
for every other result in this repository, and the odometry path is skipped entirely
rather than run with its coefficients at zero, so no published number moved.

Two new modules, `pipeline/odometry.py` and `algorithm/scan_match.py`, one further
function from `scipy.ndimage`, a module the project already depended on, and roughly
double the runtime of a mapping run: 0.4 seconds becomes 0.9.

A search window, which is a parameter with a failure mode. The default searches four
cells and two degrees around the prediction, refined twice. An odometry worse than that
per step, or a run shortened enough that each step covers several metres, walks out of
the window and the correction stops helping. `examples/pose_drift.py` at a reduced step
count shows exactly that at the largest amplitude, which is left visible rather than
tuned away.

**What remains.** Loop closure, recorded as an open limitation above: the matcher never
revises a sweep it has already integrated, so it is localisation against a growing map
rather than SLAM. The odometry model is also Gaussian and zero mean, which real odometry
is not: a slipping wheel or a scale error is a bias, and a bias is the case a matcher
finds hardest because it points the same way every step.
