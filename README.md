# ESA SpOC-4 — Luna Tomato Logistics

An independently written starter research project for the ESA **SpOC-4: Luna Tomato Logistics** challenge.

**Status**
- Matching I (25,000 decisions): working weighted 3-dimensional matching heuristic + optional full MILP.
- Matching II (92,103 decisions): same algorithm, scales to its larger instance; runtime depends on available Kaggle CPU/time.
- Trajectory Matching (8,400 decisions): working output formatter, mass calculator and assignment optimizer over **already physics-validated** trajectory candidates. **Generating and validating physical Earth-to-Moon trajectories is NOT yet implemented.** It requires the ESA evaluator, orbital dynamics, trajectory shooting and optimization.
- **No leaderboard score is claimed**. Official datasets and the ESA evaluator are required for official validation and ranking.

## Sources
- Challenge: https://optimise.esa.int/challenge/spoc-4-luna-tomato-logistics/About
- Problems: https://optimise.esa.int/challenge/spoc-4-luna-tomato-logistics/Problems
- Official ESA repository: https://github.com/esa/SpOC4
- Matching I: https://optimise.esa.int/challenge/spoc-4-luna-tomato-logistics/p/matching-i
- Matching II: https://optimise.esa.int/challenge/spoc-4-luna-tomato-logistics/p/matching-ii
- Trajectory Matching: https://optimise.esa.int/challenge/spoc-4-luna-tomato-logistics/p/trajectory-matching

## 1. Prepare the input data

Open the challenge About page and scroll to **Instance Files**. Download the matching datasets and the orbit/rate files. Copy the Beginner files to:

```
data/matching-i.txt
data/matching-ii.txt
```

Keep the exact original row order. Each row must be:

```
earth_id moon_id destination_id mass
```

The Beginner data counts expected by the ESA problem pages are **25,000** and **92,103**. Their binary decision vectors must have exactly the same length as the rows of the respective files.

For Trajectory Matching, keep ESA's `Earth_orbits.txt`, `Moon_orbits.txt`, and `LTL.txt` in `data/` and use the *official* `trajectory-matching` evaluator to validate candidate trajectories. This repository does not bundle these datasets.

## 2. Run on a laptop or Kaggle terminal

```
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.cli --input data/sample_matching.txt --problem matching-i --method heuristic --restarts 8 --allow-demo-size
python -m src.cli --input data/matching-i.txt --problem matching-i --method hybrid --seconds 300 --restarts 20
python -m src.cli --input data/matching-ii.txt --problem matching-ii --method hybrid --seconds 600 --restarts 30
```

`hybrid` first finds a feasible multi-start greedy + local-search solution, then runs a full binary MILP through SciPy's HiGHS solver. If HiGHS gives a better feasible solution, keep it; otherwise preserve the heuristic. Its `mip_gap` is useful for checking how close the best feasible solution is to a mathematically certified bound. A status of *time limit* is **not** proof of optimality.

If you need speed rather than optimization quality:

```
python -m src.cli --input data/matching-ii.txt --problem matching-ii --method heuristic --restarts 40
```

Increase `--seconds` and `--restarts` and compare outcomes; performance is dataset-dependent.

## 3. Outputs

After a successful run:

```
results/submission_matching-i.json
results/selected_matching-i.csv
results/report_matching-i.json
```

And similarly for Matching II. JSON has exactly the ESA-required list containing `decisionVector`, `problem`, and `challenge`. The report contains a positive local mass score; the ESA score is its **negative**. The CSV uses zero-based row indexes matching the original input order.

**Never submit the demo JSON**: it does not have the official required length. Use the real dataset. Always verify official local fitness and submission on the Optimise platform before claiming a result.

## 4. Algorithm in one paragraph

Each feasible transfer `(E,L,D,w)` uses one Earth orbit, one lunar orbit and one destination and delivers `w` kilograms. Let `x_i=1` when row `i` is selected. Maximize `sum(w_i*x_i)` subject to (for every E, every L and every D) the sum of x's using that resource being at most one. `x_i` is binary. The full 3-way matching problem is NP-hard; standard 2D bipartite matching is **not** the same problem. SciPy HiGHS solves the integer linear programming model; local search gives fast feasible solutions.

## 5. Advanced: scope and next research milestones

A trajectory vector has 21 entries:

```
e_id l_id d_id t0 x0 y0 z0 vx0 vy0 vz0 DVx0 DVy0 DVz0 DVx1 DVy1 DVz1 DVx2 DVy2 DVz2 T1 T2
```

A fully padded vector has `400*21=8400` numbers; inactive rows have `e_id=-1`. The advanced problem checks BCP dynamics, departure/arrival orbit elements, flight duration, exclusion zones, velocity impulses and Earth/Moon/destination uniqueness. Correct **vector shape alone is never proof of physical validity**.

Recommended real development pipeline:
1. Download the official trajectory evaluator from the ESA problem page; inspect bounds, units, constraints and its example.
2. Use an isolated Linux/conda env to install `pykep 2.6.x`, `heyoka` and any dependencies exactly supported by the evaluator (ESA discussions highlight that `pykep` may not be `pip`-installable and that versions can change results).
3. Reproduce the evaluator's known example precisely (the same fitness/validity).
4. Parameterize departure state on an allowed Earth orbit and design 2-3 impulse trajectories; propagate each arc in the BCP model.
5. Minimize transfer delta-V with differential evolution / CMA-ES / multiple shooting and constrained local refinement while targeting Moon orbit `(a,e,i)` within tolerances and avoiding forbidden regions.
6. Cache ESA-validated candidate 21-element trajectories for Earth/Moon pairs; store duration, delta-V and evaluated mass separately.
7. Optimize the final candidate matching over `(Earth, Moon, Destination)` using `advanced/assemble.py` and re-evaluate the **entire final vector** with the official evaluator.

The official Advanced score is `sum(min(5000*exp(-deltaV/(311*9.80665))-500, (200-duration_days)*c_ld))` for feasible transfers. The shortcut `advanced/trajectory.py::delivered_mass` assumes a physically valid candidate and cannot validate or generate one.

After independently validating trajectory candidates with ESA code, save an array like:

```
[
 {"vector": [21 numeric values], "delivered_mass": 1234.5, "validated_officially": true}
]
```

Then run:

```
python -m advanced.assemble --candidates data/validated_candidates.json --seconds 300
```

This is the **selection/assembly** stage only. Zero or fake candidates cannot become valid trajectories by running this command.

## 6. Kaggle workflow

1. Sign in to Kaggle; select **Code → New Notebook → Python**.
2. Set **Accelerator = None** (CPU). Use **Internet = On** only if cloning from GitHub or installing packages; Kaggle controls may depend on account verification.
3. Upload the repository as a Kaggle Dataset or clone your new public GitHub repository in a notebook cell.
4. Upload the ESA instance files as Kaggle data (`Add Input`), keeping their row order intact.
5. Run tests, inspect the row counts, run Matching I and II with `python -m src.cli ...`, inspect score/validity and output JSON.
6. Use Kaggle **Save Version → Save & Run All** to reproduce all cells from scratch, then download the generated JSON from the notebook's Output pane.

Example notebook template: `notebooks/01_kaggle_matching.ipynb`.

## 7. GitHub workflow

Open a new **public** GitHub repository (suggested name `ESA-SpOC4-Luna-Tomato-Logistics`), then from the project directory:

```
git init
git add .
git commit -m "Initial ESA SpOC4 matching research project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ESA-SpOC4-Luna-Tomato-Logistics.git
git push -u origin main
```

For later changes use `git add .`, `git commit -m "Improve matching solver"`, `git push`.

Document which datasets/seeds/solver version produced each published score. Do not commit account tokens. Consider not uploading huge experimental result blobs. Check licenses and cite ESA sources properly.

## 8. Competition deadline

Official SpOC 4 scoring ended **June 30, 2026**. ESA's README says post-deadline submissions still appear on public leaderboards, but cannot change official competition results.
