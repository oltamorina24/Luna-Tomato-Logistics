"""Adaptive large-neighborhood optimization for ESA SpOC4 Matching I/II.

Never confuses a feasible incumbent with a proven optimum.  Supports an optional
CP-SAT global warm-start and LP relaxation upper bound when dependencies exist.
"""
from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, linprog
from scipy.sparse import coo_matrix

from .matching import Transfer, score_solution


@dataclass
class Packed:
    e: np.ndarray
    l: np.ndarray
    d: np.ndarray
    w: np.ndarray
    e_rows: list[list[int]]
    l_rows: list[list[int]]
    d_rows: list[list[int]]
    e_count: int
    l_count: int
    d_count: int


def pack(rows: Sequence[Transfer]) -> Packed:
    """Compress resource IDs; original row order is preserved exactly."""
    es = {v: i for i, v in enumerate(sorted({r.e for r in rows}))}
    ls = {v: i for i, v in enumerate(sorted({r.l for r in rows}))}
    ds = {v: i for i, v in enumerate(sorted({r.d for r in rows}))}
    e = np.array([es[r.e] for r in rows], np.int32)
    l = np.array([ls[r.l] for r in rows], np.int32)
    d = np.array([ds[r.d] for r in rows], np.int32)
    w = np.array([r.weight for r in rows], float)
    e_rows = [[] for _ in es]
    l_rows = [[] for _ in ls]
    d_rows = [[] for _ in ds]
    for i in range(len(rows)):
        if w[i] > 0:
            e_rows[e[i]].append(i)
            l_rows[l[i]].append(i)
            d_rows[d[i]].append(i)
    return Packed(e, l, d, w, e_rows, l_rows, d_rows, len(es), len(ls), len(ds))


def owner_maps(p: Packed, selected: set[int]):
    oe = np.full(p.e_count, -1, np.int32)
    ol = np.full(p.l_count, -1, np.int32)
    od = np.full(p.d_count, -1, np.int32)
    for j in selected:
        if oe[p.e[j]] >= 0 or ol[p.l[j]] >= 0 or od[p.d[j]] >= 0:
            raise ValueError("Invalid incumbent: repeated resource")
        oe[p.e[j]] = j
        ol[p.l[j]] = j
        od[p.d[j]] = j
    return oe, ol, od


def _matrix(p: Packed, candidates: list[int]):
    n = len(candidates)
    ee = p.e[candidates]
    ll = p.l[candidates]
    dd = p.d[candidates]
    # Local compression keeps tiny and medium subproblems compact.
    _, e = np.unique(ee, return_inverse=True)
    _, l = np.unique(ll, return_inverse=True)
    _, d = np.unique(dd, return_inverse=True)
    ne, nl, nd = int(e.max() + 1), int(l.max() + 1), int(d.max() + 1)
    rr = np.empty(3*n, np.int32)
    rr[0::3] = e
    rr[1::3] = ne + l
    rr[2::3] = ne + nl + d
    cc = np.repeat(np.arange(n, dtype=np.int32), 3)
    A = coo_matrix((np.ones(3*n), (rr,cc)), shape=(ne+nl+nd,n)).tocsr()
    return A


def _restricted_milp(p: Packed, candidates: list[int], seconds: float,
                     rel_gap: float = .015):
    A = _matrix(p, candidates)
    n, m = len(candidates), A.shape[0]
    result = milp(c=-p.w[candidates], integrality=np.ones(n, np.uint8),
                  bounds=Bounds(np.zeros(n), np.ones(n)),
                  constraints=LinearConstraint(A, np.zeros(m), np.ones(m)),
                  options={"time_limit": max(.05, seconds), "mip_rel_gap": rel_gap,
                           "presolve": True})
    if result.x is None:
        return None
    return {candidates[j] for j in np.flatnonzero(result.x > .5)}


def lp_upper_bound(p: Packed, seconds: float = 30):
    """Full LP relaxation: theoretical upper bound only on successful solve."""
    ids = list(range(len(p.w)))
    n = len(ids)
    rr = np.empty(3*n, np.int32)
    rr[0::3] = p.e
    rr[1::3] = p.e_count + p.l
    rr[2::3] = p.e_count + p.l_count + p.d
    cc = np.repeat(np.arange(n, dtype=np.int32), 3)
    m = p.e_count + p.l_count + p.d_count
    A = coo_matrix((np.ones(3*n), (rr,cc)), shape=(m,n)).tocsr()
    t0 = time.perf_counter()
    res = linprog(-p.w, A_ub=A, b_ub=np.ones(m),
                  bounds=(0,1), method="highs",
                  options={"time_limit": seconds})
    out = {"status": int(res.status), "message": str(res.message),
           "seconds": time.perf_counter()-t0}
    if res.success:
        out["upper_bound"] = -float(res.fun)
        out["fractional_columns"] = int(np.count_nonzero((res.x > 1e-6) & (res.x < 1-1e-6)))
        # Column reduced cost can help rank promising alternatives.
        prices = -np.asarray(res.ineqlin.marginals)
        out["dual_prices"] = prices
    return out


def cp_sat_global(rows: Sequence[Transfer], incumbent: set[int],
                  seconds: float = 120, workers: int = 2,
                  seed: int = 42) -> tuple[set[int], dict]:
    """Optional: exact integer global model with incumbent hints (OR-Tools)."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return set(incumbent), {"status": "skipped: pip install ortools to enable CP-SAT"}
    p = pack(rows)
    model = cp_model.CpModel()
    n = len(rows)
    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
    for group in p.e_rows + p.l_rows + p.d_rows:
        if len(group) >= 2:
            model.AddAtMostOne(x[j] for j in group)
    # All weights in official data have three decimal places.
    model.Maximize(sum(int(round(p.w[j]*1000)) * x[j] for j in range(n)))
    for j in incumbent:
        model.AddHint(x[j], 1)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = max(1, workers)
    solver.parameters.random_seed = seed
    solver.parameters.relative_gap_limit = .0001
    t0 = time.perf_counter()
    status = solver.Solve(model)
    meta = {"status": solver.StatusName(status),
            "seconds": time.perf_counter() - t0}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return set(incumbent), meta
    selected = {j for j in range(n) if solver.BooleanValue(x[j])}
    score = score_solution(rows, sorted(selected))
    meta["score"] = score
    meta["best_bound"] = solver.BestObjectiveBound()/1000
    meta["optimal"] = status == cp_model.OPTIMAL
    return (selected if score > score_solution(rows, sorted(incumbent)) else set(incumbent)), meta


def adaptive_lns(rows: Sequence[Transfer], initial: set[int], *,
                 seconds: float = 240, rounds: int = 4000, seed: int = 43,
                 neighborhood_sizes: Sequence[int] = (55, 95, 160, 260),
                 candidate_caps: Sequence[int] = (900, 1600, 2600, 3800),
                 per_round_seconds: float = 1.2,
                 checkpoint: Callable[[set[int], float, int], None] | None = None,
                 verbose: bool = True) -> tuple[set[int], dict]:
    """Multi-scale, multi-seed MILP neighborhood search.

    Fixes all selection outside a neighborhood, exactly repairs a candidate set,
    and never discards the best feasible solution. Includes connected, random,
    low-mass, and high-contention neighborhoods to diversify search. On short
    local solves, feasible incumbents are accepted only after explicit checks.
    """
    if seconds <= 0 or rounds <= 0 or per_round_seconds <= 0:
        raise ValueError("Positive time/round budgets required")
    if len(neighborhood_sizes) != len(candidate_caps) or not neighborhood_sizes:
        raise ValueError("One candidate cap per neighborhood size is required")
    if any(x < 1 for x in neighborhood_sizes) or any(x < 10 for x in candidate_caps):
        raise ValueError("Neighborhoods/candidate caps must be positive")
    p = pack(rows)
    current = set(initial)
    start_score = score_solution(rows, sorted(current))
    current_score = best_score = start_score
    best = current.copy()
    rng = random.Random(seed)
    begin = time.perf_counter()
    attempts = improved = feasible = 0
    last_checkpoint = -1
    last_best_round = 0
    kicks = 0
    by_size = defaultdict(lambda: {"attempts": 0, "improvements": 0, "gain": 0.})
    # Pre-sort large lists once; do not create full edge lists every round.
    sorted_w = np.argsort(-p.w)

    while attempts < rounds and time.perf_counter()-begin < seconds:
        attempts += 1
        stage = rng.randrange(len(neighborhood_sizes))
        size = min(neighborhood_sizes[stage], len(current))
        cap = max(size + 5, candidate_caps[stage])
        strategy = rng.choice(("connected", "connected", "random", "weak", "star"))
        by_size[size]["attempts"] += 1
        if not size:
            break
        # Escaping local optima: after a long stall, perturb the working
        # solution while keeping the best submission untouched on disk.
        # A temporary lower-scoring state can open entirely new neighborhoods.
        if attempts - last_best_round > 260:
            trial_base = set(best)
            drop_n = max(3, min(len(trial_base)//15, 250))
            if len(trial_base) > drop_n:
                removed = set(rng.sample(sorted(trial_base), drop_n))
                trial_base -= removed
                be, bl, bd = owner_maps(p, trial_base)
                pool_kick = set(removed)
                for j in removed:
                    pool_kick.update(p.e_rows[p.e[j]])
                    pool_kick.update(p.l_rows[p.l[j]])
                    pool_kick.update(p.d_rows[p.d[j]])
                # Perturbed greedy rebuilding of the destroyed part.
                ordered = sorted(pool_kick,
                                 key=lambda j: p.w[j]*(.7+.6*rng.random()),
                                 reverse=True)
                for j in ordered:
                    if p.w[j] > 0 and be[p.e[j]] < 0 and bl[p.l[j]] < 0 and bd[p.d[j]] < 0:
                        trial_base.add(j)
                        be[p.e[j]] = bl[p.l[j]] = bd[p.d[j]] = j
                current = trial_base
                current_score = score_solution(rows, sorted(current))
                kicks += 1
                last_best_round = attempts
                if verbose:
                    print(f"Diversification kick #{kicks} at round {attempts}: "
                          f"working={current_score:.3f}, best={best_score:.3f}", flush=True)
        owners = owner_maps(p, current)
        oe, ol, od = owners
        incumbents = sorted(current)
        active: set[int] = set()

        if strategy in ("random", "weak"):
            if strategy == "weak":
                # 2/3 weaker edges, 1/3 random; shake across disconnected regions.
                weak = sorted(incumbents, key=lambda j: p.w[j])[:max(size*8, len(incumbents)//6)]
                nweak = min((size*2)//3, len(weak))
                active.update(rng.sample(weak, nweak))
            if len(active) < size:
                choices = [j for j in incumbents if j not in active]
                active.update(rng.sample(choices, min(size-len(active), len(choices))))
        else:
            pending = []
            if strategy == "star":
                for _ in range(15):
                    j = int(sorted_w[rng.randrange(min(2000, len(sorted_w)))])
                    if p.w[j] <= 0:
                        continue
                    for owner in (oe[p.e[j]], ol[p.l[j]], od[p.d[j]]):
                        if owner >= 0:
                            active.add(int(owner)); pending.append(int(owner))
                    if pending:
                        break
            if not pending:
                j = rng.choice(incumbents)
                active.add(j); pending.append(j)
            # Grow a connected component by visiting promising alternative edges.
            while len(active) < size:
                if not pending:
                    rest = [x for x in incumbents if x not in active]
                    if not rest:
                        break
                    j = rng.choice(rest)
                    active.add(j); pending.append(j)
                j = pending.pop(rng.randrange(len(pending)))
                adj = p.e_rows[p.e[j]] + p.l_rows[p.l[j]] + p.d_rows[p.d[j]]
                if len(adj) > 50:
                    adj = rng.sample(adj, 50)
                adj = sorted(adj, key=lambda i: -p.w[i])[:28]
                for k in adj:
                    for other in (oe[p.e[k]], ol[p.l[k]], od[p.d[k]]):
                        if other >= 0 and int(other) not in active:
                            active.add(int(other)); pending.append(int(other))
                            if len(active) >= size:
                                break
                    if len(active) >= size:
                        break
        if not active:
            continue

        # Candidates touching the affected resources; all incumbents retained.
        pool = set(active)
        for j in active:
            pool.update(p.e_rows[p.e[j]])
            pool.update(p.l_rows[p.l[j]])
            pool.update(p.d_rows[p.d[j]])
        ai = np.fromiter(active, dtype=np.int32)
        active_mask = np.zeros(len(p.w), dtype=bool)
        active_mask[ai] = True
        cand = np.fromiter(pool, dtype=np.int32)
        if len(cand) == 0:
            continue
        a = oe[p.e[cand]]
        b = ol[p.l[cand]]
        c = od[p.d[cand]]
        allowed = (p.w[cand] > 0) & ((a < 0) | active_mask[a.clip(min=0)]) & \
                  ((b < 0) | active_mask[b.clip(min=0)]) & \
                  ((c < 0) | active_mask[c.clip(min=0)])
        cand = cand[allowed]
        if len(cand) <= len(active):
            continue
        choices = set(int(j) for j in cand)
        choices.update(active)
        if len(choices) > cap:
            extra = np.array(list(choices - active), dtype=np.int32)
            # Approximate marginal benefit: how much mass is displaced by an edge.
            # Deduplicate owners, otherwise a row conflict on multiple resources
            # is mistakenly counted 2 or 3 times.
            ua = oe[p.e[extra]]
            ub = ol[p.l[extra]]
            uc = od[p.d[extra]]
            displaced = np.zeros(len(extra), float)
            for k in range(len(extra)):
                displaced[k] = sum(p.w[j] for j in {int(ua[k]), int(ub[k]), int(uc[k])} if j >= 0)
            quality = p.w[extra] - .33*displaced
            quality = quality + np.array([rng.random()*1.75 for _ in extra])
            keep = np.argpartition(quality, -min(cap-len(active), len(extra)))[-min(cap-len(active), len(extra)):]
            choices = set(active) | set(int(j) for j in extra[keep])
        candidates = sorted(choices)
        budget = min(per_round_seconds, seconds-(time.perf_counter()-begin))
        if budget < .1:
            break
        try:
            proposed = _restricted_milp(p, candidates, budget)
        except Exception:
            continue
        if proposed is None:
            continue
        trial = (current - active) | proposed
        # Explicit and independent legality check before accepting a change.
        try:
            new_score = score_solution(rows, sorted(trial))
        except ValueError:
            continue
        feasible += 1
        if new_score > current_score + 1e-7:
            gain = new_score-current_score
            current, current_score = trial, new_score
            improved += 1
            by_size[size]["gain"] += gain
            by_size[size]["improvements"] += 1
            is_new_best = new_score > best_score + 1e-7
            if is_new_best:
                best, best_score = trial.copy(), new_score
                last_best_round = attempts
            if verbose and is_new_best and (improved <= 10 or improved % 25 == 0):
                print(f"STRONG LNS #{improved}: {best_score:.3f} | round {attempts} "
                      f"| neighborhood {size} | gain {gain:.3f}", flush=True)
            if is_new_best and checkpoint is not None and (improved-last_checkpoint >= 20 or attempts % 50 == 0):
                checkpoint(best.copy(), best_score, attempts)
                last_checkpoint = improved
        if verbose and attempts % 100 == 0:
            print(f"Progress {attempts}/{rounds}: score={best_score:.3f} "
                  f"elapsed={time.perf_counter()-begin:.1f}s", flush=True)
    if checkpoint is not None:
        checkpoint(best.copy(), best_score, attempts)
    return best, {"method": "adaptive-multi-scale-strong-LNS", "seed": seed,
                  "start_score": start_score, "score": best_score,
                  "gain": best_score-start_score, "rounds": attempts,
                  "improvements": improved, "diversification_kicks": kicks,
                  "feasible_repairs": feasible,
                  "seconds": time.perf_counter()-begin,
                  "by_neighborhood": dict(by_size)}
