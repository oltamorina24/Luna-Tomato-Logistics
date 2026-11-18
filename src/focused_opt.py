"""Focused CP-SAT / HiGHS neighborhood repair V3 for weighted 3D matching.

Unlike global CP-SAT, restricts repairs to selected transfer neighborhoods and
unselected transfers that can be added without conflicting with fixed rows.
All proposed solutions are checked by the independent scoring function.
"""
from __future__ import annotations

from collections import defaultdict
import random
import time
from typing import Callable, Sequence

import numpy as np

from .matching import Transfer, score_solution
from .strong_opt import Packed, pack, owner_maps, _restricted_milp


def _make_neighborhood(p: Packed, incumbent: set[int], owners, rng: random.Random,
                       size: int, cap: int, best_rows: list[int], n_candidates: int):
    oe, ol, od = owners
    selected = sorted(incumbent)
    if not selected:
        return set(), []
    size = min(size, len(selected))
    cap = max(size + 50, cap)
    active: set[int] = set()
    frontier: list[int] = []
    # Opportunity-based anchor selection.  High-mass transfers are not
    # necessarily promising if they displace even higher-mass incumbents.
    # Sample unselected rows and prefer near-profitable alternatives, with
    # randomness to keep the search diverse between rounds.
    best_anchor = None
    best_quality = -float("inf")
    for _ in range(96):
        if rng.random() < .6 and best_rows:
            j = best_rows[rng.randrange(min(len(best_rows), n_candidates))]
        else:
            j = rng.randrange(len(p.w))
        if j in incumbent or p.w[j] <= 0:
            continue
        displaced = {
            int(k) for k in (oe[p.e[j]], ol[p.l[j]], od[p.d[j]]) if k >= 0
        }
        if not displaced:
            # Positive independent candidate: include it even if there is
            # no incumbent conflict, by anchoring at a nearby selected row.
            continue
        loss = sum(float(p.w[k]) for k in displaced)
        quality = float(p.w[j]) - loss + rng.uniform(-1.5, 1.5)
        if quality > best_quality:
            best_quality = quality
            best_anchor = displaced
    if best_anchor:
        active.update(best_anchor)
        frontier.extend(best_anchor)
    if not active:
        j = rng.choice(selected)
        active.add(j)
        frontier.append(j)

    # Mix graph-connected selection with occasional weak edges to diversify.
    while len(active) < size:
        if frontier and rng.random() < .9:
            j = frontier.pop(rng.randrange(len(frontier)))
            adj = p.e_rows[p.e[j]] + p.l_rows[p.l[j]] + p.d_rows[p.d[j]]
            adj = rng.sample(adj, min(len(adj), 45)) if len(adj) > 45 else adj
            for x in sorted(adj, key=lambda a: -p.w[a])[:20]:
                for k in (oe[p.e[x]], ol[p.l[x]], od[p.d[x]]):
                    if k >= 0 and int(k) not in active:
                        active.add(int(k))
                        frontier.append(int(k))
                        if len(active) >= size:
                            break
                if len(active) >= size:
                    break
        else:
            # Intentionally distribute some repairs across graph components.
            j = rng.choice(selected)
            if j not in active:
                active.add(j)
                frontier.append(j)

    a = np.fromiter(active, dtype=np.int32)
    affected = np.zeros(len(p.w), dtype=bool)
    affected[a] = True
    pool = set(active)
    for j in active:
        pool.update(p.e_rows[p.e[j]])
        pool.update(p.l_rows[p.l[j]])
        pool.update(p.d_rows[p.d[j]])
    cand = np.fromiter(pool, dtype=np.int32)
    if not len(cand):
        return active, sorted(active)
    x, y, z = oe[p.e[cand]], ol[p.l[cand]], od[p.d[cand]]
    allowed = ((x < 0) | affected[np.maximum(x, 0)]) & \
              ((y < 0) | affected[np.maximum(y, 0)]) & \
              ((z < 0) | affected[np.maximum(z, 0)]) & (p.w[cand] > 0)
    eligible = set(map(int, cand[allowed])) | active
    if len(eligible) > cap:
        extra = list(eligible - active)
        # Opportunity-aware candidate ranking, with randomized diversity.
        # Count each displaced incumbent only once even if two resources clash.
        def quality(j):
            displaced = {
                int(k) for k in (oe[p.e[j]], ol[p.l[j]], od[p.d[j]]) if k >= 0
            }
            loss = sum(float(p.w[k]) for k in displaced)
            return float(p.w[j]) - 0.38 * loss + rng.uniform(-1.5, 1.5)
        extra.sort(key=quality, reverse=True)
        quota = cap - len(active)
        high = min(len(extra), int(.75 * quota))
        chosen = extra[:high]
        remainder = extra[high:]
        chosen += rng.sample(remainder, min(len(remainder), quota - high))
        eligible = active | set(chosen)
    return active, sorted(eligible)


def _cp_sat_repair(p: Packed, active: set[int], candidates: list[int],
                   seconds: float, seed: int, workers: int):
    from ortools.sat.python import cp_model
    model = cp_model.CpModel()
    vars_ = [model.NewBoolVar(f"r{k}") for k in range(len(candidates))]
    e_groups, l_groups, d_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for idx, row in enumerate(candidates):
        e_groups[int(p.e[row])].append(vars_[idx])
        l_groups[int(p.l[row])].append(vars_[idx])
        d_groups[int(p.d[row])].append(vars_[idx])
        model.AddHint(vars_[idx], int(row in active))
    for group in (*e_groups.values(), *l_groups.values(), *d_groups.values()):
        if len(group) > 1:
            model.AddAtMostOne(group)
    weights = [int(round(float(p.w[j]) * 1000)) for j in candidates]
    objective = sum(v * w for v, w in zip(vars_, weights))
    # Keep the incumbent feasible inside every subproblem.  The previous
    # version imposed objective >= incumbent + 1, which intentionally
    # returned INFEASIBLE when no improving swap existed.  Removing it
    # distinguishes a valid neighborhood with no gain from a broken model.
    # Acceptance of a new submission still requires strict improvement.
    model.Maximize(objective)
    validator = getattr(model, "Validate", None) or getattr(model, "validate", None)
    error = validator() if validator else ""
    if error:
        return None, "MODEL_INVALID: " + error[:160]
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(.05, seconds)
    solver.parameters.num_search_workers = max(1, workers)
    solver.parameters.random_seed = int(seed)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        name = solver.StatusName(status)
        if status == cp_model.INFEASIBLE:
            name = "UNEXPECTED_INFEASIBLE"  # all-zero assignment exists
        return None, name
    return {candidates[k] for k, var in enumerate(vars_) if solver.BooleanValue(var)}, solver.StatusName(status)


def focused_search(rows: Sequence[Transfer], initial: set[int], *,
                   seconds: float = 300, rounds: int = 120, seed: int = 2026,
                   sizes: Sequence[int] = (120, 260, 450),
                   caps: Sequence[int] = (1200, 2400, 3800),
                   per_round: float = 3., workers: int = 2,
                   method: str = "cpsat",
                   checkpoint: Callable[[set[int], float, int], None] | None = None,
                   verbose: bool = True):
    if method not in ("cpsat", "milp", "hybrid"):
        raise ValueError("method must be cpsat, milp or hybrid")
    if method in ('cpsat', 'hybrid'):
        try:
            from ortools.sat.python import cp_model  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("CP-SAT needs OR-Tools; use --method milp instead") from exc
    if not sizes or len(sizes) != len(caps) or any(x <= 0 for x in sizes + tuple(caps)):
        raise ValueError('Provide one positive candidate cap per neighborhood')
    best = set(initial)
    best_score = score_solution(rows, sorted(best))
    t0 = time.perf_counter()
    p = pack(rows)
    rng = random.Random(seed)
    top = [int(j) for j in np.argsort(-p.w) if p.w[j] > 0]
    per_size = defaultdict(lambda: dict(attempts=0, improvements=0, gain=0.))
    statuses = defaultdict(int)
    attempts = improvements = 0
    while attempts < rounds and time.perf_counter() - t0 < seconds:
        budget = min(per_round, seconds - (time.perf_counter() - t0))
        if budget < .1:
            break
        stage = attempts % len(sizes) if attempts < len(sizes) * 2 else rng.randrange(len(sizes))
        attempts += 1
        size = sizes[stage]
        per_size[size]['attempts'] += 1
        owners = owner_maps(p, best)
        active, candidates = _make_neighborhood(p, best, owners, rng, size,
                                                caps[stage], top, 8000)
        if len(candidates) <= len(active):
            continue
        # The zero vector is feasible for every model consisting only of
        # AtMostOne constraints.  Therefore CP-SAT INFEASIBLE is a diagnostic
        # anomaly, not evidence that this local search is optimal.
        # Hybrid mode tries CP-SAT briefly, then falls back to HiGHS/MILP.
        try:
            if method == "cpsat":
                result, status = _cp_sat_repair(p, active, candidates, budget,
                                                seed + attempts, workers)
            elif method == "milp":
                result = _restricted_milp(p, candidates, budget, rel_gap=0.02)
                status = "HIGHS_FEASIBLE" if result is not None else "HIGHS_NO_SOLUTION"
            else:
                cp_budget = min(1.0, max(0.15, budget * 0.3))
                start_sub = time.perf_counter()
                result, cp_status = _cp_sat_repair(p, active, candidates, cp_budget,
                                                    seed + attempts, workers)
                statuses["cpsat_" + cp_status] += 1
                # If CP-SAT did not prove optimality, HiGHS is an independent
                # repair attempt. Keep whichever local assignment scores higher.
                if cp_status != "OPTIMAL":
                    left = max(.1, budget - (time.perf_counter()-start_sub))
                    milp_result = _restricted_milp(p, candidates, left, rel_gap=.02)
                    if milp_result is not None:
                        statuses["highs_feasible"] += 1
                        cp_value = sum(float(p.w[j]) for j in result) if result is not None else -1
                        milp_value = sum(float(p.w[j]) for j in milp_result)
                        if milp_value > cp_value + 1e-8:
                            result = milp_result
                            status = "HIGHS_REPAIR"
                        else:
                            status = "CPSAT_" + cp_status
                    else:
                        status = "CPSAT_" + cp_status
                else:
                    status = "CPSAT_OPTIMAL"
        except (ValueError, RuntimeError) as exc:
            statuses['error: '+type(exc).__name__] += 1
            continue
        if result is None:
            statuses[status] += 1
            continue
        proposed = (best - active) | result
        try:
            value = score_solution(rows, sorted(proposed))
        except ValueError:
            statuses['invalid_proposal'] += 1
            continue
        if value <= best_score + 1e-7:
            statuses[status + '_NO_GAIN'] += 1
            continue
        statuses[status] += 1
        if value > best_score + 1e-7:
            gain = value - best_score
            best, best_score = proposed, value
            improvements += 1
            per_size[size]['improvements'] += 1
            per_size[size]['gain'] += gain
            if checkpoint:
                checkpoint(best.copy(), best_score, attempts)
            if verbose:
                print(f"FOCUSED {method} improvement #{improvements}: "
                      f"score={best_score:.3f} round={attempts} size={size} "
                      f"gain={gain:.3f}", flush=True)
        if verbose and attempts % 10 == 0:
            print(f"FOCUSED progress {attempts}/{rounds}: "
                  f"best={best_score:.3f}, elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return best, dict(method=f"focused-{method}-neighborhood", seed=seed,
                      score=best_score, attempts=attempts, improvements=improvements,
                      seconds=time.perf_counter()-t0, statuses=dict(statuses),
                      by_size=dict(per_size))
