"""Three-dimensional weighted matching for the ESA SpOC-4 beginner tasks.

Each input row is e_id, l_id, d_id, mass; the row order defines its binary
variable index. Never reorder rows when constructing the submission vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
import time
from typing import Sequence


@dataclass(frozen=True)
class Transfer:
    e: int
    l: int
    d: int
    weight: float


def load_transfers(path: str | Path) -> list[Transfer]:
    out: list[Transfer] = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 4:
                raise ValueError(f"Line {lineno}: 4 columns required, found {len(parts)}")
            try:
                ef, lf, df = (float(x) for x in parts[:3])
                if not all(z.is_integer() for z in (ef, lf, df)):
                    raise ValueError("orbit/destination ids must be integers")
                weight = float(parts[3])
                if not math.isfinite(weight):
                    raise ValueError("non-finite weight")
            except ValueError as exc:
                raise ValueError(f"Invalid data on line {lineno}: {exc}") from exc
            out.append(Transfer(int(ef), int(lf), int(df), weight))
    if not out:
        raise ValueError("No transfers read")
    return out


def score_solution(rows: Sequence[Transfer], selected: Sequence[int]) -> float:
    seen_e, seen_l, seen_d, used = set(), set(), set(), set()
    score = 0.0
    for k in selected:
        if not isinstance(k, int) or k < 0 or k >= len(rows) or k in used:
            raise ValueError(f"Invalid/duplicate selected transfer index {k}")
        t = rows[k]
        if t.e in seen_e or t.l in seen_l or t.d in seen_d:
            raise ValueError(f"Resource conflict for row {k}: {t}")
        seen_e.add(t.e)
        seen_l.add(t.l)
        seen_d.add(t.d)
        used.add(k)
        score += t.weight
    return score


def make_vector(size: int, selected: Sequence[int]) -> list[int]:
    x = [0] * size
    for k in selected:
        if not 0 <= k < size or x[k]:
            raise ValueError(f"invalid or duplicate index {k}")
        x[k] = 1
    return x


def greedy(rows: Sequence[Transfer], rng: random.Random | None = None,
           noise: float = 0.0) -> set[int]:
    """Sorted greedy; noise randomly perturbs scores between restarts."""
    if rng is None or noise == 0:
        order = sorted(range(len(rows)), key=lambda i: rows[i].weight, reverse=True)
    else:
        order = sorted(range(len(rows)),
                       key=lambda i: rows[i].weight * (1 + noise * (2 * rng.random() - 1)),
                       reverse=True)
    occupied_e, occupied_l, occupied_d = set(), set(), set()
    chosen: set[int] = set()
    for i in order:
        t = rows[i]
        if t.weight <= 0:
            continue
        if t.e in occupied_e or t.l in occupied_l or t.d in occupied_d:
            continue
        chosen.add(i)
        occupied_e.add(t.e)
        occupied_l.add(t.l)
        occupied_d.add(t.d)
    return chosen


def local_search(rows: Sequence[Transfer], chosen: set[int],
                 max_passes: int = 8) -> set[int]:
    """Fast 1-for-up-to-3 conflicting edges interchange + repair.

    Each candidate removes at most 3 currently selected conflicting edges;
    improvements are accepted only if their total weight increases.
    """
    selected = set(chosen)
    order = sorted(range(len(rows)), key=lambda i: rows[i].weight, reverse=True)
    for _ in range(max_passes):
        owners_e = {rows[i].e: i for i in selected}
        owners_l = {rows[i].l: i for i in selected}
        owners_d = {rows[i].d: i for i in selected}
        improved = False
        for j in order:
            if j in selected or rows[j].weight <= 0:
                continue
            t = rows[j]
            conflicting = {k for k in
                           (owners_e.get(t.e), owners_l.get(t.l), owners_d.get(t.d))
                           if k is not None}
            if t.weight > sum(rows[k].weight for k in conflicting) + 1e-10:
                for k in conflicting:
                    selected.remove(k)
                    old = rows[k]
                    del owners_e[old.e]
                    del owners_l[old.l]
                    del owners_d[old.d]
                selected.add(j)
                owners_e[t.e] = j
                owners_l[t.l] = j
                owners_d[t.d] = j
                improved = True
        if not improved:
            break
    score_solution(rows, list(selected))
    return selected


def heuristic(rows: Sequence[Transfer], restarts: int = 20,
              seed: int = 2026, max_passes: int = 6) -> tuple[set[int], dict]:
    rng = random.Random(seed)
    best = set()
    best_score = -float('inf')
    t0 = time.perf_counter()
    for run in range(max(1, restarts)):
        noise = 0.0 if run == 0 else rng.choice([0.04, 0.10, 0.20, 0.35, 0.50])
        candidate = local_search(rows, greedy(rows, rng, noise), max_passes)
        value = score_solution(rows, sorted(candidate))
        if value > best_score:
            best, best_score = candidate, value
    return best, {"method": "greedy+multistart+local-search", "score": best_score,
                  "restarts": max(1, restarts), "seconds": time.perf_counter()-t0}


def milp(rows: Sequence[Transfer], seconds: float = 300, relative_gap: float = 0.0001):
    """Solve full 3D matching MILP with SciPy/HiGHS.

    Returns (feasible chosen indices or None, metadata). HiGHS may time out
    before any feasible solution; callers should keep a heuristic fallback.
    """
    import numpy as np
    from scipy.optimize import milp as scipy_milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix

    n = len(rows)
    e_ids = {x: i for i, x in enumerate(sorted({t.e for t in rows}))}
    l_ids = {x: i + len(e_ids) for i, x in enumerate(sorted({t.l for t in rows}))}
    d_ids = {x: i + len(e_ids) + len(l_ids) for i, x in enumerate(sorted({t.d for t in rows}))}
    m = len(e_ids) + len(l_ids) + len(d_ids)
    ind = np.arange(n, dtype=np.int32)
    r = np.empty(3*n, dtype=np.int32)
    r[0::3] = [e_ids[t.e] for t in rows]
    r[1::3] = [l_ids[t.l] for t in rows]
    r[2::3] = [d_ids[t.d] for t in rows]
    c = np.repeat(ind, 3)
    A = coo_matrix((np.ones(3*n), (r,c)), shape=(m,n)).tocsc()
    constraint = LinearConstraint(A, np.zeros(m), np.ones(m))
    t0 = time.perf_counter()
    res = scipy_milp(c=-np.array([t.weight for t in rows], dtype=float),
                     integrality=np.ones(n, dtype=np.int8),
                     bounds=Bounds(np.zeros(n), np.ones(n)),
                     constraints=constraint,
                     options={"time_limit": seconds, "mip_rel_gap": relative_gap})
    meta = {"method":"scipy-highs-milp", "status": int(res.status),
            "message": str(res.message), "seconds": time.perf_counter()-t0,
            "mip_gap": float(res.mip_gap) if getattr(res,'mip_gap',None) is not None else None,
            "dual_bound": -float(res.mip_dual_bound) if getattr(res,'mip_dual_bound',None) is not None else None}
    if res.x is None:
        return None, meta
    candidate = set(np.flatnonzero(res.x > 0.5).astype(int).tolist())
    try:
        meta["score"] = score_solution(rows, list(candidate))
    except ValueError as exc:
        meta["error"] = str(exc)
        return None, meta
    return candidate, meta


def solve(rows: Sequence[Transfer], method: str, seconds: float,
          restarts: int, seed: int, relative_gap: float = 1e-4):
    candidate, hm = heuristic(rows, restarts=restarts, seed=seed)
    details = {"heuristic": hm}
    if method in ("milp", "hybrid"):
        exact, em = milp(rows, seconds=seconds, relative_gap=relative_gap)
        details["milp"] = em
        if exact is not None and score_solution(rows,list(exact)) > score_solution(rows,list(candidate)):
            candidate = exact
    elif method != "heuristic":
        raise ValueError(f"Unknown method {method}")
    details["winner_score"] = score_solution(rows,list(candidate))
    details["n_selected"] = len(candidate)
    return candidate, details
