"""Elite solution pool + MILP/CP-SAT recombination for ESA SPOC-4 Matching.

Usage: python -m src.pool_merge_cli --problem matching-ii
Existing submission is the only checkpoint authority; never overwrite on loss.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .matching import load_transfers, make_vector, score_solution
from .strong_opt import pack, owner_maps, adaptive_lns, _restricted_milp
from .cli import read_incumbent, save_atomic, CHALLENGE


def perturb(p, incumbent: set[int], drop: int, rng: random.Random) -> set[int]:
    """Destroy incumbent around profitable unselected competitors and rebuild noisily."""
    if not incumbent:
        return set()
    drop = min(drop, len(incumbent))
    oe, ol, od = owner_maps(p, incumbent)
    n = len(p.w)
    # Score alternative edges by weight minus UNIQUE incumbent displacement cost.
    probe = rng.sample(range(n), min(n, 11000))
    ranked = []
    for j in probe:
        if j in incumbent or p.w[j] <= 0:
            continue
        blocked = {int(k) for k in (oe[p.e[j]], ol[p.l[j]], od[p.d[j]]) if k >= 0}
        if blocked:
            cost = sum(float(p.w[k]) for k in blocked)
            ranked.append((float(p.w[j]) - cost + rng.uniform(-3., 3.), blocked))
    ranked.sort(key=lambda item: item[0], reverse=True)

    removed: set[int] = set()
    guided = int(drop * 0.70)
    # Limit each seed's influence; spread across separate conflict regions.
    for _, blockers in ranked:
        if len(removed) >= guided:
            break
        available = sorted(blockers - removed)
        if len(available) > guided - len(removed):
            available = rng.sample(available, guided-len(removed))
        removed.update(available)
    if len(removed) < drop:
        remain = sorted(incumbent - removed)
        removed.update(rng.sample(remain, min(drop-len(removed), len(remain))))

    trial = set(incumbent) - removed
    oe, ol, od = owner_maps(p, trial)
    near = set(removed)
    for k in removed:
        near.update(p.e_rows[p.e[k]])
        near.update(p.l_rows[p.l[k]])
        near.update(p.d_rows[p.d[k]])

    # No hard refusal of incumbent edges: they can be reinserted if profitable.
    order = sorted((j for j in near if p.w[j] > 0),
                   key=lambda j: float(p.w[j]) * rng.uniform(.62, 1.38),
                   reverse=True)
    for j in order:
        if oe[p.e[j]] < 0 and ol[p.l[j]] < 0 and od[p.d[j]] < 0:
            trial.add(int(j))
            oe[p.e[j]] = j
            ol[p.l[j]] = j
            od[p.d[j]] = j
    return trial


def merge_pool(p, pool: list[set[int]], incumbent: set[int], seconds: float,
               seed: int, workers: int = 2, method: str = 'auto', cap: int = 25000):
    """Optimize the UNION of several feasible solutions, with unchanged row ids.

    Includes all incumbent rows; if the union is large, subsamples *extras*.
    Exact optimality is only claimed if the CP-SAT solver proves it.
    """
    base = set(incumbent)
    union = base.union(*pool) if pool else base
    extras = sorted(union - base)
    rng = random.Random(seed)
    if len(union) > cap:
        # Sample alternatives across the entire elite pool; preserve incumbent.
        # All weight values remain positive; all row IDs are original IDs.
        extras = rng.sample(extras, max(0, cap - len(base)))
    ids = sorted(base.union(extras))
    meta = {'method': method, 'variables': len(ids), 'pool_members': len(pool),
            'union_before_cap': len(union)}
    if len(ids) == len(base):
        meta['status'] = 'NO_ALTERNATIVES'
        return base, meta
    t0 = time.perf_counter()
    if method in ('auto', 'cpsat'):
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            if method == 'cpsat':
                raise RuntimeError('OR-Tools is not installed')
        else:
            model = cp_model.CpModel()
            x = [model.NewBoolVar(f'x{i}') for i in range(len(ids))]
            groups_e, groups_l, groups_d = {}, {}, {}
            for i, j in enumerate(ids):
                groups_e.setdefault(int(p.e[j]), []).append(x[i])
                groups_l.setdefault(int(p.l[j]), []).append(x[i])
                groups_d.setdefault(int(p.d[j]), []).append(x[i])
                model.AddHint(x[i], int(j in base))
            for group in (*groups_e.values(), *groups_l.values(), *groups_d.values()):
                if len(group) > 1:
                    model.AddAtMostOne(group)
            model.Maximize(sum(int(round(float(p.w[j])*1000)) * x[i]
                               for i, j in enumerate(ids)))
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = max(.2, seconds)
            solver.parameters.num_search_workers = max(1, workers)
            solver.parameters.random_seed = seed
            status = solver.Solve(model)
            meta.update(method='cpsat', status=solver.StatusName(status),
                        seconds=time.perf_counter()-t0)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                result = {ids[i] for i in range(len(ids)) if solver.BooleanValue(x[i])}
                return result, meta
            return base, meta
    # HiGHS fallback; may not find a feasible primal before its deadline.
    result = _restricted_milp(p, ids, seconds=max(.2, seconds), rel_gap=.005)
    meta.update(method='milp', status=('FEASIBLE' if result is not None else 'NO_SOLUTION'),
                seconds=time.perf_counter()-t0)
    return (result if result is not None else base), meta


def run(args):
    root = Path(args.root).expanduser().resolve()
    infile = root / 'data' / f'{args.problem}.txt'
    target = root / 'results' / f'submission_{args.problem}.json'
    assert infile.exists() and target.exists(), 'Project files missing: restore the backup first'
    rows = load_transfers(infile)
    expected = {'matching-i': 25000, 'matching-ii': 92103}[args.problem]
    if not args.allow_demo_size:
        assert len(rows) == expected, f'Expected {expected} rows, got {len(rows)}'
    best = read_incumbent(target, len(rows), args.problem)
    assert best is not None, 'Missing or wrong incumbent JSON'
    best = set(best)
    best_score = score_solution(rows, sorted(best))
    p = pack(rows)
    backup = target.with_name(f'before_pool_merge_{args.problem}_{int(time.time())}.json')
    shutil.copy2(target, backup)
    print(f'BASELINE: {best_score:.3f} | backup: {backup}', flush=True)
    print('PO ATTAKOJME SYNIMIN:', args.target_score, flush=True)

    rng = random.Random(args.seed)
    archive = [set(best)]
    clock = time.perf_counter()
    improvements = 0

    def accept(candidate: set[int], source: str):
        nonlocal best, best_score, improvements
        score = score_solution(rows, sorted(candidate))
        if score <= best_score + 1e-7:
            return False
        on_disk = read_incumbent(target, len(rows), args.problem)
        assert on_disk is not None
        disk_score = score_solution(rows, sorted(on_disk))
        if score <= disk_score + 1e-7:
            best, best_score = set(on_disk), disk_score
            return False
        document = [{'decisionVector': make_vector(len(rows), sorted(candidate)),
                     'problem': args.problem, 'challenge': CHALLENGE}]
        save_atomic(target, json.dumps(document, separators=(',', ':')))
        best, best_score = set(candidate), score
        improvements += 1
        print(f'*** NEW RECORD [{source}]: {best_score:.3f} | gain since start: '
              f'{best_score - initial_score:.3f}', flush=True)
        return True

    initial_score = best_score
    for trial in range(1, args.restarts + 1):
        n_drop = args.drop_sizes[(trial - 1) % len(args.drop_sizes)]
        # On some rounds start from a different elite, promoting diversity.
        source = best if trial % 3 != 0 else rng.choice(archive)
        start = perturb(p, source, n_drop, rng)
        start_score = score_solution(rows, sorted(start))
        print(f'\nRESTART {trial}/{args.restarts}: drop={n_drop} | start={start_score:.3f}', flush=True)
        cand, details = adaptive_lns(
            rows, start, seconds=args.lns_seconds, rounds=args.lns_rounds,
            seed=args.seed + 131 * trial,
            neighborhood_sizes=(550, 950, 1600, 2300),
            candidate_caps=(6500, 10500, 16000, 22000),
            per_round_seconds=2.0, verbose=False)
        value = score_solution(rows, sorted(cand))
        print(f'POST LNS {trial}: {value:.3f} | iterations={details["rounds"]}', flush=True)
        # Keep the *whole solution* in the archive even if it was slightly worse.
        if all(len(cand.symmetric_difference(other)) >= args.min_diversity for other in archive):
            archive.append(set(cand))
        accept(set(cand), f'LNS restart {trial}')
        # Include best in every merge, even if it overlaps with archived elites.
        if trial % args.merge_every == 0 or trial == args.restarts:
            merged, meta = merge_pool(
                p, archive, best, seconds=args.merge_seconds,
                seed=args.seed + 1013 * trial, workers=args.workers,
                method=args.merge_method, cap=args.merge_cap)
            merged_score = score_solution(rows, sorted(merged))
            print(f'MERGE {trial}: {merged_score:.3f} | {meta}', flush=True)
            accept(set(merged), f'pool merge {trial}')
            # A strong merged solution often benefits from a final local repair.
            if args.polish_seconds > 0:
                polished, info = adaptive_lns(
                    rows, best, seconds=args.polish_seconds,
                    rounds=args.lns_rounds,
                    seed=args.seed + 1061 * trial,
                    neighborhood_sizes=(450, 900, 1700, 2300),
                    candidate_caps=(5500, 9500, 17500, 22000),
                    per_round_seconds=2.0, verbose=False)
                print(f'POLISH {trial}: {score_solution(rows, sorted(polished)):.3f}', flush=True)
                accept(set(polished), f'polish {trial}')
        if best_score > args.target_score + 1e-7:
            print('TARGET SURPASSED LOCALLY!', flush=True)
            break

    # Re-read from disk: authoritative durable output.
    final = read_incumbent(target, len(rows), args.problem)
    assert final is not None
    end_score = score_solution(rows, sorted(final))
    print('\n' + '='*40)
    print(f'WINNER SCORE: {end_score:.3f}')
    print(f'TARGET: {args.target_score:.3f}')
    print(f'GAP TO TARGET: {args.target_score - end_score:.3f}')
    print(f'IMPROVEMENTS SAVED: {improvements}')
    print(f'ELAPSED: {time.perf_counter() - clock:.1f}s')
    print(f'SUBMISSION: {target}')
    print('='*40, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-demo-size', action='store_true')
    ap.add_argument('--root', default='/kaggle/working/Luna-Tomato-Logistics')
    ap.add_argument('--problem', choices=['matching-i', 'matching-ii'], default='matching-ii')
    ap.add_argument('--target-score', type=float, default=73805.590)
    ap.add_argument('--restarts', type=int, default=6)
    ap.add_argument('--drop-sizes', type=int, nargs='+', default=[450, 900, 1500])
    ap.add_argument('--lns-seconds', type=float, default=90)
    ap.add_argument('--lns-rounds', type=int, default=3000)
    ap.add_argument('--merge-every', type=int, default=2)
    ap.add_argument('--merge-seconds', type=float, default=100)
    ap.add_argument('--polish-seconds', type=float, default=45)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--merge-method', choices=['auto', 'cpsat', 'milp'], default='auto')
    ap.add_argument('--merge-cap', type=int, default=24000)
    ap.add_argument('--min-diversity', type=int, default=25)
    ap.add_argument('--seed', type=int, default=5909)
    args = ap.parse_args()
    assert args.restarts >= 1 and args.merge_every >= 1
    run(args)

if __name__ == '__main__':
    main()
