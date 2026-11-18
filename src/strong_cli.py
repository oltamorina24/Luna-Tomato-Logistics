"""Start from the best existing ESA submission; improve and checkpoint it safely."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from .matching import load_transfers, make_vector, score_solution, heuristic
from .cli import CHALLENGE, EXPECTED, read_incumbent, save_atomic
from .strong_opt import adaptive_lns, cp_sat_global, lp_upper_bound, pack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--problem', required=True, choices=list(EXPECTED))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--rounds', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--restarts', type=int, default=8)
    parser.add_argument('--neighborhoods', default='55,95,160,260')
    parser.add_argument('--candidate-caps', default='900,1600,2600,3800')
    parser.add_argument('--per-round', type=float, default=1.2)
    parser.add_argument('--lp-seconds', type=float, default=0,
                        help='Optional full LP relaxation bound, adds runtime')
    parser.add_argument('--cpsat-seconds', type=float, default=0,
                        help='Optional OR-Tools CP-SAT global search with hints')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--allow-demo-size', action='store_true')
    args = parser.parse_args()
    rows = load_transfers(args.input)
    if not args.allow_demo_size and len(rows) != EXPECTED[args.problem]:
        raise ValueError('Wrong instance file length')
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f'submission_{args.problem}.json'
    chosen = read_incumbent(target, len(rows), args.problem)
    if chosen is None:
        print('No incumbent: initializing from heuristic', flush=True)
        chosen, _ = heuristic(rows, restarts=args.restarts, seed=args.seed)
    else:
        print(f'Previous best: {score_solution(rows, sorted(chosen)):.3f}', flush=True)
    first_score = score_solution(rows, sorted(chosen))
    if target.exists():
        snapshot = args.out / f'previous_{args.problem}_{int(time.time())}.json'
        shutil.copy2(target, snapshot)
        print(f'Before-run backup: {snapshot}', flush=True)
    best_score = first_score
    checkpoints = 0

    def checkpoint(solution: set[int], score: float, round_number: int):
        nonlocal best_score, checkpoints
        if score < best_score-1e-6:
            raise ValueError('Attempted to overwrite stronger incumbent')
        if target.exists():
            old = read_incumbent(target, len(rows), args.problem)
            if old is not None and score_solution(rows, sorted(old)) > score+1e-6:
                print('Skipping checkpoint: disk has better result', flush=True)
                return
        doc = [{'decisionVector': make_vector(len(rows), sorted(solution)),
                'problem': args.problem, 'challenge': CHALLENGE}]
        save_atomic(target, json.dumps(doc, separators=(',', ':')))
        best_score = max(best_score, score)
        checkpoints += 1

    report = {'original_score': first_score}
    if args.lp_seconds > 0:
        bound = lp_upper_bound(pack(rows), seconds=args.lp_seconds)
        report['lp'] = {k:v for k,v in bound.items() if k != 'dual_prices'}
        if 'upper_bound' in bound:
            print(f'LP upper bound (NOT optimal integer): {bound["upper_bound"]:.3f}', flush=True)
    sizes = tuple(int(v) for v in args.neighborhoods.split(','))
    caps = tuple(int(v) for v in args.candidate_caps.split(','))
    chosen, detail = adaptive_lns(rows, chosen, seconds=args.seconds, rounds=args.rounds,
                                  seed=args.seed, neighborhood_sizes=sizes,
                                  candidate_caps=caps, per_round_seconds=args.per_round,
                                  checkpoint=checkpoint)
    report['strong_lns'] = detail
    if args.cpsat_seconds > 0:
        cp_candidate, cp_detail = cp_sat_global(rows, chosen, seconds=args.cpsat_seconds,
                                                 workers=args.workers, seed=args.seed)
        report['cpsat'] = cp_detail
        if score_solution(rows, sorted(cp_candidate)) > score_solution(rows, sorted(chosen)) + 1e-7:
            chosen = cp_candidate
            checkpoint(chosen, score_solution(rows, sorted(chosen)), detail['rounds'])
    final_score = score_solution(rows, sorted(chosen))
    checkpoint(chosen, final_score, detail['rounds'])
    report.update({'problem':args.problem, 'winner_score':final_score,
                   'esa_fitness_expected':-final_score,
                   'valid_resource_constraints':True,
                   'n_selected':len(chosen), 'checkpoints':checkpoints,
                   'submission_path':str(target)})
    save_atomic(args.out / f'report_{args.problem}_strong.json', json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
