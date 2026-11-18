"""CLI for focused CP-SAT / HiGHS matching improvement with safe backups."""
import argparse
import json
import shutil
import time
from pathlib import Path

from .matching import load_transfers, score_solution, make_vector
from .cli import read_incumbent, save_atomic, EXPECTED, CHALLENGE
from .focused_opt import focused_search


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--problem', choices=list(EXPECTED), required=True)
    p.add_argument('--out', type=Path, default=Path('results'))
    p.add_argument('--seconds', type=float, default=300)
    p.add_argument('--rounds', type=int, default=120)
    p.add_argument('--per-round', type=float, default=3.0)
    p.add_argument('--sizes', default='120,260,450')
    p.add_argument('--caps', default='1200,2400,3800')
    p.add_argument('--seed', type=int, default=2099)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--method', choices=['cpsat', 'milp', 'hybrid'], default='cpsat')
    p.add_argument('--allow-demo-size', action='store_true')
    args = p.parse_args()
    rows = load_transfers(args.input)
    if not args.allow_demo_size and len(rows) != EXPECTED[args.problem]:
        raise ValueError('Input length does not match the ESA instance')
    target = args.out / f'submission_{args.problem}.json'
    original = read_incumbent(target, len(rows), args.problem)
    if original is None:
        raise ValueError('Missing existing valid solution. First run STRONG LNS.')
    original_score = score_solution(rows, sorted(original))
    print(f'Loaded previous best: {original_score:.3f}', flush=True)
    # Back up incumbent before any optimization attempt.
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out / f'before_focused_{args.problem}_{time.time_ns()}.json'
    shutil.copy2(target, snapshot)
    print('Saved before-run backup:', snapshot, flush=True)
    best_written = original_score

    def checkpoint(solution, score, round_id):
        nonlocal best_written
        if score < best_written - 1e-7:
            raise ValueError('Refusing to overwrite better score')
        if target.exists():
            disk = read_incumbent(target, len(rows), args.problem)
            if disk is not None and score_solution(rows, sorted(disk)) > score + 1e-7:
                raise ValueError('Disk result is better than proposed result')
        body = [dict(decisionVector=make_vector(len(rows), sorted(solution)),
                     problem=args.problem, challenge=CHALLENGE)]
        save_atomic(target, json.dumps(body, separators=(',', ':')))
        best_written = score

    sizes = tuple(int(x) for x in args.sizes.split(','))
    caps = tuple(int(x) for x in args.caps.split(','))
    chosen, metadata = focused_search(rows, original, seconds=args.seconds,
                                     rounds=args.rounds, seed=args.seed,
                                     sizes=sizes, caps=caps, per_round=args.per_round,
                                     method=args.method, workers=args.workers,
                                     checkpoint=checkpoint)
    score = score_solution(rows, sorted(chosen))
    if score > best_written + 1e-7:
        checkpoint(chosen, score, metadata['attempts'])
    report = dict(problem=args.problem, prior=original_score,
                  winner_score=score, gain=score - original_score,
                  valid_resource_constraints=True,
                  esa_fitness_expected=-score, focused=metadata,
                  submission_path=str(target))
    save_atomic(args.out / f'report_{args.problem}_focused.json',
                json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
