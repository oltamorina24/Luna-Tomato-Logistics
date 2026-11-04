"""Kaggle / local command line entrypoint."""
import argparse
import csv
import json
from pathlib import Path

from .matching import load_transfers, make_vector, score_solution, solve

CHALLENGE = 'spoc-4-luna-tomato-logistics'
EXPECTED = {'matching-i':25000, 'matching-ii':92103}


def main():
    p=argparse.ArgumentParser(description='ESA SpOC-4 weighted 3D matching')
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--problem', required=True, choices=EXPECTED)
    p.add_argument('--method', default='hybrid', choices=['heuristic','milp','hybrid'])
    p.add_argument('--seconds', default=300, type=float)
    p.add_argument('--restarts', default=20, type=int)
    p.add_argument('--seed', default=2026, type=int)
    p.add_argument('--relative-gap', default=0.0001, type=float)
    p.add_argument('--out', default=Path('results'), type=Path)
    p.add_argument('--allow-demo-size', action='store_true')
    args=p.parse_args()
    rows=load_transfers(args.input)
    if len(rows)!=EXPECTED[args.problem] and not args.allow_demo_size:
        raise ValueError(f'{args.problem} should have {EXPECTED[args.problem]} rows, got {len(rows)}. '
                         'For small illustrative tests, use --allow-demo-size.')
    chosen, details=solve(rows, args.method, args.seconds, args.restarts,
                          args.seed, args.relative_gap)
    score=score_solution(rows, sorted(chosen))
    vector=make_vector(len(rows), sorted(chosen))
    assert len(vector)==len(rows) and sum(vector)==len(chosen)
    args.out.mkdir(parents=True,exist_ok=True)
    submission=[{'decisionVector':vector, 'problem':args.problem, 'challenge':CHALLENGE}]
    file=args.out/f'submission_{args.problem}.json'
    file.write_text(json.dumps(submission, separators=(',',':')), encoding='utf-8')
    details.update({'problem':args.problem,'input':str(args.input), 'rows':len(rows),
                    'score_local':score, 'esa_fitness_expected':-score,
                    'valid_resource_constraints':True, 'submission_path':str(file)})
    (args.out/f'report_{args.problem}.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    with (args.out/f'selected_{args.problem}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['row_index_0based','earth_id','moon_id','destination_id','mass'])
        for i in sorted(chosen):
            t=rows[i];writer.writerow([i,t.e,t.l,t.d,t.weight])
    print(json.dumps(details, indent=2))
    print(f'ESA submission: {file}')
    print('IMPORTANT: validate with the official ESA evaluator before submitting.')

if __name__=='__main__':
    main()
