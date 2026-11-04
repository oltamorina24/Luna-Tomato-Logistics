import argparse
import json
from pathlib import Path
from .trajectory import choose_candidates,write_submission

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidates',required=True,type=Path)
    p.add_argument('--out',default='results/submission_trajectory-matching.json',type=Path)
    p.add_argument('--seconds',default=300,type=float)
    args=p.parse_args()
    candidates=json.loads(args.candidates.read_text(encoding='utf-8'))
    rows,meta=choose_candidates(candidates,args.seconds)
    write_submission(rows,args.out)
    print(f'Saved {len(rows)} selected validated candidates to {args.out}')
    print(json.dumps(meta,indent=2))
    print('Still run the whole official ESA fitness/constraint validator!')

if __name__=='__main__':
    main()
