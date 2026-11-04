"""Trajectory Matching input/output and candidate assignment utilities.

This is not a physics/BCP trajectory optimizer. Candidate trajectory vectors
must FIRST be generated and validated with ESA's official evaluator.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

N_TRANSFERS=400
FIELDS_PER_TRANSFER=21
EXPECTED_LENGTH=N_TRANSFERS*FIELDS_PER_TRANSFER
CHALLENGE='spoc-4-luna-tomato-logistics'


def delivered_mass(delta_v_ms: float, duration_days: float, rate_kg_per_day: float) -> float:
    """Official equation for an otherwise physically VALID transfer."""
    if delta_v_ms < 0 or duration_days < 0 or duration_days > 200 or rate_kg_per_day < 0:
        return 0.0
    m_l=5000.0*math.exp(-delta_v_ms/(311.0*9.80665))-500.0
    return max(0.0,min(m_l,(200-duration_days)*rate_kg_per_day))


def make_advanced_vector(trajectories: Sequence[Sequence[float]]) -> list[float]:
    """Pad physically validated 21-number candidate rows to size 8400.

    The caller must use ESA's evaluator to verify orbital/flight constraints.
    """
    if len(trajectories)>N_TRANSFERS:
        raise ValueError('Max 400 trajectories')
    x=[]
    used_e,used_l,used_d=set(),set(),set()
    for row in trajectories:
        if len(row)!=FIELDS_PER_TRANSFER or not all(math.isfinite(float(v)) for v in row):
            raise ValueError('Each trajectory needs 21 finite values')
        e,l,d=(int(row[j]) for j in range(3))
        if any(float(row[j])!=int(row[j]) for j in range(3)):
            raise ValueError('First three fields must be integer IDs')
        if e<0 or l<0 or d<0 or e in used_e or l in used_l or d in used_d:
            raise ValueError('Duplicate/invalid resource ID')
        used_e.add(e);used_l.add(l);used_d.add(d)
        x.extend(float(v) for v in row)
    for _ in range(N_TRANSFERS-len(trajectories)):
        x.extend([-1.0]+[0.0]*(FIELDS_PER_TRANSFER-1))
    assert len(x)==EXPECTED_LENGTH
    return x


def write_submission(trajectories: Sequence[Sequence[float]], path: str|Path):
    x=make_advanced_vector(trajectories)
    doc=[{'decisionVector':x,'problem':'trajectory-matching','challenge':CHALLENGE}]
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(doc,separators=(',',':')),encoding='utf-8')
    return path


def choose_candidates(candidates: Sequence[dict], seconds: float=300):
    """Choose disjoint ESA-VALIDATED candidate trajectories via SciPy MILP.

    Each candidate: {"vector": [21 values], "delivered_mass": float,
    "validated_officially": true}. This DOES NOT generate physical trajectories.
    """
    from src.matching import Transfer, solve
    transfers=[]
    for k,c in enumerate(candidates):
        if c.get('validated_officially') is not True:
            raise ValueError(f'Candidate {k} is not marked as officially validated')
        vec=c['vector']
        if len(vec)!=21:
            raise ValueError(f'Candidate {k} needs 21 numbers')
        transfers.append(Transfer(int(vec[0]),int(vec[1]),int(vec[2]),float(c['delivered_mass'])))
    indices, meta=solve(transfers,'hybrid',seconds,20,2026)
    chosen=[candidates[i]['vector'] for i in sorted(indices)]
    make_advanced_vector(chosen)
    return chosen, meta
