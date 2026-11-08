#!/usr/bin/env python3
"""Pure offline V2.3 policy calculation to reproduce a decision; NO live uploader, API or state writes."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'agent'/'frozen_strategy'))
import icaif_official_sleeve_engine_v23 as engine

def main():

    #  Read command-line arguments
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daily-close-csv',type=Path,required=True)
    parser.add_argument('--current-weights',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()

    # Read daily close CSV and check
    with args.daily_close_csv.open(newline='',encoding='utf-8-sig') as f:
        r=csv.DictReader(f)
        if r.fieldnames is None or len(r.fieldnames)!=31 or set(r.fieldnames)!=(set(engine.SYMBOLS)|{'session'}):
            raise ValueError('Expected session and exactly the frozen 30 symbols')
        sessions=[];matrix=[]
        for row in r:
            sessions.append(row['session'])
            matrix.append([float(row[s]) for s in engine.SYMBOLS])
    if len(matrix)!=121 or not all(sessions) or not all(sessions[i]<sessions[i+1] for i in range(120)):
        raise ValueError('Expected exactly 121 strictly chronological complete sessions')

    # Calculate target weights
    target,selected,core=engine.sleeve_target_from_daily(matrix)
    action='POLICY_TARGET'

    # Optional: Simulates review day
    if args.current_weights:
        previous=json.loads(args.current_weights.read_text(encoding='utf-8'))
        if isinstance(previous,dict) and 'weights' in previous:previous=previous['weights']
        engine.validate(previous)
        desired=engine.trim_target(previous,target,initial=False)
        if desired is None:
            action='HOLD_BELOW_L1_0.03';target=None
        else:
            action='TURNOVER_CONTROLLED_TARGET';target=desired

    # Write output with action and target weights
    out={'type':'OFFLINE_POLICY_CALCULATION_NOT_A_SUBMISSION','action':action,
         'data_last_session':sessions[-1], 'reversal_selected_five':selected,
         'target_weights':target,'equity_weight':sum(target.values()) if target else None}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('ICAIF_OFFLINE_POLICY_OK',action,'five',','.join(selected), 'output',args.out)

if __name__=='__main__':main()
