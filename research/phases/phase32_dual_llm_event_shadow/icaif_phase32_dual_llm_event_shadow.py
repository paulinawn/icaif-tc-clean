#!/usr/bin/env python3
"""ICAIF Phase32: same-headline CaMeL/Laya consensus, forward-only shadow lab.

--self-test and --prepare do NOT use network or credentials.
--probe-both makes EXACTLY one CaMeL and one Laya request (paid calls possible).
Never generates or submits competition weights/trades. Provider keys stay in
~/.icaif/llm_event_lab_keys.env, managed by Phase23, NEVER printed.

An Oct9 news event first classified Oct11 is NOT an Oct9 trading feature.
The paper signal becomes observable only AFTER the later provider response.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

THIS=Path(__file__).resolve().parent
OUT=Path('artifacts/phase32_dual_llm_event_shadow')
REACTION_TERMS=(
    'shares soar','shares surged','shares fell','shares drop','stock jumps',
    'stock sinks','stock surged','stock fell','stock rallies','stock plunges',
    'stock rises','stock falls','stock up','stock down','market reacts',
)


def phase23():
    source=THIS/'icaif_phase23_financial_llm_event_lab.py'
    if not source.exists():
        raise FileNotFoundError('Download Phase23 beside Phase32 in your Downloads folder.')
    spec=importlib.util.spec_from_file_location('phase23_shadow_dependency',source)
    obj=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=obj
    assert spec.loader
    spec.loader.exec_module(obj)
    return obj


def entry_hash(ev):
    # Stable event/source key; credentials and inference time excluded.
    s='\0'.join((str(ev['event_id']),str(ev['symbol']),str(ev['headline']),str(ev['published_utc'])))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def candidate(p23,root):
    snap,digest=p23.load_snapshot(root)
    all_events=p23.pick_examples(snap,3)
    out=root/OUT
    processed=set()
    for f in out.glob('phase32_joint_*.json'):
        try:
            x=json.loads(f.read_text(encoding='utf-8'))
            if x.get('source_event_sha256'):processed.add(x['source_event_sha256'])
        except (ValueError,UnicodeError):continue
    for e in all_events:
        if entry_hash(e) not in processed:
            return e,digest
    raise RuntimeError('All three candidate headlines already analyzed. Update Phase18 snapshot to collect new events; no repeat API calls made.')


def consensus(event,camel,laya):
    """Conservative agreement label, NOT an estimated expected return.

    Laya 'unclear' is not equivalent to Camel neutral/unknown. Price-reaction
    headlines are never promoted as fresh predictive surprises.
    """
    headline=event['headline'].lower()
    direction_a=camel.get('direction')
    direction_b=laya.get('direction')
    same=direction_a in ('positive','negative') and direction_a==direction_b
    company_ok=(camel.get('is_company_specific') is True and
                float(laya.get('company_relevance_model_probability_not_calibrated',0.))>=.75)
    existing_reaction=any(v in headline for v in REACTION_TERMS)
    is_material=camel.get('materiality') in ('medium','high')
    vote=(direction_a if same and company_ok and is_material and not existing_reaction else 'abstain')
    return {'paper_consensus_vote':vote,'direction_agrees':same,
            'issuer_relevance_pass':company_ok,'materiality_pass':is_material,
            'headline_describes_price_reaction':existing_reaction,
            'estimated_forward_return':None,
            'event_predictive_alpha_validated':False,'eligible_for_official_submission':False,
            'cannot_backfill_to_publication_time':True}


def do_prepare(root):
    p23=phase23(); ev,digest=candidate(p23,root)
    print('HEADLINE_SELECTED_FOR_SAME_EVENT_DUAL_PROVIDER_LAB')
    print('SYMBOL:',ev['symbol'])
    print('SOURCE_PUBLISHED_UTC:',ev['published_utc'])
    print('FIRST_OBSERVED_UTC:',ev['observed_at_utc'])
    print('HEADLINE:',ev['headline'])
    print('SNAPSHOT_SHA256_PREFIX:',digest[:12])
    print('NO_PROVIDER_CALL_NO_CREDENTIALS_READ_NO_TRADING')


def do_probe(root):
    p23=phase23()
    event,digest=candidate(p23,root)
    # Validate key availability *locally* before making either request.
    camel_key=p23.load_key('camel');laya_key=p23.load_key('laya')
    if not camel_key or not laya_key:raise RuntimeError('Missing existing CaMeL/Laya credentials')
    print('EXPLICIT_TWO_PROVIDER_SHADOW_PROBE_STARTED',flush=True)
    body={'model':p23.CAMEL_MODEL,
          'messages':[{'role':'user','content':p23.prompt_events([event])}],
          'temperature':0,'max_tokens':1000}
    c_response=p23.transport(p23.CAMEL_URL,camel_key,body)
    ca=p23.parse_camel_response(c_response,[event])[0]
    l_response=p23.transport(p23.LAYA_URL,laya_key,
        {'state':{'ticker':event['symbol'],'headline':event['headline']},
         'questions':p23.laya_questions()})
    la=p23.parse_laya_response(l_response,event)
    observed=datetime.now(timezone.utc).isoformat()
    vote=consensus(event,ca,la)
    out=root/OUT;out.mkdir(parents=True,exist_ok=True)
    report={'research_only':True,'provider_requests':2,'both_classified_same_event':True,
            'source_event_sha256':entry_hash(event),'source_snapshot_sha256':digest,
            'source_published_utc':event['published_utc'],
            'source_first_observed_utc':event['observed_at_utc'],
            'inference_finished_at_utc':observed,'event_id':event['event_id'],
            'symbol':event['symbol'],'headline':event['headline'],
            'camel_model':p23.CAMEL_MODEL,'camel':ca,
            'laya_model_revision':'UNSPECIFIED_BY_ENDPOINT','laya':la,
            'consensus':vote,'cash_or_portfolio_impact':None,
            'historical_backtest_eligible':False,'forecast_calibrated':False,
            'no_organizer_requests':True}
    digest2=hashlib.sha256(json.dumps(report,sort_keys=True).encode('utf-8')).hexdigest()[:12]
    path=out/f'phase32_joint_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}_{digest2}.json'
    with path.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    print('CAMEL_DIRECTION:',ca['direction'],'MATERIALITY:',ca['materiality'])
    print('LAYA_DIRECTION:',la['direction'],'RELEVANCE_RUBRIC:',la['company_relevance_model_probability_not_calibrated'])
    print('PAPER_CONSENSUS_VOTE:',vote['paper_consensus_vote'])
    print('REPORT:',path)
    print('FORWARD_PAPER_RESEARCH_NOT_A_TRADE_NO_OFFICIAL_SUBMISSION')


def self_test():
    e={'symbol':'NVDA','event_id':'k1','headline':'NVIDIA raises forward revenue guidance',
       'published_utc':'2026-10-09T12:00:00Z'}
    a={'direction':'positive','is_company_specific':True,'materiality':'high'}
    b={'direction':'positive','company_relevance_model_probability_not_calibrated':.9}
    r=consensus(e,a,b)
    assert r['paper_consensus_vote']=='positive'
    assert consensus({**e,'headline':'NVIDIA stock surged after results'},a,b)['paper_consensus_vote']=='abstain'
    assert consensus(e,a,{**b,'direction':'negative'})['paper_consensus_vote']=='abstain'
    assert consensus(e,a,{**b,'company_relevance_model_probability_not_calibrated':.2})['paper_consensus_vote']=='abstain'
    assert r['eligible_for_official_submission'] is False
    assert len(entry_hash(e))==64
    print('PHASE32_SELF_TEST_PASS')
    print('SAME_HEADLINE_ISSUER_AGREEMENT_REACTION_ABSTENTION_PASS')
    print('NO_NETWORK_NO_SECRETS_NO_TRADING_NO_ORGANIZER_SUBMISSION')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--self-test',action='store_true')
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--probe-both',action='store_true',help='Explicitly send SAME headline to both providers; may consume two paid API calls.')
    args=parser.parse_args()
    try:
        if args.self_test:self_test()
        elif args.prepare:do_prepare(Path.cwd())
        else:do_probe(Path.cwd())
    except (RuntimeError,FileNotFoundError,ValueError,KeyError,TypeError) as e:
        # Provider/network errors are pre-sanitized by Phase23; private values
        # never appear in our args or logs.
        msg=str(e)
        if re.match(r'^(Missing |Configure |All three |Insufficient |PROVIDER_|CAMEL_|LAYA_|Download )',msg):
            print('PHASE32_STOP:',msg,file=sys.stderr)
        else:print('PHASE32_STOP:',type(e).__name__,file=sys.stderr)
        sys.exit(2)

if __name__=='__main__':main()
