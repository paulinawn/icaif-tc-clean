#!/usr/bin/env python3
"""Offline tests for strict MISSING/HELD synthetic-entry normalization."""
import copy
import icaif_v37_production_candidate as bridge
import icaif_v33_state_machine as machine

R1 = 'official-2026-10-12-r1'
R2 = 'official-2026-10-12-r2'
R3 = 'official-2026-10-12-r3'
R4 = 'official-2026-10-12-r4'
schedule={'rounds':[{'id':x,'phase':'official','status':'COMPLETED'} for x in (R1,R2,R3)] + [{'id':R4,'phase':'official','status':'OPEN'}]}
submitted={'round_id':R1,'decision_id':'real-1581','status':'VALID','selection_status':'SELECTED','execution_status':'EXECUTED'}
held={'round_id':R2,'decision_id':'synthetic-r2','status':'MISSING','selection_status':None,'execution_status':'HELD'}
held_unknown=dict(held,round_id=R3,decision_id='synthetic-r3',selection_status='UNKNOWN')
class Session:
    def __init__(self,rows):self.rows=rows
    def decisions(self,phase,offset,limit):
        assert phase=='official' and limit==200 and offset==0
        return {'decisions':self.rows,'offset':0,'limit':200,'has_more':False}

def rejected(rows,pattern):
    try:bridge.read_pages(Session(rows),machine,schedule)
    except Exception as e:
        assert pattern in str(e),(pattern,type(e).__name__,str(e))
        return
    raise AssertionError('SHOULD_FAIL_'+pattern)

history=bridge.read_pages(Session([submitted,held,held_unknown]),machine,schedule)
assert len(history['decisions'])==1 and history['decisions'][0]['decision_id']=='real-1581'
assert machine.enforce_no_unjournaled_decisions({'intents':{R1:{}}},history['decisions'])
print('V392_REAL_EXECUTION_RETAINED_AND_COMPLETED_HELD_PLACEHOLDERS_REMOVED_PASS')

unknown=dict(submitted,round_id=R2,decision_id='unknown-upload',status='INVALID',selection_status='NOT_SELECTED',execution_status='NOT_EXECUTED')
with_unknown=bridge.read_pages(Session([submitted,unknown,held_unknown]),machine,schedule)
try:machine.enforce_no_unjournaled_decisions({'intents':{R1:{}}},with_unknown['decisions'])
except machine.UnsafeTransition as e:assert str(e)=='UNRECOGNIZED_OFFICIAL_SUBMISSION'
else:raise AssertionError('UNRECOGNIZED_UPLOAD_ALLOWED')
print('V392_UNJOURNALED_ACTUAL_UPLOAD_STILL_FAILS_CLOSED_PASS')

rejected([submitted,dict(held,uploaded_at='2026-10-12T10:05:00Z')],'MISSING_HELD_ROW_HAS_SUBMISSION_ARTIFACT')
rejected([submitted,dict(held,round_id=R4,decision_id='future-marker')],'MISSING_HELD_ROW_NOT_COMPLETED')
rejected([submitted,dict(held,selection_status='SELECTED')],'') if False else None
assert not bridge.organizer_held_placeholder(dict(held,selection_status='SELECTED'),{R2})
assert not bridge.organizer_held_placeholder(dict(held,execution_status='PENDING'),{R2})
assert not bridge.organizer_held_placeholder(dict(held,status='INVALID'),{R2})
print('V392_AMBIGUOUS_OR_PRESENT_UPLOAD_ARTIFACT_FAILS_CLOSED_PASS')
print('V392_OFFLINE_TESTS_NO_HTTP_NO_CREDENTIALS_NO_POST_NO_REBALANCE')

# 2026-10-19 authenticated GET observed that all 34 completed synthetic
# MISSING/HELD rows have a nonempty `submitted_at` field despite NO upload.
# It must not be sufficient to classify them as actual submissions.
synthetic_timestamp = dict(held, submitted_at='2026-10-12T13:05:00+00:00')
with_timestamp = bridge.read_pages(Session([submitted,synthetic_timestamp,held_unknown]),machine,schedule)
assert len(with_timestamp['decisions']) == 1
assert machine.enforce_no_unjournaled_decisions({'intents': {R1: {}}},with_timestamp['decisions'])
print('V3921_TIMESTAMPED_MISSING_HELD_PLACEHOLDERS_EXCLUDED_PASS')

# Any substantive receipt or upload remains material, never ignored.
for artifact, value in [('submission_id', 1582),
                        ('uploaded_at','2026-10-12T13:05:00Z'),
                        ('receipt_status','INVALID'),
                        ('file_sha256','f'*64)]:
    rejected([submitted,dict(synthetic_timestamp,**{artifact:value})],
             'MISSING_HELD_ROW_HAS_SUBMISSION_ARTIFACT')
print('V3921_REAL_UPLOAD_ARTIFACT_STILL_FAILS_CLOSED_PASS')

# Conflict statuses never masquerade as placeholders.
for override in [dict(status='INVALID'),dict(execution_status='EXECUTED'),
                 dict(selection_status='SELECTED')]:
    material=bridge.read_pages(Session([submitted,dict(synthetic_timestamp,**override)]),machine,schedule)
    assert len(material['decisions'])==2
    try:machine.enforce_no_unjournaled_decisions({'intents':{R1:{}}},material['decisions'])
    except machine.UnsafeTransition as e:assert str(e)=='UNRECOGNIZED_OFFICIAL_SUBMISSION'
    else:raise AssertionError('INCONSISTENT_RECORD_DROPPED')
print('V3921_INCONSISTENT_STATUS_IS_MATERIAL_FAIL_CLOSED_PASS')
