import json
import os
from pathlib import Path
import shutil
import subprocess
import hashlib

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'inventory_replenishment.mjs'


def node(program, tmp_path):
    binary = shutil.which('node')
    if not binary:
        pytest.skip('GitHub Actions provides Node')
    env = dict(os.environ, MODULE=SCRIPT.as_uri(), GITHUB_REPOSITORY='owner/repo', GITHUB_TOKEN='test-token',
               GITHUB_REF='refs/heads/main', GITHUB_RUN_ID='55', INVENTORY_REFILL_ENABLED='true')
    env['EMPTY_LEDGER_DIGEST'] = hashlib.sha256(b'{}').hexdigest()
    result = subprocess.run([binary, '--input-type=module', '-e', program], env=env, cwd=tmp_path,
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_continuation_fullness_consumption_and_backoff(tmp_path):
    program = '''
import assert from 'node:assert/strict';
const {advance,decide}=await import(process.env.MODULE);
const now=Date.parse('2026-11-11T10:00:00Z');
let s=advance(null,{stock_count_verified:true,sent_ledger_digest:'ledger-a',stock_after:7,prepared:5,status:'stock_partial'},now,'ledger-a',55);
assert.equal(s.continue_now,true);
assert.equal(decide(s,now,'ledger-a').run,true);
s=advance(s,{stock_count_verified:true,sent_ledger_digest:'ledger-a',stock_after:20,prepared:12,status:'ok'},now,'ledger-a',56);
assert.equal(s.continue_now,false);
assert.equal(s.no_progress,0);
assert.equal(decide(s,now,'ledger-a').reason,'full');
assert.equal(decide(s,now,'ledger-b').run,true);
assert.equal(decide(s,now+4*3600000,'ledger-a').run,true);
for(let i=0;i<8;i++) {
 s=advance(s,{stock_count_verified:true,sent_ledger_digest:'ledger-a',stock_after:2,prepared:0,status:'stock_unavailable'},now,'ledger-a',60+i);
 assert.equal(s.continue_now,false);
 const wait=Date.parse(s.next_allowed_at)-now;
 assert.ok(wait>=20*60000 && wait<=120*60000);
 assert.equal(decide(s,now,'ledger-a').reason,'backoff');
}
assert.equal(decide(s,now,'ledger-a',true).reason,'sender_has_priority');
assert.equal(decide(s,now,'ledger-a',false,false).reason,'paused_by_user');
console.log('ok');
'''
    assert 'ok' in node(program, tmp_path)


def test_completion_records_state_then_only_dispatches_inventory(tmp_path):
    program = '''
import assert from 'node:assert/strict'; import fs from 'node:fs';
const {complete}=await import(process.env.MODULE); const calls=[];
fs.mkdirSync('outputs'); fs.writeFileSync('outputs/inventory-summary.json',JSON.stringify({stock_count_verified:true,sent_ledger_digest:process.env.EMPTY_LEDGER_DIGEST,stock_after:5,prepared:2,status:'stock_partial'}));
const request=async(method,path,body)=>{calls.push({method,path,body});
 if(path.includes('inventory-replenishment.json') && method==='GET') return null;
 if(path.includes('sent-paper-history.json')) return {sha:'blob',content:Buffer.from(JSON.stringify({version:1,entries:{}})).toString('base64')};
 if(path.includes('git/ref/')) return {object:{sha:'main'}};
 if(path.includes('actions/runs?')) return {total_count:0,workflow_runs:[]};
 return {};
};
await complete(request);
const put=calls.find(c=>c.method==='PUT'); assert.ok(put);
const state=JSON.parse(Buffer.from(put.body.content,'base64'));
assert.equal(state.stock_after,5);
const dispatched=calls.filter(c=>c.path.includes('/dispatches'));
assert.equal(dispatched.length,1);
assert.equal(dispatched[0].path,'actions/workflows/reading-inventory.yml/dispatches');
assert.ok(calls.indexOf(put)<calls.indexOf(dispatched[0]));
console.log('ok');
'''
    assert 'ok' in node(program, tmp_path)


def test_failed_state_write_cannot_create_an_untracked_loop(tmp_path):
    program = '''
import assert from 'node:assert/strict'; import fs from 'node:fs';
const {complete}=await import(process.env.MODULE); let dispatched=false;
fs.mkdirSync('outputs'); fs.writeFileSync('outputs/inventory-summary.json',JSON.stringify({stock_count_verified:true,sent_ledger_digest:process.env.EMPTY_LEDGER_DIGEST,stock_after:4,prepared:1,status:'stock_partial'}));
const request=async(method,path)=>{
 if(path.includes('inventory-replenishment') && method==='GET') return null;
 if(path.includes('sent-paper-history')) return {sha:'blob',content:Buffer.from(JSON.stringify({version:1,entries:{}})).toString('base64')};
 if(path.includes('git/ref/')) return {};
 if(method==='PUT') throw Error('Control HTTP 503');
 if(path.includes('dispatches')) dispatched=true;
 return {};
};
await assert.rejects(()=>complete(request)); assert.equal(dispatched,false); console.log('ok');
'''
    assert 'ok' in node(program, tmp_path)


def test_feature_validation_does_not_dispatch_or_write_control(tmp_path):
    program = '''
import assert from 'node:assert/strict';
const {complete}=await import(process.env.MODULE);
process.env.GITHUB_REF='refs/heads/codex/test'; let calls=0;
await complete(async()=>{calls++;}); assert.equal(calls,0); console.log('ok');
'''
    assert 'ok' in node(program, tmp_path)


def test_corrupt_control_is_recounted_not_reported_full(tmp_path):
    program = '''
import assert from 'node:assert/strict';
const {readState,decide}=await import(process.env.MODULE);
const old=await readState(async()=>({sha:'old-sha',content:Buffer.from('not json').toString('base64')}));
assert.equal(old.state,null); assert.equal(old.sha,'old-sha');
assert.equal(decide(old.state,Date.now(),'ledger').run,true); console.log('ok');
'''
    assert 'ok' in node(program, tmp_path)


def test_delivery_during_a_stock_run_forces_a_new_count(tmp_path):
    program = '''
import assert from 'node:assert/strict';
const {advance,decide}=await import(process.env.MODULE);
const now=Date.parse('2026-11-11T10:00:00Z');
const s=advance(null,{stock_count_verified:true,sent_ledger_digest:'before-send',stock_after:20,prepared:0,status:'ok'},now,'after-send',55);
assert.equal(s.stock_verified,false); assert.equal(s.continue_now,true);
assert.equal(decide(s,now,'after-send').run,true); console.log('ok');
'''
    assert 'ok' in node(program, tmp_path)


def test_full_reserve_keeps_scheduled_extensions_without_model_refill(tmp_path):
    program = '''
import assert from 'node:assert/strict'; import fs from 'node:fs';
const {gate}=await import(process.env.MODULE);
process.env.GITHUB_EVENT_NAME='schedule'; process.env.EVENT_SCHEDULE='15 12 * * *';
process.env.GITHUB_OUTPUT='out.txt';
const state={version:1,stock_verified:true,stock_after:20,ledger_digest:process.env.EMPTY_LEDGER_DIGEST,completed_at:new Date().toISOString(),next_allowed_at:new Date().toISOString()};
const request=async(method,path)=>{
 if(path.includes('actions/runs?')) return {total_count:0,workflow_runs:[]};
 if(path.includes('sent-paper-history')) return {content:Buffer.from(JSON.stringify({version:1,entries:{}})).toString('base64')};
 return {sha:'state',content:Buffer.from(JSON.stringify(state)).toString('base64')};
};
await gate(request); const out=fs.readFileSync('out.txt','utf8');
assert.ok(out.includes('extension_due=true')); assert.ok(out.includes('reason=full'));
assert.equal(out.trim().split(String.fromCharCode(10)).filter(x=>x.startsWith('run=')).at(-1),'run=false'); console.log('ok');
'''
    assert 'ok' in node(program, tmp_path)
