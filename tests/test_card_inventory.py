import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'card_inventory.mjs'


def run(mode, env):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is supplied by GitHub Actions')
    return subprocess.run([node, str(SCRIPT), mode], env=env, capture_output=True, text=True, timeout=10)


def test_encrypted_card_roundtrip_and_authentication(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    name = 'a' * 64 + '.json'
    marker = 'PRIVATE_CARD_CONTENT'
    (source / name).write_text(json.dumps({'card': marker}), encoding='utf-8')
    bundle = tmp_path / 'card-inventory.enc'
    env = dict(os.environ, CARD_STORE_DIR=str(source), CARD_INVENTORY_FILE=str(bundle), CARD_CACHE_SECRET='test-key')
    assert run('pack', env).returncode == 0
    original = bundle.read_bytes()
    assert marker.encode() not in original
    target = tmp_path / 'target'
    env['CARD_STORE_DIR'] = str(target)
    assert run('unpack', env).returncode == 0
    assert json.loads((target / name).read_text(encoding='utf-8')) == {'card': marker}
    shutil.rmtree(target)
    env['CARD_CACHE_SECRET'] = 'rotated-key'
    assert run('unpack', env).returncode == 0
    assert not target.exists()
    env['CARD_CACHE_SECRET'] = 'test-key'
    bundle.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    result = run('unpack', env)
    assert result.returncode == 0 and marker not in result.stdout
    assert not target.exists()


def test_missing_cache_or_secret_is_optional(tmp_path):
    env = dict(os.environ, CARD_STORE_DIR=str(tmp_path / 'missing'), CARD_INVENTORY_FILE=str(tmp_path / 'card-inventory.enc'))
    env.pop('CARD_CACHE_SECRET', None)
    assert run('pack', env).returncode == 0
    assert run('restore', env).returncode == 0
    assert not (tmp_path / 'card-inventory.enc').exists()


def node_mock(program, env):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is supplied by GitHub Actions')
    return subprocess.run([node, '--input-type=module', '-e', program], env=env,
                          capture_output=True, text=True, timeout=12, check=True)


def test_checkpoint_encrypts_and_merges_two_inventories(tmp_path):
    remote = tmp_path / 'remote'
    remote.mkdir()
    local = tmp_path / 'local'
    local.mkdir()
    first, second = 'a' * 64 + '.json', 'b' * 64 + '.json'
    (remote / first).write_text(json.dumps({'saved_at': '2026-11-10T00:00:00Z', 'card': 'REMOTE_SECRET_MARKER'}))
    (local / second).write_text(json.dumps({'saved_at': '2026-11-11T00:00:00Z', 'card': 'LOCAL_SECRET_MARKER'}))
    bundle = tmp_path / 'remote.enc'
    env = dict(os.environ, CARD_STORE_DIR=str(remote), CARD_INVENTORY_FILE=str(bundle), CARD_CACHE_SECRET='test-key')
    run('pack', env)
    env.update(CARD_STORE_DIR=str(local), GITHUB_REPOSITORY='owner/repo', GITHUB_TOKEN='token',
               SCRIPT_URL=SCRIPT.as_uri(), MOCK_REMOTE=str(bundle), MOCK_PUT=str(tmp_path / 'put.json'))
    program = '''
import fs from 'node:fs';
const remote = fs.readFileSync(process.env.MOCK_REMOTE).toString('base64');
globalThis.fetch = async (url, options) => {
  if (options.method === 'PUT') {
    fs.writeFileSync(process.env.MOCK_PUT, options.body);
    return {ok:true, status:200};
  }
  return {ok:true, status:200, json:async()=>({encoding:'base64', content:remote, sha:'old-sha'})};
};
const mod = await import(process.env.SCRIPT_URL);
await mod.checkpoint();
'''
    node_mock(program, env)
    put = (tmp_path / 'put.json').read_text()
    assert 'REMOTE_SECRET_MARKER' not in put and 'LOCAL_SECRET_MARKER' not in put
    import base64
    bundle.write_bytes(base64.b64decode(json.loads(put)['content']))
    restored = tmp_path / 'restored'
    env['CARD_STORE_DIR'] = str(restored)
    run('unpack', env)
    assert {p.name for p in restored.glob('*.json')} == {first, second}
    assert json.loads((restored / first).read_text())['card'] == 'REMOTE_SECRET_MARKER'


def test_restore_checkpoint_failure_falls_back_and_missing_env_has_no_network(tmp_path):
    env = dict(os.environ, CARD_STORE_DIR=str(tmp_path / 'cards'), CARD_CACHE_SECRET='test-key',
               GITHUB_REPOSITORY='owner/repo', GITHUB_TOKEN='token', SCRIPT_URL=SCRIPT.as_uri(),
               MOCK_CALLS=str(tmp_path / 'calls.json'))
    program = '''
import fs from 'node:fs';
const calls=[];
globalThis.fetch=async url=>{calls.push(url); if(url.includes('/contents/')) throw new Error('offline');
 return {ok:true,status:200,json:async()=>({artifacts:[]})};};
const mod=await import(process.env.SCRIPT_URL); await mod.restore();
fs.writeFileSync(process.env.MOCK_CALLS, JSON.stringify(calls));
'''
    node_mock(program, env)
    calls = json.loads((tmp_path / 'calls.json').read_text())
    assert len(calls) == 2 and '/actions/artifacts?' in calls[1]
    env.pop('GITHUB_TOKEN')
    node_mock('globalThis.fetch=()=>{throw new Error("network used")}; const m=await import(process.env.SCRIPT_URL); await m.checkpoint(); await m.restore();', env)


def test_checkpoint_retries_conflict_and_prefers_newer_remote(tmp_path):
    remote = tmp_path / 'remote'
    remote.mkdir()
    local = tmp_path / 'local'
    local.mkdir()
    name = 'c' * 64 + '.json'
    (remote / name).write_text(json.dumps({'saved_at': '2026-11-11T00:00:00Z', 'card': 'newer remote'}))
    (local / name).write_text(json.dumps({'saved_at': '2026-11-10T00:00:00Z', 'card': 'older local'}))
    bundle = tmp_path / 'remote.enc'
    env = dict(os.environ, CARD_STORE_DIR=str(remote), CARD_INVENTORY_FILE=str(bundle), CARD_CACHE_SECRET='test-key')
    run('pack', env)
    env.update(CARD_STORE_DIR=str(local), GITHUB_REPOSITORY='owner/repo', GITHUB_TOKEN='token',
               SCRIPT_URL=SCRIPT.as_uri(), MOCK_REMOTE=str(bundle), MOCK_PUT=str(tmp_path / 'put.json'))
    program = '''
import fs from 'node:fs';
const remote=fs.readFileSync(process.env.MOCK_REMOTE).toString('base64'); let puts=0;
globalThis.fetch=async (url, options)=>{
 if(options.method==='PUT') {puts++; if(puts===1) return {ok:false,status:409};
 fs.writeFileSync(process.env.MOCK_PUT, options.body); return {ok:true,status:200};}
 return {ok:true,status:200,json:async()=>({encoding:'base64',content:remote,sha:'old-sha'})};};
const mod=await import(process.env.SCRIPT_URL); await mod.checkpoint(); if(puts!==2) throw Error('retry');
'''
    node_mock(program, env)
    import base64
    bundle.write_bytes(base64.b64decode(json.loads((tmp_path / 'put.json').read_text())['content']))
    restored = tmp_path / 'restored'
    env['CARD_STORE_DIR'] = str(restored)
    run('unpack', env)
    assert json.loads((restored / name).read_text())['card'] == 'newer remote'
