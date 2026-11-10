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
