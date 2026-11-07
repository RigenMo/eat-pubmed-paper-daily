import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location('supplement_digest', Path(__file__).resolve().parents[1] / 'scripts' / 'supplement_digest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def receipt(count):
    return {'papers': [{'url': f'https://pubmed.ncbi.nlm.nih.gov/{i}/', 'journal': 'J', 'publication_date': '2026-11-01', 'jif': 12} for i in range(1, count + 1)]}


def test_complete_day_skips_network_and_send(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_TOKEN', 'test')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_RUN_ID', '1')
    monkeypatch.setattr(module, 'today_history', lambda *args: ([receipt(10)], [], 1))
    monkeypatch.setattr(module, 'send_email', lambda *args, **kwargs: pytest.fail('must not send'))
    assert module.supplement(SimpleNamespace()) == 0


def test_no_unsent_candidate_does_not_send(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_TOKEN', 'test')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_RUN_ID', '1')
    monkeypatch.setattr(module, 'today_history', lambda *args: ([receipt(8)], [123], 1))
    monkeypatch.setattr(module, 'read_candidates', lambda *args: ['1', '2'])
    monkeypatch.setattr(module, 'send_email', lambda *args, **kwargs: pytest.fail('must not send'))
    with pytest.raises(RuntimeError, match='No unsent'):
        module.supplement(SimpleNamespace())


def test_duplicate_delivery_fails_closed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_TOKEN', 'test')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'owner/repo')
    monkeypatch.setenv('GITHUB_RUN_ID', '1')
    monkeypatch.setattr(module, 'today_history', lambda *args: ([receipt(1), receipt(1)], [], 1))
    with pytest.raises(RuntimeError, match='Duplicate'):
        module.supplement(SimpleNamespace())


