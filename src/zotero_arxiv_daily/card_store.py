"""Best-effort store for complete cards built from public abstracts."""
import hashlib
import json
import math
import os
import subprocess
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from omegaconf import OmegaConf

from .paper_identity import filter_unseen, paper_keys
from .protocol import Paper
from .reading_preferences import effective_focus_mode

FIELDS = ('source', 'title', 'authors', 'abstract', 'url', 'pdf_url', 'journal',
          'title_zh', 'abstract_zh', 'reading_card', 'material_scope', 'journal_metrics',
          'reader_status', 'publication_date', 'publication_history', 'innovation_assessment',
          'discovery_channels', 'source_metadata', 'thought_card')
SCOPE = '仅据摘要（正文与补充材料未提供）'


def policy_fingerprint(config, corpus):
    settings = OmegaConf.to_container(config.recommendations, resolve=True) if OmegaConf.is_config(config.recommendations) else dict(config.recommendations)
    for key in ('card_cache_enabled', 'generation_budget_seconds'):
        settings.pop(key, None)
    references = sorted((str(p.title), str(p.abstract)) for p in corpus)
    # Only the digest is persisted; library contents never leave memory here.
    journals = config.get('journals', {})
    journal_policy = {key: journals.get(key) for key in ('enabled', 'mode', 'min_jif')}
    data = json.dumps({'version': 1, 'reader': 'abstract-reader-v1', 'effective_focus': effective_focus_mode(config.recommendations), 'recommendations': settings, 'journals': journal_policy, 'corpus': references}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def configuration_fingerprint(config):
    settings = OmegaConf.to_container(config.recommendations, resolve=True) if OmegaConf.is_config(config.recommendations) else dict(config.recommendations)
    for key in ('card_cache_enabled', 'generation_budget_seconds'):
        settings.pop(key, None)
    journals = config.get('journals', {})
    journal_policy = {key: journals.get(key) for key in ('enabled', 'mode', 'min_jif')}
    zotero = config.get('zotero', {})
    identity = {key: zotero.get(key) for key in ('user_id', 'include_path', 'ignore_path')}
    if OmegaConf.is_config(identity['include_path']):
        identity['include_path'] = OmegaConf.to_container(identity['include_path'], resolve=True)
    if OmegaConf.is_config(identity['ignore_path']):
        identity['ignore_path'] = OmegaConf.to_container(identity['ignore_path'], resolve=True)
    data = json.dumps({'version': 1, 'reader': 'abstract-reader-v1',
        'effective_focus': effective_focus_mode(config.recommendations), 'recommendations': settings,
        'journals': journal_policy, 'zotero_identity': identity}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def _directory():
    return Path(os.environ.get('CARD_STORE_DIR', '.cache/reading-cards'))


def _valid(paper, today, config):
    if paper.source != 'pubmed' or not paper.title or not paper.abstract or not paper.url or not paper.journal:
        return False
    if paper.material_scope != SCOPE or paper.reader_status != 'success':
        return False
    published = date.fromisoformat(paper.publication_date)
    days = min(90, int(config.recommendations.get("publication_days", 90)))
    if published.isoformat() != paper.publication_date or not today - timedelta(days=days) <= published <= today:
        return False
    jif = float((paper.journal_metrics or {}).get('jif', 0))
    minimum = max(10, float(config.get("journals", {}).get("min_jif", 10)))
    if not math.isfinite(jif) or not math.isfinite(minimum) or jif < minimum:
        return False
    Paper._validate_reader_translation({'abstract_zh': paper.abstract_zh})
    labels = {f'摘要段落{i}' for i in range(1, len(Paper._split_material(paper.abstract, structured_headings=True)) + 1)}
    card = Paper._validate_reader_card(paper.reading_card, SCOPE, labels)
    # Reuse the deployed reader's actual validator. "About 600 characters"
    # is a writing target; adding a new minimum would discard approved work.
    return len(Paper._format_reading_card(card)) <= 800 and bool(paper_keys(paper))


def save_ready(paper, config, corpus):
    if not config.recommendations.get('card_cache_enabled', False):
        return
    try:
        today = datetime.now(timezone(timedelta(hours=8))).date()
        if not _valid(paper, today, config):
            return
        basis = getattr(paper, '_approval_basis', 'library')
        if basis == 'cognitive_transfer':
            from .curation import validate_thought_card
            if not validate_thought_card(paper.thought_card, paper.abstract):
                return
        elif basis != 'library':
            return
        fingerprint = policy_fingerprint(config, corpus)
        data = {field: getattr(paper, field, None) for field in FIELDS}
        # Keep only validated reader fields, including verified abstract locators.
        labels = {f'摘要段落{i}' for i in range(1, len(Paper._split_material(paper.abstract, structured_headings=True)) + 1)}
        data['reading_card'] = Paper._validate_reader_card(paper.reading_card, SCOPE, labels)
        # Whitelist nested data too: recommendation matches may include Zotero quotes.
        metrics = data.get('journal_metrics') or {}
        data['journal_metrics'] = {key: metrics[key] for key in ('jif', 'year', 'source', 'quartile', 'jif_year', 'jcr_quartile', 'jcr_category', 'jcr_year', 'jcr_source', 'cas_quartile', 'cas_category', 'cas_year', 'cas_version', 'cas_source', 'metrics_provider', 'queried_at', 'verified_date') if key in metrics}
        innovation = data.get('innovation_assessment') or {}
        data['innovation_assessment'] = {key: innovation[key] for key in ('type', 'reason', 'evidence', 'scope') if key in innovation}
        data['discovery_channels'] = [v for v in (paper.discovery_channels or []) if v in ('frontier', 'cross_disciplinary', 'citations', 'counterevidence')]
        data['source_metadata'] = [{key: item[key] for key in ('provider', 'channel', 'pmid') if key in item and isinstance(item[key], str)} for item in (paper.source_metadata or []) if isinstance(item, dict)]
        if basis == 'cognitive_transfer':
            from .curation import THOUGHT_FIELDS
            data['thought_card'] = {key: paper.thought_card[key] for key in THOUGHT_FIELDS}
        else:
            data['thought_card'] = None
        directory = _directory()
        directory.mkdir(parents=True, exist_ok=True)
        key = sorted(paper_keys(paper))[0]
        payload = {'schema_version': 1, 'policy': fingerprint, 'configuration_policy': configuration_fingerprint(config), 'saved_at': datetime.now(timezone.utc).isoformat(), 'basis': 'previously_verified_same_policy', 'approval_basis': basis, 'paper': data, 'identity_keys': sorted(paper_keys(paper))}
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory, delete=False) as temp:
            json.dump(payload, temp, ensure_ascii=False)
            name = temp.name
        os.replace(name, directory / f'{key}.json')
        files = sorted(directory.glob('*.json'), key=lambda path: path.stat().st_mtime, reverse=True)
        for old in files[30:]:
            old.unlink(missing_ok=True)
        if (os.environ.get('CARD_CHECKPOINT_GITHUB') == 'true' and os.environ.get('GITHUB_TOKEN')
                and os.environ.get('GITHUB_REPOSITORY') and os.environ.get('OPENAI_API_KEY')):
            env = os.environ.copy()
            env['CARD_CACHE_SECRET'] = env['OPENAI_API_KEY']
            script = Path(__file__).resolve().parents[2] / 'scripts' / 'card_inventory.mjs'
            try:
                subprocess.run(['node', str(script), 'checkpoint'], env=env, timeout=9,
                               capture_output=True, shell=False, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
        return True
    except Exception:
        return


def load_ready(config, corpus):
    if not config.recommendations.get('card_cache_enabled', False):
        return []
    try:
        fingerprint = policy_fingerprint(config, corpus)
        now = datetime.now(timezone.utc)
        today = now.astimezone(timezone(timedelta(hours=8))).date()
        result = []
        for path in sorted(_directory().glob('*.json'), key=lambda item: item.stat().st_mtime, reverse=True)[:30]:
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                saved = datetime.fromisoformat(payload['saved_at'])
                if payload['schema_version'] != 1 or payload['policy'] != fingerprint or not timedelta(0) <= now - saved <= timedelta(days=7):
                    continue
                values = payload['paper']
                paper = Paper(**{field: values.get(field) for field in FIELDS})
                paper._aliases = payload.get('identity_keys', [])
                paper._cache_basis = 'previously_verified_same_policy'
                paper._approval_basis = payload.get('approval_basis', 'library')
                if paper._approval_basis == 'cognitive_transfer':
                    from .curation import validate_thought_card
                    if not validate_thought_card(paper.thought_card, paper.abstract):
                        continue
                elif paper._approval_basis != 'library':
                    continue
                if _valid(paper, today, config):
                    result.append(paper)
            except (ValueError, TypeError, KeyError, OSError):
                continue
        return filter_unseen(result, set())
    except Exception:
        return []


def load_offline_ready(config):
    """Recover recent cognitive cards without claiming a fresh library check."""
    if not config.recommendations.get('card_cache_enabled', False) or not config.get('zotero', {}).get('user_id'):
        return []
    try:
        from .curation import validate_thought_card
        fingerprint = configuration_fingerprint(config)
        now = datetime.now(timezone.utc)
        today = now.astimezone(timezone(timedelta(hours=8))).date()
        result = []
        for path in sorted(_directory().glob('*.json'), key=lambda item: item.stat().st_mtime, reverse=True)[:30]:
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                saved = datetime.fromisoformat(payload['saved_at'])
                if (payload['schema_version'] != 1 or payload.get('configuration_policy') != fingerprint
                        or payload.get('approval_basis') != 'cognitive_transfer'
                        or not timedelta(0) <= now - saved <= timedelta(hours=24)):
                    continue
                values = payload['paper']
                paper = Paper(**{field: values.get(field) for field in FIELDS})
                paper._aliases = payload.get('identity_keys', [])
                paper._approval_basis = 'cognitive_transfer'
                paper._cache_basis = 'approved_cognitive_stock_library_unavailable'
                if _valid(paper, today, config) and validate_thought_card(paper.thought_card, paper.abstract):
                    result.append(paper)
            except (ValueError, TypeError, KeyError, OSError):
                continue
        return filter_unseen(result, set())
    except Exception:
        return []
