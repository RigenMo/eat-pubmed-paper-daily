"""Best-effort store for complete cards built from public abstracts."""
import hashlib
import json
import math
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from omegaconf import OmegaConf

from .paper_identity import filter_unseen, paper_keys
from .protocol import Paper
from .reading_preferences import effective_focus_mode

FIELDS = ('source', 'title', 'authors', 'abstract', 'url', 'pdf_url', 'journal',
          'title_zh', 'abstract_zh', 'reading_card', 'material_scope', 'journal_metrics',
          'reader_status', 'publication_date', 'publication_history', 'innovation_assessment')
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
        directory = _directory()
        directory.mkdir(parents=True, exist_ok=True)
        key = sorted(paper_keys(paper))[0]
        payload = {'schema_version': 1, 'policy': fingerprint, 'saved_at': datetime.now(timezone.utc).isoformat(), 'basis': 'previously_verified_same_policy', 'paper': data, 'identity_keys': sorted(paper_keys(paper))}
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=directory, delete=False) as temp:
            json.dump(payload, temp, ensure_ascii=False)
            name = temp.name
        os.replace(name, directory / f'{key}.json')
        files = sorted(directory.glob('*.json'), key=lambda path: path.stat().st_mtime, reverse=True)
        for old in files[30:]:
            old.unlink(missing_ok=True)
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
                if _valid(paper, today, config):
                    result.append(paper)
            except (ValueError, TypeError, KeyError, OSError):
                continue
        return filter_unseen(result, set())
    except Exception:
        return []
