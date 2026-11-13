// Durable refill decisions only. No model calls, paper content or SMTP access.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';

const branch = 'codex/reading-card-checkpoints';
const statePath = '.github/inventory-replenishment.json';
const localState = '.cache/inventory-replenishment.json';
const senderWorkflows = ['main.yml', 'supplement.yml', 'test.yml'];

export function decide(state, now, ledgerDigest, busy = false, enabled = true) {
  if (!enabled) return {run: false, reason: 'paused_by_user'};
  if (busy) return {run: false, reason: 'sender_has_priority'};
  if (state?.stock_verified === true && state.stock_after >= 20 && state.ledger_digest === ledgerDigest &&
      now >= Date.parse(state.completed_at) && now - Date.parse(state.completed_at) < 3 * 3600000) return {run: false, reason: 'full'};
  if (Date.parse(state?.next_allowed_at) > now) return {run: false, reason: 'backoff', next: state.next_allowed_at};
  return {run: true, reason: 'refill_due'};
}

export function advance(previous, summary, now, ledgerDigest, runId) {
  const ledgerChanged = typeof summary?.sent_ledger_digest === 'string' && summary.sent_ledger_digest !== ledgerDigest;
  const verified = summary?.stock_count_verified === true && summary.sent_ledger_digest === ledgerDigest && Number.isInteger(summary.stock_after) && summary.stock_after >= 0 && summary.stock_after <= 30;
  const stock = verified ? summary.stock_after : (previous?.stock_after ?? null);
  const progress = Number.isInteger(summary?.prepared) && summary.prepared > 0;
  const status = ledgerChanged ? 'sent_ledger_changed' : (summary?.status || 'interrupted');
  const full = verified && stock >= 20;
  let streak = previous?.no_progress || 0;
  if (progress || full) streak = 0;
  else if (status !== 'paused_for_sender' && !ledgerChanged) streak = Math.min(12, streak + 1);
  let minutes = 0;
  if (!full) {
    if (status === 'paused_for_sender') minutes = 5;
    else if (ledgerChanged) minutes = 0;
    else if (status === 'stock_unavailable' || status === 'interrupted') minutes = progress ? 10 : Math.min(120, 20 * 2 ** (streak - 1));
    else if (!progress) minutes = 60;
  }
  const policy = typeof summary?.policy === 'string' ? summary.policy : previous?.policy;
  const deferred = {};
  const all = {...(previous?.policy === policy ? previous?.deferred : {}), ...summary?.deferred};
  for (const [key, stamp] of Object.entries(all).sort((a, b) => Date.parse(b[1]) - Date.parse(a[1])).slice(0, 1000)) {
    if (/^[a-f0-9]{64}$/.test(key) && Date.parse(stamp) > now) deferred[key] = stamp;
  }
  return {version: 1, stock_after: stock, stock_verified: verified, target: 20,
    completed_at: new Date(now).toISOString(), next_allowed_at: new Date(now + minutes * 60000).toISOString(),
    no_progress: streak, status, ledger_digest: ledgerDigest, run_id: String(runId || ''), policy, deferred,
    continue_now: !full && ((verified && progress) || ledgerChanged) && minutes === 0};
}

function output(fields) {
  if (process.env.GITHUB_OUTPUT) fs.appendFileSync(process.env.GITHUB_OUTPUT,
    Object.entries(fields).map(([key, value]) => key + '=' + String(value) + '\n').join(''));
}

function controlError(category, stage, reason, status = 'unknown') {
  return Object.assign(new Error('Control ' + reason), {category, stage, reason, status});
}

export function controlDiagnostic(error) {
  return JSON.stringify({endpoint: error.category || 'control', stage: error.stage || 'decision',
    reason: error.reason || 'unavailable', status: Number.isInteger(error.status) ? error.status : 'unknown'});
}

export function api(fetcher = fetch) {
  const repo = process.env.GITHUB_REPOSITORY;
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo || '') || !process.env.GITHUB_TOKEN) throw new Error('Control credentials unavailable');
  const base = 'https://api.github.com/repos/' + repo + '/';
  return async (method, endpoint, body) => {
    const category = endpoint.startsWith('actions/workflows/') ? (endpoint.includes('/runs?') ? 'sender_status' : 'inventory_dispatch') :
      endpoint.includes('sent-paper-history.json') ? 'sent_ledger' : endpoint.includes('inventory-replenishment.json') ? 'refill_state' : 'git_reference';
    const attempts = method === 'GET' ? 2 : 1;
    for (let attempt = 0; attempt < attempts; attempt++) {
      let response;
      try {
        response = await fetcher(base + endpoint, {method, redirect: 'error',
          headers: {Authorization: 'Bearer ' + process.env.GITHUB_TOKEN, Accept: 'application/vnd.github+json', 'Content-Type': 'application/json'},
          body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(5000)});
      } catch (error) {
        const transient = ['TypeError', 'TimeoutError', 'AbortError'].includes(error.name);
        if (transient && attempt + 1 < attempts) continue;
        throw controlError(category, 'request', transient ? 'network_or_timeout' : 'request_failed');
      }
      if (response.status === 404 && method === 'GET') return null;
      if (!response.ok) {
        if ((response.status === 429 || response.status >= 500) && attempt + 1 < attempts) continue;
        const error = controlError(category, 'response', 'http_error', response.status);
        error.message = 'Control HTTP ' + response.status;
        throw error;
      }
      try { return response.status === 204 ? {} : await response.json(); }
      catch (error) {
        const transient = ['TypeError', 'TimeoutError', 'AbortError'].includes(error.name);
        if (transient && attempt + 1 < attempts) continue;
        throw controlError(category, 'decode', transient ? 'network_or_timeout' : 'invalid_json');
      }
    }
  };
}

export async function readState(request) {
  const item = await request('GET', 'contents/' + statePath + '?ref=' + encodeURIComponent(branch));
  if (!item) return {state: null, sha: null};
  try {
    const state = JSON.parse(Buffer.from(item.content, 'base64').toString('utf8'));
    if (state.version !== 1 || !Object.hasOwn(state, 'stock_after') || !Number.isFinite(Date.parse(state.completed_at)) || !Number.isFinite(Date.parse(state.next_allowed_at))) throw new Error('Invalid refill control');
    const cooldown = Date.parse(state.next_allowed_at) - Date.parse(state.completed_at);
    if (cooldown < 0 || cooldown > 120 * 60000 || Date.parse(state.completed_at) > Date.now() + 60000) throw new Error('Invalid control timing');
    return {state, sha: item.sha};
  } catch {
    // A broken control snapshot cannot claim a full reserve. Recount from the
    // independently validated card store, then repair this file with its CAS SHA.
    return {state: null, sha: item.sha};
  }
}

async function ledger(request) {
  const item = await request('GET', 'contents/.github/sent-paper-history.json?ref=codex%2Fsent-paper-history');
  if (!item || typeof item.content !== 'string') throw controlError('sent_ledger', 'validate', 'ledger_unavailable');
  let value;
  try { value = JSON.parse(Buffer.from(item.content, 'base64').toString('utf8')); }
  catch { throw controlError('sent_ledger', 'decode', 'invalid_ledger_json'); }
  if (!value || value.version !== 1 || !value.entries || typeof value.entries !== 'object' || Array.isArray(value.entries)) throw controlError('sent_ledger', 'validate', 'invalid_ledger');
  // Reconciliation timestamps change even when nothing was sent. Only actual
  // sent identities/dates invalidate a counted reserve.
  const sorted = Object.entries(value.entries).sort(([a], [b]) => a.localeCompare(b));
  const serialized = JSON.stringify(Object.fromEntries(sorted));
  return createHash('sha256').update(serialized).digest('hex');
}

async function busy(request) {
  for (const workflow of senderWorkflows) {
    for (const status of ['in_progress', 'queued']) {
      const data = await request('GET', 'actions/workflows/' + workflow + '/runs?status=' + status + '&per_page=1');
      // One matching run proves the sender is busy, even when more pages exist.
      // Missing or contradictory filtered results cannot prove it is idle.
      if (!data || !Array.isArray(data.workflow_runs) || !Number.isInteger(data.total_count) || data.total_count < 0 ||
          data.workflow_runs.length > 1 || data.total_count < data.workflow_runs.length ||
          (data.total_count === 0) !== (data.workflow_runs.length === 0) ||
          data.workflow_runs.some(run => typeof run?.path !== 'string' || run.path.split('@')[0] !== '.github/workflows/' + workflow || run.status !== status)) {
        throw controlError('sender_status', 'validate', 'invalid_sender_response');
      }
      if (data.total_count > 0) return true;
    }
  }
  return false;
}

export async function saveState(request, state) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const old = await readState(request);
    if (!old.sha) {
      const ref = await request('GET', 'git/ref/heads/' + branch);
      if (!ref) {
        const main = await request('GET', 'git/ref/heads/main');
        await request('POST', 'git/refs', {ref: 'refs/heads/' + branch, sha: main.object.sha});
      }
    }
    try {
      await request('PUT', 'contents/' + statePath, {branch, message: 'Update bounded stock replenishment status',
        content: Buffer.from(JSON.stringify(state)).toString('base64'), ...(old.sha ? {sha: old.sha} : {})});
      return;
    } catch (error) {
      if (!/409|422/.test(error.message) || attempt === 2) throw error;
    }
  }
}

export async function gate(request) {
  const validation = process.env.GITHUB_REF !== 'refs/heads/main';
  output({run: false, validation, extension_due: false});
  if (process.env.INVENTORY_REFILL_ENABLED === 'false') return console.log('Stock replenishment paused by repository variable.');
  if (await busy(request)) { output({reason: 'sender_has_priority'}); return console.log('Stock paused: sender has priority.'); }
  if (validation) { output({run: true}); return; }
  output({extension_due: process.env.GITHUB_EVENT_NAME === 'schedule' && process.env.EVENT_SCHEDULE === '15 12 * * *'});
  const old = await readState(request);
  const digest = await ledger(request);
  let decision = decide(old.state, Date.now(), digest, false, process.env.INVENTORY_REFILL_ENABLED !== 'false');
  // A deployed producer fix should be tried once without waiting for the old
  // failure's cooldown; sender priority, user pause and full reserves still win.
  if (process.env.GITHUB_EVENT_NAME === 'push' && decision.reason === 'backoff') {
    decision = {run: true, reason: 'deployment_recheck'};
  }
  fs.mkdirSync('.cache', {recursive: true});
  fs.writeFileSync(localState, JSON.stringify({...old.state, checked_ledger_digest: digest}));
  output({run: decision.run, reason: decision.reason});
  console.log('Stock control: ' + JSON.stringify(decision));
}

export async function complete(request) {
  if (process.env.GITHUB_REF !== 'refs/heads/main') return;
  const old = await readState(request);
  let summary = null;
  try { summary = JSON.parse(fs.readFileSync('outputs/inventory-summary.json', 'utf8')); } catch {}
  const state = advance(old.state, summary, Date.now(), await ledger(request), process.env.GITHUB_RUN_ID);
  await saveState(request, state);
  console.log('Stock replenishment: ' + JSON.stringify(state));
  if (state.continue_now && process.env.INVENTORY_REFILL_ENABLED !== 'false' && !await busy(request)) {
    await request('POST', 'actions/workflows/reading-inventory.yml/dispatches', {ref: 'main'});
    console.log('Next bounded refill round requested; completed cards are retained.');
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    if (process.argv[2] === 'gate') await gate(api());
    else if (process.argv[2] === 'complete') await complete(api());
    else throw new Error('Unknown refill action');
  } catch (error) {
    output({run: false, reason: 'control_unavailable'});
    console.log('Stock control unavailable ' + controlDiagnostic(error) + '; the next heartbeat will retry.');
    process.exitCode = 1;
  }
}
