const API = 'https://api.github.com/repos/';

export async function check(env, fetcher = fetch) {
  if (!env.GH_ACTIONS_TOKEN || env.REPOSITORY !== 'RigenMo/eat-pubmed-paper-daily') {
    throw new Error('WATCHDOG_CONFIGURATION_ERROR');
  }
  const base = API + env.REPOSITORY + '/actions/';
  const headers = {
    Authorization: `Bearer ${env.GH_ACTIONS_TOKEN}`,
    Accept: 'application/vnd.github+json',
    'Content-Type': 'application/json',
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': 'daily-email-watchdog',
  };
  async function request(path, options = {}) {
    let response;
    try {
      response = await fetcher(base + path, {
        ...options, headers, redirect: 'error', signal: AbortSignal.timeout(10000),
      });
    } catch {
      throw new Error('WATCHDOG_GITHUB_UNAVAILABLE');
    }
    if (!response.ok) throw new Error('WATCHDOG_GITHUB_ERROR');
    return response;
  }
  // Fail closed when the active-run list is truncated or malformed.
  for (const status of ['queued', 'in_progress']) {
    const response = await request(`runs?status=${status}&per_page=100`);
    let data;
    try { data = await response.json(); } catch { throw new Error('WATCHDOG_INVALID_RESPONSE'); }
    if (!Number.isInteger(data.total_count) || !Array.isArray(data.workflow_runs) ||
        data.total_count !== data.workflow_runs.length ||
        data.workflow_runs.some(run => typeof run.path !== 'string')) {
      throw new Error('WATCHDOG_INVALID_RESPONSE');
    }
    if (data.workflow_runs.some(run => /^\.github\/workflows\/(main|supplement)\.yml(?:@.*)?$/.test(run.path))) {
      return 'active_sender';
    }
  }
  const response = await request('workflows/keep-alive.yml');
  let workflow;
  try { workflow = await response.json(); } catch { throw new Error('WATCHDOG_INVALID_RESPONSE'); }
  if (typeof workflow.state !== 'string') throw new Error('WATCHDOG_INVALID_RESPONSE');
  if (workflow.state !== 'active') {
    await request('workflows/keep-alive.yml/enable', { method: 'PUT' });
    return 'enabled_supervisor';
  }
  await request('workflows/keep-alive.yml/dispatches', {
    method: 'POST', body: JSON.stringify({ ref: 'main', inputs: { recover: 'true' } }),
  });
  return 'requested_supervisor';
}

export default {
  async scheduled(_event, env) {
    await check(env);
  },
  fetch() {
    return new Response('Not found', { status: 404 });
  },
};
