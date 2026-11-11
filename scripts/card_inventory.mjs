// Optional encrypted transport; cache failures never authorize or block SMTP.
import fs from 'node:fs';
import path from 'node:path';
import {createCipheriv, createDecipheriv, hkdfSync, randomBytes} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const magic = Buffer.from('EPCARD1');
const limit = 4 * 1024 * 1024;
const directory = process.env.CARD_STORE_DIR || '.cache/reading-cards';
const bundle = process.env.CARD_INVENTORY_FILE || '.cache/card-inventory.enc';
const secret = process.env.CARD_CACHE_SECRET;
const validName = name => /^[a-f0-9]{64}\.json$/.test(name);
const branch = 'codex/reading-card-checkpoints';
const remotePath = '.github/card-inventory.enc';

function key(salt) {
  if (!secret) throw new Error('Missing cache key');
  return hkdfSync('sha256', secret, salt, 'eat-paper-private-card-inventory-v1', 32);
}

function seal(payload) {
  const plain = Buffer.from(JSON.stringify(payload));
  if (plain.length > limit) throw new Error('Inventory too large');
  const salt = randomBytes(16), nonce = randomBytes(12);
  const cipher = createCipheriv('aes-256-gcm', key(salt), nonce);
  cipher.setAAD(magic);
  const encrypted = Buffer.concat([cipher.update(plain), cipher.final()]);
  return Buffer.concat([magic, salt, nonce, cipher.getAuthTag(), encrypted]);
}

function unseal(data) {
  const offset = magic.length;
  if (data.length < offset + 44 || data.length > limit + 100 || !data.subarray(0, offset).equals(magic)) throw new Error('Invalid inventory');
  const decipher = createDecipheriv('aes-256-gcm', key(data.subarray(offset, offset + 16)), data.subarray(offset + 16, offset + 28));
  decipher.setAAD(magic);
  decipher.setAuthTag(data.subarray(offset + 28, offset + 44));
  const payload = JSON.parse(Buffer.concat([decipher.update(data.subarray(offset + 44)), decipher.final()]));
  if (payload.version !== 1 || !Array.isArray(payload.files) || payload.files.length > 30) throw new Error('Invalid inventory schema');
  for (const item of payload.files) {
    if (!validName(item.name) || typeof item.content !== 'string' || item.content.length > 200000) throw new Error('Invalid inventory entry');
    JSON.parse(item.content);
  }
  return payload;
}

function install(payload) {
  fs.mkdirSync(directory, {recursive: true, mode: 0o700});
  for (const item of payload.files) {
    const target = path.join(directory, item.name);
    // Compare saved times so a stale local snapshot cannot replace a newer one.
    if (fs.existsSync(target)) {
      const local = {name: item.name, content: fs.readFileSync(target, 'utf8')};
      if (savedAt(local) >= savedAt(item)) continue;
    }
    const temporary = target + '.tmp';
    fs.writeFileSync(temporary, item.content, {mode: 0o600});
    fs.renameSync(temporary, target);
  }
}

function savedAt(item) {
  try { return Date.parse(JSON.parse(item.content).saved_at) || 0; } catch { return 0; }
}

function localFiles() {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory).filter(validName)
    .sort((a, b) => fs.statSync(path.join(directory, b)).mtimeMs - fs.statSync(path.join(directory, a)).mtimeMs)
    .slice(0, 30).map(name => ({name, content: fs.readFileSync(path.join(directory, name), 'utf8')}));
}

function mergeFiles(left, right) {
  const files = new Map();
  for (const item of [...left, ...right]) {
    if (!validName(item.name) || typeof item.content !== 'string') throw new Error('Invalid inventory entry');
    JSON.parse(item.content);
    if (!files.has(item.name) || savedAt(item) > savedAt(files.get(item.name))) files.set(item.name, item);
  }
  return [...files.values()].sort((a, b) => savedAt(b) - savedAt(a)).slice(0, 30);
}

function githubBase() {
  const repo = process.env.GITHUB_REPOSITORY;
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo || '') || !process.env.GITHUB_TOKEN || !secret) return null;
  return 'https://api.github.com/repos/' + repo;
}

async function githubRequest(url, deadline, options = {}, timeout = 3000) {
  if (!url.startsWith('https://api.github.com/repos/' + process.env.GITHUB_REPOSITORY + '/')) throw new Error('Invalid GitHub URL');
  const remaining = deadline - Date.now();
  if (remaining <= 0) throw new Error('Inventory deadline');
  return fetch(url, {...options, headers: {Authorization: 'Bearer ' + process.env.GITHUB_TOKEN,
    Accept: 'application/vnd.github+json', 'Content-Type': 'application/json'}, redirect: 'manual',
    signal: AbortSignal.timeout(Math.min(timeout, remaining))});
}

async function remoteInventory(base, deadline) {
  const response = await githubRequest(base + '/contents/' + remotePath + '?ref=' + encodeURIComponent(branch), deadline);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error('Checkpoint unavailable');
  let data = await response.json();
  if (data.encoding === 'none' && /^[a-f0-9]{40}$/i.test(data.sha || '')) {
    const blob = await githubRequest(base + '/git/blobs/' + data.sha, deadline);
    if (!blob.ok) throw new Error('Checkpoint blob unavailable');
    const blobData = await blob.json();
    data = {...blobData, sha: data.sha};
  }
  if (data.encoding !== 'base64' || typeof data.content !== 'string' || typeof data.sha !== 'string') throw new Error('Invalid checkpoint');
  return {sha: data.sha, payload: unseal(Buffer.from(data.content.replace(/\s/g, ''), 'base64'))};
}

export async function checkpoint() {
  const base = githubBase();
  if (!base) return;
  const deadline = Date.now() + 8000;
  if (!localFiles().length) return;
  for (let attempt = 0; attempt < 3; attempt++) {
    const remote = await remoteInventory(base, deadline);
    if (!remote) {
      const main = await githubRequest(base + '/git/ref/heads/main', deadline);
      if (!main.ok) throw new Error('Main branch unavailable');
      const mainData = await main.json();
      const sha = mainData.object?.sha;
      if (typeof sha !== 'string' || !/^[a-f0-9]{40}$/i.test(sha)) throw new Error('Invalid main reference');
      const created = await githubRequest(base + '/git/refs', deadline, {method: 'POST', body: JSON.stringify({ref: 'refs/heads/' + branch, sha})});
      if (!created.ok && created.status !== 422) throw new Error('Checkpoint branch unavailable');
    }
    const files = mergeFiles(remote?.payload.files || [], localFiles());
    if (!files.length) return;
    const payload = {message: 'Update encrypted reading-card inventory', branch,
      content: seal({version: 1, files}).toString('base64')};
    if (remote) payload.sha = remote.sha;
    const put = await githubRequest(base + '/contents/' + remotePath, deadline,
      {method: 'PUT', body: JSON.stringify(payload)});
    if (put.ok) return;
    if (![409, 422].includes(put.status) || attempt === 2) throw new Error('Checkpoint write unavailable');
  }
}

export async function restore() {
  const repo = process.env.GITHUB_REPOSITORY;
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo || '') || !process.env.GITHUB_TOKEN || !secret) return;
  const deadline = Date.now() + 60000;
  const base = 'https://api.github.com/repos/' + repo;
  try {
    const remote = await remoteInventory(base, deadline);
    if (remote) install(remote.payload);
  } catch (error) {
    console.log('Card checkpoint skipped (' + error.name + ')');
  }
  async function request(url, authenticated = true) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) throw new Error('Inventory deadline');
    return fetch(url, {headers: authenticated ? {Authorization: 'Bearer ' + process.env.GITHUB_TOKEN, Accept: 'application/vnd.github+json'} : {}, redirect: 'manual', signal: AbortSignal.timeout(Math.min(15000, remaining))});
  }
  const response = await request(base + '/actions/artifacts?per_page=100');
  if (!response.ok) throw new Error('Inventory listing unavailable');
  const data = await response.json();
  const artifacts = (data.artifacts || []).filter(a => !a.expired && /^reading-card-inventory-\d+-\d+$/.test(a.name) && a.workflow_run?.head_branch === 'main').sort((a, b) => b.id - a.id).slice(0, 3);
  let recovered = 0;
  for (const artifact of artifacts) {
    try {
      let download = await request(base + '/actions/artifacts/' + artifact.id + '/zip');
      if (download.status === 302) {
        const location = new URL(download.headers.get('location'));
        if (location.protocol !== 'https:' || !/(^|\.)(blob\.core\.windows\.net|githubusercontent\.com)$/.test(location.hostname)) throw new Error('Invalid artifact redirect');
        download = await request(location.href, false);
      }
      if (!download.ok) throw new Error('Inventory download unavailable');
      const chunks = [];
      let size = 0;
      for await (const chunk of download.body) {
        size += chunk.length;
        if (size > limit + 100000) throw new Error('Archive too large');
        chunks.push(chunk);
      }
      // stdlib ZIP reader extracts exactly one fixed encrypted entry, no paths.
      const python = "import io,sys,zipfile; z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())); e=[i for i in z.infolist() if i.filename=='card-inventory.enc']; assert len(e)==1 and e[0].file_size<4300000; sys.stdout.buffer.write(z.read(e[0]))";
      const result = spawnSync('python3', ['-c', python], {input: Buffer.concat(chunks), timeout: 5000, maxBuffer: limit + 100000});
      if (result.status !== 0) throw new Error('Invalid inventory archive');
      const payload = unseal(result.stdout);
      install(payload);
      recovered += payload.files.length;
    } catch (error) {
      console.log('Card inventory snapshot skipped (' + error.name + ')');
    }
  }
  console.log('Card inventory restored entries: ' + recovered);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) try {
  const mode = process.argv[2];
  if (mode === 'restore') await restore();
  else if (mode === 'checkpoint') await checkpoint();
  else if (mode === 'pack') {
    if (fs.existsSync(directory) && secret) {
      const names = fs.readdirSync(directory).filter(validName).sort((a, b) => fs.statSync(path.join(directory, b)).mtimeMs - fs.statSync(path.join(directory, a)).mtimeMs).slice(0, 30);
      const files = names.map(name => ({name, content: fs.readFileSync(path.join(directory, name), 'utf8')}));
      if (files.length) {
        fs.mkdirSync(path.dirname(bundle), {recursive: true});
        fs.writeFileSync(bundle, seal({version: 1, files}), {mode: 0o600});
        console.log('Card inventory encrypted entries: ' + files.length);
      }
    }
  } else if (mode === 'unpack') install(unseal(fs.readFileSync(bundle)));
  else throw new Error('Unknown inventory mode');
} catch (error) {
  console.log('Optional card inventory unavailable (' + error.name + ')');
  // No plaintext fallback; the sender can always rebuild through its old path.
}
