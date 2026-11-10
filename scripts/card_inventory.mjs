// Optional encrypted transport; cache failures never authorize or block SMTP.
import fs from 'node:fs';
import path from 'node:path';
import {createCipheriv, createDecipheriv, hkdfSync, randomBytes} from 'node:crypto';
import {spawnSync} from 'node:child_process';

const magic = Buffer.from('EPCARD1');
const limit = 4 * 1024 * 1024;
const directory = process.env.CARD_STORE_DIR || '.cache/reading-cards';
const bundle = process.env.CARD_INVENTORY_FILE || '.cache/card-inventory.enc';
const secret = process.env.CARD_CACHE_SECRET;
const validName = name => /^[a-f0-9]{64}\.json$/.test(name);

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
    // Restores newest artifacts first; an older snapshot cannot replace them.
    if (fs.existsSync(target)) continue;
    const temporary = target + '.tmp';
    fs.writeFileSync(temporary, item.content, {mode: 0o600});
    fs.renameSync(temporary, target);
  }
}

async function restore() {
  const repo = process.env.GITHUB_REPOSITORY;
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo || '') || !process.env.GITHUB_TOKEN || !secret) return;
  const deadline = Date.now() + 60000;
  const base = 'https://api.github.com/repos/' + repo;
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

try {
  const mode = process.argv[2];
  if (mode === 'restore') await restore();
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
