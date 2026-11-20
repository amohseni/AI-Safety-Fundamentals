// Link check: every external href in docs/**/*.html must answer with a 2xx or 3xx status.
// Usage: node tools/check-links.mjs [--json]
// Each URL gets a HEAD request, then a GET if the HEAD fails. 6 requests run at a time, each with
// a 20 s timeout. A 401, 403, or 429 (or a 202 challenge page) from a host that blocks scripts is
// reported as "unverified, check by hand" and does not fail the run. A doi.org link whose publisher
// blocks scripts counts as ok when doi.org itself confirms that the DOI is registered.
// Exit code 1 if any link is broken.
// Uses curl, so it follows the proxy and certificate settings of the shell it runs in.
import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const docs = path.join(root, 'docs');
const CONCURRENCY = 6;
const TIMEOUT_S = 20;
const UA = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36';

function htmlFiles(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) return e.name === 'vendor' ? [] : htmlFiles(p);
    return e.name.endsWith('.html') ? [p] : [];
  });
}

const where = new Map(); // url -> files that link to it
for (const file of htmlFiles(docs)) {
  const text = fs.readFileSync(file, 'utf8');
  for (const m of text.matchAll(/href="(https?:\/\/[^"]+)"/g)) {
    const url = m[1].replace(/&amp;/g, '&');
    if (!where.has(url)) where.set(url, new Set());
    where.get(url).add(path.relative(docs, file));
  }
}

function curl(url, head) {
  const args = ['-s', '-L', '-o', '/dev/null', '-m', String(TIMEOUT_S), '-A', UA, '-w', '%{http_code} %{url_effective}'];
  if (head) args.push('-I');
  args.push(url);
  return new Promise((resolve) => {
    execFile('curl', args, { timeout: (TIMEOUT_S + 5) * 1000 }, (err, stdout) => {
      const [code, final] = String(stdout || '').trim().split(' ');
      resolve({ status: Number(code) || 0, final: final || url });
    });
  });
}

// doi.org's handle API answers {"responseCode": 1, ...} for a registered DOI.
function doiRegistered(doi) {
  const api = 'https://doi.org/api/handles/' + doi.split('/').map(encodeURIComponent).join('/');
  return new Promise((resolve) => {
    execFile('curl', ['-s', '-m', String(TIMEOUT_S), api], { timeout: (TIMEOUT_S + 5) * 1000 }, (err, stdout) => {
      try { resolve(JSON.parse(stdout).responseCode === 1); } catch { resolve(false); }
    });
  });
}

const good = (s) => s >= 200 && s < 400 && s !== 202;
const blocked = (s) => s === 401 || s === 403 || s === 429 || s === 202;

async function check(url) {
  let r = await curl(url, true);
  if (!good(r.status)) {
    const g = await curl(url, false);
    if (good(g.status) || !blocked(r.status) || g.status !== 0) r = g;
  }
  let verdict = good(r.status) ? 'ok' : blocked(r.status) ? 'unverified' : 'broken';
  let note = '';
  if (verdict === 'unverified' && url.startsWith('https://doi.org/') && await doiRegistered(url.slice(16))) {
    verdict = 'ok';
    note = 'DOI registered; the publisher page answered ' + r.status;
  }
  return { url, status: r.status, final: r.final, verdict, note, files: [...where.get(url)] };
}

const urls = [...where.keys()].sort();
const results = [];
let next = 0;
await Promise.all(Array.from({ length: CONCURRENCY }, async () => {
  while (next < urls.length) {
    const url = urls[next++];
    results.push(await check(url));
  }
}));
results.sort((a, b) => a.url.localeCompare(b.url));

const count = (v) => results.filter((r) => r.verdict === v).length;
if (process.argv.includes('--json')) {
  console.log(JSON.stringify(results, null, 2));
} else {
  for (const v of ['broken', 'unverified']) {
    for (const r of results.filter((x) => x.verdict === v)) {
      const label = v === 'broken' ? 'BROKEN    ' : 'unverified';
      const status = r.status || 'no answer';
      console.log(`${label} ${status}  ${r.url}  (${r.files.join(', ')})${v === 'unverified' ? '  check by hand' : ''}`);
    }
  }
  const viaDoi = results.filter((r) => r.note).length;
  console.log(`${urls.length} external links in ${new Set(results.flatMap((r) => r.files)).size} files: ${count('ok')} ok` +
    (viaDoi ? ` (${viaDoi} of them confirmed as registered DOIs because the publisher blocks scripts)` : '') +
    `, ${count('unverified')} unverified (check by hand), ${count('broken')} broken`);
}
process.exit(count('broken') ? 1 : 0);
