// Runs every tools/test-*.mjs in sequence and prints one line per script: name, pass or fail, seconds.
// Exits nonzero if any script fails. A script passes when it exits with code 0.
// Run: node tools/run-tests.mjs            all scripts (about eleven minutes on 2 cores)
//      node tools/run-tests.mjs --fast     skips the scripts in SLOW (about four and a half minutes)
//      node tools/run-tests.mjs 2-6 8-1    only the scripts whose names contain one of these strings
// On a failure the script's FAIL lines (or the end of its output) are printed under its line.
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

// Scripts that take more than about 60 seconds on a 2-core machine (measured 2026-11-19, Node 22).
const SLOW = ['test-page-2-3.mjs', 'test-page-5-5.mjs', 'test-page-6-3.mjs', 'test-page-6-4.mjs', 'test-page-7-3.mjs'];

const dir = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2), fast = args.includes('--fast'), only = args.filter((a) => !a.startsWith('--'));
let files = fs.readdirSync(dir).filter((f) => /^test-.*\.mjs$/.test(f)).sort();
if (only.length) files = files.filter((f) => only.some((o) => f.includes(o)));
const skipped = fast ? files.filter((f) => SLOW.includes(f)) : [];
files = files.filter((f) => !skipped.includes(f));

const failed = [], t00 = Date.now();
for (const f of files) {
  const t0 = Date.now();
  const r = spawnSync(process.execPath, [path.join(dir, f)], { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 });
  const secs = ((Date.now() - t0) / 1000).toFixed(1), ok = r.status === 0;
  console.log(`${f.padEnd(24)} ${ok ? 'pass' : 'FAIL'} ${secs.padStart(6)} s`);
  if (!ok) {
    failed.push(f);
    const out = `${r.stdout || ''}${r.stderr || ''}`.trimEnd().split('\n');
    const lines = out.filter((l) => l.startsWith('FAIL'));
    for (const l of lines.length ? lines : out.slice(-15)) console.log(`    ${l.length > 400 ? l.slice(0, 400) + ' ...' : l}`);
    if (r.status === null) console.log(`    (no exit code: ${r.signal || r.error})`);
  }
}
for (const f of skipped) console.log(`${f.padEnd(24)} skipped (slow)`);
console.log(`\n${files.length - failed.length} of ${files.length} scripts passed in ${((Date.now() - t00) / 1000).toFixed(0)} s` +
  (skipped.length ? `; ${skipped.length} slow scripts skipped` : '') + (failed.length ? `; failed: ${failed.join(', ')}` : ''));
process.exit(failed.length ? 1 : 0);
