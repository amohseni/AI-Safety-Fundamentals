// Numerical tests behind the claims on page 7.2. Run: node tools/test-page-7-2.mjs [directions.json]
// With an argument, also writes each probe's direction for tools/truth/steer.py.
import fs from 'node:fs';
import { decodeTruth } from '../docs/js/lib/s6l-truth.js';
import { trainProbe, scorecard, TRAIN_SETS } from '../docs/js/lib/s6l-belief.js';
import { rng } from '../docs/js/lib/s6l-probe.js';

const T = decodeTruth(JSON.parse(fs.readFileSync(new URL('../docs/data/truth/statements.json', import.meta.url))));
const pct = (v) => (v === null ? '  -  ' : `${(v * 100).toFixed(1)}`);
const dirs = {};
for (const [method, sets] of Object.entries(TRAIN_SETS)) {
  for (const [set] of sets) {
    const t0 = Date.now();
    const p = trainProbe(T, method, set);
    const c = scorecard(T, p);
    console.log(`${method}-${set} (${Date.now() - t0} ms)${p.loss !== undefined ? ` loss ${p.loss.toFixed(4)}` : ''}: truth ${pct(c.truth)} | neg ${pct(c.neg)} para ${pct(c.para)} |P+Pneg-1| ${c.sum === null ? '-' : c.sum.toFixed(3)} mean sum ${c.sumMean === null ? '-' : c.sumMean.toFixed(3)} | spanish ${pct(c.spanish)} elements ${pct(c.elements)} spanishNeg ${pct(c.spanishNeg)} elementsNeg ${pct(c.elementsNeg)}${c.alice !== null ? ` | agrees with Alice ${pct(c.alice)}` : ''}`);
    dirs[`${method}-${set}`] = { basis: p.basis, w: Array.from(p.w) };
  }
}
if (process.argv[2]) {
  // A random direction in the statements' reduced space, as a control for steering.
  const r = rng(5);
  dirs.random = { basis: 'statements', w: Array.from({ length: T.k }, () => r() - 0.5) };
  // Four more random directions (2026-11-06 review), to show how much one random direction varies.
  for (let n = 2; n <= 5; n++) dirs[`random${n}`] = { basis: 'statements', w: Array.from({ length: T.k }, () => r() - 0.5) };
  const test = T.S.filter((s) => s.topic === 'cities' && s.form === 'plain' && !T.isTrain(s)).map((s) => s.i);
  // 56 held-out city statements: both statements (true and false) of every fourth held-out city.
  const pick = test.filter((_, j) => j % 8 < 2);
  const train = T.S.filter((s) => s.topic === 'cities' && s.form === 'plain' && T.isTrain(s)).map((s) => s.i);
  fs.writeFileSync(process.argv[2], JSON.stringify({ strengths: [-3, -2, -1, 0, 1, 2, 3], test_rows: pick, train_rows: train, directions: dirs }));
  console.log('wrote', process.argv[2], pick.length, 'test statements');
}

// Task 1: how the plain-trained probe treats negated statements, by whether the country is right.
{
  const p = trainProbe(T, 'supervised', 'plain');
  const neg = T.S.filter((s) => s.topic === 'cities' && s.form === 'neg' && !T.isTrain(s));
  const rightCountry = neg.filter((s) => s.label === 0), wrongCountry = neg.filter((s) => s.label === 1);
  const callTrue = (l) => l.filter((s) => p.credence(s) > 0.5).length / l.length;
  console.log(`plain-trained probe on held-out negations: "not in <its own country>" called true ${pct(callTrue(rightCountry))}; "not in <another country>" called true ${pct(callTrue(wrongCountry))}`);
}

// Use: the steering runs (docs/data/7-2/use.json, from tools/truth/steer.py).
{
  const use = JSON.parse(fs.readFileSync(new URL('../docs/data/7-2/use.json', import.meta.url)));
  if (use.TEMP) { console.log('use.json is a placeholder'); process.exit(1); }
  if (process.argv[2]) process.exit(0);
  const lab = use.rows.map((i) => T.S[i].label);
  const acc = (d) => d.filter((v, k) => (v > 0 ? 1 : 0) === lab[k]).length / d.length;
  console.log(`unsteered: model answers ${pct(acc(use.base))}% correctly; mean P(TRUE)-P(FALSE) ${(use.base.reduce((a, b) => a + b, 0) / use.base.length).toFixed(3)}`);
  for (const [name, P] of Object.entries(use.probes)) {
    const parts = use.strengths.map((s, j) => {
      if (s === 0) return '0: -';
      const want = s > 0 ? 1 : -1;
      let n = 0, m = 0;
      use.base.forEach((v, k) => { if (Math.sign(v) === -want) { n++; if (Math.sign(P.diff[j][k]) === want) m++; } });
      return `${s}: moved ${m}/${n} acc ${pct(acc(P.diff[j]))}`;
    });
    console.log(`  ${name} (sep ${P.sep}): ${parts.join(' | ')}`);
  }
  // Task 3 notice: the probe's +2 count lies inside the spread of the five random directions.
  const movedAt = (P, st) => { const j = use.strengths.indexOf(st), want = Math.sign(st); let m = 0; use.base.forEach((v, k) => { if (Math.sign(v) === -want && Math.sign(P.diff[j][k]) === want) m++; }); return m; };
  const rnd = Object.keys(use.probes).filter((k) => k.startsWith('random')).map((k) => movedAt(use.probes[k], 2)).sort((x, y) => x - y);
  const sp = movedAt(use.probes['supervised-plain'], 2);
  const ok = sp === 21 && rnd.join(',') === '0,8,9,23,28' && movedAt(use.probes['supervised-plain'], -2) === 11;
  console.log(`${ok ? 'ok  ' : 'FAIL'} task 3: probe turns ${sp}/31 at +2; random directions ${rnd.join(', ')}`);
  if (!ok) process.exitCode = 1;
}
