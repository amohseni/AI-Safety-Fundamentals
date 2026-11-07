// Numerical tests behind the claims on page 6.3. Run: node tools/test-page-6-3.mjs [quick]
import fs from 'node:fs';
import { decodeProbeData, split, PROPERTIES } from '../docs/js/lib/s6l-tasks.js';
import { logistic, mlp, accuracy } from '../docs/js/lib/s6l-probe.js';
import { PROBE_DEFAULTS } from '../docs/js/lib/s6l-probe-settings.js';

const D = decodeProbeData(JSON.parse(fs.readFileSync(new URL('../docs/data/6-3/probe-data.json', import.meta.url))));
const pct = (v) => `${(v * 100).toFixed(1)}`;
const quick = process.argv[2] === 'quick';
const onlyDirections = process.argv[2] === 'directions';
for (const prop of onlyDirections ? [] : Object.keys(PROPERTIES)) {
  const t = D.tasks[prop];
  console.log(`\n${PROPERTIES[prop]}: ${t.idx.length} examples, ${t.train.filter(Boolean).length} for training, positives ${t.y.filter(Boolean).length}`);
  for (const L of D.layers) {
    const s = split(t, D.acts(prop, L));
    const line = [];
    for (const [name, train] of [
      ['linear', (X, y) => logistic(X, y, PROBE_DEFAULTS.linear)],
      ...(quick ? [] : [['mlp', (X, y) => mlp(X, y, PROBE_DEFAULTS.mlp)]]),
    ]) {
      const p = train(s.Xtr, s.ytr), c = train(s.Xtr, s.yctr);
      const real = accuracy(p, s.Xte, s.yte), ctl = accuracy(c, s.Xte, s.ycte);
      line.push(`${name} train ${pct(accuracy(p, s.Xtr, s.ytr))} test ${pct(real)} control ${pct(ctl)} (train ${pct(accuracy(c, s.Xtr, s.yctr))}) sel ${pct(real - ctl)}`);
    }
    console.log(`  layer ${String(L).padStart(2)}: ${line.join(' | ')}`);
  }
}

// With "directions <file>", write the default linear probe's direction for each property and layer,
// and a random direction per layer, for tools/truth/ablate.py.
if (process.argv[2] === 'directions') {
  const { rng } = await import('../docs/js/lib/s6l-probe.js');
  const out = { layers: D.layers, probes: {}, random: {} };
  const r = rng(9);
  for (const prop of Object.keys(PROPERTIES)) {
    out.probes[prop] = {};
    for (const L of D.layers) {
      const s = split(D.tasks[prop], D.acts(prop, L));
      out.probes[prop][L] = Array.from(logistic(s.Xtr, s.ytr, PROBE_DEFAULTS.linear).w);
    }
  }
  for (const L of D.layers) out.random[L] = Array.from({ length: D.k }, () => r() - 0.5);
  fs.writeFileSync(process.argv[3], JSON.stringify(out));
  console.log('wrote', process.argv[3]);
}

// Ablation results (docs/data/6-3/ablation.json, from tools/truth/ablate.py), as the page reports them.
if (!onlyDirections) {
  const A = JSON.parse(fs.readFileSync(new URL('../docs/data/6-3/ablation.json', import.meta.url)));
  if (A.TEMP) { console.log('ablation.json is a placeholder'); process.exit(1); }
  console.log(`\nAblation. Base: loss all ${A.base.loss[0].toFixed(3)}, before nouns ${A.base.loss[1].toFixed(3)}, before closing quotes ${A.base.loss[2].toFixed(3)}; judgment accuracy ${pct(A.base.judge[0])}`);
  for (const L of A.layers) {
    const p = A.probe[L], r = A.random[L];
    if (!p || p.noun === null) continue;
    console.log(`  layer ${String(L).padStart(2)}: noun ${p.noun[1].toFixed(2)} (random ${r.noun[1].toFixed(2)}) | quote ${p.quote[2].toFixed(2)} (random ${r.quote[2].toFixed(2)}) | truth ${pct(p.truth[0])} (random ${pct(r.truth[0])})`);
  }
}
