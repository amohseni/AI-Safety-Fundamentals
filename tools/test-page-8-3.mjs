// Numbers quoted on page 8.3, recomputed with the page's own agent and settings (docs/js/lib/8-coinrun.js):
// 6,000 training episodes, training runs with seeds 1 to 10, success on 500 standard levels (seed 7),
// and on 2,000 random test levels (seed 999). Run: node tools/test-page-8-3.mjs
import { train, evaluate } from '../docs/js/lib/8-coinrun.js';

for (const p of [0, 0.02, 0.1, 0.5]) {
  const rows = [];
  for (let seed = 1; seed <= 10; seed++) {
    const th = train(p, { seed });
    const tr = evaluate(th, { pRandom: 0, levels: 500, seed: 7 });
    const te = evaluate(th, { levels: 2000, record: true });
    rows.push([tr.success, te.success, te.toEnd]);
  }
  const col = (i) => rows.map((r) => Math.round(100 * r[i]));
  const rng = (a) => `${Math.min(...a)} to ${Math.max(...a)}`;
  console.log(`${(100 * p).toFixed(0).padStart(2)}% randomized | standard levels ${rng(col(0))}% | test success ${col(1).join(' ')} | reaches right end ${col(2).join(' ')}`);
}
console.log('Page: 0% -> test 22 to 30, right end about 80; 2% -> 9 of 10 runs 84 to 99, one run 32; 10% -> all at least 95.');
// Training samples actions, so last-bit differences in floating point between JavaScript engines can change a run.
// In Chromium (2026-10-16) the 2% runs gave 87 97 93 98 95 84 96 98 32 99; Node 22 gives the values above.
