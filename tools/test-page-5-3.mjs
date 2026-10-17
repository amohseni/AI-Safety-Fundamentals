// Numbers quoted on page 5.3. Run: node tools/test-page-5-3.mjs
// Default noise 0.3; feature draws use seeds 1, 2, ... as the tool's "New features" button does.
import { makeData, makeFeatures, fit, P_GRID, N_TRAIN } from '../docs/js/lib/s45-features.js';
let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) fail++; };
const NOISE = 0.3, DRAWS = 20;
const data = makeData(NOISE);
const peaks = [], logMean = P_GRID.map(() => 0), rows = [];
for (let s = 1; s <= DRAWS; s++) {
  const F = makeFeatures(data.X, s);
  const res = P_GRID.map((p) => fit(data, F, p, 0));
  const te = res.map((r) => r.test);
  te.forEach((v, i) => { logMean[i] += Math.log(v) / DRAWS; });
  const ip = te.indexOf(Math.max(...te));
  peaks.push(P_GRID[ip]);
  const firstInterp = P_GRID.find((p, i) => res[i].train < 1e-6);
  const left = te.filter((_, i) => P_GRID[i] < 30), right = te.filter((_, i) => P_GRID[i] >= 200);
  rows.push({ s, peak: P_GRID[ip], top: te[ip], minLeft: Math.min(...left), at40: te[P_GRID.indexOf(40)], at400: te[P_GRID.indexOf(400)], train40: res[P_GRID.indexOf(40)].train, firstInterp, rightMax: Math.max(...right) });
}
rows.forEach((r) => console.log(`     draw ${r.s}: peak at p = ${r.peak} (test error ${r.top.toFixed(2)}); p = 40: ${r.at40.toFixed(2)} (train ${r.train40.toExponential(0)}); p = 400: ${r.at400.toFixed(3)}; best p < 30: ${r.minLeft.toFixed(3)}; first p with train error < 1e-6: ${r.firstInterp}`));
const near = peaks.filter((p) => p >= 36 && p <= 44).length;
check(near >= 15, `peak within 36 to 44 (n = ${N_TRAIN}) in ${near} of ${DRAWS} draws; peaks: ${peaks.join(', ')}`);
const im = logMean.indexOf(Math.max(...logMean));
check(Math.abs(P_GRID[im] - N_TRAIN) <= 2, `geometric mean of test error over ${DRAWS} draws peaks at p = ${P_GRID[im]} (${Math.exp(logMean[im]).toFixed(2)}); at p = 400 ${Math.exp(logMean[P_GRID.indexOf(400)]).toFixed(3)}; best below 30 ${Math.exp(Math.min(...logMean.slice(0, P_GRID.indexOf(30)))).toFixed(3)}`);
check(rows.filter((r) => r.at400 < r.at40).length === DRAWS, 'p = 400 beats p = 40 in every draw');
console.log(`     p = 400 vs best p < 30: p = 400 lower in ${rows.filter((r) => r.at400 < r.minLeft).length} of ${DRAWS} draws`);

// Ridge removes the peak: largest test error over p, with ridge.
for (const ridge of [1e-4, 1e-3, 1e-2]) {
  const tops = [];
  for (let s = 1; s <= DRAWS; s++) {
    const F = makeFeatures(data.X, s);
    const te = P_GRID.map((p) => fit(data, F, p, ridge).test);
    tops.push(Math.max(...te.filter((_, i) => P_GRID[i] >= 20)));
  }
  console.log(`     ridge ${ridge}: largest test error for p >= 20, over draws: median ${tops.sort((a, b) => a - b)[DRAWS / 2].toFixed(3)}, max ${tops.at(-1).toFixed(3)}`);
  if (ridge === 1e-3) check(tops.at(-1) < 0.1, `ridge 0.001 removes the peak in every draw (largest error for p >= 20: ${tops.at(-1).toFixed(3)})`);
}
// Smaller noise: peak smaller.
for (const nz of [0, 0.1]) {
  const d = makeData(nz), F = makeFeatures(d.X, 1);
  const te = P_GRID.map((p) => fit(d, F, p, 0).test);
  console.log(`     noise ${nz}, draw 1: largest test error ${Math.max(...te).toFixed(3)} at p = ${P_GRID[te.indexOf(Math.max(...te))]}`);
}
process.exit(fail ? 1 : 0);
