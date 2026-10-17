// Numbers quoted on page 8.4, recomputed with the page's own models (docs/js/lib/8-power.js), same seeds as the page.
// Run: node tools/test-page-8-4.mjs
import { defaultGraph, SHUTDOWN, sampleRewards, analyze, allocate } from '../docs/js/lib/8-power.js';
import { graphValues } from '../docs/js/lib/gridworld.js';

const pct = (v) => `${(100 * v).toFixed(1)}%`;
const g = defaultGraph();
const withOff = { nodes: [...g.nodes, SHUTDOWN], edges: [...g.edges, ['start', 'off']] };
function shares(G, dist, gamma, seed = 1) {
  const a = analyze(G, sampleRewards(dist, G.nodes.length, 1000, seed), gamma);
  return Object.fromEntries([...a.share].map(([t, v]) => [G.nodes[t].id, v]));
}
// Check the exact solver against plain value iteration on a few samples.
{
  const { succ } = analyze(g, [], 0.9);
  let worst = 0;
  for (const r of sampleRewards('normal', g.nodes.length, 50, 3)) {
    const V = graphValues(succ, r, 0.95);
    let W = Float64Array.from(r);
    for (let k = 0; k < 3000; k++) W = Float64Array.from(r, (x, s) => x + 0.95 * Math.max(...succ[s].map((t) => W[t])));
    worst = Math.max(worst, ...V.map((v, i) => Math.abs(v - W[i])));
  }
  console.log('Exact solver vs value iteration, largest difference:', worst.toExponential(1));
}
const s1 = shares(g, 'uniform', 0.9);
console.log('Task 1 (uniform, 0.9): hub', pct(s1.hub), 'dead end', pct(s1.dead), 'stay', pct(s1.start), '(page: 69, 16, 15)');
console.log('        normal 0.9 hub', pct(shares(g, 'normal', 0.9).hub), '(page 71); sparse 0.9 hub', pct(shares(g, 'sparse', 0.9).hub), '(page 73)');
console.log('Task 2 (uniform, 0.1): hub', pct(shares(g, 'uniform', 0.1).hub), '(page 36); sparse 0.1 hub', pct(shares(g, 'sparse', 0.1).hub), '(page 73)');
console.log('        hub share by discount:', [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99].map((x) => `${x}: ${pct(shares(g, 'uniform', x).hub)}`).join(', '));
for (const gamma of [0.1, 0.9, 0.99]) { const s = shares(withOff, 'uniform', gamma); console.log(`Task 3 (uniform, ${gamma}): shutdown first ${pct(s.off)}, avoided ${pct(1 - s.off)}`); }
console.log('        page: 13% at 0.9 (87% avoid), 90% avoid at 0.99, 78% at 0.1');
for (const [shape, floor] of [['diminishing', -3], ['diminishing', -6], ['linear', -3]]) {
  const tr = allocate({ L: 4, J: 3, shape, floor });
  let peak = -Infinity, pt = 0; tr.forEach((s, t) => { if (s.true > peak) { peak = s.true; pt = t; } });
  const e = tr[tr.length - 1];
  console.log(`Task 4 (${shape}, floor ${floor}): omitted ends at ${e.x[3].toFixed(2)}, in proxy ${e.x[0].toFixed(2)}; true utility peaks ${peak.toFixed(2)} (step ${pt}), ends ${e.true.toFixed(2)}`);
}
console.log('        page: omitted to -3; utility 0 -> 2.20 -> -16.4; floor -6 ends -400; linear stays at peak');
// Check question 2: 1 absorbing state versus a region of 20, no staying at the start, gamma near 1.
{
  const n = 23; // 0 start, 1 lone state, 2 region entry, 3..22 region states
  const succ = [[1, 2], [1], [2, ...Array.from({ length: 20 }, (_, i) => 3 + i)], ...Array.from({ length: 20 }, (_, i) => [3 + i, 2])];
  let region = 0; const S = sampleRewards('uniform', n, 4000, 5);
  for (const r of S) { const V = graphValues(succ, r, 0.999); if (V[2] > V[1]) region++; }
  console.log('Check 2: share choosing the region', pct(region / S.length), '(page: about 95%; 20 of 21 is 95.2%)');
}
