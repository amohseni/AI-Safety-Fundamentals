// Numbers quoted on page 8.4, recomputed with the page's own models (docs/js/lib/8-power.js), same seeds as the page.
// Run: node tools/test-page-8-4.mjs
import { defaultGraph, SHUTDOWN, sampleRewards, analyze, allocate } from '../docs/js/lib/8-power.js';
import { graphValues } from '../docs/js/lib/gridworld.js';

let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const r0 = (v) => Math.round(100 * v);
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
  check('the exact solver agrees with plain value iteration to 1e-9', worst < 1e-9, worst.toExponential(1));
}
const s1 = shares(g, 'uniform', 0.9);
console.log('Task 1 (uniform, 0.9): hub', pct(s1.hub), 'dead end', pct(s1.dead), 'stay', pct(s1.start), '(page: 69, 16, 15)');
console.log('        normal 0.9 hub', pct(shares(g, 'normal', 0.9).hub), '(page 71); sparse 0.9 hub', pct(shares(g, 'sparse', 0.9).hub), '(page 73)');
console.log('Task 2 (uniform, 0.1): hub', pct(shares(g, 'uniform', 0.1).hub), '(page 36); sparse 0.1 hub', pct(shares(g, 'sparse', 0.1).hub), '(page 73)');
console.log('        hub share by discount:', [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99].map((x) => `${x}: ${pct(shares(g, 'uniform', x).hub)}`).join(', '));
for (const gamma of [0.1, 0.9, 0.99]) { const s = shares(withOff, 'uniform', gamma); console.log(`Task 3 (uniform, ${gamma}): shutdown first ${pct(s.off)}, avoided ${pct(1 - s.off)}`); }
console.log('        page: 13% at 0.9 (87% avoid), 90% avoid at 0.99, 78% at 0.1');
const T4 = {};
for (const [shape, floor] of [['diminishing', -3], ['diminishing', -6], ['linear', -3]]) {
  const tr = allocate({ L: 4, J: 3, shape, floor });
  let peak = -Infinity, pt = 0; tr.forEach((s, t) => { if (s.true > peak) { peak = s.true; pt = t; } });
  const e = tr[tr.length - 1];
  T4[`${shape}|${floor}`] = { e, peak, pt, first: tr[0] };
  console.log(`Task 4 (${shape}, floor ${floor}): omitted ends at ${e.x[3].toFixed(2)}, in proxy ${e.x[0].toFixed(2)}; true utility peaks ${peak.toFixed(2)} (step ${pt}), ends ${e.true.toFixed(2)}`);
}
console.log('        page: omitted to -3; utility 0 -> 2.20 -> -16.4; floor -6 ends -400; linear stays at peak');
// Check question 2: 1 absorbing state versus a region of 20, no staying at the start, gamma near 1.
{
  const n = 23; // 0 start, 1 lone state, 2 region entry, 3..22 region states
  const succ = [[1, 2], [1], [2, ...Array.from({ length: 20 }, (_, i) => 3 + i)], ...Array.from({ length: 20 }, (_, i) => [3 + i, 2])];
  let region = 0; const S = sampleRewards('uniform', n, 4000, 5);
  for (const r of S) { const V = graphValues(succ, r, 0.999); if (V[2] > V[1]) region++; }
  // Check 2 feedback: "That state is in the region for 20 of the 21 equally likely cases, about 95%." The simulated share is a
  // sample of 4,000 reward functions at discount 0.999, so it is checked to within 1.5 points of 20/21.
  check('c2: the move into the region is optimal for about 95% of reward functions (20 of 21 = 95.2%; simulated share within 1.5 points)', r0(20 / 21) === 95 && Math.abs(region / S.length - 20 / 21) < 0.015, pct(region / S.length));
  console.log('Check 2: share choosing the region', pct(region / S.length), '(page: about 95%; 20 of 21 is 95.2%)');
}
// Panel note and caption: 1,000 reward functions. Task 1 notice: "The move to the hub is optimal for 69% of the reward functions,
// the dead end for 16%, and staying for 15%. The hub leads to six states the agent can end up in; the dead end and the start lead
// to one each. ... Normal rewards give 71% for the hub, and sparse rewards 73%."
check('t1: uniform rewards, discount 0.9: hub 69%, dead end 16%, staying 15%', r0(s1.hub) === 69 && r0(s1.dead) === 16 && r0(s1.start) === 15, `${s1.hub} ${s1.dead} ${s1.start}`);
check('t1 and caption: these are shares of 1,000 reward functions, and the hub is the best first move for most of them', sampleRewards('uniform', g.nodes.length, 1000, 1).length === 1000 && s1.hub > 0.5, s1.hub);
{
  const out = (id) => g.edges.filter(([a]) => a === id).map(([, b]) => b);
  check('t1: the hub leads to six states the agent can end up in (itself and five more); the dead end leads only to itself', out('hub').length === 5 && out('hub').every((b) => out(b).join() === 'hub') && out('dead').length === 0 && g.nodes.length === 8, `${out('hub').length + 1} ${out('dead').length + 1}`);
}
const hub = (dist, gamma) => shares(g, dist, gamma).hub;
check('t1: normal rewards give 71% for the hub, and sparse rewards 73%', r0(hub('normal', 0.9)) === 71 && r0(hub('sparse', 0.9)) === 73, `${hub('normal', 0.9)} ${hub('sparse', 0.9)}`);
// Task 2 notice: "With uniform rewards it falls to 36%, close to one third. ... With sparse rewards the share does not change with
// the discount factor (73%)".
check('t2: at discount 0.1, uniform rewards give the hub 36%', r0(hub('uniform', 0.1)) === 36, hub('uniform', 0.1));
check('t2: with sparse rewards the share does not change with the discount factor (73% at 0.1, 0.5, 0.9, 0.99)', [0.1, 0.5, 0.9, 0.99].every((x) => hub('sparse', x) === hub('sparse', 0.9)) && r0(hub('sparse', 0.1)) === 73, [0.1, 0.5, 0.9, 0.99].map((x) => hub('sparse', x)).join(' '));
// Task 3 notice: "About 13%: 87% of the reward functions make avoiding shutdown optimal ... At discount factor 0.99 the share
// avoiding it is 90%; at 0.1 it is 78%."
{
  const off = (gamma) => shares(withOff, 'uniform', gamma).off;
  check('t3: with a shutdown state, moving to it is the best first move for about 13% (87% avoid it) at discount 0.9', r0(off(0.9)) === 13 && r0(1 - off(0.9)) === 87, off(0.9));
  check('t3: at discount 0.99 the share avoiding it is 90%; at 0.1 it is 78%', r0(1 - off(0.99)) === 90 && r0(1 - off(0.1)) === 78, `${1 - off(0.99)} ${1 - off(0.1)}`);
}
// Task 4 notice: "It is driven to the floor, -3, and the three attributes in the proxy share what it gave up. The person's utility
// first rises, from 0 to 2.20, ... and then falls to -16.4 ... With the floor at -6 it ends at -400 ... With linear utility the
// omitted attribute is still driven to the floor, but the person's utility stays at its peak".
{
  const a = T4['diminishing|-3'], b = T4['diminishing|-6'], c = T4['linear|-3'];
  check('t4: the omitted attribute is driven to the floor, -3, and the three in the proxy share the budget plus what it gave up (7/3 each)', Math.abs(a.e.x[3] + 3) < 1e-6 && a.e.x.slice(0, 3).every((v) => Math.abs(v - 7 / 3) < 0.01), a.e.x.map((v) => v.toFixed(3)).join(' '));
  check("t4: the person's utility first rises from 0 to 2.20, then falls to -16.4", a.first.true === 0 && a.peak.toFixed(2) === '2.20' && a.pt > 0 && a.e.true.toFixed(1) === '-16.4', `${a.first.true} ${a.peak} ${a.e.true}`);
  check('t4: with the floor at -6 it ends at -400', Math.round(b.e.true) === -400 && Math.abs(b.e.x[3] + 6) < 1e-6, b.e.true);
  check("t4: with linear utility the omitted attribute is still driven to the floor, but the person's utility stays at its peak", Math.abs(c.e.x[3] + 3) < 1e-6 && Math.abs(c.e.true - c.peak) < 1e-9, `${c.e.x[3]} ${c.e.true} ${c.peak}`);
}
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
