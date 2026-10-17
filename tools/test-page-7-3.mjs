// Numbers quoted on page 7.3, recomputed with the page's own network and code.
// Run: node tools/test-page-7-3.mjs
import fs from 'node:fs';
import path from 'node:path';
import * as M from '../docs/js/lib/s73-maze.js';
import * as P from '../docs/js/lib/s73-policy.js';
import { enumerate, valueIteration } from '../docs/js/lib/gridworld.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const net = P.load(JSON.parse(fs.readFileSync(path.join(root, 'docs/data/7-3/policy.json'))));
const pct = (a, b) => `${a}/${b} (${((100 * a) / b).toFixed(1)}%)`;
let failures = 0;
const expect = (ok, msg) => { if (!ok) { failures++; console.log(`FAIL ${msg}`); } };

// 1. The shortest distances in s73-maze.js agree with value iteration in gridworld.js.
{
  const r = M.rng(5);
  let agree = 0;
  for (let i = 0; i < 50; i++) {
    const env = M.randomEnv(r), free = M.freeCells(env), goal = free[Math.floor(r() * free.length)];
    const d = M.distances(env, goal);
    const mdp = enumerate({ start: { pos: free[0], hasKey: false }, nA: M.NA, key: (s) => `${s.pos}|${s.hasKey}`,
      step: (s, a) => (s.pos === goal ? null : { next: M.step(env, s, a) }) });
    const R = new Float64Array(mdp.nS * mdp.nA).fill(-1);
    const { V } = valueIteration(mdp, R, { gamma: 1, maxIter: 200 });
    agree += mdp.states.every((s, j) => { const dd = d[(s.hasKey ? M.N : 0) + s.pos]; return Number.isFinite(dd) ? Math.abs(-V[j] - dd) < 1e-9 : V[j] < -150; }) ? 1 : 0;
  }
  console.log(`distances agree with gridworld.js value iteration in ${agree}/50 environments`);
  expect(agree === 50, 'distances');
}

// Random test episodes: environment, start, goal, and a second goal, all reachable.
function episodes(n, seed) {
  const r = M.rng(seed), out = [];
  while (out.length < n) {
    const env = M.randomEnv(r), free = M.freeCells(env);
    const goal = free[Math.floor(r() * free.length)];
    let start; do { start = free[Math.floor(r() * free.length)]; } while (start === goal);
    let goal2; do { goal2 = free[Math.floor(r() * free.length)]; } while (goal2 === goal || goal2 === start);
    const d = M.distances(env, goal), d2 = M.distances(env, goal2);
    if (!Number.isFinite(d[start]) || !Number.isFinite(d2[start])) continue;
    out.push({ env, goal, start, goal2, opt: d[start], opt2: d2[start] });
  }
  return out;
}
const EP = episodes(1000, 99);
function randomBasis(k, d, seed) {
  const r = M.rng(seed), B = [];
  while (B.length < k) {
    const v = Float32Array.from({ length: d }, () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r()));
    for (const b of B) { let c = 0; for (let j = 0; j < d; j++) c += b[j] * v[j]; for (let j = 0; j < d; j++) v[j] -= c * b[j]; }
    const n = Math.hypot(...v); for (let j = 0; j < d; j++) v[j] /= n;
    B.push(v);
  }
  return B;
}
const netRandGoal = { ...net, goalBasis: randomBasis(net.kGoal, 128, 31) };
const netRandDoor = { ...net, doorBasis: randomBasis(net.kDoor, 128, 37) };
console.log(`subspaces: goal ${net.kGoal} directions, doors ${net.kDoor} directions, of ${net.sizes[1]} first-layer activations`);

// 2. The trained network on random episodes.
{
  let ok = 0, short = 0;
  for (const e of EP) { const res = M.run(e.env, e.start, e.goal, P.policy(net, e.env, e.goal)); if (res.reached) ok++; if (res.reached && res.steps === e.opt) short++; }
  console.log(`network, 1000 random episodes: reached goal ${pct(ok, 1000)}, by a shortest route ${pct(short, 1000)}`);
  expect(ok >= 995, 'network success');
}

// 3. Probe accuracy on fresh random states.
{
  const r = M.rng(123);
  let g = 0, d0 = 0, d1 = 0, n = 0;
  while (n < 5000) {
    const env = M.randomEnv(r), B = M.blocked(env), free = M.freeCells(env, B);
    const goal = free[Math.floor(r() * free.length)];
    const pos = free[Math.floor(r() * free.length)];
    if (pos === goal) continue;
    const st = { pos, hasKey: env.key >= 0 && r() < 0.3 };
    const h = P.hidden1(net, M.encode(env, st, goal, undefined, B));
    if (P.argmax(P.readGoal(net, h)) === goal) g++;
    const doors = P.readDoors(net, h);
    if (P.argmax(doors[0]) === env.doors[0]) d0++;
    if (P.argmax(doors[1]) === env.doors[1]) d1++;
    n++;
  }
  console.log(`probes on 5000 fresh states: goal ${pct(g, n)}, upper door ${pct(d0, n)}, lower door ${pct(d1, n)}`);
  expect(g / n > 0.98 && d0 / n > 0.98 && d1 / n > 0.98, 'probe accuracy');
}

// 4. Many means, one end: the six environments.
{
  const paths = new Set(), dpaths = new Set();
  let ok = 0, dok = 0;
  for (const E of M.ENVIRONMENTS) {
    const a = M.run(E.env, M.START, M.GOAL, P.policy(net, E.env, M.GOAL));
    const d = M.run(E.env, M.START, M.GOAL, M.directPolicy(M.GOAL));
    const opt = M.distances(E.env, M.GOAL)[M.START];
    if (a.reached) { ok++; paths.add(a.path.join(',')); }
    if (d.reached) { dok++; dpaths.add(d.path.join(',')); }
    const doorsUsed = M.DOORS.map(([x, y]) => a.path.includes(M.cell(x, y)));
    console.log(`  ${E.name}: network ${a.reached ? `reached in ${a.steps} steps (shortest ${opt})` : 'failed'}, through the ${doorsUsed[0] ? 'upper' : 'lower'} door${a.keys.at(-1) ? ', with the key' : ''}; direct controller ${d.reached ? `reached in ${d.steps}` : `blocked${d.stuck ? ', pushing against a wall' : ''}`}`);
  }
  console.log(`six environments: network reached ${ok}/6 by ${paths.size} distinct paths; direct controller reached ${dok}/6 by ${dpaths.size} distinct path(s)`);
  expect(ok === 6 && paths.size === 6 && dok === 2 && dpaths.size === 1, 'means count');
}

// 5. Removing the goal subspace (mean ablation), and a random subspace of the same size.
{
  let rm = 0, rmR = 0;
  const finals = new Map();
  for (const e of EP) {
    const b = M.run(e.env, e.start, e.goal, P.policy(net, e.env, e.goal, { goalMode: 'remove' }));
    if (b.reached) rm++;
    finals.set(b.path.at(-1), (finals.get(b.path.at(-1)) || 0) + 1);
    if (M.run(e.env, e.start, e.goal, P.policy(netRandGoal, e.env, e.goal, { goalMode: 'remove' })).reached) rmR++;
  }
  const top = Math.max(...finals.values());
  console.log(`remove goal subspace: reached goal ${pct(rm, 1000)}; most common final cell holds ${pct(top, 1000)} of runs; random ${net.kGoal}-dim subspace removed: reached ${pct(rmR, 1000)}`);
  const E = M.ENVIRONMENTS[0];
  const res = M.run(E.env, M.START, M.GOAL, P.policy(net, E.env, M.GOAL, { goalMode: 'remove' }));
  const cells = new Set(res.path);
  console.log(`  in "${E.name}" with the goal removed: visits ${cells.size} cells (${[...cells].map((c) => M.xy(c).join(',')).join(' ')}), reached ${res.reached}`);
  expect(rm < 150 && rmR > 500 && !res.reached, 'removal');
}

// 6. Swapping in another goal's activations along the goal subspace.
{
  let sw = 0, swShort = 0, swR = 0, readNew = 0, readSteps = 0;
  for (const e of EP) {
    const cfg = { goalMode: 'swap', goal2: e.goal2 };
    const a = M.run(e.env, e.start, e.goal2, P.policy(net, e.env, e.goal, cfg));
    if (a.reached) sw++;
    if (a.reached && a.steps === e.opt2) swShort++;
    if (M.run(e.env, e.start, e.goal2, P.policy(netRandGoal, e.env, e.goal, cfg)).reached) swR++;
    // what the goal probe reads during the swapped run
    a.path.slice(0, -1).forEach((c, t) => {
      const h = P.compute(net, e.env, { pos: c, hasKey: a.keys[t] }, e.goal, cfg).h;
      readSteps++; if (P.argmax(P.readGoal(net, h)) === e.goal2) readNew++;
    });
  }
  console.log(`swap goal: reached the new goal ${pct(sw, 1000)}, by a shortest route ${pct(swShort, 1000)}; random subspace: ${pct(swR, 1000)}; goal probe reads the new goal on ${pct(readNew, readSteps)} steps`);
  let tot = 0, ok = 0;
  for (const E of M.ENVIRONMENTS) {
    const free = M.freeCells(E.env).filter((c) => c !== M.GOAL && c !== M.START);
    let k = 0;
    for (const g2 of free) if (M.run(E.env, M.START, g2, P.policy(net, E.env, M.GOAL, { goalMode: 'swap', goal2: g2 })).reached) k++;
    tot += free.length; ok += k;
    console.log(`  ${E.name}: swapped agent reached the new goal from ${k} of ${free.length} placements`);
  }
  console.log(`  six environments together: ${pct(ok, tot)}`);
  expect(sw > 900 && swR < 300, 'swap');
  const E = M.ENVIRONMENTS[0], g2 = M.cell(1, 6);
  const a = M.run(E.env, M.START, g2, P.policy(net, E.env, M.GOAL, { goalMode: 'swap', goal2: g2 }));
  console.log(`  default substitute (1,6) in "${E.name}": reached ${a.reached} in ${a.steps} steps (shortest ${M.distances(E.env, g2)[M.START]})`);
  expect(a.reached, 'default swap');
  {
    const cfg = { goalMode: 'swap', goal2: g2 };
    const reads = a.path.map((c, t) => P.argmax(P.readGoal(net, P.compute(net, E.env, { pos: c, hasKey: a.keys[t] }, M.GOAL, cfg).h)) === g2);
    console.log(`  goal probe reads the substitute at ${reads.filter(Boolean).length} of ${reads.length} steps of that run`);
    expect(reads.every(Boolean), 'probe reads substitute');
    // The door probes at the last step of that run (the notice quotes the upper door's reading).
    const hl = P.compute(net, E.env, { pos: a.path.at(-1), hasKey: a.keys.at(-1) }, M.GOAL, cfg).h;
    const up = P.readDoors(net, hl)[0];
    console.log(`  door probes at the last step: upper door read as ${M.DOOR_STATES[P.argmax(up)]} with probability ${up[P.argmax(up)].toFixed(2)}`);
    expect(P.argmax(up) === 2 && Math.abs(up[2] - 0.65) < 0.005, 'door probe after swap');
    const U = M.ENVIRONMENTS.find((x) => x.id === 'upper-locked');
    const right = M.freeCells(U.env).filter((c) => M.xy(c)[0] > M.WALL_X && c !== M.GOAL);
    let viaKey = 0, reached = 0;
    for (const c of right) {
      const b = M.run(U.env, M.START, c, P.policy(net, U.env, M.GOAL, { goalMode: 'swap', goal2: c }));
      if (b.reached) reached++;
      if (b.reached && b.path.includes(U.env.key)) viaKey++;
    }
    console.log(`  "${U.name}", substitute in the right-hand room: reached from ${reached} of ${right.length} cells, all ${viaKey === reached ? '' : 'but some '}by way of the key (${viaKey})`);
    expect(viaKey === reached && reached >= right.length - 4, 'key first');
  }
  for (const E2 of M.ENVIRONMENTS) {
    const b = M.run(E2.env, M.START, g2, P.policy(net, E2.env, M.GOAL, { goalMode: 'swap', goal2: g2 }));
    console.log(`    ${E2.name}: reached ${b.reached} in ${b.steps} (shortest ${M.distances(E2.env, g2)[M.START]})`);
  }
}

// 7. Changing the belief about the doors.
{
  const r = M.rng(11), DC = M.DOORS.map(([x, y]) => M.cell(x, y));
  const side = (c) => Math.sign(M.xy(c)[0] - M.WALL_X);
  let n = 0, nOpen = 0, through = 0, nClosed = 0, stays = 0, crossed = 0, throughR = 0, staysR = 0, readOk = 0, readN = 0;
  while (n < 1000) {
    const doors = [Math.floor(r() * 2), Math.floor(r() * 2)];
    if (doors[0] && doors[1]) continue;
    const env = { doors, walls: [0, 1, 2, 3].map(() => r() < 0.5), key: -1 };
    const free = M.freeCells(env);
    const start = free[Math.floor(r() * free.length)], goal = free[Math.floor(r() * free.length)];
    if (side(start) * side(goal) !== -1 || !Number.isFinite(M.distances(env, goal)[start])) continue;
    const believed = Math.floor(r() * 2), belief = believed === 0 ? [0, 1] : [1, 0];
    n++;
    const judge = (nt) => {
      const res = M.run(env, start, goal, P.policy(nt, env, goal, { doorBelief: belief }));
      const viaBelieved = res.path.includes(DC[believed]), viaOther = res.path.includes(DC[1 - believed]);
      const [bx, by] = M.DOORS[believed];
      const last = res.path.slice(-10);
      const atDoor = last.every((c) => { const [x, y] = M.xy(c); return side(c) === side(start) && Math.abs(x - bx) + Math.abs(y - by) <= 2; });
      return { res, viaBelieved, viaOther, atDoor };
    };
    const j = judge(net), jr = judge(netRandDoor);
    if (j.viaOther) crossed++;
    if (env.doors[believed] === 0) { nOpen++; if (j.viaBelieved && j.res.reached && !j.viaOther) through++; if (jr.viaBelieved && jr.res.reached && !jr.viaOther) throughR++; }
    else { nClosed++; if (j.atDoor && !j.viaOther) stays++; if (jr.atDoor && !jr.viaOther) staysR++; }
    const h = P.compute(net, env, { pos: start, hasKey: false }, goal, { doorBelief: belief }).h;
    const rd = P.readDoors(net, h).map(P.argmax);
    readN++; if (rd[0] === belief[0] && rd[1] === belief[1]) readOk++;
  }
  console.log(`door belief, ${n} episodes with start and goal on opposite sides:`);
  console.log(`  believed-open door really open: went through it to the goal ${pct(through, nOpen)} (random ${net.kDoor}-dim subspace: ${pct(throughR, nOpen)})`);
  console.log(`  believed-open door really closed: stayed in front of it for the last 10 steps ${pct(stays, nClosed)} (random subspace: ${pct(staysR, nClosed)})`);
  console.log(`  crossed the door it was made to believe closed: ${crossed}; door probes read the set belief at the start in ${pct(readOk, readN)}`);
  expect(through / nOpen > 0.95 && stays / nClosed > 0.85 && crossed === 0, 'door belief');
  for (const [id, belief] of [['upper-closed', [0, 1]], ['open', [1, 0]], ['open', [0, 1]]]) {
    const E = M.ENVIRONMENTS.find((x) => x.id === id);
    const res = M.run(E.env, M.START, M.GOAL, P.policy(net, E.env, M.GOAL, { doorBelief: belief }));
    console.log(`  "${E.name}", belief upper ${M.DOOR_STATES[belief[0]]}, lower ${M.DOOR_STATES[belief[1]]}: ${res.reached ? `reached in ${res.steps}` : `not reached; last cells ${res.path.slice(-4).map((c) => M.xy(c).join(',')).join(' ')}`}, via ${res.path.includes(M.cell(4, 1)) ? 'upper' : res.path.includes(M.cell(4, 5)) ? 'lower' : 'no'} door`);
  }
}
// 9 (run here). Share of the change in the first layer, caused by moving only the goal, that lies in the goal subspace.
{
  const r = M.rng(7);
  let inn = 0, tot = 0;
  for (let i = 0; i < 3000; i++) {
    const env = M.randomEnv(r), free = M.freeCells(env), g = free[Math.floor(r() * free.length)];
    let gb; do { gb = free[Math.floor(r() * free.length)]; } while (gb === g);
    const st = { pos: free[Math.floor(r() * free.length)], hasKey: false };
    const h = P.hidden1(net, M.encode(env, st, g)), h2 = P.hidden1(net, M.encode(env, st, gb));
    const d = h2.map((v, j) => v - h[j]);
    for (const v of d) tot += v * v;
    for (const u of net.goalBasis.slice(0, net.kGoal)) { let c = 0; for (let j = 0; j < d.length; j++) c += u[j] * d[j]; inn += c * c; }
  }
  console.log(`moving only the goal: ${(100 * inn / tot).toFixed(1)}% of the squared change in the first layer lies in the ${net.kGoal}-dim goal subspace (3000 random states)`);
  expect(Math.abs(inn / tot - 0.82) < 0.01, 'goal subspace share');
}
// 8. Controls: the same interventions along five random subspaces of the same sizes.
{
  const rem = [], swp = [];
  for (const seed of [41, 42, 43, 44, 45]) {
    const nr = { ...net, goalBasis: randomBasis(net.kGoal, 128, seed) };
    let a = 0, b = 0;
    for (const e of EP) {
      if (M.run(e.env, e.start, e.goal, P.policy(nr, e.env, e.goal, { goalMode: 'remove' })).reached) a++;
      if (M.run(e.env, e.start, e.goal2, P.policy(nr, e.env, e.goal, { goalMode: 'swap', goal2: e.goal2 })).reached) b++;
    }
    rem.push(a / 10); swp.push(b / 10);
  }
  console.log(`five random ${net.kGoal}-dim subspaces: removal leaves the goal reached in ${rem.join('%, ')}% of the 1000 episodes; swapping reaches the new goal in ${swp.join('%, ')}%`);
  expect(Math.min(...rem) > 50 && Math.max(...swp) < 30, 'random controls');
}
console.log(failures ? `${failures} check(s) failed` : 'all checks passed');
process.exit(failures ? 1 : 0);
