// Sanity checks for docs/js/lib/gridworld.js. Run: node tools/test-gridworld.mjs
import { enumerate, valueIteration, finiteHorizon, rollout, bestActions, grid, DIRS, graphValues } from '../docs/js/lib/gridworld.js';

// A 4 by 3 grid; reward 1 for stepping onto the goal at (3, 0), which is absorbing.
const g = grid(4, 3), goal = g.idx(3, 0);
const mdp = enumerate({
  start: g.idx(0, 2), nA: 4, key: (s) => s,
  step: (s, a) => {
    if (s === goal) return { next: s, r: 0 };
    const [x, y] = g.xy(s), nx = x + DIRS[a][0], ny = y + DIRS[a][1];
    if (!g.inside(nx, ny)) return null;
    const t = g.idx(nx, ny);
    return { next: t, r: t === goal ? 1 : 0 };
  },
});
const R = Float64Array.from(mdp.info, (r) => (r ? r.r : 0));
const { V, Q, iters } = valueIteration(mdp, R, { gamma: 0.9 });
const s0 = 0; // start
console.log('states', mdp.nS, 'iterations', iters, 'V(start)', V[s0].toFixed(4), 'expected 0.9^4 =', (0.9 ** 4).toFixed(4));
console.log('best first actions from start (0 up, 1 right):', bestActions(Array.from(Q.slice(s0 * 4, s0 * 4 + 4))));
const fh = finiteHorizon(mdp, R, 10);
const path = rollout(mdp, fh.policy, s0, 10);
console.log('finite-horizon path reaches goal:', path.some((p) => mdp.states[p.s] === goal), 'in', path.findIndex((p) => mdp.states[p.s] === goal), 'steps (expected 5)');
// Graph solver: two-state cycle with rewards 1 and 0, gamma 0.5: V = (1, 0.5)/(1 - 0.25) from the cycle.
const Vg = graphValues([[1], [0]], Float64Array.from([1, 0]), 0.5);
console.log('graph cycle values', Array.from(Vg).map((v) => v.toFixed(4)).join(' '), 'expected 1.3333 0.6667');
