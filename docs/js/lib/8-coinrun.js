// A grid version of CoinRun for page 8.3. Pure JavaScript, no DOM, so Node tests can import it.
//
// The level is a W by H grid. The agent starts at the left end of the middle row. The episode ends with
// reward 1 when the agent reaches the coin, or with reward 0 after MAX_STEPS steps.
// Standard levels: the coin sits at the right end of the middle row. Randomized levels: the coin is placed
// uniformly at random on any cell except the start.
//
// The policy is linear-softmax over four actions (up, right, down, left). Its input has two parts: the direction
// of the coin from the agent (one-hot: left, same column, right; and above, same row, below), and the agent's
// own position in the level (one-hot column and one-hot row), plus a constant 1.
// It is trained by REINFORCE with a running baseline.
import { rng } from './mlp.js';

export const W = 10, H = 5, MAX_STEPS = 30, START = [0, 2], END = [W - 1, 2];
export const MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]];
export const NF = 6 + W + H + 1, NA = 4;

export function features(ax, ay, cx, cy, out = new Float64Array(NF)) {
  out.fill(0);
  out[cx < ax ? 0 : cx === ax ? 1 : 2] = 1; // coin to the left, same column, to the right
  out[3 + (cy < ay ? 0 : cy === ay ? 1 : 2)] = 1; // coin above, same row, below
  out[6 + ax] = 1; // the agent's column
  out[6 + W + ay] = 1; // the agent's row
  out[NF - 1] = 1; // constant
  return out;
}

export function probs(theta, x, out = new Float64Array(NA)) {
  let m = -Infinity;
  for (let a = 0; a < NA; a++) { let z = 0; for (let f = 0; f < NF; f++) z += theta[a * NF + f] * x[f]; out[a] = z; if (z > m) m = z; }
  let s = 0;
  for (let a = 0; a < NA; a++) { out[a] = Math.exp(out[a] - m); s += out[a]; }
  for (let a = 0; a < NA; a++) out[a] /= s;
  return out;
}

export function randomCoin(rand) {
  for (;;) {
    const cx = Math.floor(rand() * W), cy = Math.floor(rand() * H);
    if (cx !== START[0] || cy !== START[1]) return [cx, cy];
  }
}

export function makeLevel(rand, pRandom) {
  return rand() < pRandom ? randomCoin(rand) : END.slice();
}

// One episode; returns { got, steps, path } where path lists visited cells.
export function episode(theta, coin, rand, { greedy = false, record = false } = {}) {
  let [x, y] = START;
  const xs = new Float64Array(NF), p = new Float64Array(NA);
  const path = record ? [[x, y]] : null;
  const traj = [];
  for (let t = 0; t < MAX_STEPS; t++) {
    features(x, y, coin[0], coin[1], xs);
    probs(theta, xs, p);
    let a;
    if (greedy) { a = 0; for (let b = 1; b < NA; b++) if (p[b] > p[a]) a = b; }
    else { const r = rand(); let c = 0; a = NA - 1; for (let b = 0; b < NA; b++) { c += p[b]; if (r < c) { a = b; break; } } }
    traj.push([Float64Array.from(xs), Float64Array.from(p), a]);
    const nx = x + MOVES[a][0], ny = y + MOVES[a][1];
    if (nx >= 0 && nx < W && ny >= 0 && ny < H) { x = nx; y = ny; }
    if (record) path.push([x, y]);
    if (x === coin[0] && y === coin[1]) return { got: true, steps: t + 1, traj, path };
  }
  return { got: false, steps: MAX_STEPS, traj, path };
}

// REINFORCE with a running baseline. makeTrainer returns { theta, done, run(n) }; run(n) trains n more episodes,
// so a page can train in slices and show progress. train() runs all episodes at once.
export const EPISODES = 6000;
export function makeTrainer(pRandom, { seed = 1, episodes = EPISODES, lr = 0.1, gamma = 0.95 } = {}) {
  const rand = rng(seed);
  const theta = new Float64Array(NA * NF);
  let baseline = 0, e = 0;
  return {
    theta, episodes,
    get done() { return e; },
    run(n) {
      const stop = Math.min(episodes, e + n);
      for (; e < stop; e++) {
        const coin = makeLevel(rand, pRandom);
        const { got, steps, traj } = episode(theta, coin, rand);
        for (let t = 0; t < traj.length; t++) {
          const G = got ? Math.pow(gamma, steps - 1 - t) : 0;
          const adv = G - baseline;
          const [xs, p, a] = traj[t];
          for (let b = 0; b < NA; b++) {
            const g = (b === a ? 1 : 0) - p[b];
            for (let f = 0; f < NF; f++) theta[b * NF + f] += lr * adv * g * xs[f];
          }
        }
        baseline += 0.01 * ((got ? Math.pow(gamma, steps - 1) : 0) - baseline);
      }
      return e >= episodes;
    },
  };
}
export function train(pRandom, opts = {}) {
  const tr = makeTrainer(pRandom, opts);
  tr.run(tr.episodes);
  return tr.theta;
}

// Success rate on test levels (coin anywhere except the start), sampling actions from the policy.
export function evaluate(theta, { levels = 2000, seed = 999, pRandom = 1, record = false } = {}) {
  const rand = rng(seed);
  let got = 0, toEnd = 0;
  const visits = record ? new Float64Array(W * H) : null;
  for (let i = 0; i < levels; i++) {
    const coin = pRandom === 1 ? randomCoin(rand) : makeLevel(rand, pRandom);
    const r = episode(theta, coin, rand, { record });
    if (r.got) got++;
    if (record) { const seen = new Set(); for (const [x, y] of r.path) { const k = y * W + x; if (!seen.has(k)) { seen.add(k); visits[k]++; } } }
    // Did the agent reach the right end of the middle row at some point?
    if (record && r.path.some(([x, y]) => x === END[0] && y === END[1])) toEnd++;
  }
  return { success: got / levels, toEnd: record ? toEnd / levels : null, visits };
}
