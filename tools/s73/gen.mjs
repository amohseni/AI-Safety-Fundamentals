// Training data for the page 7.3 policy network: random two-room environments, goals, and
// agent states, each labeled with the set of actions that lie on a shortest route to the goal.
// Usage: node tools/s73/gen.mjs <out-dir> [pairs=80000] [perPair=20] [seed=1]
// Writes obs.bin (316 input bits per sample, packed 8 per byte, high bit first), mask.bin
// (one byte per sample, bit a set when action a is optimal), and meta.bin (8 bytes per sample:
// goal, agent, upper door, lower door, has key, wall bits, key cell + 1, unused).
import fs from 'node:fs';
import path from 'node:path';
import * as M from '../../docs/js/lib/s73-maze.js';

const [out, pairsArg, perArg, seedArg] = process.argv.slice(2);
const pairs = Number(pairsArg || 80000), per = Number(perArg || 20), r = M.rng(Number(seedArg || 1));
const n = pairs * per, bytes = Math.ceil(M.OBS / 8);
const obs = new Uint8Array(n * bytes), mask = new Uint8Array(n), meta = new Uint8Array(n * 8);
const x = new Float32Array(M.OBS);
let k = 0;
while (k < n) {
  const env = M.randomEnv(r), B = M.blocked(env), free = M.freeCells(env, B);
  const goal = free[Math.floor(r() * free.length)];
  // The agent may also stand in a doorway or on the key's cell (having just picked it up).
  const cells = [];
  for (let c = 0; c < M.N; c++) if (!B[c] && c !== goal) cells.push(c);
  const dist = M.distances(env, goal, B);
  for (let j = 0; j < per && k < n; j++) {
    const pos = cells[Math.floor(r() * cells.length)];
    const d = M.isDoor(pos);
    const hasKey = pos === env.key || (d >= 0 && env.doors[d] === 2) || (env.key >= 0 && r() < 0.3);
    const st = { pos, hasKey };
    const m = M.optimalMask(env, st, goal, dist, B);
    if (!m) continue;
    M.encode(env, st, goal, x, B);
    for (let i = 0; i < M.OBS; i++) if (x[i]) obs[k * bytes + (i >> 3)] |= 0x80 >> (i & 7);
    mask[k] = m;
    meta.set([goal, pos, env.doors[0], env.doors[1], hasKey ? 1 : 0, env.walls.reduce((s, w, i) => s | ((w ? 1 : 0) << i), 0), env.key + 1, 0], k * 8);
    k++;
  }
}
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, 'obs.bin'), obs);
fs.writeFileSync(path.join(out, 'mask.bin'), mask);
fs.writeFileSync(path.join(out, 'meta.bin'), meta);
console.log(`${n} samples, ${bytes} bytes each, written to ${out}`);
