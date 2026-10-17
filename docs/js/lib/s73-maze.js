// The two-room gridworld of page 7.3. Pure JavaScript, no DOM, so Node tests and the
// training-data generator (tools/s73/gen.mjs) use the same rules as the page.
//
// The grid is 9 cells wide and 7 high. A wall runs down column 4 with two doors in it,
// the upper door at (4, 1) and the lower door at (4, 5). Each door is open, closed, or
// locked; a locked door opens for an agent that carries the key, which it picks up by
// stepping onto the key's cell. Four optional wall segments vary the routes.
//
// env   = { doors: [upper, lower] with 0 open, 1 closed, 2 locked; walls: [4 booleans]; key: cell or -1 }
// state = { pos: cell, hasKey: boolean }; cell = y * W + x, with y growing downward.
// Actions: 0 up, 1 right, 2 down, 3 left. A move into a wall, a closed door, a locked
// door without the key, or off the grid leaves the agent where it is.

export const W = 9, H = 7, N = W * H;
export const WALL_X = 4;
export const DOORS = [[4, 1], [4, 5]];
export const DOOR_NAMES = ['upper door', 'lower door'];
export const DOOR_STATES = ['open', 'closed', 'locked'];
export const SEGMENTS = [[[2, 1], [2, 2], [2, 3]], [[6, 3], [6, 4], [6, 5]], [[1, 5], [2, 5]], [[6, 1], [7, 1]]];
export const MOVES = [[0, -1], [1, 0], [0, 1], [-1, 0]];
export const NA = 4;
export const OBS = 5 * N + 1; // channels: blocked, locked door, key, agent, goal; then one has-key unit

export const cell = (x, y) => y * W + x;
export const xy = (c) => [c % W, Math.floor(c / W)];
const DOOR_CELLS = DOORS.map(([x, y]) => cell(x, y));
export const isDoor = (c) => DOOR_CELLS.indexOf(c);

// 1 where a cell can never be entered: the middle wall, closed doors, and wall segments that are on.
export function blocked(env) {
  const b = new Uint8Array(N);
  for (let y = 0; y < H; y++) b[cell(WALL_X, y)] = 1;
  DOOR_CELLS.forEach((c, i) => { b[c] = env.doors[i] === 1 ? 1 : 0; });
  SEGMENTS.forEach((seg, i) => { if (env.walls[i]) seg.forEach(([x, y]) => { b[cell(x, y)] = 1; }); });
  return b;
}

export function step(env, st, a, B = blocked(env)) {
  const [x, y] = xy(st.pos);
  const nx = x + MOVES[a][0], ny = y + MOVES[a][1];
  if (nx < 0 || ny < 0 || nx >= W || ny >= H) return st;
  const c = cell(nx, ny);
  if (B[c]) return st;
  const d = isDoor(c);
  if (d >= 0 && env.doors[d] === 2 && !st.hasKey) return st;
  return { pos: c, hasKey: st.hasKey || c === env.key };
}

// The network's input for one step. `out` may be passed to avoid allocation.
export function encode(env, st, goal, out = new Float32Array(OBS), B = blocked(env)) {
  out.fill(0);
  for (let c = 0; c < N; c++) if (B[c]) out[c] = 1;
  DOOR_CELLS.forEach((c, i) => { if (env.doors[i] === 2) out[N + c] = 1; });
  if (env.key >= 0 && !st.hasKey) out[2 * N + env.key] = 1;
  out[3 * N + st.pos] = 1;
  out[4 * N + goal] = 1;
  out[5 * N] = st.hasKey ? 1 : 0;
  return out;
}

// Shortest number of steps to the goal from every (cell, has key) pair; Infinity if unreachable.
// dist[k * N + c] for k = 0 (no key) or 1 (key in hand).
export function distances(env, goal, B = blocked(env)) {
  const dist = new Float64Array(2 * N).fill(Infinity);
  dist[goal] = 0; dist[N + goal] = 0;
  for (let changed = true; changed;) {
    changed = false;
    for (let k = 0; k < 2; k++) for (let c = 0; c < N; c++) {
      if (B[c] || c === goal) continue;
      const st = { pos: c, hasKey: k === 1 };
      let best = dist[k * N + c];
      for (let a = 0; a < NA; a++) {
        const s2 = step(env, st, a, B);
        if (s2.pos === c && s2.hasKey === st.hasKey) continue;
        const v = 1 + dist[(s2.hasKey ? N : 0) + s2.pos];
        if (v < best) best = v;
      }
      if (best < dist[k * N + c]) { dist[k * N + c] = best; changed = true; }
    }
  }
  return dist;
}

// The actions on some shortest route to the goal, as a 4-bit mask (bit a set for action a).
export function optimalMask(env, st, goal, dist, B = blocked(env)) {
  const here = dist[(st.hasKey ? N : 0) + st.pos];
  let m = 0;
  if (!Number.isFinite(here) || here === 0) return 0;
  for (let a = 0; a < NA; a++) {
    const s2 = step(env, st, a, B);
    if (s2.pos === st.pos && s2.hasKey === st.hasKey) continue;
    if (dist[(s2.hasKey ? N : 0) + s2.pos] === here - 1) m |= 1 << a;
  }
  return m;
}

// Runs a controller from `start` for at most maxSteps. policy(st, t) returns an action.
// Returns { path: [cells], reached, steps, stuck } where stuck means the last 4 steps did not move.
export function run(env, start, goal, policy, maxSteps = 40) {
  const B = blocked(env);
  let st = { pos: start, hasKey: false };
  const path = [st.pos], keys = [false];
  let reached = st.pos === goal, t = 0;
  while (!reached && t < maxSteps) {
    const a = policy(st, t);
    st = step(env, st, a, B);
    path.push(st.pos); keys.push(st.hasKey);
    t++;
    if (st.pos === goal) reached = true;
  }
  const n = path.length;
  const stuck = !reached && n > 4 && path.slice(n - 5).every((c) => c === path[n - 1]);
  return { path, keys, reached, steps: t, stuck, final: st };
}

// The direct controller: moves along the row toward the goal's column, then along the column
// toward the goal. It has one way of closing the gap and pushes against whatever blocks it.
export function directPolicy(goal) {
  const [gx, gy] = xy(goal);
  return (st) => {
    const [x, y] = xy(st.pos);
    if (x < gx) return 1;
    if (x > gx) return 3;
    return y < gy ? 2 : 0;
  };
}

// Random environments, starts, and goals for training and testing.
export function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

export function randomEnv(r) {
  let doors;
  do { doors = [Math.floor(r() * 3), Math.floor(r() * 3)]; } while (doors[0] === 1 && doors[1] === 1);
  const walls = [0, 1, 2, 3].map(() => r() < 0.5);
  const env = { doors, walls, key: -1 };
  if (doors.includes(2)) {
    const B = blocked(env);
    const free = [];
    for (let c = 0; c < N; c++) if (!B[c] && isDoor(c) < 0) free.push(c);
    env.key = free[Math.floor(r() * free.length)];
  }
  return env;
}

export function freeCells(env, B = blocked(env)) {
  const out = [];
  for (let c = 0; c < N; c++) if (!B[c] && isDoor(c) < 0 && c !== env.key) out.push(c);
  return out;
}

// Same doors, walls, and key, read as a string (for counting distinct routes and environments).
export const envKey = (env) => `${env.doors.join('')}|${env.walls.map(Number).join('')}|${env.key}`;

// The six environments of the page. The agent starts at START and its goal is GOAL in all six.
export const START = cell(0, 1), GOAL = cell(7, 4);
export const ENVIRONMENTS = [
  { id: 'open', name: 'Both doors open', short: 'Both open', env: { doors: [0, 0], walls: [0, 0, 0, 0], key: -1 } },
  { id: 'lower-closed', name: 'Lower door closed', short: 'Lower closed', env: { doors: [0, 1], walls: [0, 0, 0, 0], key: -1 } },
  { id: 'upper-closed', name: 'Upper door closed', short: 'Upper closed', env: { doors: [1, 0], walls: [0, 0, 0, 0], key: -1 } },
  { id: 'upper-locked', name: 'Upper door locked, lower closed', short: 'Upper locked', env: { doors: [2, 1], walls: [0, 0, 0, 0], key: cell(1, 5) } },
  { id: 'lower-locked', name: 'Lower door locked, upper closed', short: 'Lower locked', env: { doors: [1, 2], walls: [0, 0, 0, 0], key: cell(2, 3) } },
  { id: 'walls', name: 'Both doors open, more walls', short: 'More walls', env: { doors: [0, 0], walls: [1, 1, 0, 1], key: -1 } },
];
