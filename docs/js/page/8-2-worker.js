// Page 8.2: solves the cleaning robot's MDP off the main thread, so the page never freezes.
import { buildRoom, solve } from '../lib/8-cleaning.js';

const rooms = {};
self.onmessage = (ev) => {
  const { id, weights, plant } = ev.data;
  const key = plant ? 'plant' : 'plain';
  if (!rooms[key]) rooms[key] = buildRoom({ plant });
  const room = rooms[key];
  const { path, summary } = solve(room, weights);
  self.postMessage({ id, nS: room.nS, summary, path: path.map(({ s, a, t }) => ({ a, t, d: a >= 0 ? room.dur[s * room.nA + a] : 0, st: room.states[s] })) });
};
