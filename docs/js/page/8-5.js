// Page 8.5: training cannot see the difference (toy model).
import { chart } from '../lib/chart.js';
import { slider, button } from '../lib/ui.js';
import * as S from '../lib/s7-select.js';

const X0 = [1 / 3, 1 / 3, 1 / 3];
const START = { q: 1, h: 0, T: 200 };
const state = { ...START };
const pct = (x) => {
  const v = 100 * x;
  if (v === 0) return '0%';
  if (v < 0.01) return '<0.01%';
  if (v < 1) return v.toFixed(2) + '%';
  return v.toFixed(1) + '%';
};

const ctl = document.getElementById('controls');
const qS = slider(ctl, { name: 'Detection probability', min: 0, max: 1, step: 0.01, value: state.q, format: (v) => v.toFixed(2), onInput: (v) => { state.q = v; draw(); } });
const hS = slider(ctl, { name: 'Honeypots, share of inputs', min: 0, max: 0.2, step: 0.005, value: state.h, format: (v) => `${(100 * v).toFixed(1)}%`, onInput: (v) => { state.h = v; draw(); } });
const tS = slider(ctl, { name: 'Training rounds', min: 0, max: 500, step: 10, value: state.T, format: (v) => String(v), onInput: (v) => { state.T = v; draw(); } });
button(ctl, 'Reset', () => { Object.assign(state, START); qS.set(state.q); hS.set(state.h); tS.set(state.T); draw(); });

const c = chart(document.getElementById('chart'), {
  x: { domain: [0, 200], label: 'training round' }, y: { domain: [0, 1], label: 'share of the population', format: '.0%' },
  title: 'Share of each type of policy over training', margin: { top: 34, right: 112, bottom: 52, left: 60 },
});
const out = document.getElementById('readout');

function draw() {
  const T = state.T;
  c.setX([0, Math.max(T, 10)]);
  const traj = S.trajectory(X0, state, T);
  const lines = [['aligned', 0, 'accent', 'aligned', null], ['open', 1, 'quiet', 'open', null], ['deceptive', 2, 'second', 'deceptive', '6 4']];
  for (const [id, i, color, label, dash] of lines) c.line(id, traj.map((x, t) => [t, x[i]]), { color, label, width: 2, dash });
  c.spreadLabels(lines.map((l) => l[0]), 20);
  const r = S.rewards(state), end = traj[T];
  out.innerHTML = `Expected reward per round: aligned 1, open 0, deceptive ${r[2].toFixed(3)}. ` +
    `At deployment, after ${T} rounds: <span class="c-accent">aligned ${pct(end[0])}</span>, open ${pct(end[1])}, <span class="c-second">deceptive ${pct(end[2])}</span>. ` +
    `Misaligned share deployed: <b>${pct(end[1] + end[2])}</b>.`;
}
draw();

// ---- Static figure: the inner alignment problem as a flow ----
(function flow() {
  const fig = document.getElementById('fig-flow');
  const W = Math.max(300, Math.min(600, fig.clientWidth || 600)), H = 400;
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'flow'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.setAttribute('height', H);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Flow from the training objective through the training process to a learned system with its own goal, branching at deployment into goal matches and goal differs');
  const el = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); if (text) e.textContent = text; svg.append(e); return e; };
  const lines = (x, y, anchor, rows) => rows.forEach(([t, cls], k) => el('text', { x, y: y + 27 * k, 'text-anchor': anchor, class: cls || '' }, t));
  const arrow = (x1, y1, x2, y2) => {
    el('line', { x1, y1, x2, y2 });
    const a = Math.atan2(y2 - y1, x2 - x1), s = 8;
    const p = [[x2, y2], [x2 - s * Math.cos(a - 0.4), y2 - s * Math.sin(a - 0.4)], [x2 - s * Math.cos(a + 0.4), y2 - s * Math.sin(a + 0.4)]];
    el('path', { class: 'head', d: `M${p.map((q) => q.map((v) => v.toFixed(1)).join(',')).join('L')}Z` });
  };
  const mid = W / 2;
  lines(mid, 24, 'middle', [['Training objective']]);
  arrow(mid, 36, mid, 70);
  lines(mid, 92, 'middle', [['Training process'], ['selects on behavior', 'quiet']]);
  arrow(mid, 130, mid, 164);
  lines(mid, 186, 'middle', [['Learned system'], ['with its own goal', 'quiet']]);
  const lx = W * 0.25, rx = W * 0.75;
  arrow(mid - 20, 226, lx, 266);
  arrow(mid + 20, 226, rx, 266);
  lines(lx, 292, 'middle', [['Goal matches'], ['the objective'], ['in deployment:', 'quiet'], ['as intended', 'quiet']]);
  lines(rx, 292, 'middle', [['Goal differs'], ['in training:', 'quiet'], ['behaves well;', 'quiet'], ['later, may not', 'quiet']]);
  fig.append(svg);
})();
