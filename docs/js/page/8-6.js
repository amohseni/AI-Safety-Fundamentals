// Page 8.6: control protocol game.
import { chart } from '../lib/chart.js';
import { slider, button } from '../lib/ui.js';
import * as C from '../lib/s7-control.js';

const NEVER = 4.5; // hand-off slider value meaning "never hand off"
const START = { mu: 2, budget: 0.02, tA: 2.05, tD: NEVER };
const state = { ...START };
const tDval = () => (state.tD >= NEVER ? Infinity : state.tD);
const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;

const ctl = document.getElementById('controls');
const S = {
  mu: slider(ctl, { name: 'Monitor quality', min: 0, max: 4, step: 0.1, value: state.mu, format: (v) => v.toFixed(1), onInput: (v) => { state.mu = v; update(); } }),
  budget: slider(ctl, { name: 'Audit budget, share of tasks', min: 0.005, max: 0.1, step: 0.005, value: state.budget, format: (v) => pct(v), onInput: (v) => { state.budget = v; update(); } }),
  tA: slider(ctl, { name: 'Audit threshold', min: -1, max: 4, step: 0.01, value: state.tA, format: (v) => v.toFixed(2), onInput: (v) => { state.tA = v; update(); } }),
  tD: slider(ctl, { name: 'Hand-off threshold', min: -3, max: NEVER, step: 0.01, value: state.tD, format: (v) => (v >= NEVER ? 'never' : v.toFixed(2)), onInput: (v) => { state.tD = v; update(); } }),
};
button(ctl, 'Reset', () => { Object.assign(state, START); Object.keys(S).forEach((k) => S[k].set(state[k])); update(); });

const dist = chart(document.getElementById('dist'), {
  height: 230, x: { domain: [-4, 7.5], label: 'suspicion score' }, y: { domain: [0, 0.45], ticks: 2 },
  title: 'Distributions of the monitor’s scores for normal work and for sabotage', margin: { top: 30, right: 20, bottom: 52, left: 44 },
});
const trade = chart(document.getElementById('trade'), {
  height: 300, x: { domain: [0, 1], label: 'usefulness', format: '.0%', ticks: 4 }, y: { domain: [0, 1], label: 'safety', format: '.0%' },
  title: 'Safety against usefulness', margin: { top: 34, right: 24, bottom: 52, left: 60 },
});
const out = document.getElementById('readout');
const xs = Array.from({ length: 231 }, (_, i) => -4 + i * 0.05);

let curveTimer = null;
function update() {
  const tD = tDval();
  // Score distributions and thresholds.
  dist.line('normal', xs.map((x) => [x, C.density(x, 0)]), { color: 'ink', label: null });
  dist.line('sab', xs.map((x) => [x, C.density(x, state.mu)]), { color: 'second', label: null });
  dist.text('normal:l', -1.1, 0.3, 'normal', { color: 'ink', anchor: 'end' });
  dist.text('sab:l', state.mu + 1.1, 0.3, 'sabotage', { color: 'second', anchor: 'start' });
  dist.vline('tA', state.tA, { color: 'accent', dash: null, label: 'audit', labelY: -16 }); // above the plot, clear of the curves
  // The hand-off label goes above the plot, ending at its line, so it never meets the audit label or the curves.
  if (Number.isFinite(tD)) {
    dist.vline('tD', tD, { color: 'accent', dash: '4 3' });
    if (tD < state.tA && tD > -2) dist.text('tD:l', tD - 0.1, 0.47, 'hand off', { color: 'accent', anchor: 'end' });
    else if (state.tA - tD > 3.3) dist.text('tD:l', tD + 0.1, 0.47, 'hand off', { color: 'accent', anchor: 'start' });
    else dist.text('tD:l', tD + 0.1, 0.3, 'hand off', { color: 'accent', anchor: 'start' });
  } else { dist.remove('tD'); dist.remove('tD:l'); }

  const e = C.evaluate({ mu: state.mu, budget: state.budget, tA: state.tA, tD });
  trade.point('now', e.usefulness, e.safety, { color: 'accent', r: 6, label: 'current protocol', anchor: e.usefulness > 0.6 ? 'end' : 'start', dx: e.usefulness > 0.6 ? -10 : 10, dy: e.safety > 0.85 ? 40 : -10 });
  const flagged = 1 - C.Phi(state.tA);
  const lasts = flagged * 1000 > e.B ? `the ${e.B} audits last about ${Math.round(e.B / flagged)} tasks` : `about ${Math.round(e.audits)} of the ${e.B} audits are used`;
  out.innerHTML = `The audit threshold flags ${pct(flagged)} of normal work and ${pct(e.catchRate)} of sabotage; with an honest model, ${lasts}. ` +
    `<span class="c-accent">Safety ${pct(e.safety)}</span>: the probability that no sabotage succeeds. Usefulness ${pct(e.usefulness)}: the share of tasks done by the untrusted model.`;
  // The trade-off curve over all hand-off thresholds is slower; draw it once the controls settle.
  clearTimeout(curveTimer);
  curveTimer = setTimeout(curve, 120);
}
function curve() {
  const pts = [];
  for (let tD = -3; tD <= NEVER + 1e-9; tD += 0.25) {
    const e = C.evaluate({ mu: state.mu, budget: state.budget, tA: state.tA, tD: tD >= NEVER ? Infinity : tD });
    pts.push([e.usefulness, e.safety]);
  }
  pts.sort((a, b) => a[0] - b[0]);
  trade.line('curve', pts, { color: 'quiet', label: null });
  // Label just above the plot, over the curve's left end, which is always near 100% safety.
  trade.text('curve:l', 0.03, 1.07, 'every hand-off threshold', { color: 'quiet', anchor: 'start' });
}
update();
curve();
