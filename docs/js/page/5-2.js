// Page 5.2: learning and selection on a simplex.
import { slider, segmented, button } from '../lib/ui.js';
import { simplex } from '../lib/s45-simplex.js';
import { learn, integrate, replicatorRHS, pgRHS, runAndMeasure } from '../lib/s45-selection.js';

const RULES = [['cross', 'Cross'], ['exp', 'Exponential weights'], ['pg', 'Softmax policy gradient']];
const STEPS = ['0.1', '0.03', '0.01', '0.003', '0.001'];
const NAMES = { fixed: ['action 1', 'action 2', 'action 3'], rps: ['rock', 'paper', 'scissors'] };
const X0 = { fixed: [0.05, 0.15, 0.8], rps: [0.5, 0.3, 0.2] };
const HORIZON = { fixed: 60, rps: 120 };

// ---- Static figure: replicator flow with one learning path ----
(function figure() {
  const box = document.createElement('div');
  document.getElementById('fig-simplex').append(box);
  const game = { kind: 'fixed', pay: [0.9, 0.6, 0.3] };
  const s = simplex(box, { width: 460, labels: NAMES.fixed, title: 'Replicator flow with a learning path' });
  s.flowField((x) => replicatorRHS(game, x));
  s.path('rep', integrate(replicatorRHS, game, X0.fixed, 60).map((p) => p.x), { color: 'ink', width: 2 });
  s.path('run', learn('cross', game, X0.fixed, 0.01, 60, 3).map((p) => p.x), { color: 'accent', width: 1.2 });
  s.point('start', X0.fixed, { color: 'ink', r: 4, label: 'start', dx: -8, anchor: 'end' });
})();

// ---- The tool ----
const st = { kind: 'fixed', pay: [0.9, 0.6, 0.3], rule: 'cross', h: 0.1, runs: 5, x0: X0.fixed.slice() };
const game = () => (st.kind === 'fixed' ? { kind: 'fixed', pay: st.pay.slice() } : { kind: 'rps' });
const ctl = document.getElementById('controls'), out = document.getElementById('readout');
const S = simplex(document.getElementById('stage'), { width: 520, labels: NAMES.fixed, title: 'Probabilities of three actions, with the replicator flow and learning paths' });
S.svg.attr('tabindex', 0).attr('aria-label', 'Simplex of action probabilities. Click or use the arrow keys to move the starting point.');

segmented(ctl, { name: 'Payoffs', options: [['fixed', 'Three fixed payoffs'], ['rps', 'Rock-paper-scissors']], value: st.kind,
  onChange: (k) => { st.kind = k; st.x0 = X0[k].slice(); S.setLabels(NAMES[k]); paySliders.forEach((p) => { p.el.disabled = k !== 'fixed'; }); reset(); } });
const paySliders = [0, 1, 2].map((i) => slider(ctl, { name: `Payoff of action ${i + 1}`, min: 0, max: 1, step: 0.01, value: st.pay[i],
  format: (v) => v.toFixed(2), onInput: (v) => { st.pay[i] = v; reset(); } }));
segmented(ctl, { name: 'Learning rule', options: RULES, value: st.rule, onChange: (k) => { st.rule = k; reset(); } });
segmented(ctl, { name: 'Step size', options: STEPS.map((s) => [s, s]), value: String(st.h), onChange: (k) => { st.h = Number(k); reset(); } });
slider(ctl, { name: 'Number of runs', min: 1, max: 10, step: 1, value: st.runs, format: String, onInput: (v) => { st.runs = v; reset(); } });
const row = document.createElement('div'); row.className = 'seg'; ctl.append(row);
button(row, 'Run', run, 'primary');
button(row, 'Clear', reset);

let refRep = [], refPG = [];
function reset() {
  const g = game(), T = HORIZON[st.kind];
  S.clearPaths('run');
  S.flowField((x) => replicatorRHS(g, x));
  refRep = integrate(replicatorRHS, g, st.x0, T);
  S.path('rep', refRep.map((p) => p.x), { color: 'ink', width: 2.2 });
  if (st.rule === 'pg') {
    refPG = integrate(pgRHS, g, st.x0, T);
    S.path('pgc', refPG.map((p) => p.x), { color: 'second', width: 2, dash: '6 4' });
  } else S.remove('pgc');
  S.point('start', st.x0, { color: 'ink', r: 4.5, label: 'start', dx: 8, dy: -8 });
  out.innerHTML = `The ink curve solves the replicator equation from the start point for ${T} units of time.` +
    (st.rule === 'pg' ? ' The <span class="c-second">dashed curve</span> is where softmax policy gradient goes when its step size shrinks to zero.' : '') +
    ` Press Run for ${st.runs} learning run${st.runs > 1 ? 's' : ''} of ${Math.round(T / st.h).toLocaleString()} steps each.`;
}

function run() {
  const g = game(), T = HORIZON[st.kind];
  S.clearPaths('run');
  const { paths, gRep: gr, gPG: gp } = runAndMeasure(st.rule, g, st.x0, st.h, T, st.runs);
  paths.forEach((p, k) => S.path(`run${k + 1}`, p.map((q) => q.x), { color: 'accent', width: 1, opacity: 0.6 }));
  S.path('rep', refRep.map((p) => p.x), { color: 'ink', width: 2.2 }); // keep the curve on top
  out.innerHTML = `${RULES.find((r) => r[0] === st.rule)[1]}, step size ${st.h}, ${st.runs} run${st.runs > 1 ? 's' : ''}. ` +
    `Largest distance of a <span class="c-accent">learning path</span> from the replicator curve, averaged over runs: <b>${gr.toFixed(3)}</b>` +
    (gp !== null ? `; from the <span class="c-second">dashed curve</span>: <b>${gp.toFixed(3)}</b>.` : '.');
}

// Click, or arrow keys, to move the starting point.
const setStart = (x) => {
  let y = x.map((v) => Math.max(0.01, v)); const z = y.reduce((a, b) => a + b, 0); y = y.map((v) => v / z);
  st.x0 = y; reset();
};
S.svg.on('click', (ev) => {
  const r = S.svg.node().getBoundingClientRect();
  const px = ((ev.clientX - r.left) * S.W) / r.width, py = ((ev.clientY - r.top) * S.H) / r.height;
  const x = S.fromXY([px, py]);
  if (x.every((v) => v >= -0.02)) setStart(x);
});
S.svg.on('keydown', (ev) => {
  const d = { ArrowUp: [0, -8], ArrowDown: [0, 8], ArrowLeft: [-8, 0], ArrowRight: [8, 0] }[ev.key];
  if (!d) return;
  ev.preventDefault();
  const [px, py] = S.toXY(st.x0);
  const x = S.fromXY([px + d[0], py + d[1]]);
  if (x.every((v) => v >= 0.005)) setStart(x);
});
reset();
