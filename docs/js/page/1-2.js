// Page 1.2: perceptron playground.
import { Playground, datasets, perceptronPass, step, sum } from '../lib/perceptron.js';
import { chart } from '../lib/chart.js';
import { slider, segmented, button, reducedMotion } from '../lib/ui.js';

const START = { w: [-0.5, 1], b: 0.3 }; // starting line (tools/test-page-1-2.mjs uses the same)
const MAX_PASSES = 100;
const NAMES = { and: 'AND', or: 'OR', xor: 'XOR', separable: 'Random, separable', nonseparable: 'Random, not separable' };

// ---- Static figure: three lines that fail on XOR ----
(function figure() {
  const fig = document.getElementById('fig-xor');
  const lines = [[[1, 0], -0.5, 'A vertical line'], [[1, 1], -0.5, 'A line cutting off (0, 0)'], [[1, -1], -0.3, 'A diagonal line']];
  for (const [w, b, title] of lines) {
    const box = document.createElement('div');
    const d = datasets.xor();
    const wrong = d.X.filter((x, i) => step(sum(w, b, x)) !== d.Y[i]).length;
    box.innerHTML = `<p class="small-title">${title}: ${wrong} wrong</p>`;
    const c = document.createElement('canvas');
    c.className = 'mini';
    c.setAttribute('aria-hidden', 'true');
    box.append(c);
    fig.append(box);
    const pg = new Playground(c, { w, b });
    pg.setData(d);
    requestAnimationFrame(() => pg.draw());
  }
})();

// ---- The tool ----
const state = { data: 'and', lr: 1, mode: 'single', ix: 0, passMistakes: 0, hist: [], updates: 0, running: null, last: '' };
const canvas = document.getElementById('plane');
const pg = new Playground(canvas, { w: START.w, b: START.b, onChange: () => { state.last = 'You moved the line.'; resetHistory(); draw(); } });
const out = document.getElementById('readout');
const ctl = document.getElementById('controls');

const errsChart = chart(document.getElementById('errs'), {
  height: 200, margin: { top: 34, right: 30, bottom: 50, left: 46 },
  x: { domain: [1, 10], label: 'pass', ticks: 5, format: 'd' }, y: { domain: [0, 4], label: 'mistakes in the pass', ticks: 4, format: 'd' },
  title: 'Mistakes per pass through the data',
});

function resetHistory() { stopRun(); state.ix = 0; state.passMistakes = 0; state.hist = []; state.updates = 0; }

function setData(k) {
  state.data = k;
  pg.setData(datasets[k]());
  resetHistory();
  state.last = '';
  draw();
}

function drawChart() {
  const n = state.data === 'and' || state.data === 'or' || state.data === 'xor' ? 4 : 24;
  errsChart.setY([0, n]);
  errsChart.setX([1, Math.max(10, state.hist.length)]);
  if (!state.hist.length) { errsChart.remove('h'); errsChart.remove('end'); return; }
  const pts = state.hist.map((e, i) => [i + 1, e]);
  errsChart.line('h', pts, { color: 'accent' });
  const lastP = pts[pts.length - 1];
  errsChart.point('end', lastP[0], lastP[1], { color: 'accent', r: 3.5, label: String(lastP[1]), dx: 6, dy: -6 });
}

function draw() {
  pg.draw();
  drawChart();
  const X = pg.X, wrong = pg.mistakes();
  const fmt = (v) => (Math.abs(v) < 5e-3 ? '0' : v.toFixed(2));
  const pass = state.hist.length;
  let s = '';
  if (state.mode === 'hidden') {
    s = `Network with a hidden layer. Points labeled wrongly: <b>${wrong}</b> of ${X.length}.`;
  } else {
    s = `\\(w\\) = (${fmt(pg.w[0])}, ${fmt(pg.w[1])}), \\(b\\) = ${fmt(pg.b)}. Points labeled wrongly now: <b>${wrong}</b> of ${X.length}. ` +
      (pass ? `Passes completed: ${pass}; updates so far: ${state.updates}.` : 'No passes yet.') +
      (state.ix > 0 ? ` In pass ${pass + 1}, ${state.ix} of ${X.length} examples seen.` : '') +
      (state.last ? ` ${state.last}` : '');
  }
  out.innerHTML = s;
  if (window.renderMathInElement) window.renderMathInElement(out, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
}

// One example of the learning rule.
function stepOne() {
  const { X, Y } = pg, i = state.ix, x = X[i];
  const pred = step(sum(pg.w, pg.b, x));
  if (pred !== Y[i]) {
    const d = state.lr * (Y[i] - pred);
    pg.w = [pg.w[0] + d * x[0], pg.w[1] + d * x[1]]; pg.b += d;
    state.passMistakes++; state.updates++;
    state.last = `Example (${x.map((v) => +v.toFixed(2)).join(', ')}) has label ${Y[i]} but got ${pred}, so the rule ${d > 0 ? 'added' : 'subtracted'} ${Math.abs(d)} times the example to the weights.`;
  } else {
    state.last = `Example (${x.map((v) => +v.toFixed(2)).join(', ')}) has label ${Y[i]} and got ${pred}: no change.`;
  }
  state.ix++;
  if (state.ix >= X.length) { state.hist.push(state.passMistakes); state.ix = 0; state.passMistakes = 0; }
}
function finishPass() {
  if (state.ix > 0) { while (state.ix > 0) stepOne(); return; }
  const s = { w: pg.w.slice(), b: pg.b };
  const m = perceptronPass(s, pg.X, pg.Y, state.lr);
  const same = Math.abs(s.w[0] - pg.w[0]) + Math.abs(s.w[1] - pg.w[1]) + Math.abs(s.b - pg.b) < 1e-9;
  pg.w = s.w; pg.b = s.b;
  state.hist.push(m); state.updates += m;
  state.last = m === 0 ? 'This pass made no mistakes, so the weights no longer change.'
    : same ? `This pass made ${m} updates and ended with the same weights it started with.` : '';
}
function done() { return state.hist.length > 0 && state.hist[state.hist.length - 1] === 0 && state.ix === 0; }
function stopRun() { clearTimeout(state.running); state.running = null; if (runBtn) runBtn.textContent = 'Run'; }
function run() {
  if (state.running) { stopRun(); return; }
  if (reducedMotion()) {
    while (!done() && state.hist.length < MAX_PASSES) finishPass();
    if (!done()) state.last = `Stopped after ${MAX_PASSES} passes.`;
    draw(); return;
  }
  runBtn.textContent = 'Pause';
  const tick = () => {
    if (done()) { stopRun(); draw(); return; }
    if (state.hist.length >= MAX_PASSES) { state.last = `Stopped after ${MAX_PASSES} passes.`; stopRun(); draw(); return; }
    finishPass(); draw();
    state.running = setTimeout(tick, state.hist.length < 12 ? 350 : 60);
  };
  tick();
}

segmented(ctl, { name: 'Dataset', options: Object.entries(NAMES), value: state.data, onChange: setData });
segmented(ctl, { name: 'Learning rate', options: [['0.1', '0.1'], ['0.25', '0.25'], ['0.5', '0.5'], ['1', '1']], value: '1',
  onChange: (k) => { state.lr = Number(k); } });
segmented(ctl, { name: 'Network', options: [['single', 'One perceptron'], ['hidden', 'With a hidden layer']], value: 'single',
  onChange: (k) => setMode(k) });
const btns = document.createElement('div');
btns.className = 'seg';
ctl.append(btns);
const stepBtn = button(btns, 'Step', () => { stopRun(); stepOne(); draw(); });
const passBtn = button(btns, 'Pass', () => { stopRun(); finishPass(); draw(); });
const runBtn = button(btns, 'Run', run, 'primary');
const resetBtn = button(btns, 'Reset line', () => { pg.set(START.w, START.b); resetHistory(); state.last = ''; draw(); });

// Hidden layer: nine sliders, set by hand.
const hc = document.createElement('div');
hc.id = 'hidden-controls';
ctl.append(hc);
const H = { W: [[1, 0], [0, 1]], b: [0, 0], v: [1, 1], c: 0 };
const unit = (label) => { const p = document.createElement('p'); p.className = 'unit'; p.textContent = label; hc.append(p); };
const hs = (name, get, set) => slider(hc, { name, min: -2, max: 2, step: 0.5, value: get(), format: (v) => v.toFixed(1), onInput: (v) => { set(v); draw(); } });
unit('Hidden unit h₁');
hs('weight on x₁', () => H.W[0][0], (v) => { H.W[0][0] = v; });
hs('weight on x₂', () => H.W[0][1], (v) => { H.W[0][1] = v; });
hs('bias', () => H.b[0], (v) => { H.b[0] = v; });
unit('Hidden unit h₂');
hs('weight on x₁', () => H.W[1][0], (v) => { H.W[1][0] = v; });
hs('weight on x₂', () => H.W[1][1], (v) => { H.W[1][1] = v; });
hs('bias', () => H.b[1], (v) => { H.b[1] = v; });
unit('Output unit');
hs('weight on h₁', () => H.v[0], (v) => { H.v[0] = v; });
hs('weight on h₂', () => H.v[1], (v) => { H.v[1] = v; });
hs('bias', () => H.c, (v) => { H.c = v; });

function setMode(k) {
  stopRun();
  state.mode = k;
  pg.hidden = k === 'hidden' ? H : null;
  hc.classList.toggle('on', k === 'hidden');
  document.getElementById('errs').style.display = k === 'hidden' ? 'none' : '';
  [stepBtn, passBtn, runBtn, resetBtn].forEach((b) => { b.disabled = k === 'hidden'; });
  draw();
}

window.addEventListener('resize', () => pg.draw());
setData('and');
