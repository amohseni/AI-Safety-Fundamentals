// Page 6.4: toy model of superposition (Elhage et al. 2022), trained live.
import { makeRuns, pickBest, TRAIN } from '../lib/s6-toymodel.js';
import { ArrowView, heatmap, bars } from '../lib/s6-views.js';
import { slider, segmented, button, reducedMotion } from '../lib/ui.js';

const SPARSITY = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99];
const state = { n: 5, m: 2, sIx: 0, r: 0.8, group: 0, bars: 'len' };
const S = () => SPARSITY[state.sIx];

// ---- Static figure: importance falling by 0.8, four sparsities. Parameters from tools/test-page-6-4.mjs (group 0). ----
const FIG = {
  0: [0.935, -0.355, 0.002, -0.003, 0.007, -0.353, -0.935, 0.018, 0.007, -0.001],
  0.5: [-0.492, -0.865, 0.813, 0.004, -0.002, 0.874, -0.488, 0.459, -0.015, 0],
  0.8: [-0.594, 0.587, -0.825, 0.812, -0.003, -0.826, 0.813, 0.595, -0.588, -0.001],
  0.9: [-0.408, 0.9, -1.086, 0.86, -0.28, -1.037, 0.656, 0.117, -0.643, 0.978],
};
const figViews = [];
(function figure() {
  const fig = document.getElementById('fig-sparsity');
  for (const s of [0, 0.5, 0.8, 0.9]) {
    const W = FIG[s];
    const box = document.createElement('div');
    const stored = [0, 1, 2, 3, 4].filter((i) => Math.hypot(W[i], W[5 + i]) > 0.5).length;
    box.innerHTML = `<p class="small-title">S = ${s}: ${stored} stored</p>`;
    const c = document.createElement('canvas');
    c.setAttribute('aria-hidden', 'true');
    box.append(c);
    fig.append(box);
    const v = new ArrowView(c, { dim: 2, interactive: false, maxSide: 260 });
    v.extent = 1.25;
    v.items = [0, 1, 2, 3, 4].map((i) => ({ vec: [W[i], W[5 + i]], width: 1 + 3 * 0.8 ** i, label: String(i + 1) }));
    figViews.push(v);
  }
})();

// ---- The tool ----
const canvas = document.getElementById('arrows');
const view = new ArrowView(canvas, { dim: 2 });
const barBox = document.getElementById('bars');
const heatBox = document.getElementById('heat');
const barChart = bars(barBox, { title: 'One bar per feature' });
const heat = heatmap(heatBox, { title: 'Dot products between feature directions' });
const out = document.getElementById('readout');
const barNote = document.getElementById('bars-note');

let runs = [], shown = null, ema = [], raf = null, training = false, done = false;

function reset() {
  stop();
  runs = makeRuns({ n: state.n, m: state.m, S: S(), r: state.r, group: state.group });
  ema = runs.map(() => NaN);
  shown = runs[0];
  done = false;
  trainBtn.disabled = false;
  view.setDim(state.m);
  draw();
  start();
}

function start() {
  if (done) return;
  training = true;
  trainBtn.textContent = 'Pause';
  raf = requestAnimationFrame(loop);
}
function stop() {
  training = false;
  cancelAnimationFrame(raf);
  trainBtn.textContent = done ? 'Trained' : 'Train';
}

function loop() {
  const t0 = performance.now();
  const budget = reducedMotion() ? 40 : 14;
  while (performance.now() - t0 < budget && runs[0].steps < TRAIN.steps) {
    for (let k = 0; k < runs.length; k++) {
      const l = runs[k].step();
      ema[k] = Number.isNaN(ema[k]) ? l : 0.98 * ema[k] + 0.02 * l;
    }
  }
  let b = 0;
  ema.forEach((v, k) => { if (v < ema[b]) b = k; });
  shown = runs[b];
  if (runs[0].steps >= TRAIN.steps) {
    shown = pickBest(runs);
    done = true;
    stop();
    trainBtn.disabled = true;
    draw();
    return;
  }
  if (!reducedMotion()) draw();
  else out.textContent = `Training: step ${runs[0].steps.toLocaleString()} of ${TRAIN.steps.toLocaleString()}.`;
  raf = requestAnimationFrame(loop);
}

function draw() {
  const t = shown, n = t.n, m = t.m;
  const norms = t.norms();
  view.axes = Array.from({ length: m }, (_, k) => `coordinate ${k + 1}`);
  view.highlight = state.bars === 'len' ? -1 : Number(state.bars);
  view.extent = Math.max(1.25, ...norms) * 1.05;
  view.items = Array.from({ length: n }, (_, i) => ({ vec: t.dir(i), width: 1 + 3 * t.importance(i), label: String(i + 1) }));
  view.draw();

  const every = n > 10 ? 5 : 1;
  if (state.bars === 'len') {
    barChart.update(norms, { domain: [0, 1.5], yLabel: 'length of direction', xLabel: 'feature', ref: [{ v: 0.5, label: 'stored above 0.5' }], every });
    barNote.textContent = 'Length of each feature’s direction.';
  } else {
    const k = Number(state.bars);
    const w = Array.from({ length: n }, (_, i) => t.W[k * n + i]);
    barChart.update(w, { domain: [-1.5, 1.5], yLabel: `component on coordinate ${k + 1}`, xLabel: 'feature', ref: [{ v: 0.3, label: '0.3' }, { v: -0.3, label: '−0.3' }], every });
    const big = w.filter((v) => Math.abs(v) >= 0.3).length;
    barNote.textContent = `Component of each feature’s direction on coordinate ${k + 1}: the value of coordinate ${k + 1} when that feature alone is 1. ${big} of ${n} components have size at least 0.3.`;
  }
  const G = t.gram();
  heat.update(G, { max: Math.max(1, ...G.map((row, i) => Math.abs(row[i]))), every, top: 'feature', left: 'feature' });

  const stored = norms.filter((v) => v > 0.5).length;
  const step = t.steps;
  const loss = done ? t.evalLoss() : Math.min(...ema);
  out.innerHTML = `${done ? 'Trained' : 'Training'}: step <b>${step.toLocaleString()}</b> of ${TRAIN.steps.toLocaleString()}; loss ${Number.isFinite(loss) ? loss.toFixed(4) : '…'}. ` +
    `Stored: <b>${stored} of ${n}</b> features in ${m} dimensions (direction longer than 0.5).` +
    (m === 3 ? ' Drag the picture, or select it and use the arrow keys, to turn it.' : '');
}

// ---- Controls ----
const ctl = document.getElementById('controls');
slider(ctl, { name: 'Features n', min: 5, max: 20, step: 1, value: state.n, onInput: (v) => { state.n = v; schedule(); } });
segmented(ctl, { name: 'Hidden dimensions m', options: [['2', '2'], ['3', '3']], value: '2', onChange: (k) => {
  state.m = Number(k);
  if (state.bars !== 'len' && Number(state.bars) >= state.m) state.bars = 'len';
  barSeg.rebuild();
  reset();
} });
slider(ctl, { name: 'Sparsity S (chance a feature is 0)', min: 0, max: SPARSITY.length - 1, step: 1, value: state.sIx,
  format: (i) => String(SPARSITY[i]), onInput: (v) => { state.sIx = v; schedule(); } });
slider(ctl, { name: 'Importance falloff r', min: 0.7, max: 1, step: 0.05, value: state.r, format: (v) => v.toFixed(2),
  onInput: (v) => { state.r = Math.round(v * 100) / 100; schedule(); } });
const btnRow = document.createElement('div');
btnRow.className = 'seg';
ctl.append(btnRow);
const trainBtn = button(btnRow, 'Pause', () => { if (done) return; training ? stop() : start(); }, 'primary');
button(btnRow, 'New start', () => { state.group++; reset(); });

// Bars mode: lengths, or the components on one hidden coordinate.
const barCtl = document.getElementById('bar-controls');
const barSeg = (() => {
  let seg = null;
  return {
    rebuild() {
      barCtl.innerHTML = '';
      const opts = [['len', 'Length'], ...Array.from({ length: state.m }, (_, k) => [String(k), `Coordinate ${k + 1}`])];
      seg = segmented(barCtl, { name: 'Bars show', options: opts, value: state.bars, onChange: (k) => { state.bars = k; draw(); } });
    },
  };
})();
barSeg.rebuild();

// Sliders restart training once the reader stops moving them.
let pending = null;
function schedule() { clearTimeout(pending); stop(); pending = setTimeout(reset, 250); }

let resizeT = null;
window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(() => { draw(); figViews.forEach((v) => v.draw()); }, 150); });
if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { draw(); figViews.forEach((v) => v.draw()); });

requestAnimationFrame(() => { figViews.forEach((v) => v.draw()); reset(); });
