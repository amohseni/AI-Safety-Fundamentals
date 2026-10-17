// Page 6.5: a sparse autoencoder trained live on the hidden activations of the pentagon toy model from page 6.4.
import { ToyModel, SAE, sampleFeatures } from '../lib/s6-toymodel.js';
import { rng } from '../lib/mlp.js';
import { ArrowView, heatmap, bars } from '../lib/s6-views.js';
import { slider, segmented, button, reducedMotion } from '../lib/ui.js';

// The pentagon model of page 6.4 (5 features, 2 dimensions, S = 0.9, equal importance, group 0),
// as trained by tools/test-page-6-5.mjs.
const toy = new ToyModel({ n: 5, m: 2, S: 0.9, r: 1 });
toy.W.set([-0.1893, 0.7875, -1.0884, 0.9732, -0.4878, -1.0844, 0.7733, -0.1586, -0.5101, 0.9896]);
toy.b.set([-0.2125, -0.2122, -0.2127, -0.2098, -0.2097]);

const PENALTY = [0, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1, 2];
const STEPS = 3000;
const LETTERS = 'ABCDEFGHIJKLMNOPQRST';
const state = { pIx: 2, size: 5, seed: 1 };

// Activation points to show: 600 inputs, of which the nonzero ones are drawn.
const pts = (() => {
  const X = sampleFeatures(rng(77), 5, 0.9, 600);
  const out = [];
  for (let s = 0; s < 600; s++) { const h = toy.hidden(X, s * 5); if (h[0] || h[1]) out.push([h[0], h[1]]); }
  return out;
})();

const figDraw = [];
// ---- Static figure: one activation decomposed by a dictionary of 5 (penalty 0.5, starting point 1). ----
(function figure() {
  const D = [0.7044, -0.9794, -0.2458, -0.5374, 0.76, 0.7098, -0.2022, -0.9693, 0.8433, -0.6499];
  const f = [0, 0.682, 0.897, 0, 0];
  const h = toy.hidden([0.9, 0, 0.6, 0, 0]);
  const fig = document.getElementById('fig-decomp');
  const left = document.createElement('div');
  left.innerHTML = '<p class="small-title">The activation and its two parts</p>';
  const c = document.createElement('canvas');
  c.setAttribute('aria-hidden', 'true');
  left.append(c);
  const right = document.createElement('div');
  right.innerHTML = '<p class="small-title">The code: one number per unit</p>';
  fig.append(left, right);
  const v = new ArrowView(c, { dim: 2, interactive: false, maxSide: 300 });
  v.extent = 1.3;
  const dB = [D[1] * f[1], D[6] * f[1]], dC = [D[2] * f[2], D[7] * f[2]];
  v.items = [
    { vec: [h[0], h[1]], color: 'ink', width: 2, label: 'h' },
    { vec: dC, color: 'accent', width: 2, label: 'C' },
    { vec: dB, color: 'accent', width: 2, label: 'B' },
  ];
  // Unit B's part drawn again from the tip of unit C's part, to show the sum.
  const drawSum = () => {
    v.draw();
    const ctx = v.ctx, side = parseFloat(c.style.width), s = (side / 2 - 34) / v.extent;
    const X = (p) => side / 2 + p[0] * s, Y = (p) => side / 2 - p[1] * s;
    ctx.save();
    ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--accent');
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(X(dC), Y(dC)); ctx.lineTo(X([dC[0] + dB[0], dC[1] + dB[1]]), Y([dC[0] + dB[0], dC[1] + dB[1]])); ctx.stroke();
    ctx.restore();
  };
  const chartBox = document.createElement('div');
  right.append(chartBox);
  const b = bars(chartBox, { title: 'Code values for units A to E', height: 230, maxWidth: 300 });
  const drawAll = () => { drawSum(); b.update(f, { domain: [0, 1], yLabel: 'unit activation', xLabel: 'unit', tickLabel: (i) => LETTERS[i] }); };
  figDraw.push(drawAll);
})();

// ---- The tool ----
const canvas = document.getElementById('plane');
const view = new ArrowView(canvas, { dim: 2, interactive: false });
const heat = heatmap(document.getElementById('sim'), { title: 'Cosine similarity between true features and dictionary units', maxWidth: 520 });
const out = document.getElementById('readout');
const matches = document.getElementById('matches');
let sae = null, raf = null, training = false, last = null;

function reset() {
  stop();
  sae = new SAE(toy, { size: state.size, lambda: PENALTY[state.pIx], seed: state.seed, lr: 0.005 });
  last = null;
  trainBtn.disabled = false;
  draw();
  start();
}
function start() { if (sae.steps >= STEPS) return; training = true; trainBtn.textContent = 'Pause'; raf = requestAnimationFrame(loop); }
function stop() { training = false; cancelAnimationFrame(raf); trainBtn.textContent = sae && sae.steps >= STEPS ? 'Trained' : 'Train'; }
function loop() {
  const t0 = performance.now(), budget = reducedMotion() ? 40 : 14;
  while (performance.now() - t0 < budget && sae.steps < STEPS) sae.step();
  if (sae.steps >= STEPS) { stop(); trainBtn.disabled = true; draw(); return; }
  if (!reducedMotion()) draw(true);
  else out.textContent = `Training: step ${sae.steps.toLocaleString()} of ${STEPS.toLocaleString()}.`;
  raf = requestAnimationFrame(loop);
}

function draw(quick = false) {
  const ev = sae.evaluate(quick ? 800 : 4000);
  last = ev;
  view.extent = 1.5;
  view.points = pts;
  view.axes = null;
  const items = [];
  for (let i = 0; i < 5; i++) items.push({ vec: toy.dir(i), color: 'ink', width: 1.2, label: String(i + 1), head: false });
  for (let k = 0; k < sae.size; k++) {
    const d = sae.dir(k).map((x) => x * 1.4);
    items.push(ev.alive[k] ? { vec: d, color: 'accent', width: 2, label: LETTERS[k] } : { vec: d, color: 'quiet', width: 1, dash: [3, 3], label: LETTERS[k] });
  }
  view.items = items;
  view.draw();

  const rec = sae.recovery(0.5, ev.alive);
  const G = Array.from({ length: 5 }, (_, i) => {
    const w = toy.dir(i), L = Math.hypot(...w);
    return Array.from({ length: sae.size }, (_, k) => { const d = sae.dir(k); return (w[0] * d[0] + w[1] * d[1]) / L; });
  });
  heat.update(G, { max: 1, colLabel: (j) => LETTERS[j], top: 'dictionary unit', every: 1 });
  const found = rec.filter((r) => r.cos > 0.95).length;
  const mean = rec.reduce((a, r) => a + r.cos, 0) / rec.length;
  out.innerHTML = `${sae.steps >= STEPS ? 'Trained' : 'Training'}: step <b>${sae.steps.toLocaleString()}</b> of ${STEPS.toLocaleString()}. ` +
    `Rebuild error <b>${ev.fvu.toFixed(3)}</b>. Active units per input <b>${ev.l0.toFixed(2)}</b> (true features active per input: 0.50). ` +
    `Recovered <b>${found} of 5</b> features; average best cosine ${mean.toFixed(2)}.` + (ev.dead ? ` ${ev.dead} of ${sae.size} units never fire (dashed).` : '');
  matches.textContent = 'Best live unit for each feature: ' + rec.map((r) => `${r.feature + 1} → ${r.unit >= 0 ? LETTERS[r.unit] : 'none'} (${r.cos.toFixed(2)})`).join(', ') + '.';
}

const ctl = document.getElementById('controls');
slider(ctl, { name: 'Sparsity penalty λ', min: 0, max: PENALTY.length - 1, step: 1, value: state.pIx, format: (i) => String(PENALTY[i]),
  onInput: (v) => { state.pIx = v; schedule(); } });
segmented(ctl, { name: 'Dictionary size (units)', options: [['3', '3'], ['5', '5'], ['10', '10'], ['20', '20']], value: String(state.size),
  onChange: (k) => { state.size = Number(k); reset(); } });
const row = document.createElement('div');
row.className = 'seg';
ctl.append(row);
const trainBtn = button(row, 'Pause', () => { if (sae.steps >= STEPS) return; training ? stop() : start(); }, 'primary');
button(row, 'New start', () => { state.seed++; reset(); });

let pending = null;
function schedule() { clearTimeout(pending); stop(); pending = setTimeout(reset, 250); }

const redrawAll = () => { if (sae) draw(); figDraw.forEach((f) => f()); };
let rt = null;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(redrawAll, 150); });
if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawAll);
requestAnimationFrame(() => { figDraw.forEach((f) => f()); reset(); });
