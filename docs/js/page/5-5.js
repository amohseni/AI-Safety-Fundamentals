// Page 5.5: five seeds. Panel 1 trains five spirals networks live; panel 2 shows
// five precomputed modular-addition runs (tools/modadd).
import { chart } from '../lib/chart.js';
import { segmented, button, cssVar, reducedMotion } from '../lib/ui.js';
import { loadSeeds, loadEmbeddings, embedding } from '../lib/modadd.js';
import { SEEDS, STEPS, DATA, makeNet, agreement, responses, bestMatches, median } from './5-5-measure.js';

const pct = (v) => `${Math.round(v * 100)}%`;
const pct1 = (v) => `${(v * 100).toFixed(1)}%`;

// ---- Drawing helpers for the plane ----
function prep(canvas) {
  const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight || W;
  if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  return { ctx, W, H };
}
// The plane [-1, 1]^2 mapped onto the canvas.
const toPx = (W, v) => ((v + 1) / 2) * W;

function shadeBoundary(canvas, net, n = 36) {
  const { ctx, W } = prep(canvas);
  const a = cssVar('--accent'), b = cssVar('--second'), cw = W / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const x = [-1 + (2 * i + 1) / n, 1 - (2 * j + 1) / n];
    const p = net.predict(x);
    ctx.fillStyle = p > 0.5 ? a : b;
    ctx.globalAlpha = 0.12 + 0.3 * Math.min(1, Math.abs(p - 0.5) * 2);
    ctx.fillRect(i * cw, j * cw, cw + 0.5, cw + 0.5);
  }
  ctx.globalAlpha = 1;
  dots(ctx, W, 1.3);
}

function dots(ctx, W, r) {
  const a = cssVar('--accent'), b = cssVar('--second');
  DATA.X.forEach((x, k) => {
    ctx.beginPath();
    ctx.arc(toPx(W, x[0]), toPx(W, -x[1]), r, 0, 2 * Math.PI);
    ctx.fillStyle = DATA.Y[k] ? a : b;
    ctx.fill();
  });
}

// A hidden unit's response over the plane, from -1 (paper) to 1 (ink).
function shadeUnit(canvas, net, layer, u, n = 36) {
  const { ctx, W } = prep(canvas);
  ctx.fillStyle = cssVar('--ink');
  const cw = W / n;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const h = net.stages([-1 + (2 * i + 1) / n, 1 - (2 * j + 1) / n])[2 * layer][u];
    ctx.globalAlpha = 0.85 * (h + 1) / 2;
    ctx.fillRect(i * cw, j * cw, cw + 0.5, cw + 0.5);
  }
  ctx.globalAlpha = 1;
}

// The decision boundary (output = 0.5) as line segments, by marching squares.
function boundarySegments(net, n) {
  const f = [];
  for (let i = 0; i <= n; i++) { f.push([]); for (let j = 0; j <= n; j++) f[i].push(net.predict([-1 + (2 * i) / n, 1 - (2 * j) / n]) - 0.5); }
  const segs = [];
  const P = (i, j) => [-1 + (2 * i) / n, 1 - (2 * j) / n];
  const lerp = (p, q, fp, fq) => { const t = fp / (fp - fq); return [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]; };
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const c = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]];
    const v = c.map(([a, b]) => f[a][b]);
    const pts = [];
    for (let e = 0; e < 4; e++) {
      const k = (e + 1) % 4;
      if ((v[e] > 0) !== (v[k] > 0)) pts.push(lerp(P(...c[e]), P(...c[k]), v[e], v[k]));
    }
    if (pts.length === 2) segs.push(pts);
    if (pts.length === 4) segs.push([pts[0], pts[1]], [pts[2], pts[3]]);
  }
  return segs;
}

// ---- Panel 1: spirals ----
const state = { layer: 2, selected: new Set([0, 1, 2, 3, 4]), steps: 0, training: false };
let nets = SEEDS.map(makeNet);
const out1 = document.getElementById('spiral-readout');
const overlay = document.getElementById('overlay');
const rowsEl = document.getElementById('rows');

function buildRows() {
  rowsEl.innerHTML = '';
  const head = document.createElement('div');
  head.className = 'unit-row head';
  head.innerHTML = `<span></span><span>Output</span>${[1, 2, 3].map((u) => `<span>Unit ${u}</span>`).join('')}`;
  rowsEl.append(head);
  for (const r of [...state.selected].sort()) {
    const row = document.createElement('div');
    row.className = 'unit-row';
    row.innerHTML = `<span class="run-name"><span class="wide">Run </span>${r + 1}</span>` + [0, 1, 2, 3].map(() => '<canvas class="mini" aria-hidden="true"></canvas>').join('');
    row.dataset.run = r;
    rowsEl.append(row);
  }
}

function drawSpirals(full = true) {
  const { ctx, W } = prep(overlay);
  ctx.strokeStyle = cssVar('--rule'); ctx.lineWidth = 1;
  dots(ctx, W, 2.2);
  ctx.strokeStyle = cssVar('--ink'); ctx.lineWidth = 1.3;
  for (const r of state.selected) {
    ctx.beginPath();
    for (const [p, q] of boundarySegments(nets[r], full ? 90 : 50)) { ctx.moveTo(toPx(W, p[0]), toPx(W, -p[1])); ctx.lineTo(toPx(W, q[0]), toPx(W, -q[1])); }
    ctx.stroke();
  }
  rowsEl.querySelectorAll('.unit-row[data-run]').forEach((row) => {
    const net = nets[Number(row.dataset.run)];
    const cs = row.querySelectorAll('canvas');
    shadeBoundary(cs[0], net, full ? 40 : 24);
    for (let u = 0; u < 3; u++) shadeUnit(cs[u + 1], net, state.layer, u, full ? 40 : 24);
  });
  readout1();
}

function readout1() {
  const accs = nets.map((n) => n.accuracy(DATA.X, DATA.Y));
  let s = `Training step <b>${state.steps.toLocaleString('en-US')}</b> of ${STEPS.toLocaleString('en-US')}. Accuracy on the training points: ${accs.map((a, r) => `run ${r + 1} ${pct(a)}`).join(', ')}.`;
  const sel = [...state.selected].sort();
  if (!state.training && state.steps > 0 && sel.length >= 2) {
    const ag = [], m = [];
    const R = new Map(sel.map((r) => [r, responses(nets[r], state.layer)]));
    for (const a of sel) for (const b of sel) {
      if (a < b) ag.push(agreement(nets[a], nets[b]));
      if (a !== b) m.push(...bestMatches(R.get(a), R.get(b)));
    }
    s += ` On 2,000 new points drawn from the same spirals, any two of the selected runs give the same class to between <b>${pct1(Math.min(...ag))}</b> and <b>${pct1(Math.max(...ag))}</b> of the points. For each layer-${state.layer} unit of one run, the most similar unit of another run has a correlation (ignoring sign) of median <b>${median(m).toFixed(2)}</b> over these points; ${m.filter((v) => v < 0.9).length} of the ${m.length} units have no match at 0.9 or above.`;
  } else if (sel.length < 2 && state.steps > 0) {
    s += ' Select at least two runs to compare them.';
  }
  out1.innerHTML = s;
}

let raf = null;
function loop() {
  const k = Math.min(40, STEPS - state.steps);
  for (const n of nets) for (let i = 0; i < k; i++) n.step(DATA.X, DATA.Y);
  state.steps += k;
  if (state.steps >= STEPS) { stopTraining(); drawSpirals(true); return; }
  if (!reducedMotion() && state.steps % 400 === 0) drawSpirals(false);
  else out1.innerHTML = `Training step <b>${state.steps.toLocaleString('en-US')}</b> of ${STEPS.toLocaleString('en-US')}…`;
  raf = setTimeout(loop, 0);
}
function startTraining() {
  if (state.training || state.steps >= STEPS) return;
  state.training = true; trainBtn.textContent = 'Pause';
  raf = setTimeout(loop, 0);
}
function stopTraining() { state.training = false; clearTimeout(raf); trainBtn.textContent = state.steps >= STEPS ? 'Trained' : 'Continue training'; trainBtn.disabled = state.steps >= STEPS; }

const ctl1 = document.getElementById('spiral-controls');
const btns = document.createElement('div'); btns.className = 'seg'; ctl1.append(btns);
const trainBtn = button(btns, 'Train all five', () => (state.training ? (stopTraining(), drawSpirals(true)) : startTraining()), 'primary');
button(btns, 'Reset', () => { stopTraining(); nets = SEEDS.map(makeNet); state.steps = 0; trainBtn.disabled = false; trainBtn.textContent = 'Train all five'; drawSpirals(true); });
segmented(ctl1, { name: 'Hidden units shown', options: [['1', 'Layer 1'], ['2', 'Layer 2']], value: '2',
  onChange: (k) => { state.layer = Number(k); buildRows(); drawSpirals(!state.training); } });
// Run picker: several runs can be selected at once.
{
  const wrap = document.createElement('div');
  wrap.className = 'ctl'; wrap.setAttribute('role', 'group'); wrap.setAttribute('aria-label', 'Runs shown');
  wrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Runs shown</span></span>';
  const seg = document.createElement('span'); seg.className = 'seg';
  SEEDS.forEach((_, r) => {
    const b = document.createElement('button'); b.type = 'button'; b.textContent = `Run ${r + 1}`; b.setAttribute('aria-pressed', 'true');
    b.addEventListener('click', () => {
      if (state.selected.has(r)) state.selected.delete(r); else state.selected.add(r);
      b.setAttribute('aria-pressed', String(state.selected.has(r)));
      buildRows(); drawSpirals(!state.training);
    });
    seg.append(b);
  });
  wrap.append(seg); ctl1.append(wrap);
}

buildRows();
drawSpirals(true);
let rt;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { if (!state.training) drawSpirals(true); }, 150); });

// ---- Panel 2: modular addition ----
(async function modadd() {
  const data = await loadSeeds();
  const box = document.getElementById('freq-multiples');
  const yMax = Math.max(...data.seeds.flatMap((s) => s.spectrum)) * 1.12;
  const K = (data.p - 1) / 2;
  data.seeds.forEach((s, r) => {
    const div = document.createElement('div');
    div.innerHTML = `<p class="small-title">Run ${r + 1}</p>`;
    box.append(div);
    const c = chart(div, { width: 300, height: 200, margin: { top: 24, right: 14, bottom: 56, left: 36 },
      x: { domain: [0, K + 1], ticks: 3, label: 'frequency k' }, y: { domain: [0, yMax], ticks: 2 },
      title: `Run ${r + 1}: size of each frequency in the embedding; key frequencies ${s.key_freqs.join(', ')}` });
    s.spectrum.forEach((v, i) => c.line(`k${i}`, [[i + 1, 0], [i + 1, v]], { color: s.key_freqs.includes(i + 1) ? 'accent' : 'quiet', width: 3 }));
    // Number the key frequencies, nudging labels apart when two are close.
    let lastX = -Infinity;
    s.key_freqs.forEach((k) => {
      const px = c.x(k), shift = px - lastX < 30 ? 30 - (px - lastX) : 0;
      lastX = px + shift;
      c.text(`t${k}`, c.x.invert(px + shift), s.spectrum[k - 1] + yMax * 0.05, String(k), { color: 'accent', anchor: 'middle' });
    });
  });
  const lines = data.seeds.map((s, r) => `run ${r + 1}: ${s.key_freqs.join(', ')}`).join('; ');
  document.getElementById('mod-readout').innerHTML =
    `Every run answers ${data.seeds.every((s) => s.test_acc === 1) ? '100%' : `between ${pct1(Math.min(...data.seeds.map((s) => s.test_acc)))} and ${pct1(Math.max(...data.seeds.map((s) => s.test_acc)))}`} of the test pairs correctly. Key frequencies, <span class="c-accent">in blue</span>: ${lines}.`;
})();

// ---- Static figure: the trained embedding as a heat map ----
(async function figure() {
  const emb = await loadEmbeddings();
  const p = emb.p, d = emb.d_model;
  const E = embedding(emb.embeddings[0].W_E, p, d);
  const canvas = document.getElementById('fig-weights');
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = W * p / d;
  canvas.style.height = `${H}px`;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  let mx = 0; E.forEach((row) => row.forEach((v) => { mx = Math.max(mx, Math.abs(v)); }));
  const pos = cssVar('--accent'), neg = cssVar('--second'), cw = W / d, ch = H / p;
  for (let a = 0; a < p; a++) for (let j = 0; j < d; j++) {
    const v = E[a][j];
    ctx.fillStyle = v >= 0 ? pos : neg;
    ctx.globalAlpha = Math.min(1, Math.abs(v) / (0.6 * mx));
    ctx.fillRect(j * cw, a * ch, cw + 0.3, ch + 0.3);
  }
  ctx.globalAlpha = 1;
})();
