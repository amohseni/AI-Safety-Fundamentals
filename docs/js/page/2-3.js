// Page 2.3: collapse.
import { Net, ACTS, countRegions } from '../lib/s2-net.js';
import { datasets } from '../lib/mlp.js';
import { chart } from '../lib/chart.js';
import { slider, segmented, button, cssVar } from '../lib/ui.js';
import { toRGB } from '../lib/perceptron.js';

const d3 = window.d3;
const DATA = datasets.spirals();
const SEEDS = [1, 2, 3, 4, 5]; // the starts measured in tools/test-page-2-3.mjs
const MAX_STEPS = 2000, LR = 0.01, LO = -1.2, HI = 1.2, GRID = 300;

// ---- Static figure ----
(function figure() {
  const fig = document.getElementById('fig-acts');
  const curves = [['ReLU', ACTS.relu.f, [-1, 3]], ['GELU', ACTS.gelu.f, [-1, 3]], ['sigmoid', ACTS.sigmoid.f, [-0.2, 1.2]]];
  for (const [name, f, yd] of curves) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${name}</p>`;
    fig.append(box);
    const c = chart(box, { width: 300, height: 190, margin: { top: 16, right: 14, bottom: 34, left: 40 }, x: { domain: [-3, 3], ticks: 3 }, y: { domain: yd, ticks: 3 }, title: `${name} activation function` });
    c.line('f', d3.range(-3, 3.001, 0.05).map((z) => [z, f(z)]), { color: 'ink', width: 2 });
  }
  // Softmax of the scores (2, 1, 0).
  const box = document.createElement('div');
  box.innerHTML = '<p class="small-title">softmax of 2, 1, 0</p>';
  fig.append(box);
  const s = [2, 1, 0].map(Math.exp), tot = s.reduce((a, b) => a + b), p = s.map((v) => v / tot);
  const W = 220, H = 190, m = { top: 16, right: 14, bottom: 34, left: 40 };
  const svg = d3.select(box).append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .attr('role', 'img').attr('aria-label', 'Softmax turns the scores 2, 1, 0 into probabilities 0.67, 0.24, 0.09');
  const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const x = d3.scaleBand().domain(['2', '1', '0']).range([0, iw]).padding(0.4), y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).tickSizeOuter(0));
  g.selectAll('rect').data(p).join('rect').attr('x', (v, i) => x(['2', '1', '0'][i])).attr('width', x.bandwidth()).attr('y', (v) => y(v)).attr('height', (v) => ih - y(v)).attr('fill', 'var(--ink)').attr('fill-opacity', 0.75);
  g.selectAll('text.v').data(p).join('text').attr('class', 'v label-ink').attr('text-anchor', 'middle').attr('x', (v, i) => x(['2', '1', '0'][i]) + x.bandwidth() / 2).attr('y', (v) => y(v) - 5).text((v) => v.toFixed(2));
})();

// ---- The tool ----
const state = { on: [true, true, true], act: 'relu', width: 8, seedIx: 0, steps: 0, training: false, pieces: null };
let net, raf = null;
const canvas = document.getElementById('plane');
const ctx = canvas.getContext('2d');
const out = document.getElementById('readout');
const ctl = document.getElementById('controls');

const acts = () => state.on.map((o) => (o ? state.act : 'none'));
function reset() {
  stop();
  net = new Net([2, state.width, state.width, state.width, 1], acts(), { seed: SEEDS[state.seedIx % SEEDS.length], lr: LR });
  state.steps = 0;
  update();
}
const piecewise = () => state.act === 'relu' || state.on.every((o) => !o);
function update() {
  state.pieces = piecewise() ? (state.on.some((o) => o) ? countRegions(net, GRID, LO, HI) : 1) : null;
  draw(true);
}

// Render shading, piece edges, the boundary, and the points.
function draw(full) {
  const dpr = window.devicePixelRatio || 1, S = canvas.clientWidth;
  if (canvas.width !== Math.round(S * dpr)) { canvas.width = Math.round(S * dpr); canvas.height = Math.round(S * dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, S, S);
  const col = { a: cssVar('--accent'), b: cssVar('--second'), ink: cssVar('--ink'), quiet: cssVar('--quiet') };
  const n = full ? GRID : 120;
  const ca = toRGB(col.a), cb = toRGB(col.b), cq = toRGB(col.quiet);
  const off = document.createElement('canvas'); off.width = n; off.height = n;
  const octx = off.getContext('2d'), img = octx.createImageData(n, n);
  const logits = new Float64Array(n * n), pats = full && state.act === 'relu' && state.on.some((o) => o) ? new Array(n * n) : null;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = [LO + ((HI - LO) * (i + 0.5)) / n, HI - ((HI - LO) * (j + 0.5)) / n];
    const k = j * n + i;
    logits[k] = net.logit(x);
    if (pats) pats[k] = net.pattern(x);
  }
  for (let k = 0; k < n * n; k++) {
    const c = logits[k] > 0 ? ca : cb;
    img.data[4 * k] = c[0]; img.data[4 * k + 1] = c[1]; img.data[4 * k + 2] = c[2]; img.data[4 * k + 3] = 26;
  }
  if (pats) {
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const k = j * n + i;
      if ((i + 1 < n && pats[k] !== pats[k + 1]) || (j + 1 < n && pats[k] !== pats[k + n])) {
        img.data[4 * k] = cq[0]; img.data[4 * k + 1] = cq[1]; img.data[4 * k + 2] = cq[2]; img.data[4 * k + 3] = 110;
      }
    }
  }
  octx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, S, S);
  // Boundary: where the logit is zero, from a grid one cell larger on every side, so the outline
  // d3.contours closes along the grid's border falls outside the canvas.
  const m2 = 122, lg = new Float64Array(m2 * m2), step = (HI - LO) / 120;
  for (let j = 0; j < m2; j++) for (let i = 0; i < m2; i++) lg[j * m2 + i] = net.logit([LO + (i - 0.5) * step, HI - (j - 0.5) * step]);
  const cont = d3.contours().size([m2, m2]).thresholds([0])(Array.from(lg));
  const sc = S / 120;
  const proj = d3.geoTransform({ point(x, y) { this.stream.point((x - 1) * sc, (y - 1) * sc); } });
  ctx.beginPath(); d3.geoPath(proj, ctx)(cont[0]);
  ctx.strokeStyle = col.ink; ctx.lineWidth = 1.8; ctx.stroke();
  // Points
  const px = (v) => ((v - LO) / (HI - LO)) * S, py = (v) => S - ((v - LO) / (HI - LO)) * S;
  DATA.X.forEach((x, i) => {
    ctx.beginPath(); ctx.arc(px(x[0]), py(x[1]), 2.8, 0, 2 * Math.PI);
    ctx.fillStyle = DATA.Y[i] ? col.a : col.b; ctx.fill();
  });
  readout();
}

function readout() {
  const acc = Math.round(net.accuracy(DATA.X, DATA.Y) * 100);
  let s = `Training step <b>${state.steps.toLocaleString()}</b>. Accuracy on the training points: <b>${acc}%</b>.`;
  if (state.on.every((o) => !o)) {
    const { v, c } = net.collapse();
    s += ` With no activation functions, the whole network computes the sigmoid of one linear function: \\(${v[0].toFixed(2)}\\,x_1 ${v[1] < 0 ? '-' : '+'} ${Math.abs(v[1]).toFixed(2)}\\,x_2 ${c < 0 ? '-' : '+'} ${Math.abs(c).toFixed(2)}\\).`;
  } else if (state.pieces !== null) {
    s += state.training ? ' Linear pieces: counted when training pauses.' : ` Linear pieces: <b>${state.pieces.toLocaleString()}</b>.`;
  } else {
    s += ` With ${ACTS[state.act].name} the network is not piecewise linear, so there are no pieces to count.`;
  }
  if (state.steps >= MAX_STEPS) s += ' Training stopped at 2,000 steps.';
  out.innerHTML = s;
  if (window.renderMathInElement) window.renderMathInElement(out, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
}

function loop() {
  const t0 = performance.now();
  while (performance.now() - t0 < 14 && state.steps < MAX_STEPS) { net.step(DATA.X, DATA.Y); state.steps++; }
  if (state.steps >= MAX_STEPS) { stop(); return; }
  draw(false);
  raf = requestAnimationFrame(loop);
}
function start() {
  if (state.training || state.steps >= MAX_STEPS) return;
  state.training = true; trainBtn.textContent = 'Pause'; raf = requestAnimationFrame(loop);
}
function stop() {
  const was = state.training;
  state.training = false; cancelAnimationFrame(raf);
  if (trainBtn) trainBtn.textContent = 'Train';
  if (was) update();
}

for (let l = 0; l < 3; l++) {
  segmented(ctl, { name: `Layer ${l + 1} activation function`, options: [['on', 'on'], ['off', 'off']], value: 'on',
    onChange: (k) => { state.on[l] = k === 'on'; reset(); } });
}
segmented(ctl, { name: 'Activation function, where on', options: [['relu', 'ReLU'], ['gelu', 'GELU'], ['tanh', 'tanh'], ['sigmoid', 'sigmoid']], value: 'relu',
  onChange: (k) => { state.act = k; reset(); } });
slider(ctl, { name: 'Width (units per hidden layer)', min: 2, max: 16, step: 1, value: state.width, format: (v) => String(v),
  onInput: (v) => { state.width = v; reset(); } });
const btns = document.createElement('div');
btns.className = 'seg';
ctl.append(btns);
const trainBtn = button(btns, 'Train', () => (state.training ? stop() : start()), 'primary');
button(btns, 'New start', () => { state.seedIx++; reset(); });

let rT;
window.addEventListener('resize', () => { clearTimeout(rT); rT = setTimeout(() => draw(!state.training), 150); });
reset();
