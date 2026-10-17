// Page 5.3: through the threshold.
import { chart } from '../lib/chart.js';
import { slider, segmented, button } from '../lib/ui.js';
import { makeData, makeFeatures, fit, truth, P_GRID, N_TRAIN } from '../lib/s45-features.js';

const d3 = window.d3;
const XS = d3.range(-1, 1.0001, 0.005);
const RIDGES = ['0', '0.00001', '0.0001', '0.001', '0.01', '0.1'];
const st = { p: 10, noise: 0.3, ridge: 0, seed: 1 };
let data = makeData(st.noise), F = makeFeatures(data.X, st.seed), curve = [];
const clampE = (v) => Math.min(1e3, Math.max(1e-4, v));

// ---- Static figure: geometric mean of test error over 8 draws, computed in small chunks ----
(function figure() {
  const box = document.createElement('div');
  document.getElementById('fig-dd').append(box);
  const c = chart(box, { width: 560, height: 300, margin: { top: 30, right: 70, bottom: 52, left: 64 },
    x: { domain: [1, 1000], log: true, ticks: 4, format: '~g', label: 'number of features p' }, y: { domain: [0.01, 10], log: true, ticks: 3, format: '~g', label: 'test error' },
    title: 'Test error against the number of features, averaged over 8 draws' });
  c.vline('n', N_TRAIN, { color: 'quiet', label: 'p = 40' });
  const d = makeData(0.3), acc = P_GRID.map(() => 0);
  let s = 1;
  const step = () => {
    const Fs = makeFeatures(d.X, s);
    P_GRID.forEach((p, i) => { acc[i] += Math.log(fit(d, Fs, p, 0).test) / 8; });
    if (++s <= 8) setTimeout(step, 30);
    else c.line('t', P_GRID.map((p, i) => [p, Math.exp(acc[i])]), { color: 'second', label: 'test' });
  };
  setTimeout(step, 400);
})();

// ---- The tool ----
const ctl = document.getElementById('controls'), out = document.getElementById('readout');
const pS = slider(ctl, { name: 'Number of features p', min: 0, max: 3, step: 0.005, value: Math.log10(st.p),
  format: (v) => String(Math.round(10 ** v)), onInput: (v) => { st.p = Math.round(10 ** v); drawFit(); } });
const quick = document.createElement('div'); quick.className = 'seg quick'; ctl.append(quick);
for (const p of [10, 40, 400]) button(quick, `p = ${p}`, () => { st.p = p; pS.set(Math.log10(p)); drawFit(); });
slider(ctl, { name: 'Label noise (standard deviation)', min: 0, max: 0.6, step: 0.05, value: st.noise, format: (v) => v.toFixed(2),
  onInput: (v) => { st.noise = v; data = makeData(v); schedule(); } });
segmented(ctl, { name: 'Ridge strength', options: RIDGES.map((r) => [r, r]), value: '0',
  onChange: (k) => { st.ridge = Number(k); schedule(); } });
button(ctl, 'New features', () => { st.seed++; F = makeFeatures(data.X, st.seed); schedule(); });

const fc = chart(document.getElementById('fit-chart'), { height: 300, margin: { top: 34, right: 70, bottom: 52, left: 52 },
  x: { domain: [-1, 1], label: 'x' }, y: { domain: [-2.5, 2.5], label: 'y' }, title: 'Training points, true curve, and fitted curve' });
const ec = chart(document.getElementById('err-chart'), { height: 300, margin: { top: 34, right: 70, bottom: 52, left: 64 },
  x: { domain: [1, 1000], log: true, ticks: 4, format: '~g', label: 'number of features p' }, y: { domain: [1e-4, 1e3], log: true, ticks: 4, format: '~g', label: 'error' },
  title: 'Training and test error against the number of features' });
ec.vline('n', N_TRAIN, { color: 'quiet', label: 'p = 40', labelY: 0 });
const dots = fc.g.append('g');

function drawCurve() {
  curve = P_GRID.map((p) => { const r = fit(data, F, p, st.ridge); return { p, train: r.train, test: r.test }; });
  ec.line('train', curve.map((v) => [v.p, clampE(v.train)]), { color: 'ink', label: 'training', labelAt: 0.15, dy: 18 });
  ec.line('test', curve.map((v) => [v.p, clampE(v.test)]), { color: 'second', label: 'test' });
  drawFit();
}

function drawFit() {
  const r = fit(data, F, st.p, st.ridge);
  dots.selectAll('circle').data(data.X).join('circle').attr('r', 3.5).attr('fill', 'var(--ink)')
    .attr('cx', (x) => fc.x(x)).attr('cy', (x, i) => fc.y(data.Y[i]));
  fc.line('true', XS.map((x) => [x, truth(x)]), { color: 'quiet', dash: '4 3', label: 'true', labelAt: 0.1, dy: 18 });
  fc.line('fit', XS.map((x) => [x, Math.max(-4, Math.min(4, r.f(x)))]), { color: 'accent', width: 1.8, label: 'fit', labelAt: 0.9 });
  ec.vline('p', st.p, { color: 'accent', dash: null });
  ec.point('pt', st.p, clampE(r.test), { color: 'accent', r: 4.5, label: null });
  const first = curve.find((v) => v.train < 1e-6);
  out.innerHTML = `With <span class="c-accent">\\(p\\) = <b>${st.p}</b></span> features${st.ridge ? ` and ridge ${st.ridge}` : ''}: training error <b>${r.train < 1e-6 ? '0.000' : r.train.toFixed(3)}</b>, ` +
    `<span class="c-second">test error <b>${r.test.toFixed(3)}</b></span>.` +
    (st.ridge === 0 && first ? ` With these features the model first fits every training point exactly at \\(p\\) = ${first.p}.` : '') +
    ` Feature draw ${st.seed}.`;
  if (window.renderMathInElement) window.renderMathInElement(out, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
}

let t = null;
function schedule() { clearTimeout(t); out.textContent = 'Fitting…'; t = setTimeout(drawCurve, 60); }
drawCurve();
