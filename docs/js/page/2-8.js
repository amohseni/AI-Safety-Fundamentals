// Page 2.8: degree slider.
import { sample, fit, mse, truth } from '../lib/s2-poly.js';
import { chart } from '../lib/chart.js';
import { slider, button } from '../lib/ui.js';

const d3 = window.d3;
const LO = 1e-4, HI = 1e6;
const clampE = (v) => Math.max(LO, Math.min(HI, v));
const fmtE = (v) => (v < 1e-4 ? '0.0000' : v < 100 ? v.toFixed(v < 1 ? 3 : 2) : v < 1e6 ? Math.round(v).toLocaleString() : v.toExponential(1).replace(/e\+?(-?\d+)/, ' × 10<sup>$1</sup>'));
const xs = d3.range(-1, 1.0001, 0.004);
const clipY = (v) => Math.max(-3.4, Math.min(3.4, v));

function fitChart(el, data, d, { width = 420, small = false } = {}) {
  const c = chart(el, { width, height: small ? 200 : 300, margin: small ? { top: 14, right: 12, bottom: 34, left: 34 } : { top: 34, right: 20, bottom: 50, left: 46 },
    x: { domain: [-1, 1], ticks: small ? 3 : 5, label: small ? null : 'x' }, y: { domain: [-3, 3], ticks: small ? 3 : 5, label: small ? null : 'y' },
    title: `Polynomial fit of degree ${d}` });
  c.line('truth', xs.map((x) => [x, truth(x)]), { color: 'quiet', dash: '4 4', width: 1.2 });
  c.pts = c.g.append('g');
  c.pts.selectAll('circle').data(data.X).join('circle').attr('cx', (x) => c.x(x)).attr('cy', (x, i) => c.y(data.Y[i])).attr('r', 3.5).attr('fill', 'var(--ink)');
  return c;
}

// ---- Static figure ----
(function figure() {
  const fig = document.getElementById('fig-fits'), data = sample(12, 0.3, 1);
  for (const d of [1, 3, 15]) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">Degree ${d}</p>`;
    fig.append(box);
    const c = fitChart(box, data, d, { width: 260, small: true });
    const f = fit(data, d).f;
    c.line('fit', xs.map((x) => [x, clipY(f(x))]), { color: 'accent', width: 1.8 });
  }
})();

// ---- The tool ----
const state = { deg: 3, n: 12, noise: 0.3, seed: 1 };
let data, test, fits, errs, fc, ec;
const ctl = document.getElementById('controls');
const out = document.getElementById('readout');
slider(ctl, { name: 'Degree', min: 0, max: 30, step: 1, value: state.deg, format: String, onInput: (v) => { state.deg = v; draw(); } });
slider(ctl, { name: 'Number of points', min: 6, max: 30, step: 1, value: state.n, format: String, onInput: (v) => { state.n = v; recompute(); } });
slider(ctl, { name: 'Noise (standard deviation)', min: 0, max: 0.6, step: 0.05, value: state.noise, format: (v) => v.toFixed(2), onInput: (v) => { state.noise = v; recompute(); } });
const bw = document.createElement('div'); bw.className = 'seg'; ctl.append(bw);
button(bw, 'New points', () => { state.seed++; recompute(); });

function recompute() {
  data = sample(state.n, state.noise, state.seed);
  test = sample(2000, state.noise, 999);
  fits = [];
  errs = [];
  for (let d = 0; d <= 30; d++) { const f = fit(data, d).f; fits.push(f); errs.push([mse(f, data), mse(f, test)]); }
  build();
}

function build() {
  for (const id of ['fit-chart', 'err-chart']) document.getElementById(id).innerHTML = '';
  // The stages start empty, so their width is read from the tool; charts are drawn at their true width.
  const avail = document.getElementById('tool').clientWidth || 420;
  fc = fitChart(document.getElementById('fit-chart'), data, state.deg, { width: Math.min(420, avail) });
  const thr = state.n - 1;
  ec = chart(document.getElementById('err-chart'), { width: Math.min(420, avail), height: 300, margin: { top: 34, right: 54, bottom: 50, left: 58 },
    x: { domain: [0, 30], label: 'degree', ticks: 6 }, y: { domain: [LO, HI], log: true, ticks: 5, format: pow10, label: 'mean squared error' },
    title: 'Training and test error against degree' });
  if (thr <= 30) ec.vline('thr', thr, { color: 'quiet', label: null });
  ec.line('train', errs.map((e, d) => [d, clampE(e[0])]), { color: 'accent', label: 'train' });
  ec.line('test', errs.map((e, d) => [d, clampE(e[1])]), { color: 'second', label: 'test' });
  ec.spreadLabels(['train', 'test'], 22);
  draw();
}

const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
function pow10(d) { const e = Math.round(Math.log10(d)); return e % 2 !== 0 ? '' : e === 0 ? '1' : `10${String(e).split('').map((c) => SUP[c]).join('')}`; }

function draw() {
  const f = fits[state.deg], [tr, te] = errs[state.deg];
  fc.line('fit', xs.map((x) => [x, clipY(f(x))]), { color: 'accent', width: 2 });
  ec.vline('cur', state.deg, { color: 'ink', dash: null, label: null });
  ec.point('tr', state.deg, clampE(tr), { color: 'accent', r: 4 });
  ec.point('te', state.deg, clampE(te), { color: 'second', r: 4 });
  const best = errs.reduce((b, e, d) => (e[1] < errs[b][1] ? d : b), 0);
  const p = state.deg + 1;
  const regime = p < state.n ? `${p} coefficients for ${state.n} points: an ordinary least-squares fit` : p === state.n ? `${p} coefficients for ${state.n} points: the one polynomial through every point` : `${p} coefficients for ${state.n} points: of the many polynomials through every point, the one with the smallest coefficients`;
  out.innerHTML = `Degree <b>${state.deg}</b> (${regime}). Training error <b>${fmtE(tr)}</b>; test error <b>${fmtE(te)}</b>. Lowest test error for these points: degree ${best}, ${fmtE(errs[best][1])}. The vertical gray line marks degree ${state.n - 1}, where the number of coefficients equals the number of points.`;
}

let rT;
window.addEventListener('resize', () => { clearTimeout(rT); rT = setTimeout(build, 200); });
recompute();
