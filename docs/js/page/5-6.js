// Page 5.6: make emergence appear.
import { chart } from '../lib/chart.js';
import { slider, segmented, button } from '../lib/ui.js';
import { SIZES, perDigit, metric, sizeAt, riseWidth } from '../lib/s45-emergence.js';

const d3 = window.d3;
const st = { alpha: 0.5, k: 1, kind: 'exact', log: true };
const METRICS = [['exact', 'Exact match'], ['digit', 'Per-digit accuracy'], ['loglik', 'Log-likelihood']];
// On the linear axis, model sizes are plotted in units of 10^12.
const X = (N) => (st.log ? N : N / 1e12);
const sci = (n) => { const e = Math.floor(Math.log10(n)); return `${(n / 10 ** e).toFixed(1)} \u00d7 10${String(e).split('').map((d) => SUP[d]).join('')}`; };
const fine = (lo, hi) => d3.range(0, 201).map((i) => 10 ** (lo + ((hi - lo) * i) / 200));
const NS = fine(7, 12);
// Tick labels as powers of ten, 10⁷ and so on; other ticks are left unlabeled.
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const pow10 = (n) => { const e = Math.round(Math.log10(n)); return Math.abs(Math.log10(n) - e) < 1e-6 ? '10' + String(e).split('').map((d) => SUP[d]).join('') : ''; };

// ---- Static figure: per-digit and exact match (k = 10) ----
(function figure() {
  const fig = document.getElementById('fig-em');
  for (const [kind, title] of [['digit', 'Per-digit accuracy'], ['exact', 'Exact match, 10 digits']]) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${title}</p>`;
    fig.append(box);
    const c = chart(box, { width: 340, height: 230, margin: { top: 14, right: 30, bottom: 52, left: 46 },
      x: { domain: [1e7, 1e12], log: true, ticks: 3, format: pow10, label: 'parameters' }, y: { domain: [0, 1], ticks: 2 }, title });
    c.line('l', NS.map((N) => [N, metric(kind, N, 0.5, 10)]), { color: 'ink', width: 1 });
    SIZES.forEach((N, i) => c.point(`p${i}`, N, metric(kind, N, 0.5, 10), { color: 'accent', r: 3.5, label: null }));
  }
})();

// ---- The tool ----
const ctl = document.getElementById('controls'), out = document.getElementById('readout');
segmented(ctl, { name: 'Metric', options: METRICS, value: st.kind, onChange: (k) => { st.kind = k; build(); } });
const kS = slider(ctl, { name: 'Answer length k (digits)', min: 1, max: 100, step: 1, value: st.k, format: String, onInput: (v) => { st.k = v; draw(); } });
const q = document.createElement('div'); q.className = 'seg'; ctl.append(q);
for (const k of [1, 10, 20, 100]) button(q, `k = ${k}`, () => { st.k = k; kS.set(k); draw(); });
slider(ctl, { name: 'Steepness α', min: 0.2, max: 1.5, step: 0.05, value: st.alpha, format: (v) => v.toFixed(2), onInput: (v) => { st.alpha = v; draw(); } });
segmented(ctl, { name: 'Model size axis', options: [['log', 'Logarithmic'], ['lin', 'Linear']], value: 'log', onChange: (k) => { st.log = k === 'log'; build(); } });

let left, right;
const xOpt = () => (st.log ? { domain: [1e7, 1e12], log: true, ticks: 5, format: pow10, label: 'parameters N' } : { domain: [0, 1], ticks: 4, label: 'parameters N, in units of 10\u00b9\u00b2' });
function build() {
  const a = document.getElementById('digit-chart'), b = document.getElementById('metric-chart');
  a.innerHTML = ''; b.innerHTML = '';
  left = chart(a, { height: 260, margin: { top: 34, right: 40, bottom: 52, left: 52 }, x: xOpt(), y: { domain: [0, 1], ticks: 2, label: 'per-digit accuracy p' }, title: 'Per-digit accuracy against model size' });
  const ylab = METRICS.find((m) => m[0] === st.kind)[1].toLowerCase();
  right = chart(b, { height: 260, margin: { top: 34, right: 40, bottom: 52, left: 52 }, x: xOpt(), y: { domain: [0, 1], ticks: 2, label: ylab }, title: `${ylab} against model size` });
  draw();
}
function draw() {
  const { alpha, k, kind } = st;
  left.line('l', NS.map((N) => [X(N), perDigit(N, alpha)]), { color: 'ink', width: 1 });
  SIZES.forEach((N, i) => left.point(`p${i}`, X(N), perDigit(N, alpha), { color: 'ink', r: 3.5, label: null }));
  const vals = SIZES.map((N) => metric(kind, N, alpha, k));
  if (kind === 'loglik') right.setY([Math.min(-1, Math.floor(Math.min(...NS.map((N) => metric(kind, N, alpha, k))))), 0]);
  else right.setY([0, 1]);
  right.line('l', NS.map((N) => [X(N), metric(kind, N, alpha, k)]), { color: 'accent', width: 1 });
  SIZES.forEach((N, i) => right.point(`p${i}`, X(N), vals[i], { color: 'accent', r: 3.5, label: null }));
  const f = (v) => (kind === 'loglik' ? v.toFixed(2) : v.toFixed(3));
  let txt = `Answer length <b>${k}</b>, steepness ${alpha.toFixed(2)}. ${METRICS.find((m) => m[0] === kind)[1]} at the eleven models: ${vals.map(f).join(', ')}.`;
  if (kind === 'exact') {
    const lo = sizeAt(0.1, alpha, k), hi = sizeAt(0.9, alpha, k);
    txt += ` Exact match passes 10% at ${sci(lo)} parameters and 90% at ${sci(hi)}, a span of ${riseWidth(alpha).toFixed(2)} powers of ten.`;
  }
  out.innerHTML = txt;
  if (window.renderMathInElement) window.renderMathInElement(out, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
}
build();
