// Page 3.8: spend a compute budget, with the Chinchilla loss fit.
import { chart } from '../lib/chart.js';
import { slider, segmented, button } from '../lib/ui.js';
import { loss, tokensFor, optimum, optimumNumeric, FIT } from '../lib/3-models.js';

const d3 = window.d3;
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const pow10 = (e) => '10' + String(e).split('').map((c) => SUP[c]).join('');
const sci = (x) => { const e = Math.floor(Math.log10(x) + 1e-9), m = x / 10 ** e; return `${m.toFixed(2)} × ${pow10(e)}`; };
const big = (x) => (x >= 1e12 ? `${x >= 1e14 ? Math.round(x / 1e12).toLocaleString('en-US') : (x / 1e12).toFixed(2)} trillion` : x >= 1e9 ? `${(x / 1e9).toFixed(1)} billion` : x >= 1e6 ? `${(x / 1e6).toFixed(1)} million` : x >= 1e3 ? `${(x / 1e3).toFixed(1)} thousand` : x.toFixed(0));
const isPow = (v) => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-6;
const logTick = (v) => (isPow(v) ? String(Math.round(Math.log10(v))) : '');
// Rewrites tick labels "18" as 10 with a raised 18, at the same type size.
const raise = (c) => c.svg.selectAll('.axis text').each(function () { const t = this.textContent; if (/^\d+$/.test(t)) this.innerHTML = `10<tspan dy="-7">${t}</tspan>`; });
const sizeTick = (v) => (isPow(v) ? ({ 7: '10M', 8: '100M', 9: '1B', 10: '10B', 11: '100B', 12: '1T', 13: '10T' })[Math.round(Math.log10(v))] : '');
const C_CHIN = 5.76e23;
const NMIN = 1e7, NMAX = 1e13;

// ---- Static figure: loss against compute for five model sizes, with the frontier ----
(function figure() {
  const fig = document.getElementById('fig-frontier');
  const thin = (fig.clientWidth || 640) < 480;
  const fmt = thin ? (v) => (isPow(v) && (Math.round(Math.log10(v)) - 18) % 4 === 0 ? logTick(v) : '') : logTick;
  const c = chart(fig, { width: 640, height: 340, margin: { top: 34, right: 70, bottom: 52, left: 52 },
    x: { domain: [1e18, 1e26], log: true, ticks: 8, format: fmt, label: thin ? 'compute (FLOPs)' : 'training compute (FLOPs)' }, y: { domain: [1.5, 4], ticks: 4, label: 'predicted loss' },
    title: 'Predicted loss against compute for five model sizes' });
  const Cs = d3.range(18, 26.001, 0.02).map((e) => 10 ** e);
  const sizes = [[1e8, '100M'], [1e9, '1B'], [1e10, '10B'], [1e11, '100B'], [1e12, '1T']];
  sizes.forEach(([N, lab]) => c.line(`n${N}`, Cs.map((C) => [C, loss(N, tokensFor(C, N))]), { color: 'quiet', width: 1.2, label: lab }));
  c.spreadLabels(sizes.map(([N]) => `n${N}`), 17);
  c.line('frontier', Cs.map((C) => { const o = optimum(C); return [C, loss(o.N, o.D)]; }), { color: 'accent', width: 2.2 });
  c.line('floor', [[1e18, FIT.E], [1e26, FIT.E]], { color: 'quiet', dash: '3 3' });
  c.text('floor-l', 1.3e18, FIT.E + 0.06, 'E, the floor');
  raise(c);
})();

// ---- The tool ----
const state = { lc: Math.log10(C_CHIN), ln: Math.log10(70e9), serve: false, ls: 13 };
const ctl = document.getElementById('controls');
const out = document.getElementById('readout');
const cS = slider(ctl, { name: 'Budget C, FLOPs', min: 18, max: 26, step: 0.0001, value: state.lc, format: (v) => sci(10 ** v),
  onInput: (v) => { state.lc = v; draw(); } });
const r1 = document.createElement('div'); r1.className = 'btns'; ctl.append(r1);
button(r1, 'Chinchilla’s budget', () => { state.lc = Math.log10(C_CHIN); cS.set(state.lc); draw(); });
button(r1, 'Budget × 10', () => { state.lc = Math.min(26, state.lc + 1); cS.set(state.lc); draw(); });
const nS = slider(ctl, { name: 'Model size N', min: 7, max: 13, step: 0.001, value: state.ln, format: (v) => big(10 ** v),
  onInput: (v) => { state.ln = v; draw(); } });
const r2 = document.createElement('div'); r2.className = 'btns'; ctl.append(r2);
const setN = (N) => { state.ln = Math.log10(N); nS.set(state.ln); draw(); };
button(r2, '70B', () => setN(70e9));
button(r2, '280B', () => setN(280e9));
button(r2, 'Best size', () => setN(best().N));
segmented(ctl, { name: 'Count the cost of serving', options: [['off', 'Off'], ['on', 'On']], value: 'off',
  onChange: (k) => { state.serve = k === 'on'; sS.el.disabled = !state.serve; draw(); } });
const sS = slider(ctl, { name: 'Tokens served', min: 10, max: 14, step: 0.01, value: state.ls, format: (v) => big(10 ** v),
  onInput: (v) => { state.ls = v; draw(); } });
sS.el.disabled = true;

const S = () => (state.serve ? 10 ** state.ls : 0);
const best = () => optimumNumeric(10 ** state.lc, S(), FIT, 6, 14, 0.002);

const stage = document.getElementById('chart');
const narrow = (stage.clientWidth || 560) < 440;
const sizeTickNarrow = (v) => (isPow(v) && Math.round(Math.log10(v)) % 2 === 1 ? sizeTick(v) : '');
const c = chart(stage, {
  x: { domain: [NMIN, NMAX], log: true, ticks: 6, format: narrow ? sizeTickNarrow : sizeTick, label: narrow ? 'parameters' : 'model size N (parameters)' }, y: { domain: [1.8, 2.4], ticks: 4, label: 'predicted loss' },
  margin: { top: 34, right: narrow ? 24 : 110, bottom: 52, left: 56 }, title: 'Predicted loss against model size at the chosen budget',
});
const curve = (C, s) => d3.range(7, 13.0001, 0.01).map((e) => { const N = 10 ** e, D = tokensFor(C, N, s); return [N, D > 0 ? loss(N, D) : Infinity]; });

// Drag on the chart to choose N.
const fromPointer = (ev) => {
  const [mx] = d3.pointer(ev, c.g.node());
  state.ln = Math.max(7, Math.min(13, Math.log10(c.x.invert(Math.max(0, Math.min(c.iw, mx))))));
  nS.set(state.ln); draw();
};
c.svg.on('pointerdown', (ev) => { fromPointer(ev); c.svg.on('pointermove', fromPointer); })
  .on('pointerup pointerleave', () => c.svg.on('pointermove', null));

function draw() {
  const C = 10 ** state.lc, s = S(), N = 10 ** state.ln, D = tokensFor(C, N, s);
  const b = best(), b0 = optimumNumeric(C, 0, FIT, 6, 14, 0.002);
  const feasible = Number.isFinite(b.L);
  const lo = feasible ? Math.min(b.L, b0.L) : b0.L;
  c.setY([Math.floor((lo - 0.04) * 20) / 20, Math.floor((lo - 0.04) * 20) / 20 + 0.6]);
  c.line('train', curve(C, 0), { color: 'ink', label: state.serve && !narrow ? 'training only' : null, labelAt: 0.97 });
  if (!state.serve || narrow) c.remove('train:label');
  if (state.serve) {
    const pts = curve(C, s);
    const visible = pts.some((p) => Number.isFinite(p[1]) && p[1] <= c.y.domain()[1]);
    c.line('serve', pts, { color: 'second', label: narrow || !visible ? null : 'with serving', labelAt: 'end' });
    if (narrow || !visible) c.remove('serve:label');
  }
  else c.remove('serve');
  c.spreadLabels(['train', 'serve'], 18);
  if (feasible) c.point('best', b.N, b.L, { color: 'quiet', r: 4, label: 'lowest', dx: 0, dy: 22, anchor: 'middle' });
  else c.remove('best');
  const L = D > 0 ? loss(N, D) : Infinity;
  c.point('me', N, L, { color: 'accent', r: 6, label: null });
  const ratio = D / N;
  out.innerHTML = (D > 0
    ? `<span class="c-accent">\\(N\\) = ${big(N)} parameters</span>, trained on ${big(D)} tokens (${ratio >= 10 ? ratio.toFixed(0) : ratio.toPrecision(2)} per parameter): predicted loss <b>${L.toFixed(3)}</b>. `
    : `<span class="c-accent">\\(N\\) = ${big(N)} parameters</span>: serving alone uses up the budget, so no tokens are left for training. `) +
    (feasible ? `Lowest at this budget${state.serve ? ', counting serving' : ''}: ${b.L.toFixed(3)}, at ${big(b.N)} parameters and ${big(b.D)} tokens (${(b.D / b.N).toFixed(0)} per parameter).`
      : 'This budget cannot serve that many tokens with any model of a million parameters or more.');
  if (window.renderMathInElement) window.renderMathInElement(out, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
}
draw();
window.addEventListener('load', draw);
