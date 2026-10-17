// Page 1.3: where did the gradient go?
import { DeepNet } from '../lib/deepnet.js';
import { datasets } from '../lib/mlp.js';
import { slider, segmented, button } from '../lib/ui.js';

const d3 = window.d3;
const DATA = datasets.circles();
const WIDTH = 16;
const state = { act: 'sigmoid', depth: 10, gain: 1, seed: 1 };

function sci(v, html = true) {
  if (!Number.isFinite(v) || v === 0) return '0';
  const k = Math.floor(Math.log10(v));
  if (k >= -2 && k <= 3) return v.toPrecision(2);
  const e = String(k).replace('-', '\u2212');
  return html ? `${(v / 10 ** k).toFixed(1)} \u00d7 10<span class="exp">${e}</span>` : `${(v / 10 ** k).toFixed(1)} times 10 to the power ${e}`;
}
// Writes m × 10^k into an SVG text element, with the exponent raised (same type size).
function setPow(t, mant, k) {
  t.text(null);
  const lead = Math.abs(mant - 1) < 1e-9 ? '10' : `${mant.toFixed(1)} \u00d7 10`;
  t.append('tspan').text(lead);
  t.append('tspan').attr('dy', '-0.45em').text(String(k).replace('-', '\u2212'));
  t.append('tspan').attr('dy', '0.45em').text('\u200a');
}
function times(r) {
  if (r >= 1e6) return `${(r / 1e6).toPrecision(2)} million`;
  if (r >= 1e4) return Math.round(r / 1000).toLocaleString() + ',000';
  if (r >= 10) return Math.round(r).toLocaleString();
  return r.toFixed(1);
}

// ---- Static figure: a small network with two hidden layers ----
(function figure() {
  const box = document.getElementById('fig-net');
  const W = 300, H = 190;
  const svg = d3.select(box).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H).attr('aria-hidden', 'true');
  const layers = [2, 4, 4, 1];
  const xs = layers.map((_, i) => 30 + i * 80);
  const pos = layers.map((n, i) => d3.range(n).map((k) => [xs[i], H / 2 + (k - (n - 1) / 2) * 38]));
  for (let i = 0; i < layers.length - 1; i++) {
    for (const a of pos[i]) for (const b of pos[i + 1]) {
      svg.append('line').attr('x1', a[0]).attr('y1', a[1]).attr('x2', b[0]).attr('y2', b[1]).attr('stroke', 'var(--quiet)').attr('stroke-width', 0.8);
    }
  }
  pos.flat().forEach(([x, y]) => svg.append('circle').attr('cx', x).attr('cy', y).attr('r', 7).attr('fill', 'var(--paper)').attr('stroke', 'var(--ink)'));
})();

// ---- The tool ----
const ctl = document.getElementById('controls');
const out = document.getElementById('readout');
segmented(ctl, { name: 'Activation function', options: [['sigmoid', 'Sigmoid'], ['tanh', 'tanh'], ['relu', 'ReLU']], value: state.act,
  onChange: (k) => { state.act = k; draw(); } });
slider(ctl, { name: 'Hidden layers', min: 2, max: 20, step: 1, value: state.depth, format: (v) => String(v),
  onInput: (v) => { state.depth = v; draw(); } });
slider(ctl, { name: 'Initial weight scale g', min: 0.25, max: 4, step: 0.05, value: state.gain, format: (v) => v.toFixed(2),
  onInput: (v) => { state.gain = v; draw(); } });
const note = document.createElement('p');
note.className = 'ctl-name';
note.style.margin = '0';
note.innerHTML = 'Each weight starts as a random draw with standard deviation <i>g</i>/\u221a<i>n</i>, where <i>n</i> is the number of inputs to its layer.';
ctl.append(note);
button(ctl, 'New random start', () => { state.seed += 1; draw(); });

const stage = document.getElementById('chart');
let svg, g, iw, ih, xAxis, yAxis, bars, valLabels;
const m = { top: 46, right: 12, bottom: 52, left: 64 };
function build() {
  stage.innerHTML = '';
  const W = Math.max(280, Math.min(600, stage.clientWidth || 560)), H = 340;
  iw = W - m.left - m.right; ih = H - m.top - m.bottom;
  svg = d3.select(stage).append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .attr('role', 'img');
  g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  xAxis = g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`);
  yAxis = g.append('g').attr('class', 'axis');
  g.append('text').attr('x', -m.left + 4).attr('y', -28).text('gradient size');
  g.append('text').attr('x', iw).attr('y', ih + 46).attr('text-anchor', 'end').text('layer');
  bars = g.append('g');
  valLabels = g.append('g');
}

function draw() {
  if (!svg) build();
  const net = new DeepNet({ depth: state.depth, width: WIDTH, act: state.act, gain: state.gain, seed: state.seed });
  const { rms } = net.gradients(DATA.X, DATA.Y);
  const L = state.depth;
  const names = d3.range(1, L + 1).map(String).concat(['out']);
  const vals = rms.map((v) => Math.max(v, 1e-300));
  // y domain: whole powers of ten around the data.
  const lo = Math.floor(Math.log10(d3.min(vals))) - 1, hi = Math.ceil(Math.log10(d3.max(vals))) + 1;
  const y = d3.scaleLog().domain([10 ** lo, 10 ** hi]).range([ih, 0]);
  const x = d3.scaleBand().domain(names).range([0, iw]).padding(0.25);
  const span = hi - lo;
  const stepT = span > 16 ? 4 : span > 8 ? 2 : 1;
  const ticks = d3.range(lo, hi + 1).filter((k) => (k - lo) % stepT === 0).map((k) => 10 ** k);
  yAxis.call(d3.axisLeft(y).tickValues(ticks).tickFormat(() => '').tickSizeOuter(0));
  yAxis.selectAll('.tick text').each(function (v) { setPow(d3.select(this), 1, Math.round(Math.log10(v))); });
  const every = iw / names.length < 26 ? 5 : 1;
  const narrow = x.bandwidth() / 0.75 < 32;
  const shown = names.filter((n, i) => (n === 'out' && !narrow) || i === 0 || i === L - 1 || ((i + 1) % every === 0 && (every === 1 || L - (i + 1) > 2)));
  xAxis.call(d3.axisBottom(x).tickValues(shown).tickSizeOuter(0));
  bars.selectAll('rect').data(vals).join('rect')
    .attr('x', (_, i) => x(names[i])).attr('width', x.bandwidth())
    .attr('y', (v) => y(v)).attr('height', (v) => ih - y(v))
    .attr('fill', (_, i) => (i === 0 || i === L - 1 ? 'var(--accent)' : 'var(--quiet)')).attr('fill-opacity', (_, i) => (i === 0 || i === L - 1 ? 0.9 : 0.45));
  const lab = [[0, `layer 1`], [L - 1, `layer ${L}`]];
  valLabels.selectAll('text').data(lab).join('text').attr('class', 'label-accent')
    .attr('x', ([i]) => x(names[i]) + x.bandwidth() / 2)
    .attr('y', ([i]) => Math.max(14, y(vals[i]) - 8))
    .attr('text-anchor', ([i]) => (i === 0 ? 'start' : 'end'))
    .each(function ([i]) { const k = Math.floor(Math.log10(vals[i])); setPow(d3.select(this), vals[i] / 10 ** k, k); })
    // Raise each value label above every bar it spans, so it never sits on top of a bar.
    .each(function ([i]) {
      const len = this.getComputedTextLength();
      const cx = x(names[i]) + x.bandwidth() / 2;
      const a = i === 0 ? cx : cx - len, b = i === 0 ? cx + len : cx;
      const tops = names.map((n, j) => [x(n), x(n) + x.bandwidth(), y(vals[j])]).filter(([u, v]) => v > a && u < b).map((t) => t[2]);
      d3.select(this).attr('y', Math.max(14, Math.min(...tops) - 8));
    });
  svg.attr('aria-label', `Gradient size for each of ${L} hidden layers and the output layer, log scale. Layer 1: ${sci(vals[0], false)}; layer ${L}: ${sci(vals[L - 1], false)}.`);

  const r = vals[0] / vals[L - 1];
  const actName = { sigmoid: 'sigmoid', tanh: 'tanh', relu: 'ReLU' }[state.act];
  const cmp = r < 0.5 ? `<b>${times(1 / r)} times smaller</b> than` : r > 2 ? `<b>${times(r)} times larger</b> than` : `<b>${r.toFixed(2)} times</b>`;
  out.innerHTML = `${actName}, ${L} hidden layers, scale ${state.gain.toFixed(2)}, random start ${state.seed}. The gradient at <span class="c-accent">layer 1</span> is ${sci(vals[0])}, ${cmp} the gradient at <span class="c-accent">layer ${L}</span> (${sci(vals[L - 1])}). The last bar is the output layer.`;
}

if (document.readyState === 'complete') draw(); else window.addEventListener('load', draw);
let rt;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { svg = null; draw(); }, 150); });
