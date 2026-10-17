// Page 2.1: fit by hand.
import { MODELS, loss } from '../lib/s2-fit.js';
import { chart } from '../lib/chart.js';
import { slider, segmented } from '../lib/ui.js';
import { bands } from '../lib/s2-contour.js';

const d3 = window.d3;

// ---- Static figure: the box with knobs ----
(function figure() {
  // Drawn at its true width (at most 300 pixels) so its labels stay at the text size.
  const W = Math.max(260, Math.min(300, document.getElementById('fig-box').clientWidth || 300)), H = 170;
  const svg = d3.select('#fig-box').append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .attr('role', 'img').attr('aria-label', 'A box labeled f of x and theta, with an input arrow x, an output arrow y, and four knobs on top labeled theta');
  const quiet = 'var(--quiet)', ink = 'var(--ink)';
  svg.append('defs').append('marker').attr('id', 'arr21').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
    .attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto').append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', quiet);
  const bx = 80, by = 70, bw = 140, bh = 70;
  svg.append('rect').attr('x', bx).attr('y', by).attr('width', bw).attr('height', bh).attr('fill', 'none').attr('stroke', ink);
  svg.append('text').attr('x', bx + bw / 2).attr('y', by + bh / 2 + 7).attr('text-anchor', 'middle').attr('class', 'label-ink').text('f(x; θ)');
  svg.append('line').attr('x1', 22).attr('y1', by + bh / 2).attr('x2', bx - 4).attr('y2', by + bh / 2).attr('stroke', quiet).attr('marker-end', 'url(#arr21)');
  svg.append('text').attr('x', 14).attr('y', by + bh / 2 + 7).attr('text-anchor', 'middle').attr('class', 'label-ink').text('x');
  svg.append('line').attr('x1', bx + bw).attr('y1', by + bh / 2).attr('x2', W - 24).attr('y2', by + bh / 2).attr('stroke', quiet).attr('marker-end', 'url(#arr21)');
  svg.append('text').attr('x', W - 12).attr('y', by + bh / 2 + 7).attr('text-anchor', 'middle').attr('class', 'label-ink').text('y');
  const angles = [-40, 25, 70, -10];
  angles.forEach((ang, i) => {
    const cx = bx + 22 + i * 32, cy = by - 16;
    svg.append('line').attr('x1', cx).attr('y1', cy + 11).attr('x2', cx).attr('y2', by).attr('stroke', quiet);
    svg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', 11).attr('fill', 'none').attr('stroke', ink);
    const r = (ang * Math.PI) / 180;
    svg.append('line').attr('x1', cx).attr('y1', cy).attr('x2', cx + 9 * Math.sin(r)).attr('y2', cy - 9 * Math.cos(r)).attr('stroke', ink).attr('stroke-width', 1.6);
  });
  svg.append('text').attr('x', bx + bw / 2).attr('y', 22).attr('text-anchor', 'middle').text('θ');
})();

// ---- The tool ----
const state = { model: 'line', p: {} };
for (const [k, m] of Object.entries(MODELS)) state.p[k] = Object.fromEntries(m.params.map((q) => [q.k, q.start]));
const ctl = document.getElementById('controls');
const out = document.getElementById('readout');
let sliders = {}, dataC, mapC, mapKey = '';
const sliderBox = document.createElement('div');
sliderBox.style.display = 'contents';

segmented(ctl, { name: 'Model', options: [['line', 'Line: y = a x + b'], ['wave', 'Wave: y = a sin(b x) + c']], value: state.model,
  onChange: (k) => { state.model = k; build(); } });
ctl.append(sliderBox);

function build() {
  const m = MODELS[state.model], p = state.p[state.model];
  sliderBox.innerHTML = '';
  sliders = {};
  for (const q of m.params) {
    sliders[q.k] = slider(sliderBox, { name: q.name, min: q.min, max: q.max, step: q.step, value: p[q.k], format: (v) => v.toFixed(q.step < 0.01 ? 3 : 2),
      onInput: (v) => { p[q.k] = v; draw(); } });
  }
  for (const id of ['data-chart', 'map-chart']) document.getElementById(id).innerHTML = '';
  // The stages start empty, so their width is read from the tool; charts are drawn at their true width.
  const avail = document.getElementById('tool').clientWidth || 420;
  dataC = chart(document.getElementById('data-chart'), { width: Math.min(420, avail), height: 300, margin: { top: 34, right: 20, bottom: 50, left: 46 },
    x: { domain: m.xDomain, label: 'x' }, y: { domain: m.yDomain, label: 'y' }, title: 'The data and the model' });
  dataC.res = dataC.g.insert('g', ':first-child');
  dataC.pts = dataC.g.append('g');
  dataC.pts.selectAll('circle').data(m.data.X).join('circle').attr('cx', (x) => dataC.x(x)).attr('cy', (x, i) => dataC.y(m.data.Y[i])).attr('r', 3.5).attr('fill', 'var(--ink)');
  const [kx, ky] = m.map;
  const qx = m.params.find((q) => q.k === kx), qy = m.params.find((q) => q.k === ky);
  mapC = chart(document.getElementById('map-chart'), { width: Math.min(340, avail), height: 300, margin: { top: 34, right: 20, bottom: 50, left: 46 },
    x: { domain: [qx.min, qx.max], label: qx.name }, y: { domain: [qy.min, qy.max], label: qy.name }, title: 'Map of the loss over two parameters' });
  mapC.bands = mapC.g.insert('g', ':first-child');
  const setFromPointer = (ev) => {
    const [mx, my] = d3.pointer(ev, mapC.g.node());
    const vx = Math.max(qx.min, Math.min(qx.max, mapC.x.invert(mx))), vy = Math.max(qy.min, Math.min(qy.max, mapC.y.invert(my)));
    p[kx] = Math.round(vx / qx.step) * qx.step; p[ky] = Math.round(vy / qy.step) * qy.step;
    sliders[kx].set(p[kx]); sliders[ky].set(p[ky]);
    draw();
  };
  mapC.svg.style('cursor', 'crosshair').on('pointerdown', (ev) => { setFromPointer(ev); mapC.svg.on('pointermove', setFromPointer); })
    .on('pointerup pointerleave', () => mapC.svg.on('pointermove', null));
  mapKey = '';
  draw();
}

// Banded map of the loss: each band is a region below a loss level, drawn with low opacity, so lower ground is darker.
function drawMap(m, p) {
  const [kx, ky] = m.map;
  const others = m.params.filter((q) => q.k !== kx && q.k !== ky).map((q) => `${q.k}=${p[q.k]}`).join(',');
  const key = `${state.model}:${others}`;
  if (key === mapKey) return;
  mapKey = key;
  const qx = m.params.find((q) => q.k === kx), qy = m.params.find((q) => q.k === ky);
  const q = { ...p };
  bands(mapC.bands, {
    f: (x, y) => { q[kx] = x; q[ky] = y; return -Math.log(loss(m, q)); },
    xd: [qx.min, qx.max], yd: [qy.min, qy.max], iw: mapC.iw, ih: mapC.ih, n: 120,
    // The line's loss rises steadily away from its bowl, so its bands reach up to 30; the wave's bands stop at 2.5 so its
    // separate low points stand out against the higher ground.
    thresholds: d3.range(-Math.log(state.model === 'line' ? 30 : 2.5), -Math.log(0.06), Math.log(1.15)), opacity: state.model === 'line' ? 0.03 : 0.045,
  });
}

function draw() {
  const m = MODELS[state.model], p = state.p[state.model];
  const xs = d3.range(m.xDomain[0], m.xDomain[1] + 1e-9, (m.xDomain[1] - m.xDomain[0]) / 200);
  dataC.line('model', xs.map((x) => [x, m.f(p, x)]), { color: 'accent', width: 2 });
  dataC.res.selectAll('line').data(m.data.X).join('line').attr('x1', (x) => dataC.x(x)).attr('x2', (x) => dataC.x(x))
    .attr('y1', (x, i) => dataC.y(m.data.Y[i])).attr('y2', (x) => dataC.y(Math.max(m.yDomain[0], Math.min(m.yDomain[1], m.f(p, x)))))
    .attr('stroke', 'var(--quiet)').attr('stroke-width', 1);
  drawMap(m, p);
  const [kx, ky] = m.map;
  mapC.point('you', p[kx], p[ky], { color: 'accent', r: 5, label: 'you', dx: 8, dy: -8 });
  const L = loss(m, p);
  const vals = m.params.map((q) => `${q.k} = ${p[q.k].toFixed(q.step < 0.01 ? 3 : 2)}`).join(', ');
  out.innerHTML = `Model ${m.formula} with ${vals}. Loss (mean squared error over the 20 points): <b>${L.toFixed(3)}</b>.` +
    (state.model === 'wave' ? ` The map shows \\(a\\) and \\(b\\) with \\(c\\) held at ${p.c.toFixed(2)}.` : '');
  if (window.renderMathInElement) window.renderMathInElement(out, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
}

build();
let rT;
window.addEventListener('resize', () => { clearTimeout(rT); rT = setTimeout(build, 200); });
