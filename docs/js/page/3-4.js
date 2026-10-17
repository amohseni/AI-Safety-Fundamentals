// Page 3.4: attention by hand, with draggable 2D queries and keys.
import { segmented, button } from '../lib/ui.js';
import { attention, mix } from '../lib/3-models.js';
import { START, LIMIT, FIGURE } from './3-4-data.js';

const d3 = window.d3;
const N = 4;
const copy = (a) => a.map((v) => v.slice());
const state = { q: copy(START.q), k: copy(START.k), v: copy(START.v), sel: 3, causal: true, scale: true, table: 'weights' };
// Labels get a paper-colored outline so lines passing under them do not cut through the letters.
const halo = (sel) => sel.attr('stroke', 'var(--paper)').attr('stroke-width', 5).attr('stroke-linejoin', 'round').attr('paint-order', 'stroke');
const SUB = (name, i) => `${name}<tspan dy="5">${i}</tspan>`;

// A square plane in an SVG drawn at its true pixel width.
function plane(container, { range = 3.4, maxW = 400 } = {}) {
  const W = Math.max(240, Math.min(maxW, container.clientWidth || maxW));
  const svg = d3.select(container).append('svg').attr('class', 'chart').attr('width', W).attr('height', W).attr('viewBox', `0 0 ${W} ${W}`);
  const s = d3.scaleLinear().domain([-range, range]).range([0, W]);
  const t = d3.scaleLinear().domain([-range, range]).range([W, 0]);
  svg.append('line').attr('x1', s(-range)).attr('x2', s(range)).attr('y1', t(0)).attr('y2', t(0)).attr('stroke', 'var(--rule)');
  svg.append('line').attr('x1', s(0)).attr('x2', s(0)).attr('y1', t(-range)).attr('y2', t(range)).attr('stroke', 'var(--rule)');
  return { svg, s, t, W, range };
}

// Draws an arrow from the origin to p; returns the group.
function arrow(g, P, p, color, label, faint = false, head = true) {
  const x0 = P.s(0), y0 = P.t(0), x1 = P.s(p[0]), y1 = P.t(p[1]);
  const ang = Math.atan2(y1 - y0, x1 - x0), len = Math.hypot(x1 - x0, y1 - y0), h = head ? Math.min(11, len * 0.5) : 0;
  g.append('line').attr('x1', x0).attr('y1', y0).attr('x2', x1 - Math.cos(ang) * h * 0.8).attr('y2', y1 - Math.sin(ang) * h * 0.8)
    .attr('stroke', color).attr('stroke-width', 2).attr('stroke-dasharray', faint ? '4 4' : null);
  const a1 = ang + Math.PI - 0.4, a2 = ang + Math.PI + 0.4;
  if (head) g.append('path').attr('d', `M${x1},${y1}L${x1 + h * Math.cos(a1)},${y1 + h * Math.sin(a1)}L${x1 + h * Math.cos(a2)},${y1 + h * Math.sin(a2)}Z`).attr('fill', color);
  // Label beyond the tip, pushed away from the origin.
  const ux = len > 1 ? (x1 - x0) / len : 1, uy = len > 1 ? (y1 - y0) / len : 0;
  g.append('text').attr('x', x1 + ux * (head ? 16 : 24)).attr('y', y1 + uy * (head ? 16 : 24) + 6).attr('text-anchor', 'middle')
    .attr('class', color === 'var(--accent)' ? 'label-accent' : faint ? '' : 'label-ink').html(label).call(halo);
}

function valuesPlane(g, P, values, w, allowed, sel) {
  const out = mix(w, values);
  values.forEach((v, j) => {
    if (allowed[j] && w[j] > 0.002) g.append('line').attr('x1', P.s(out[0])).attr('y1', P.t(out[1])).attr('x2', P.s(v[0])).attr('y2', P.t(v[1]))
      .attr('stroke', 'var(--accent)').attr('stroke-opacity', 0.55).attr('stroke-width', 0.6 + 9 * w[j]);
  });
  values.forEach((v, j) => {
    g.append('circle').attr('cx', P.s(v[0])).attr('cy', P.t(v[1])).attr('r', 5).attr('fill', allowed[j] ? 'var(--ink)' : 'var(--rule)');
    g.append('text').attr('x', P.s(v[0]) + (v[0] < 0 ? -10 : 10)).attr('y', P.t(v[1]) + (v[1] < 0 ? 22 : -10))
      .attr('text-anchor', v[0] < 0 ? 'end' : 'start').attr('class', allowed[j] ? 'label-ink' : '').html(SUB('v', j + 1)).call(halo);
  });
  g.append('circle').attr('cx', P.s(out[0])).attr('cy', P.t(out[1])).attr('r', 6).attr('fill', 'var(--accent)');
  g.append('text').attr('x', P.s(out[0])).attr('y', P.t(out[1]) + (out[1] > 0 ? 26 : -12)).attr('text-anchor', 'middle').attr('class', 'label-accent').text(`output ${sel + 1}`).call(halo);
  return out;
}

// ---- Static figure: one position, step by step ----
(function figure() {
  const fig = document.getElementById('fig-attn');
  const box = (title) => { const d = document.createElement('div'); d.innerHTML = `<p class="small-title">${title}</p>`; fig.append(d); return d; };
  const q3 = FIGURE.q;
  const { scores, weights } = attention([q3, q3, q3], FIGURE.k);
  const w = weights[2], sc = scores[2];
  const b1 = box('Query and keys');
  const P1 = plane(b1, { maxW: 280, range: 2.6 });
  const g1 = P1.svg.append('g');
  FIGURE.k.forEach((k, j) => arrow(g1, P1, k, 'var(--ink)', SUB('k', j + 1)));
  arrow(g1, P1, q3, 'var(--accent)', SUB('q', 3));
  P1.svg.attr('aria-hidden', 'true');

  const b2 = box('Weights, with scores below');
  const W = Math.max(240, Math.min(280, b2.clientWidth || 280)), H = W;
  const svg = d3.select(b2).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`).attr('aria-hidden', 'true');
  const m = { top: 30, bottom: 64, left: 10, right: 10 };
  const x = d3.scaleBand().domain([0, 1, 2]).range([m.left, W - m.right]).padding(0.4);
  const y = d3.scaleLinear().domain([0, 1]).range([H - m.bottom, m.top]);
  w.forEach((v, j) => {
    svg.append('rect').attr('x', x(j)).attr('width', x.bandwidth()).attr('y', y(v)).attr('height', y(0) - y(v)).attr('fill', 'var(--accent)').attr('fill-opacity', 0.8);
    svg.append('text').attr('x', x(j) + x.bandwidth() / 2).attr('y', y(v) - 8).attr('text-anchor', 'middle').attr('class', 'label-accent').text(v.toFixed(2));
    svg.append('text').attr('x', x(j) + x.bandwidth() / 2).attr('y', H - m.bottom + 22).attr('text-anchor', 'middle').attr('class', 'label-ink').text(`token ${j + 1}`);
    svg.append('text').attr('x', x(j) + x.bandwidth() / 2).attr('y', H - m.bottom + 46).attr('text-anchor', 'middle').text(sc[j].toFixed(2).replace('-', '−'));
  });
  svg.append('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', y(0)).attr('y2', y(0)).attr('stroke', 'var(--rule)');

  const b3 = box('Values and the output');
  const P3 = plane(b3, { maxW: 280, range: 2.6 });
  valuesPlane(P3.svg.append('g'), P3, FIGURE.v, w, [true, true, true], 2);
  P3.svg.attr('aria-hidden', 'true');
})();

// ---- The tool ----
const ctl = document.getElementById('controls');
const selCtl = segmented(ctl, { name: 'Selected token', options: [0, 1, 2, 3].map((i) => [String(i), `${i + 1}`]), value: String(state.sel),
  onChange: (k) => { state.sel = Number(k); draw(); } });
segmented(ctl, { name: 'Causal mask', options: [['on', 'On'], ['off', 'Off']], value: 'on', onChange: (k) => { state.causal = k === 'on'; draw(); } });
segmented(ctl, { name: 'Divide scores by √2', options: [['on', 'On'], ['off', 'Off']], value: 'on', onChange: (k) => { state.scale = k === 'on'; draw(); } });
segmented(ctl, { name: 'Table shows', options: [['weights', 'Weights'], ['scores', 'Scores']], value: 'weights', onChange: (k) => { state.table = k; draw(); } });
button(ctl, 'Start over', () => { state.q = copy(START.q); state.k = copy(START.k); draw(); });

const qkBox = document.getElementById('qk'), voBox = document.getElementById('vo');
const QK = plane(qkBox, { range: LIMIT + 0.4, maxW: 416 });
QK.svg.attr('role', 'group').attr('aria-label', 'Queries and keys in the plane; each arrow tip can be moved');
const qkLayer = QK.svg.append('g'), handleLayer = QK.svg.append('g');
const VO = plane(voBox, { range: 2.2, maxW: 416 });
VO.svg.attr('role', 'img');
const voLayer = VO.svg.append('g');
const tableBox = document.getElementById('table');
const out = document.getElementById('readout');
const clamp = (v) => Math.max(-LIMIT, Math.min(LIMIT, Math.round(v * 100) / 100));

// Handles: one for the selected query, one per key. Each is draggable and keyboard operable.
function setPoint(kind, j, p) {
  p = p.map(clamp);
  if (kind === 'k') {
    for (let m = 0; m < N; m++) if (m !== j && Math.hypot(p[0] - state.k[m][0], p[1] - state.k[m][1]) < 0.2) p = state.k[m].slice();
    state.k[j] = p;
  } else state.q[j] = p;
  draw();
}
let dragging = null;
function makeHandle(kind, j) {
  const g = handleLayer.append('g').attr('class', 'handle').attr('tabindex', 0).attr('role', 'button');
  g.append('circle').attr('r', 18).attr('fill', 'transparent');
  g.append('circle').attr('class', 'ring').attr('r', 7).attr('fill', 'var(--paper)').attr('stroke', kind === 'q' ? 'var(--accent)' : 'var(--ink)').attr('stroke-width', 1.5);
  g.on('pointerdown', (ev) => { dragging = { kind, j: kind === 'q' ? state.sel : j }; ev.target.setPointerCapture?.(ev.pointerId); ev.preventDefault(); });
  g.on('keydown', (ev) => {
    const jj = kind === 'q' ? state.sel : j, p = (kind === 'q' ? state.q : state.k)[jj].slice(), d = ev.shiftKey ? 0.5 : 0.1;
    if (ev.key === 'ArrowLeft') p[0] -= d; else if (ev.key === 'ArrowRight') p[0] += d;
    else if (ev.key === 'ArrowUp') p[1] += d; else if (ev.key === 'ArrowDown') p[1] -= d; else return;
    ev.preventDefault();
    setPoint(kind, jj, p);
  });
  return g;
}
const keyHandles = [0, 1, 2, 3].map((j) => makeHandle('k', j));
const qHandle = makeHandle('q', 0);
QK.svg.on('pointermove', (ev) => {
  if (!dragging) return;
  const [mx, my] = d3.pointer(ev, QK.svg.node());
  setPoint(dragging.kind, dragging.j, [QK.s.invert(mx), QK.t.invert(my)]);
});
QK.svg.on('pointerup pointercancel', () => { dragging = null; });
QK.svg.style('touch-action', 'none');

function draw() {
  const { scores, weights } = attention(state.q, state.k, { causal: state.causal, scale: state.scale });
  const i = state.sel, allowed = [0, 1, 2, 3].map((j) => !state.causal || j <= i);
  // Arrows: keys first (faint when masked for the selected token), then the selected query.
  qkLayer.selectAll('*').remove();
  state.k.forEach((k, j) => arrow(qkLayer, QK, k, allowed[j] ? 'var(--ink)' : 'var(--rule)', SUB('k', j + 1), !allowed[j], false));
  arrow(qkLayer, QK, state.q[i], 'var(--accent)', SUB('q', i + 1), false, false);
  keyHandles.forEach((h, j) => h.attr('transform', `translate(${QK.s(state.k[j][0])},${QK.t(state.k[j][1])})`)
    .attr('aria-label', `Key ${j + 1} tip at ${state.k[j][0].toFixed(1)}, ${state.k[j][1].toFixed(1)}; arrow keys move it`));
  qHandle.attr('transform', `translate(${QK.s(state.q[i][0])},${QK.t(state.q[i][1])})`)
    .attr('aria-label', `Query ${i + 1} tip at ${state.q[i][0].toFixed(1)}, ${state.q[i][1].toFixed(1)}; arrow keys move it`);
  qHandle.raise();
  if (dragging && dragging.kind === 'q') dragging.j = i;

  voLayer.selectAll('*').remove();
  const o = valuesPlane(voLayer, VO, state.v, weights[i], allowed, i);
  document.getElementById('vo-title').textContent = `Values and the output of token ${i + 1}`;
  VO.svg.attr('aria-label', `Output of token ${i + 1} at ${o[0].toFixed(2)}, ${o[1].toFixed(2)}`);

  // Table: one row per token; click a row to select that token.
  const M = state.table === 'weights' ? weights : scores;
  const cell = (r, j) => {
    if (state.causal && j > r) return '<td class="masked" aria-label="masked">·</td>';
    const v = M[r][j];
    const bg = state.table === 'weights' ? ` style="background: color-mix(in srgb, var(--accent) ${Math.round(v * 45)}%, transparent)"` : '';
    return `<td${bg}>${v.toFixed(2).replace('-', '−')}</td>`;
  };
  tableBox.innerHTML = `<p class="small-title">${state.table === 'weights' ? 'Weights' : `Scores ${state.scale ? 'q<sub>i</sub> · k<sub>j</sub> / √2' : 'q<sub>i</sub> · k<sub>j</sub>'}`}: row i attends to column j${state.causal ? '; a dot marks a masked pair' : ''}</p>
    <table class="wts"><thead><tr><th></th>${[1, 2, 3, 4].map((j) => `<th scope="col">to ${j}</th>`).join('')}</tr></thead>
    <tbody>${[0, 1, 2, 3].map((r) => `<tr data-r="${r}" class="${r === i ? 'sel' : ''}"><th scope="row">token ${r + 1}</th>${[0, 1, 2, 3].map((j) => cell(r, j)).join('')}</tr>`).join('')}</tbody></table>`;
  tableBox.querySelectorAll('tbody tr').forEach((tr) => tr.addEventListener('click', () => { state.sel = Number(tr.dataset.r); selCtl.set(String(state.sel)); draw(); }));

  const w = weights[i];
  out.innerHTML = `Token ${i + 1}: weights ${w.map((x, j) => (allowed[j] ? `${x.toFixed(2)} on ${j + 1}` : null)).filter(Boolean).join(', ')}` +
    `${state.causal && i < 3 ? ` (later tokens masked)` : ''}. Its output is (${o.map((v) => v.toFixed(2).replace('-', '−')).join(', ')}).`;
}
draw();
