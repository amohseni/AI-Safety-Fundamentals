// Page 2.2: the neuron (perceptron playground, neuron mode).
import { Playground, datasets, ACT, sum } from '../lib/perceptron.js';
import { slider, segmented } from '../lib/ui.js';

const d3 = window.d3;

// ---- Static figure: a neuron with three inputs ----
(function figure() {
  // Drawn at its true width (at most 300 pixels) so its labels stay at the text size.
  const W = Math.max(260, Math.min(300, document.getElementById('fig-neuron').clientWidth || 300)), H = 230;
  const svg = d3.select('#fig-neuron').append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .attr('role', 'img').attr('aria-label', 'Diagram of a neuron with three inputs, weights, a bias, a sum, an activation function, and an output');
  const ink = 'var(--ink)', quiet = 'var(--quiet)';
  const sx = 136, sy = 120;
  const ins = [[26, 40, 'x₁', 'w₁'], [26, 120, 'x₂', 'w₂'], [26, 200, 'x₃', 'w₃']];
  svg.append('defs').append('marker').attr('id', 'arr').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
    .attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto').append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', quiet);
  for (const [x, y, lab, wl] of ins) {
    svg.append('text').attr('x', x - 6).attr('y', y + 6).attr('text-anchor', 'end').attr('class', 'label-ink').text(lab);
    const ex = sx - 24, ey = sy + (y - sy) * 0.2;
    svg.append('line').attr('x1', x).attr('y1', y).attr('x2', ex).attr('y2', ey).attr('stroke', quiet).attr('marker-end', 'url(#arr)');
    if (y !== sy) svg.append('text').attr('x', x + (ex - x) * 0.5).attr('y', y + (ey - y) * 0.5 + (y < sy ? -8 : 20)).attr('text-anchor', 'middle').text(wl);
    else svg.append('text').attr('x', x + (ex - x) * 0.5).attr('y', y - 8).attr('text-anchor', 'middle').text(wl);
  }
  svg.append('circle').attr('cx', sx).attr('cy', sy).attr('r', 22).attr('fill', 'none').attr('stroke', ink);
  svg.append('text').attr('x', sx).attr('y', sy + 7).attr('text-anchor', 'middle').attr('class', 'label-ink').text('Σ');
  svg.append('text').attr('x', sx).attr('y', sy - 32).attr('text-anchor', 'middle').text('+ b');
  const ax = 184, aw = 46;
  svg.append('line').attr('x1', sx + 22).attr('y1', sy).attr('x2', ax - 4).attr('y2', sy).attr('stroke', quiet).attr('marker-end', 'url(#arr)');
  svg.append('text').attr('x', (sx + 22 + ax) / 2 - 2).attr('y', sy - 8).attr('text-anchor', 'middle').text('z');
  const curve = d3.range(-6, 6.01, 0.25).map((z) => [ax + ((z + 6) / 12) * aw, sy + 22 - 44 / (1 + Math.exp(-z))]);
  svg.append('path').attr('d', d3.line()(curve)).attr('fill', 'none').attr('stroke', ink).attr('stroke-width', 1.6);
  svg.append('text').attr('x', ax + aw / 2).attr('y', sy + 46).attr('text-anchor', 'middle').text('f');
  svg.append('line').attr('x1', ax + aw + 4).attr('y1', sy).attr('x2', W - 4).attr('y2', sy).attr('stroke', quiet).attr('marker-end', 'url(#arr)');
  svg.append('text').attr('x', W - 2).attr('y', sy + 28).attr('text-anchor', 'end').attr('class', 'label-ink').text('output');
})();

// ---- The tool ----
const state = { data: 'and', act: 'step' };
const canvas = document.getElementById('plane');
const pg = new Playground(canvas, { mode: 'neuron', w: [1, 1], b: -1.5, act: 'step', onChange: () => { syncSliders(); draw(); } });
const out = document.getElementById('readout');
const ctl = document.getElementById('controls');
const clampW = (v) => Math.max(-6, Math.min(6, v));

segmented(ctl, { name: 'Dataset', options: [['and', 'AND'], ['or', 'OR'], ['xor', 'XOR']], value: state.data,
  onChange: (k) => { state.data = k; pg.setData(datasets[k]()); draw(); } });
segmented(ctl, { name: 'Activation function', options: [['step', 'step'], ['sigmoid', 'sigmoid'], ['relu', 'ReLU']], value: state.act,
  onChange: (k) => { state.act = k; pg.act = k; draw(); } });
const w1S = slider(ctl, { name: 'Weight w₁', min: -6, max: 6, step: 0.05, value: 1, format: (v) => v.toFixed(2), onInput: (v) => { pg.w = [v, pg.w[1]]; draw(); } });
const w2S = slider(ctl, { name: 'Weight w₂', min: -6, max: 6, step: 0.05, value: 1, format: (v) => v.toFixed(2), onInput: (v) => { pg.w = [pg.w[0], v]; draw(); } });
slider(ctl, { name: 'Bias b', min: -6, max: 6, step: 0.1, value: -1.5, format: (v) => v.toFixed(1), onInput: (v) => { pg.b = v; draw(); } });

function syncSliders() {
  pg.w = pg.w.map(clampW);
  w1S.set(pg.w[0]); w2S.set(pg.w[1]);
}

function draw() {
  pg.draw();
  const f = ACT[pg.act].f, X = pg.X;
  const len = Math.hypot(...pg.w);
  const rows = X.map((x, i) => {
    const z = sum(pg.w, pg.b, x);
    return `<tr><td>(${x.join(', ')})</td><td>${pg.Y[i]}</td><td>${(Math.abs(z) < 5e-3 ? 0 : z).toFixed(2)}</td><td>${f(z).toFixed(2)}</td><td>${(z > 0 ? 1 : 0) === pg.Y[i] ? '' : 'wrong side'}</td></tr>`;
  }).join('');
  out.innerHTML = `\\(w\\) = (${pg.w[0].toFixed(2)}, ${pg.w[1].toFixed(2)}), length ${len.toFixed(2)}; \\(b\\) = ${pg.b.toFixed(1)}. Points on the wrong side of the line: <b>${pg.mistakes()}</b> of 4.` +
    `<table class="sums"><thead><tr><th>input</th><th>label</th><th>sum \\(z\\)</th><th>output</th><th></th></tr></thead><tbody>${rows}</tbody></table>`;
  if (window.renderMathInElement) window.renderMathInElement(out, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
}

window.addEventListener('resize', () => pg.draw());
pg.setData(datasets.and());
draw();
