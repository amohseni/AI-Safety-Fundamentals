// Page 4.3: tilt the distribution.
import { chart } from '../lib/chart.js';
import { slider, segmented, button } from '../lib/ui.js';
import { candidates, tilt, expect, kl, sweep } from '../lib/s45-tilt.js';

const d3 = window.d3;
const state = { lb: 1, hack: true, showTrue: false }; // lb = log10 beta
const beta = () => 10 ** state.lb;
const fmtB = (b) => (b >= 1 ? b.toFixed(1) : b >= 0.1 ? b.toFixed(2) : b.toFixed(3));

// ---- Static figure: the two rewards against KL divergence, with and without the hack ----
(function figure() {
  const fig = document.getElementById('fig-kl');
  for (const [hack, title] of [[true, 'With the hack response'], [false, 'Without it']]) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${title}</p>`;
    fig.append(box);
    const s = sweep(candidates(hack));
    const c = chart(box, { width: 340, height: 250, margin: { top: 16, right: 70, bottom: 58, left: 46 },
      x: { domain: [0, 3.6], ticks: 4, label: 'KL divergence' }, y: { domain: [0, 1], ticks: 3 }, title: `${title}: rewards against KL divergence` });
    c.line('p', s.map((v) => [v.kl, v.proxy]), { color: 'ink', label: 'score' });
    c.line('t', s.map((v) => [v.kl, v.truth]), { color: 'second', label: 'true' });
    c.spreadLabels(['p', 't'], 18);
  }
})();

// ---- The tool ----
const ctl = document.getElementById('controls');
const bS = slider(ctl, { name: 'Strength of the KL penalty β', min: -2, max: 1, step: 0.01, value: state.lb,
  format: (v) => fmtB(10 ** v), onInput: (v) => { state.lb = v; draw(); } });
segmented(ctl, { name: 'Hack response', options: [['in', 'Included'], ['out', 'Removed']], value: 'in',
  onChange: (k) => { state.hack = k === 'in'; build(); } });
const trueBtn = button(ctl, 'Show the true rewards', () => {
  state.showTrue = !state.showTrue;
  trueBtn.textContent = state.showTrue ? 'Hide the true rewards' : 'Show the true rewards';
  draw();
});
const resetBtn = button(ctl, 'Back to β = 10', () => { state.lb = 1; bS.set(1); draw(); });
resetBtn.setAttribute('aria-label', 'Set beta back to 10');

const barsEl = document.getElementById('bars');
const curveEl = document.getElementById('curve');
const out = document.getElementById('readout');
let rs, sw, svg, curve;

function build() {
  rs = candidates(state.hack);
  sw = sweep(rs);
  // Bars: one row per response, label above its bar; score and true reward at the right.
  barsEl.innerHTML = '';
  const W = Math.max(280, Math.min(560, barsEl.clientWidth || 560));
  const rowH = 54, top = 26, H = top + rowH * rs.length;
  const colTrue = W, colScore = W - 64, barMax = W - 150;
  svg = d3.select(barsEl).append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .attr('role', 'img').attr('aria-label', 'Probability of each response under the starting model and under the policy');
  svg.append('text').attr('x', 0).attr('y', 16).text('probability');
  svg.append('text').attr('x', colScore).attr('y', 16).attr('text-anchor', 'end').text('score');
  svg.append('text').attr('class', 'label-second').attr('x', colTrue).attr('y', 16).attr('text-anchor', 'end').text('true');
  const x = d3.scaleLinear().domain([0, 1]).range([0, barMax]);
  const g = svg.selectAll('g.row').data(rs).join('g').attr('class', 'row').attr('transform', (d, i) => `translate(0,${top + i * rowH})`);
  g.append('text').attr('class', 'name').attr('x', 0).attr('y', 18).text((d) => d.label);
  g.append('rect').attr('class', 'pol').attr('x', 0).attr('y', 28).attr('height', 14).attr('fill', 'var(--accent)').attr('fill-opacity', 0.8);
  g.append('rect').attr('class', 'ref').attr('x', 0).attr('y', 25).attr('height', 20).attr('width', (d) => x(d.ref))
    .attr('fill', 'none').attr('stroke', 'var(--ink)').attr('stroke-width', 1);
  g.append('text').attr('class', 'pv label-accent').attr('y', 41);
  g.append('text').attr('class', 'num').attr('x', colScore).attr('y', 41).attr('text-anchor', 'end').text((d) => d.proxy.toFixed(2));
  g.append('text').attr('class', 'true').attr('x', colTrue).attr('y', 41).attr('text-anchor', 'end');
  svg.node().__x = x;

  // Curve: both expected rewards as beta decreases (beta on a log scale, falling to the right).
  curveEl.innerHTML = '';
  curve = chart(curveEl, { width: 560, height: 300, margin: { top: 34, right: 70, bottom: 52, left: 52 },
    x: { domain: [10, 0.01], log: true, ticks: 4, format: '~g', label: 'β, falling to the right' },
    y: { domain: [0, 1], ticks: 4, label: 'expected reward' }, title: 'Expected reward model score and expected true reward as beta decreases' });
  curve.line('p', sw.map((v) => [v.beta, v.proxy]), { color: 'ink', label: 'score' });
  curve.line('t', sw.map((v) => [v.beta, v.truth]), { color: 'second', label: 'true' });
  curve.spreadLabels(['p', 't'], 18);
  draw();
}

function draw() {
  const b = beta(), pi = tilt(rs, b), x = svg.node().__x;
  svg.selectAll('rect.pol').data(pi).attr('width', (p) => x(p));
  svg.selectAll('text.pv').data(pi).attr('x', (p, i) => Math.max(x(p), x(rs[i].ref)) + 6).text((p) => p.toFixed(2));
  svg.selectAll('text.true').data(rs).text((d) => (state.showTrue ? d.truth.toFixed(2) : '?'));
  const P = expect(pi, rs, 'proxy'), T = expect(pi, rs, 'truth'), K = kl(pi, rs);
  curve.vline('b', b, { color: 'accent', dash: null });
  curve.point('bp', b, P, { color: 'ink', r: 3.5, label: null });
  curve.point('bt', b, T, { color: 'second', r: 3.5, label: null });
  out.innerHTML = `At β = <b>${fmtB(b)}</b>: expected reward model score <b>${P.toFixed(2)}</b>; ` +
    `<span class="c-second">expected true reward <b>${T.toFixed(2)}</b></span>; KL divergence from the starting model ${K.toFixed(2)}.`;
}

let resizeT;
window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(build, 150); });
build();
