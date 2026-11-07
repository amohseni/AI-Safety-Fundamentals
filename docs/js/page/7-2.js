// Page 7.2: test a belief detector.
import { slider, segmented } from '../lib/ui.js';
import { decodeTruth } from '../lib/s6l-truth.js';
import { trainProbe, scorecard, TRAIN_SETS } from '../lib/s6l-belief.js';

const d3 = window.d3;
const [truthJson, use] = await Promise.all([
  fetch(new URL('../../data/truth/statements.json', import.meta.url)).then((r) => r.json()),
  fetch(new URL('../../data/7-2/use.json', import.meta.url)).then((r) => r.json()),
]);
const T = decodeTruth(truthJson);
const pct = (v) => (v === null || v === undefined ? '<span class="quiet">not measured</span>' : `${Math.round(v * 100)}%`);
const state = { method: 'supervised', set: 'plain', shown: 'cities-plain', strength: 0, row: 0 };
const cache = new Map();
const probe = () => {
  const k = `${state.method}-${state.set}`;
  if (!cache.has(k)) cache.set(k, trainProbe(T, state.method, state.set));
  return cache.get(k);
};

// ---- Controls ----
const ctl = document.getElementById('controls');
segmented(ctl, { name: 'Probe method', options: [['supervised', 'Supervised'], ['ccs', 'Contrast-consistent search']], value: state.method,
  onChange: (k) => { state.method = k; state.set = 'plain'; buildSetControl(); update(); } });
const setBox = document.createElement('div'); ctl.append(setBox);
function buildSetControl() {
  setBox.innerHTML = '';
  segmented(setBox, { name: 'Training set (city statements)', options: TRAIN_SETS[state.method], value: state.set, onChange: (k) => { state.set = k; update(); } });
}
buildSetControl();
const SHOWN = [['cities-plain', 'Cities, held out'], ['cities-neg', 'Cities, negated'], ['spanish-plain', 'Spanish words'], ['elements-plain', 'Chemical elements']];
segmented(ctl, { name: 'Statements shown', options: SHOWN, value: state.shown, onChange: (k) => { state.shown = k; update(); } });

// ---- Plot: credence on the horizontal axis; dots spread up and down at random so they do not overlap ----
const plotEl = document.getElementById('plot');
const W = Math.max(260, Math.min(520, plotEl.clientWidth || 520)), H = 240, mx = 24;
const svg = d3.select(plotEl).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
  .attr('role', 'img').attr('aria-label', 'Statements placed left to right by the probe\u2019s credence that they are true');
const xs = d3.scaleLinear().domain([0, 1]).range([mx, W - mx]);
svg.append('line').attr('x1', mx).attr('x2', W - mx).attr('y1', H - 30).attr('y2', H - 30).attr('stroke', 'var(--rule)');
svg.append('line').attr('x1', xs(0.5)).attr('x2', xs(0.5)).attr('y1', 26).attr('y2', H - 30).attr('stroke', 'var(--ink)').attr('stroke-dasharray', '3 3');
svg.append('text').attr('x', mx).attr('y', H - 6).text('credence 0');
svg.append('text').attr('x', W - mx).attr('y', H - 6).attr('text-anchor', 'end').text('1');
svg.append('text').attr('x', xs(0.5)).attr('y', H - 6).attr('text-anchor', 'middle').text('0.5');
const gDots = svg.append('g');
// A fixed vertical position for each statement, from a hash of its index.
const jitter = (i) => { let h = Math.imul(i + 1, 2654435761) >>> 0; h ^= h >>> 15; return (h % 1000) / 1000; };

function shownList() {
  const [topic, form] = state.shown.split('-');
  return T.S.filter((s) => s.topic === topic && s.form === form && (topic !== 'cities' || !T.isTrain(s)));
}

// ---- Scorecard ----
const card = document.getElementById('scorecard');
function renderCard(sc) {
  const u = useRow();
  const rows = [
    ['Truth', 'held-out city statements', pct(sc.truth)],
    ['Coherence', 'negated city statements', pct(sc.neg)],
    ['', 'same verdict on a paraphrase', pct(sc.para)],
    ['', 'credence in A plus credence in not-A, average distance from 1', sc.sum === null ? pct(null) : sc.sum.toFixed(2)],
    ['Use', state.strength === 0 ? 'steering along the direction: answers moved toward the steered side' : `steering at strength ${state.strength > 0 ? '+' : ''}${state.strength}: answers moved toward the steered side`, u],
    ['Uniformity', 'Spanish words', pct(sc.spanish)],
    ['', 'chemical elements', pct(sc.elements)],
    ['', 'negated Spanish words and negated elements', sc.spanishNeg === null ? pct(null) : `${pct(sc.spanishNeg)} and ${pct(sc.elementsNeg)}`],
  ];
  if (sc.alice !== null) rows.push(['', 'agreement with Alice’s opinion, held-out cities', pct(sc.alice)]);
  card.innerHTML = `<thead><tr><th>Criterion</th><th>Test</th><th class="num">Result</th></tr></thead><tbody>${rows.map(([a, b, c]) => `<tr><th class="crit">${a}</th><td>${b}</td><td class="num">${c}</td></tr>`).join('')}</tbody>`;
}

// ---- Use: precomputed steering (tools/truth/steer.py) ----
const sIx = (s) => use.strengths.indexOf(s);
function useRow() {
  const key = `${state.method}-${state.set}`;
  const P = use.probes[key];
  if (!P) return pct(null);
  if (state.strength === 0) return '<span class="quiet">set a strength</span>';
  return `${Math.round(moved(P) * 100)}%<br><span class="quiet">random direction ${Math.round(moved(use.probes.random) * 100)}%</span>`;
}
// Of the statements answered on the other side without steering, the share answered on the steered
// side at this strength. (A few statements have P(TRUE) = P(FALSE) at the model's bfloat16 precision.)
function moved(P) {
  const d = P.diff[sIx(state.strength)], b = use.base;
  const want = state.strength > 0 ? 1 : -1;
  let n = 0, m = 0;
  b.forEach((v, k) => { if (Math.sign(v) === -want) { n++; if (Math.sign(d[k]) === want) m++; } });
  return n ? m / n : 0;
}

const stSel = document.getElementById('statement');
use.rows.forEach((i, k) => { const o = document.createElement('option'); o.value = k; o.textContent = T.S[i].text; stSel.append(o); });
const start = use.rows.findIndex((i) => T.S[i].text.includes('Tokyo'));
state.row = start >= 0 ? start : 0;
stSel.value = state.row;
stSel.addEventListener('change', () => { state.row = Number(stSel.value); update(); });
const stCtl = document.getElementById('steer-controls');
slider(stCtl, { name: 'Steering strength', min: use.strengths[0], max: use.strengths[use.strengths.length - 1], step: 1, value: 0,
  format: (v) => (v > 0 ? `+${v}` : String(v)), onInput: (v) => { state.strength = v; update(); } });
const stOut = document.getElementById('steer-readout');

function update() {
  const p = probe();
  const list = shownList();
  const pts = [];
  for (const s of list) {
    const c = p.credence(s);
    if (c !== null) pts.push({ c, s });
  }
  gDots.selectAll('circle').data(pts).join('circle')
    .attr('cx', (q) => xs(q.c)).attr('cy', (q) => 30 + jitter(q.s.i) * (H - 70)).attr('r', 3.6)
    .attr('fill', (q) => (q.s.label ? 'var(--accent)' : 'none')).attr('fill-opacity', 0.75)
    .attr('stroke', (q) => (q.s.label ? 'none' : 'var(--second)')).attr('stroke-width', 1.2);
  const sc = scorecard(T, p);
  renderCard(sc);
  const P = use.probes[`${state.method}-${state.set}`];
  const k = state.row, base = use.base[k];
  const ans = (v) => (v > 0 ? 'TRUE' : v < 0 ? 'FALSE' : 'TRUE and FALSE with equal probability');
  const s = T.S[use.rows[k]];
  if (state.strength === 0 || !P) {
    stOut.innerHTML = `&ldquo;${s.text}&rdquo; is ${s.label ? 'true' : 'false'}. Without steering the model answers <b>${ans(base)}</b> (P(TRUE) &minus; P(FALSE) = ${base.toFixed(2)}).`;
  } else {
    const v = P.diff[sIx(state.strength)][k];
    stOut.innerHTML = `&ldquo;${s.text}&rdquo; is ${s.label ? 'true' : 'false'}. Without steering the model answers ${ans(base)} (${base.toFixed(2)}); steered at ${state.strength > 0 ? '+' : ''}${state.strength} along this probe&rsquo;s direction it answers <b>${ans(v)}</b> (${v.toFixed(2)}).`;
  }
}
update();
window.__s72 = { state, update };

// ---- Static figure: the probing setup and the four criteria ----
(function figure() {
  const fig = document.getElementById('fig-setup-svg');
  const w = Math.max(300, Math.min(620, fig.clientWidth || 620)), h = 260;
  const svg = d3.select(fig).append('svg').attr('class', 'chart').attr('width', w).attr('height', h).attr('viewBox', `0 0 ${w} ${h}`)
    .attr('role', 'img').attr('aria-label', 'A statement goes into the model; the activation after one layer goes to a linear probe, which outputs a credence.');
  const cols = 5, rowsN = 4, gx = Math.min(64, (w - 170) / cols), gy = 34, x0 = 44, y0 = 30;
  const words = ['Lord', 'Byron', 'died', 'in', '1824.'];
  for (let r = 0; r < rowsN; r++) for (let c = 0; c < cols; c++) {
    svg.append('rect').attr('x', x0 + c * gx).attr('y', y0 + (rowsN - 1 - r) * gy).attr('width', gx - 8).attr('height', gy - 12)
      .attr('fill', 'none').attr('stroke', r === 2 && c === cols - 1 ? 'var(--accent)' : 'var(--rule)').attr('stroke-width', r === 2 && c === cols - 1 ? 2 : 1);
  }
  words.forEach((t, c) => svg.append('text').attr('x', x0 + c * gx + (gx - 8) / 2).attr('y', y0 + rowsN * gy + 14).attr('text-anchor', 'end')
    .attr('transform', `rotate(-40 ${x0 + c * gx + (gx - 8) / 2} ${y0 + rowsN * gy + 14})`).text(t));
  const ax = x0 + cols * gx, ay = y0 + 1 * gy + (gy - 12) / 2;
  svg.append('line').attr('x1', ax - 6).attr('x2', ax + 40).attr('y1', ay).attr('y2', ay).attr('stroke', 'var(--accent)').attr('stroke-width', 1.5);
  svg.append('text').attr('x', ax + 46).attr('y', ay - 4).attr('class', 'label-accent').text('probe');
  svg.append('text').attr('x', ax + 46).attr('y', ay + 20).attr('class', 'label-accent').text('credence');
  svg.append('text').attr('x', x0).attr('y', 18).text('layers');
})();
