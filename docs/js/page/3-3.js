// Page 3.3: sample the next token, from GPT-2 small's next-token distributions computed in advance.
import { slider, button, fmt } from '../lib/ui.js';
import { decodeTree, distribution, storedMass, draw as drawToken } from '../lib/s3j-sample.js';

const d3 = window.d3;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const showTok = (t) => t.replace(/ /g, '·').replace(/\n/g, '↵');
const pct = (v) => (v >= 0.995 ? `${(v * 100).toFixed(0)}%` : v >= 0.1 ? `${(v * 100).toFixed(0)}%` : v >= 0.001 ? `${(v * 100).toFixed(1)}%` : '<0.1%');
const prob = (v) => (v >= 0.001 ? v.toFixed(3) : v > 0 ? '<0.001' : '0');

const out = document.getElementById('s-readout');
let doc;
try { doc = await (await fetch('../data/3-3/tree.json')).json(); } catch (e) { out.textContent = 'The stored distributions could not be loaded.'; throw e; }
const TEMPS = doc.temps;
const trees = doc.prompts.map((p) => decodeTree(p.tree, doc.k, TEMPS.length));
const word = (id) => doc.vocab[String(id)] ?? `#${id}`;

// ---- Static figure: the ten most likely tokens after "The capital of France is" at three temperatures ----
(function figure() {
  const fig = document.getElementById('fig-temps');
  const tree = trees[0];
  for (const T of [0.5, 1, 2]) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">T = ${T}</p>`;
    fig.append(box);
    const rows = distribution(tree, 0, TEMPS, T, 1).rows.slice(0, 10);
    barChart(box, rows, { width: 260, max: 0.4, compact: true });
  }
})();

// Horizontal bars: one row per token, label on the left, value on the right.
function barChart(container, rows, { width = 520, max = null, compact = false, style = () => 'kept' } = {}) {
  const rowH = 26, m = { top: 4, right: 64, bottom: 6, left: 0 };
  const W = Math.max(240, Math.min(width, container.clientWidth || width));
  const probe = document.createElement('canvas').getContext('2d');
  probe.font = '21.33px et-book, Georgia, serif';
  m.left = Math.ceil(Math.max(...rows.map((r) => probe.measureText(showTok(word(r.id))).width))) + 12;
  const H = rows.length * rowH + m.top + m.bottom;
  const svg = d3.select(container).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`);
  const top = max ?? Math.max(...rows.map((r) => r.pT), 1e-9);
  const x = d3.scaleLinear().domain([0, top]).range([0, W - m.left - m.right]);
  rows.forEach((r, k) => {
    const y = m.top + k * rowH;
    const s = style(r);
    const wBar = Math.max(s === 'cut' ? 0 : 1, x(Math.min(r.pT, top)));
    svg.append('text').attr('x', m.left - 8).attr('y', y + rowH / 2 + 6).attr('text-anchor', 'end').attr('class', s === 'cut' ? '' : 'label-ink').text(showTok(word(r.id)));
    if (s === 'open') svg.append('rect').attr('x', m.left + 0.5).attr('y', y + 4.5).attr('width', Math.max(0, wBar - 1)).attr('height', rowH - 9).attr('fill', 'none').attr('stroke', 'var(--accent)');
    else svg.append('rect').attr('x', m.left).attr('y', y + 4).attr('width', wBar).attr('height', rowH - 8).attr('fill', s === 'cut' ? 'var(--rule)' : 'var(--accent)').attr('fill-opacity', s === 'cut' ? 1 : 0.85);
    svg.append('text').attr('x', m.left + wBar + 6).attr('y', y + rowH / 2 + 6).attr('class', s === 'cut' ? '' : 'label-ink').text(prob(r.pT));
  });
  svg.attr('role', 'img').attr('aria-label', rows.map((r) => `${word(r.id)} ${prob(r.pT)}`).join(', '));
  if (compact) svg.attr('aria-hidden', 'true');
  return svg;
}

// ---- The tool ----
const state = { prompt: 0, T: 1, p: 1, path: [0], chosen: [] };   // path: node indices; chosen: [{id, pT, T, p}]
const ctl = document.getElementById('s-controls');
const pWrap = document.createElement('label');
pWrap.className = 'ctl';
pWrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Prompt</span></span>';
const sel = document.createElement('select');
sel.innerHTML = doc.prompts.map((p, k) => `<option value="${k}">${esc(p.text)}</option>`).join('');
pWrap.append(sel);
ctl.append(pWrap);
sel.addEventListener('change', () => { state.prompt = Number(sel.value); reset(); });
slider(ctl, { name: 'Temperature T', min: 0, max: 2, step: 0.25, value: state.T, format: (v) => (v === 0 ? '0 (greedy)' : fmt(2)(v)),
  onInput: (v) => { state.T = v; draw(); } });
slider(ctl, { name: 'Top-p', min: 0.1, max: 1, step: 0.1, value: state.p, format: fmt(1), onInput: (v) => { state.p = Math.round(v * 10) / 10; draw(); } });
const btns = document.createElement('div');
btns.className = 'seg';
ctl.append(btns);
const sampleBtn = button(btns, 'Sample', () => step('sample'), 'primary');
const greedyBtn = button(btns, 'Greedy', () => step('greedy'));
button(btns, 'Reset', () => reset());

const textBox = document.getElementById('s-text');
const chartBox = document.getElementById('s-chart');
const drawsBox = document.getElementById('s-draws');

// Keys of the stored continuations, written by Python as "1.0|0.9".
function sampleKey(T, p) {
  const py = (v) => (Number.isInteger(v) ? v.toFixed(1) : String(v));
  return `${py(T)}|${py(Math.round(p * 10) / 10)}`;
}

function reset() { state.path = [0]; state.chosen = []; draw(); }
function step(kind) {
  const tree = trees[state.prompt], node = state.path[state.path.length - 1];
  const dist = distribution(tree, node, TEMPS, kind === 'greedy' ? 0 : state.T, kind === 'greedy' ? 1 : state.p);
  const r = drawToken(dist, Math.random());
  if (!r) return;
  state.chosen.push({ id: r.id, pT: r.pT, T: kind === 'greedy' ? 0 : state.T, p: kind === 'greedy' ? 1 : state.p });
  state.path.push(tree.children[node].get(r.id));
  draw();
}

function draw() {
  const tree = trees[state.prompt], node = state.path[state.path.length - 1];
  const P = doc.prompts[state.prompt];
  // Generated text so far, each token with its probability when it was chosen.
  textBox.innerHTML = `<span class="ptext">${esc(P.text)}</span>` + state.chosen.map((c) =>
    `<span class="gtok"><span class="t">${esc(showTok(word(c.id)))}</span><span class="pr">${prob(c.pT)}</span></span>`).join('');
  const dist = distribution(tree, node, TEMPS, state.T, state.p);
  const leaf = tree.children[node].size === 0;
  chartBox.innerHTML = '';
  barChart(chartBox, dist.rows, { style: (r) => (!r.keep ? 'cut' : r.stored ? 'kept' : 'open') });
  const held = dist.rows.reduce((s, r) => s + r.pT, 0);
  const mass = storedMass(dist);
  const keepText = state.T === 0 ? 'At T = 0 the sampler always takes the most likely token.'
    : state.p >= 1 ? 'Top-p is 1, so no token is cut.'
      : dist.nucleusInside ? `Top-p ${state.p.toFixed(1)} keeps the ${dist.kept === 1 ? 'single most likely token' : `${dist.kept} most likely tokens`}; the gray bars are cut.`
        : `Top-p ${state.p.toFixed(1)} keeps all 20 of these tokens and more beyond them.`;
  const after = state.chosen.length ? `After ${state.chosen.length} generated token${state.chosen.length === 1 ? '' : 's'}` : 'After the prompt';
  out.innerHTML = `${after}, the 20 most likely next tokens hold <b>${pct(held)}</b> of GPT-2&rsquo;s probability at T = ${state.T}. ${keepText} ` +
    (leaf ? 'This is as far as the prefixes computed in advance go: press Reset to start again.'
      : `The filled bars are the tokens after which the next step was computed in advance; together they hold <b>${pct(mass)}</b> of the probability, and Sample chooses among them in proportion to their probabilities.`);
  sampleBtn.disabled = leaf || mass <= 0;
  greedyBtn.disabled = leaf;
  // Whole continuations sampled in advance at these settings.
  if (state.T === 0) {
    drawsBox.innerHTML = `<p class="small-title">At T = 0, GPT-2&rsquo;s greedy continuation (30 tokens)</p><p class="draw"><span class="ptext">${esc(P.text)}</span>${esc(P.greedy)}</p>`;
  } else {
    const list = P.samples[sampleKey(state.T, state.p)];
    drawsBox.innerHTML = list
      ? `<p class="small-title">Four continuations GPT-2 sampled at T = ${state.T}, top-p ${state.p.toFixed(1)} (30 tokens each, computed in advance)</p>` +
        list.map((t) => `<p class="draw"><span class="ptext">${esc(P.text)}</span>${esc(t)}</p>`).join('')
      : `<p class="small-title">Whole continuations were sampled in advance only for top-p 0.5, 0.9, and 1.</p>`;
  }
}
draw();
let rt;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
