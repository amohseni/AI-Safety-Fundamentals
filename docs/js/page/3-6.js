// Page 3.6: the logit lens on GPT-2 small, from precomputed data (tools/3-6/lens.py),
// and the block diagram. Numbers quoted in the tasks are checked by tools/test-page-3-6.mjs.
import { segmented, cssVar } from '../lib/ui.js';
import { loadJSON, palette, showToken, escapeHTML } from '../lib/s3k-gpt2.js';

const d3 = window.d3;
const readout = document.getElementById('readout');

// ---- Static figure: the block diagram (drawn first; it needs no data) ----
function blockDiagram() {
  const fig = document.getElementById('fig-block');
  fig.innerHTML = '';
  const W = Math.max(300, Math.min(520, fig.clientWidth || 520));
  const wide = W >= 440;
  const sx = 34, bx = 92, step = 50;
  const parts = [];
  for (let L = 0; L < 3; L++) parts.push([`${wide ? 'layer norm, then ' : ''}attention, layer ${L}`], [`${wide ? 'layer norm, then ' : ''}MLP, layer ${L}`]);
  const top = 104, H = top + parts.length * step + 140;
  const svg = d3.select(fig).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
    .attr('role', 'img').attr('aria-label', 'Block diagram of three transformer layers around the residual stream');
  const line = (x1, y1, x2, y2, o = {}) => svg.append('line').attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2)
    .attr('stroke', o.c || 'var(--ink)').attr('stroke-width', o.w || 1.4);
  // A box sized to its text.
  const box = (x, y, h, text) => {
    const r = svg.append('rect').attr('x', x).attr('y', y).attr('height', h).attr('fill', 'var(--paper)').attr('stroke', 'var(--quiet)').attr('rx', 3);
    const t = svg.append('text').attr('x', x + 10).attr('y', y + h / 2 + 7).attr('class', 'label-ink').text(text);
    const w = (t.node().getComputedTextLength ? t.node().getComputedTextLength() : text.length * 9) + 20;
    r.attr('width', w);
    return w;
  };
  const arrowHead = (x, y, dir) => {
    const d = dir === 'left' ? `M${x},${y}l8,-5l0,10z` : `M${x},${y}l-5,-8l10,0z`;
    svg.append('path').attr('d', d).attr('fill', 'var(--ink)');
  };
  box(4, 8, 34, 'token and position embeddings');
  line(sx, 42, sx, H - 96, { w: 2.2, c: 'var(--accent)' });
  svg.append('text').attr('x', sx + 10).attr('y', 72).attr('class', 'label-accent').text('residual stream');
  parts.forEach(([name], i) => {
    const y = top + i * step;
    line(sx, y, bx, y, { c: 'var(--quiet)' });
    const bw = box(bx, y - 15, 30, name);
    const mx = bx + Math.min(bw / 2, 60);
    line(mx, y + 15, mx, y + 28, { c: 'var(--quiet)' });
    line(mx, y + 28, sx + 10, y + 28, { c: 'var(--quiet)' });
    arrowHead(sx + 10, y + 28, 'left');
    svg.append('circle').attr('cx', sx).attr('cy', y + 28).attr('r', 9).attr('fill', 'var(--paper)').attr('stroke', 'var(--ink)');
    svg.append('text').attr('x', sx).attr('y', y + 34).attr('text-anchor', 'middle').attr('class', 'label-ink').text('+');
  });
  const yb = top + parts.length * step;
  svg.append('text').attr('x', sx + 14).attr('y', yb + 12).text('… 9 more layers');
  arrowHead(sx, H - 88, 'down');
  box(4, H - 86, 34, wide ? 'final layer norm, then unembedding' : 'layer norm, unembedding');
  svg.append('text').attr('x', 10).attr('y', H - 18).attr('class', 'label-ink').text('a score for every token');
}
blockDiagram();

let data;
try {
  data = await loadJSON('../../data/3-6/lens.json', import.meta.url);
} catch (e) {
  readout.textContent = 'The model readings could not be loaded, so the tool cannot run.';
  throw e;
}

const NAMES = ['Eiffel Tower', 'Colosseum', 'Japan', 'Michael Jordan', 'Days of the week', 'Water'];
const st = { p: 0, row: 12, col: null };
let scrollEnd = true; // show the last position first on narrow screens
const ctl = document.getElementById('controls');
segmented(ctl, { name: 'Prompt', options: data.prompts.map((p, i) => [String(i), NAMES[i] || p.prompt]), value: '0',
  onChange: (k) => { st.p = Number(k); st.row = 12; st.col = null; scrollEnd = true; draw(); } });

const table = document.getElementById('lens');
const rowName = (s) => (s === 0 ? 'embedding' : `layer ${s - 1}`);
const rowLong = (s) => (s === 0 ? 'at the embedding' : `after layer ${s - 1}`);
const tk = (t) => escapeHTML(showToken(t));
const q = (t) => `&ldquo;${escapeHTML(showToken(t).trim() || t)}&rdquo;`;

function draw() {
  const P = data.prompts[st.p];
  const n = P.tokens.length;
  if (st.col === null) st.col = n - 1;
  const shade = palette('--ink');
  const paper = cssVar('--paper'), ink = cssVar('--ink');
  let html = '<thead><tr><th scope="col"><span style="color: var(--quiet)">input</span></th>';
  for (let i = 1; i < n; i++) html += `<th scope="col">${tk(P.tokens[i])}</th>`;
  html += '</tr></thead><tbody>';
  for (let s = 0; s < 13; s++) {
    html += `<tr><th scope="row">${rowName(s)}</th>`;
    for (let i = 1; i < n; i++) {
      const c = P.cells[s][i], p = c.p[0];
      const sel = s === st.row && i === st.col;
      html += `<td tabindex="${sel ? 0 : -1}" data-s="${s}" data-i="${i}" class="${sel ? 'sel' : ''}" style="background:${shade(p)};color:${Math.sqrt(p) > 0.55 ? paper : ink}" aria-label="${rowLong(s)}, position ${i}: ${escapeHTML(c.t[0])}, ${p.toFixed(2)}">${tk(c.t[0])}</td>`;
    }
    html += '</tr>';
  }
  table.innerHTML = html + '</tbody>';
  if (scrollEnd) { const sc = table.parentElement; sc.scrollLeft = sc.scrollWidth; scrollEnd = false; }
  detail();
}

function select(s, i, focus) {
  st.row = s; st.col = i;
  draw();
  if (focus) { const td = table.querySelector('td.sel'); if (td) td.focus(); }
}
table.addEventListener('click', (e) => {
  const td = e.target.closest('td');
  if (td) select(Number(td.dataset.s), Number(td.dataset.i), true);
});
table.addEventListener('keydown', (e) => {
  const mv = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
  if (!mv) return;
  e.preventDefault();
  const n = data.prompts[st.p].tokens.length;
  select(Math.max(0, Math.min(12, st.row + mv[0])), Math.max(1, Math.min(n - 1, st.col + mv[1])), true);
});

const topBox = document.getElementById('top10');
function detail() {
  const P = data.prompts[st.p];
  const c = P.cells[st.row][st.col];
  document.getElementById('top-title').innerHTML = `Top 10 next tokens ${rowLong(st.row)}, at ${q(P.tokens[st.col])}`;
  topBox.innerHTML = '';
  const W = Math.max(260, Math.min(420, topBox.clientWidth || 420)), rowH = 26, H = rowH * 10 + 6, lw = 130, vw = 52;
  const svg = d3.select(topBox).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
    .attr('role', 'img').attr('aria-label', 'Top 10 predicted next tokens and their probabilities');
  const x = d3.scaleLinear().domain([0, 1]).range([lw, W - vw]);
  c.t.forEach((t, k) => {
    const y = k * rowH;
    svg.append('text').attr('x', lw - 8).attr('y', y + 19).attr('text-anchor', 'end').attr('class', k === 0 ? 'label-accent' : 'label-ink').text(showToken(t));
    svg.append('rect').attr('x', lw).attr('y', y + 6).attr('width', Math.max(1, x(c.p[k]) - lw)).attr('height', rowH - 10)
      .attr('fill', k === 0 ? 'var(--accent)' : 'var(--quiet)').attr('fill-opacity', k === 0 ? 0.85 : 0.5);
    svg.append('text').attr('x', x(c.p[k]) + 6).attr('y', y + 19).text(c.p[k] < 0.01 ? c.p[k].toFixed(3) : c.p[k].toFixed(2));
  });
  let s = `Position ${st.col}, ${q(P.tokens[st.col])}, ${rowLong(st.row)}: the top prediction is <span class="c-accent">${q(c.t[0])}</span>, with probability <b>${c.p[0].toFixed(2)}</b>.`;
  if (c.chg) {
    const [a, m, b] = c.chg, L = st.row - 1;
    const f = (v) => (v < 0.01 ? v.toFixed(3) : v.toFixed(2));
    const da = m - a, dm = b - m;
    s += ` Layer ${L} changed the probability of ${q(c.t[0])} from <b>${f(a)}</b> before the layer to <b>${f(m)}</b> after its attention part, then to <b>${f(b)}</b> after its MLP.`;
    s += Math.abs(da) > Math.abs(dm) ? ' The attention part made the larger change.' : ' The MLP made the larger change.';
  } else {
    s += ' This row is the embedding, before any layer.';
  }
  readout.innerHTML = s;
}
draw();

let rw;
window.addEventListener('resize', () => { clearTimeout(rw); rw = setTimeout(() => { blockDiagram(); draw(); }, 150); });
const mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
if (mq && mq.addEventListener) mq.addEventListener('change', draw);
