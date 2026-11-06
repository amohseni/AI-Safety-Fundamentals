// Page 3.2: embedding space (word2vec vectors), neighbors, analogies, and a country-capital figure.
import { segmented, button, cssVar } from '../lib/ui.js';
import { decodeVectors, nearest, row, analogy, pca2, mapNeighbors } from '../lib/s3j-embed.js';

const d3 = window.d3;
const FONT = '21.33px et-book, Georgia, serif';
const EXAMPLES = [
  ['king', 'man', 'woman'],
  ['Tokyo', 'Japan', 'France'],
  ['Paris', 'France', 'Italy'],
  ['Oslo', 'Norway', 'Austria'],
  ['bigger', 'big', 'small'],
];
const FIG_PAIRS = [['France', 'Paris'], ['Germany', 'Berlin'], ['Poland', 'Warsaw'], ['Russia', 'Moscow'], ['Egypt', 'Cairo'], ['Japan', 'Tokyo']];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

const out = document.getElementById('map-readout');
const lists = document.getElementById('lists');
let E;
try {
  const res = await fetch('../data/3-2/vectors.json');
  E = decodeVectors(await res.json());
} catch (e) {
  out.textContent = 'The word vectors could not be loaded.';
  throw e;
}

// ---- Static figure: six countries and capitals, PCA of their twelve vectors ----
(function figure() {
  const fig = document.getElementById('fig-capitals');
  const ids = FIG_PAIRS.flat().map((w) => E.index.get(w));
  const [c1, c2] = pca2(E, ids);
  const W = Math.max(300, Math.min(560, fig.clientWidth || 560)), H = Math.round(W * 0.72);
  const m = { top: 24, right: 24, bottom: 24, left: 24 };
  const xs = d3.extent(c1), ys = d3.extent(c2);
  // Keep the two axes on the same scale, so directions are not distorted.
  const span = Math.max(xs[1] - xs[0], (ys[1] - ys[0]) * (W - m.left - m.right) / (H - m.top - m.bottom));
  const cx = (xs[0] + xs[1]) / 2, cy = (ys[0] + ys[1]) / 2;
  const k = (W - m.left - m.right) / (span * 1.25);
  const X = (v) => W / 2 + (v - cx) * k, Y = (v) => H / 2 - (v - cy) * k;
  const svg = d3.select(fig).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`).attr('aria-hidden', 'true');
  svg.append('defs').append('marker').attr('id', 'capital-head').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
    .attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto-start-reverse')
    .append('path').attr('d', 'M0,0L10,5L0,10Z').attr('fill', 'var(--accent)');
  FIG_PAIRS.forEach(([country, capital], p) => {
    const i = 2 * p, j = 2 * p + 1;
    const x1 = X(c1[i]), y1 = Y(c2[i]), x2 = X(c1[j]), y2 = Y(c2[j]);
    const len = Math.hypot(x2 - x1, y2 - y1), ux = (x2 - x1) / len, uy = (y2 - y1) / len;
    svg.append('line').attr('x1', x1 + ux * 6).attr('y1', y1 + uy * 6).attr('x2', x2 - ux * 6).attr('y2', y2 - uy * 6)
      .attr('stroke', 'var(--accent)').attr('stroke-width', 1.4).attr('marker-end', 'url(#capital-head)');
    svg.append('circle').attr('cx', x1).attr('cy', y1).attr('r', 3).attr('fill', 'var(--ink)');
    svg.append('circle').attr('cx', x2).attr('cy', y2).attr('r', 3).attr('fill', 'var(--accent)');
    // Labels sit beyond each end of the arrow, along its direction.
    svg.append('text').attr('x', x1 - ux * 14).attr('y', y1 - uy * 14 + 6 + (uy < 0 ? 8 : -8)).attr('text-anchor', 'middle').attr('class', 'label-ink').text(country);
    svg.append('text').attr('x', x2 + ux * 14).attr('y', y2 + uy * 14 + 6 + (uy < 0 ? -8 : 8)).attr('text-anchor', 'middle').attr('class', 'label-accent').text(capital);
  });
})();

// ---- The tool ----
const state = { proj: 'umap', query: 'car', abc: EXAMPLES[0].slice(), exclude: true, mode: 'search', view: 'near' };
const ctl = document.getElementById('map-controls');

function textBox(parent, { name, value, list }) {
  const wrap = document.createElement('label');
  wrap.className = 'ctl';
  if (name) wrap.innerHTML = `<span class="ctl-head"><span class="ctl-name">${name}</span></span>`;
  const input = document.createElement('input');
  Object.assign(input, { type: 'text', value, spellcheck: false, autocomplete: 'off' });
  input.setAttribute('autocapitalize', 'off');
  if (list) input.setAttribute('list', list);
  wrap.append(input);
  parent.append(wrap);
  return input;
}
const dl = document.createElement('datalist');
dl.id = 'words-list';
dl.innerHTML = E.words.map((w) => `<option value="${esc(w)}">`).join('');
document.body.append(dl);

const qBox = textBox(ctl, { name: 'Search for a word', value: state.query, list: 'words-list' });
qBox.addEventListener('input', () => { state.query = qBox.value.trim(); state.mode = 'search'; draw(); });

segmented(ctl, { name: 'Projection', options: [['pca', 'PCA'], ['umap', 'UMAP']], value: state.proj, onChange: (k) => { state.proj = k; draw(); } });
const viewCtl = segmented(ctl, { name: 'Map shows', options: [['near', 'The highlighted words'], ['all', 'All words']], value: state.view, onChange: (k) => { state.view = k; draw(); } });

const anWrap = document.createElement('div');
anWrap.className = 'ctl';
anWrap.setAttribute('role', 'group');
anWrap.setAttribute('aria-label', 'Analogy a minus b plus c');
anWrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Analogy: a &minus; b + c</span></span>';
const abcRow = document.createElement('div');
abcRow.className = 'abc';
anWrap.append(abcRow);
const abcBoxes = [0, 1, 2].map((k) => {
  const lab = document.createElement('span');
  lab.textContent = ['a', 'minus b', 'plus c'][k];
  abcRow.append(lab);
  const input = document.createElement('input');
  Object.assign(input, { type: 'text', value: state.abc[k], spellcheck: false, autocomplete: 'off' });
  input.setAttribute('autocapitalize', 'off');
  input.setAttribute('list', 'words-list');
  input.setAttribute('aria-label', ['a', 'b', 'c'][k]);
  abcRow.append(input);
  input.addEventListener('input', () => { state.abc[k] = input.value.trim(); });
  input.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') { state.mode = 'analogy'; draw(); } });
  return input;
});
ctl.append(anWrap);
const exSel = document.createElement('label');
exSel.className = 'ctl';
exSel.innerHTML = '<span class="ctl-head"><span class="ctl-name">Examples</span></span>';
const sel = document.createElement('select');
sel.style.font = 'inherit';
sel.style.fontSize = 'var(--text)';
sel.style.color = 'var(--ink)';
sel.style.background = 'transparent';
sel.innerHTML = EXAMPLES.map((e, k) => `<option value="${k}">${e[0]} − ${e[1]} + ${e[2]}</option>`).join('');
exSel.append(sel);
ctl.append(exSel);
sel.addEventListener('change', () => {
  state.abc = EXAMPLES[Number(sel.value)].slice();
  abcBoxes.forEach((b, k) => { b.value = state.abc[k]; });
  state.mode = 'analogy'; draw();
});
segmented(ctl, { name: 'Leave out a, b, c', options: [['on', 'On'], ['off', 'Off']], value: 'on',
  onChange: (k) => { state.exclude = k === 'on'; if (state.mode === 'analogy') draw(); } });
button(ctl, 'Compute the analogy', () => { state.mode = 'analogy'; draw(); }, 'primary');

// Canvas map.
const stage = document.getElementById('map-stage');
const canvas = document.createElement('canvas');
canvas.setAttribute('role', 'img');
stage.append(canvas);
let W = 0, H = 0;
function size() {
  const avail = Math.min(640, stage.clientWidth || 560);
  W = Math.max(260, avail); H = Math.round(W * 0.8);
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
  canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function wordIndex(w) {
  if (!w) return undefined;
  if (E.index.has(w)) return E.index.get(w);
  const lower = w.toLowerCase();
  return E.words.findIndex((x) => x.toLowerCase() === lower) >= 0 ? E.words.findIndex((x) => x.toLowerCase() === lower) : undefined;
}

function draw() {
  const ctx = size();
  const P = state.proj === 'pca' ? E.pca : E.umap;
  const hi = [];           // [index, role] role: 'query' | 'nn' | 'input' | 'answer'
  let html = '', tableHtml = '';
  if (state.mode === 'search') {
    const i = wordIndex(state.query);
    if (i === undefined) {
      html = state.query ? `<span class="err">&ldquo;${esc(state.query)}&rdquo; is not among the 4,039 stored words.</span> Most stored words are lowercase, and names are capitalized.` : 'Type a word.';
    } else {
      const nn = nearest(E, row(E, i), 10, [i]);
      hi.push([i, 'query'], ...nn.map((r) => [r.i, 'nn']));
      const kept = mapNeighbors(P, i, nn.map((r) => r.i));
      html = `Ten nearest neighbors of <span class="c-accent">${esc(E.words[i])}</span>, by cosine similarity over all 300 numbers. On the ${state.proj === 'pca' ? 'PCA' : 'UMAP'} map, <b>${kept}</b> of these ten ${kept === 1 ? 'is' : 'are'} among the ten dots nearest to it.`;
      tableHtml = `<div><p class="small-title">Nearest to ${esc(E.words[i])}</p><table class="nn"><tbody>${nn.map((r) => `<tr><td>${esc(r.word)}</td><td class="num">${r.sim.toFixed(3)}</td></tr>`).join('')}</tbody></table></div>`;
    }
  } else {
    const missing = state.abc.filter((w) => wordIndex(w) === undefined);
    if (missing.length) {
      html = `<span class="err">Not among the stored words: ${missing.map((w) => `&ldquo;${esc(w)}&rdquo;`).join(', ')}.</span>`;
    } else {
      const ws = state.abc.map((w) => E.words[wordIndex(w)]);
      const res = analogy(E, ws[0], ws[1], ws[2], { k: 5, exclude: state.exclude });
      const ids = ws.map((w) => E.index.get(w));
      ids.forEach((i) => hi.push([i, 'input']));
      res.forEach((r) => { if (!ids.includes(r.i)) hi.push([r.i, 'answer']); });
      const top = res[0];
      const isInput = ids.includes(top.i);
      html = `${esc(ws[0])} &minus; ${esc(ws[1])} + ${esc(ws[2])}, with the input words ${state.exclude ? 'left out' : 'allowed'}: the closest word is <span class="${isInput ? 'c-second' : 'c-accent'}">${esc(top.word)}</span>, cosine similarity ${top.sim.toFixed(3)}${isInput ? ', one of the inputs' : ''}.`;
      tableHtml = `<div><p class="small-title">Closest to ${esc(ws[0])} &minus; ${esc(ws[1])} + ${esc(ws[2])}</p><table class="nn"><tbody>${res.map((r) => `<tr class="${ids.includes(r.i) ? 'input' : ''}"><td>${esc(r.word)}${ids.includes(r.i) ? ' (input)' : ''}</td><td class="num">${r.sim.toFixed(3)}</td></tr>`).join('')}</tbody></table></div>`;
    }
  }
  out.innerHTML = html;
  lists.innerHTML = tableHtml;

  // View: all words, or a window around the highlighted words.
  let x0 = 0, x1 = 1, y0 = 0, y1 = 1;
  if (state.view === 'near' && hi.length) {
    const xs = hi.map(([i]) => P[i][0]), ys = hi.map(([i]) => P[i][1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const half = Math.max(0.04, (Math.max(...xs) - Math.min(...xs)) / 2, ((Math.max(...ys) - Math.min(...ys)) / 2) * (W / H)) * 1.35;
    x0 = cx - half; x1 = cx + half; y0 = cy - half * (H / W); y1 = cy + half * (H / W);
  }
  const pad = 14;
  const sx = (v) => pad + ((v - x0) / (x1 - x0)) * (W - 2 * pad);
  const sy = (v) => H - pad - ((v - y0) / (y1 - y0)) * (H - 2 * pad);
  const colors = { ink: cssVar('--ink'), quiet: cssVar('--quiet'), rule: cssVar('--rule'), accent: cssVar('--accent'), second: cssVar('--second'), paper: cssVar('--paper') };
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = colors.rule;
  for (let i = 0; i < E.n; i++) {
    const x = sx(P[i][0]), y = sy(P[i][1]);
    if (x < -2 || x > W + 2 || y < -2 || y > H + 2) continue;
    ctx.fillRect(x - 1, y - 1, 2, 2);
  }
  // Highlighted dots, then labels placed to avoid overlaps.
  const roleColor = { query: colors.accent, nn: colors.accent, input: colors.second, answer: colors.accent };
  hi.forEach(([i, role]) => {
    ctx.fillStyle = roleColor[role];
    ctx.beginPath(); ctx.arc(sx(P[i][0]), sy(P[i][1]), role === 'query' || role === 'input' ? 5 : 3.5, 0, 2 * Math.PI); ctx.fill();
  });
  ctx.font = FONT;
  ctx.textBaseline = 'middle';
  const boxes = [];
  const order = [...hi].sort((a, b) => (a[1] === 'query' || a[1] === 'input' ? -1 : 0) - (b[1] === 'query' || b[1] === 'input' ? -1 : 0));
  for (const [i, role] of order) {
    const x = sx(P[i][0]), y = sy(P[i][1]);
    const w = ctx.measureText(E.words[i]).width, h = 20;
    const tries = [[8, 0], [-8 - w, 0], [-w / 2, -16], [-w / 2, 16], [8, -16], [8, 16], [-8 - w, -16], [-8 - w, 16], [-w / 2, -32], [-w / 2, 32]];
    let placed = null;
    for (const [dx, dy] of tries) {
      const b = { x: x + dx, y: y + dy - h / 2, w, h };
      if (b.x < 0 || b.x + w > W || b.y < 0 || b.y + h > H) continue;
      if (!boxes.some((o) => b.x < o.x + o.w + 2 && o.x < b.x + b.w + 2 && b.y < o.y + o.h && o.y < b.y + b.h)) { placed = b; break; }
    }
    if (!placed) continue;
    boxes.push(placed);
    ctx.lineWidth = 5; ctx.strokeStyle = colors.paper; ctx.lineJoin = 'round';
    ctx.strokeText(E.words[i], placed.x, placed.y + h / 2);
    ctx.fillStyle = roleColor[role];
    ctx.fillText(E.words[i], placed.x, placed.y + h / 2);
  }
  canvas.setAttribute('aria-label', `${state.proj === 'pca' ? 'PCA' : 'UMAP'} map of 4,039 words. Highlighted: ${hi.map(([i]) => E.words[i]).join(', ') || 'none'}.`);
}
draw();
let rt;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 120); });
window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', draw);
