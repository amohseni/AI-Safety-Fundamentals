// Page 3.7: step through one GPT-2 small forward pass, recorded in advance.
import { button, cssVar } from '../lib/ui.js';
import { steps, shapeOf, paramsThrough, tensorAt } from '../lib/s3j-gpt2.js';

const d3 = window.d3;
const FONT = '21.33px et-book, Georgia, serif';
const S = steps();
const fmtN = (v) => v.toLocaleString('en-US');
const showTok = (t) => t.replace(/ /g, '·').replace(/\n/g, '↵');
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const shapeText = (sh) => sh.map((v, i) => (i === 0 && sh.length === 2 ? 'n' : fmtN(v))).join(' × ');

const stage = document.getElementById('fp-stage');
let index;
try { index = await (await fetch('../data/3-7/index.json')).json(); } catch (e) { stage.textContent = 'The recorded forward pass could not be loaded.'; throw e; }
const C = index.counts;
const cache = new Map();
const load = (k) => { if (!cache.has(k)) cache.set(k, fetch(`../data/3-7/${index.prompts[k].file}`).then((r) => r.json())); return cache.get(k); };

// ---- Static figure: the pipeline with shapes ----
(function figure() {
  const el = document.getElementById('fig-pipe');
  const parts = [['op', 'text'], ['arr', 'n tokens'], ['op', 'token and position embeddings'], ['arr', 'n × 768'],
    ['rep', '12 layers: normalize, attention, add; normalize, MLP, add'], ['arr', 'n × 768'], ['op', 'normalize'], ['arr', 'n × 768'],
    ['op', 'unembedding'], ['arr', 'n × 50,257'], ['op', 'softmax'], ['arr', 'n × 50,257'], ['op', 'next-token probabilities']];
  el.innerHTML = parts.map(([c, t]) => (c === 'arr' ? `<span class="arr">→ ${t} →</span>` : `<span class="op${c === 'rep' ? ' rep' : ''}">${t}</span>`)).join('');
  el.setAttribute('aria-label', parts.map((p) => p[1]).join(', '));
})();

// ---- Controls ----
const state = { prompt: 0, i: 0 };
const top = document.getElementById('fp-top');
const pWrap = document.createElement('label');
pWrap.className = 'ctl';
pWrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Prompt</span></span>';
const sel = document.createElement('select');
sel.innerHTML = index.prompts.map((p, k) => `<option value="${k}">${esc(p.text)}</option>`).join('');
pWrap.append(sel);
top.append(pWrap);
sel.addEventListener('change', () => { state.prompt = Number(sel.value); draw(); });
const back = button(top, 'Back', () => go(state.i - 1));
const next = button(top, 'Next', () => go(state.i + 1), 'primary');
button(top, 'Start over', () => go(0));
const lWrap = document.createElement('label');
lWrap.className = 'ctl';
lWrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Layer</span><output></output></span>';
const lRange = document.createElement('input');
Object.assign(lRange, { type: 'range', min: 1, max: 12, step: 1, value: 1 });
lWrap.append(lRange);
top.append(lWrap);
lRange.addEventListener('input', () => {
  const L = Number(lRange.value), cur = S[state.i];
  const sub = cur.layer >= 1 && cur.layer <= 12 ? S.findIndex((s) => s.layer === cur.layer && s.key === cur.key) - S.findIndex((s) => s.layer === cur.layer) : 0;
  go(S.findIndex((s) => s.layer === L) + sub);
});
function go(i) { state.i = Math.max(0, Math.min(S.length - 1, i)); draw(); }
document.getElementById('tool').addEventListener('keydown', (ev) => {
  if (ev.target.tagName === 'SELECT' || ev.target.tagName === 'INPUT') return;
  if (ev.key === 'ArrowRight') { go(state.i + 1); ev.preventDefault(); }
  if (ev.key === 'ArrowLeft') { go(state.i - 1); ev.preventDefault(); }
});

// ---- Heat maps ----
function rgb(hex) {
  const h = hex.replace('#', '');
  const f = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  return [0, 2, 4].map((k) => parseInt(f.slice(k, k + 2), 16));
}
const mixc = (a, b, t) => a.map((x, k) => Math.round(x + (b[k] - x) * t));
function colors() { return { paper: rgb(cssVar('--paper')), accent: rgb(cssVar('--accent')), second: rgb(cssVar('--second')), ink: cssVar('--ink'), quiet: cssVar('--quiet'), rule: cssVar('--rule') }; }

function canvasFor(parent, W, H) {
  const c = document.createElement('canvas');
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(W * dpr); c.height = Math.round(H * dpr);
  c.style.width = `${W}px`; c.style.height = `${H}px`;
  parent.append(c);
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.font = FONT; ctx.textBaseline = 'middle';
  return { c, ctx };
}

// Rows = positions, labeled with their tokens; columns = numbers, pooled to the pixel width: each pixel
// column shows the entry of largest absolute value among the numbers it covers, so no spike is averaged away.
function rowMap(parent, rec, cols, value, color, note) {
  const n = rec.tokens.length;
  const avail = Math.min(640, parent.clientWidth || 560);
  const m = document.createElement('canvas').getContext('2d');
  m.font = FONT;
  const left = Math.ceil(Math.max(...rec.tokens.map(([t]) => m.measureText(showTok(t)).width))) + 12;
  const W = Math.max(240, avail), rowH = 30, H = n * rowH;
  const { c, ctx } = canvasFor(parent, W, H);
  const PW = Math.max(1, Math.floor(W - left));
  const img = new ImageData(PW, n);
  for (let r = 0; r < n; r++) for (let px = 0; px < PW; px++) {
    const j0 = Math.floor((px * cols) / PW), j1 = Math.max(j0 + 1, Math.floor(((px + 1) * cols) / PW));
    let best = value(r, j0);
    for (let j = j0 + 1; j < j1; j++) { const v = value(r, j); if (Math.abs(v) > Math.abs(best)) best = v; }
    const [R, G, B] = color(best);
    const o = (r * PW + px) * 4;
    img.data[o] = R; img.data[o + 1] = G; img.data[o + 2] = B; img.data[o + 3] = 255;
  }
  const off = document.createElement('canvas');
  off.width = PW; off.height = n;
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, PW, n, left, 0, PW, n * rowH);
  const K = colors();
  ctx.fillStyle = K.ink; ctx.textAlign = 'right';
  rec.tokens.forEach(([t], r) => ctx.fillText(showTok(t), left - 8, r * rowH + rowH / 2));
  c.setAttribute('role', 'img');
  const p = document.createElement('p');
  p.className = 'what';
  p.textContent = note;
  parent.append(p);
  return c;
}

function drawTensor(box, rec, st) {
  const K = colors();
  const n = rec.tokens.length;
  if (st.key === 'tokens') {
    const div = document.createElement('div');
    div.className = 'toks';
    div.innerHTML = rec.tokens.map(([t, id]) => `<span class="tok"><span class="t">${t.split('').map((ch) => (ch === ' ' ? '<span class="sp">·</span>' : esc(ch))).join('')}</span><span class="id">${id}</span></span>`).join('');
    box.append(div);
    return;
  }
  const T = tensorAt(rec, S, state.i);
  if (T.kind === 'signed') {
    const cols = T.shape[1];
    const col = (v) => (v >= 0 ? mixc(K.paper, K.accent, v / 7) : mixc(K.paper, K.second, -v / 7));
    const cv = rowMap(box, rec, cols, (r, j) => T.values[r * cols + j], col, `${fmtN(cols)} numbers per position. Colors are cut off at ±${T.info.clip.toFixed(2)}; the values run from ${T.info.min.toFixed(1).replace('-', '−')} to ${T.info.max.toFixed(1).replace('-', '−')}.`);
    cv.setAttribute('aria-label', `Heat map, ${n} rows by ${cols} columns. Values range from ${T.info.min} to ${T.info.max}.`);
    return;
  }
  if (T.kind === 'pattern') {
    const wrap = document.createElement('div');
    wrap.style.display = 'flex'; wrap.style.flexWrap = 'wrap'; wrap.style.gap = '0.6rem 1rem';
    box.append(wrap);
    const cell = n <= 3 ? 20 : 16;
    for (let h = 0; h < 12; h++) {
      const d = document.createElement('div');
      const size = Math.max(n * cell, 64);
      const { c, ctx } = canvasFor(d, size, n * cell + 28);
      for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
        const w = T.values[h * n * n + i * n + j] / 255;
        const [R, G, B] = mixc(K.paper, K.accent, w);
        ctx.fillStyle = `rgb(${R},${G},${B})`;
        ctx.fillRect(j * cell, i * cell, cell - 1, cell - 1);
      }
      ctx.strokeStyle = K.rule; ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, n * cell - 1, n * cell - 1);
      ctx.fillStyle = K.quiet; ctx.textAlign = 'left';
      ctx.fillText(`head ${h + 1}`, 0, n * cell + 15);
      c.setAttribute('role', 'img');
      c.setAttribute('aria-label', `Head ${h + 1} attention weights`);
      wrap.append(d);
    }
    const note = document.createElement('p');
    note.className = 'what';
    note.textContent = `Row i of each square holds the weights position i puts on positions 1 to i (a stronger blue is more). Cells above the diagonal are masked.`;
    box.append(note);
    return;
  }
  // Scores or probabilities over the vocabulary, in bins of 64 token ids.
  const cols = T.shape[1];
  const isP = st.key === 'probs';
  rowMap(box, rec, cols, (r, j) => T.values[r * cols + j] / 255, (v) => mixc(K.paper, K.accent, v),
    isP ? 'Columns: the 50,257 tokens in order of id. A stronger blue means a higher probability.' : 'Columns: the 50,257 tokens in order of id. A stronger blue means a higher score.')
    .setAttribute('aria-label', `Heat map, ${n} rows by 50,257 tokens, in ${cols} bins`);
  const ul = document.createElement('div');
  ul.innerHTML = `<p class="small-title">${isP ? 'Most likely next tokens, with probabilities' : 'Highest scores'}</p>` +
    rec.top.map((row, r) => `<p style="margin:0">after “${esc(showTok(rec.tokens[r][0]))}”: ${row.slice(0, 3).map(([t, , z, p]) => `“${esc(showTok(t))}” ${isP ? p.toFixed(3) : z.toFixed(1).replace('-', '−')}`).join(', ')}</p>`).join('');
  box.append(ul);
}

// ---- Parameter strip ----
const SEGS = [{ key: 'wte', n: C.wte, label: 'token embedding matrix' }, { key: 'wpe', n: C.wpe }];
const stepSeg = new Map();      // step index -> segment index
S.forEach((s, i) => {
  if (s.key === 'tok_emb') stepSeg.set(i, 0);
  else if (s.key === 'pos_emb') stepSeg.set(i, 1);
  else if (s.params(C) > 0) { stepSeg.set(i, SEGS.length); SEGS.push({ key: s.key, n: s.params(C), layer: s.layer }); }
});
function drawStrip() {
  const box = document.getElementById('fp-strip');
  box.innerHTML = `<p style="margin:0">Parameters used so far: <b>${fmtN(paramsThrough(S, state.i, C))}</b> of ${fmtN(C.total)}</p>`;
  const W = Math.max(240, Math.min(760, box.clientWidth || 640)), H = 70;
  const svg = d3.select(box).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
    .attr('role', 'img');
  const x = d3.scaleLinear().domain([0, C.total]).range([0, W - 2]);
  const usedSeg = new Set([...stepSeg].filter(([i]) => i <= state.i).map(([, k]) => k));
  const cur = S[state.i].key === 'logits' ? 0 : stepSeg.get(state.i);
  let acc = 0;
  const pos = SEGS.map((s) => { const a = acc; acc += s.n; return [a, acc]; });
  SEGS.forEach((s, k) => {
    const [a, b] = pos[k];
    svg.append('rect').attr('x', x(a) + 1).attr('y', 6).attr('width', Math.max(0.5, x(b) - x(a) - (s.n > 1e5 ? 0.6 : 0))).attr('height', 22)
      .attr('fill', usedSeg.has(k) ? 'var(--accent)' : 'var(--rule)');
  });
  if (cur !== undefined) {
    const [a, b] = pos[cur];
    svg.append('rect').attr('x', x(a) + 0.5).attr('y', 2).attr('width', Math.max(3, x(b) - x(a))).attr('height', 30)
      .attr('fill', 'none').attr('stroke', 'var(--ink)').attr('stroke-width', 1.5);
  }
  const used = paramsThrough(S, state.i, C);
  const L1 = pos[2][0], L12 = pos[pos.length - 2][1];
  svg.append('text').attr('x', x(0) + 1).attr('y', 52).text(W > 330 ? 'token embeddings' : 'embeddings');
  svg.append('text').attr('x', x(L12)).attr('y', 52).attr('text-anchor', 'end').text(W > 420 ? '12 layers, attention and MLP' : '12 layers');
  svg.append('line').attr('x1', x(L1)).attr('x2', x(L1)).attr('y1', 30).attr('y2', 38).attr('stroke', 'var(--quiet)');
  svg.attr('aria-label', `Parameter strip: ${fmtN(used)} of ${fmtN(C.total)} parameters used so far.`);
}

// ---- Step list (the pipeline) ----
const list = document.getElementById('fp-steps');
function drawList() {
  const cur = S[state.i];
  const L = cur.layer >= 1 && cur.layer <= 12 ? cur.layer : (cur.layer === 0 ? 1 : 12);
  const items = [];
  S.forEach((s, i) => {
    if (s.layer >= 1 && s.layer <= 12 && s.layer !== L) return;
    if (s.layer === L && s.key === 'ln1') items.push(`<li class="layer-head">Layer ${L} of 12</li>`);
    items.push(`<li class="${s.layer >= 1 && s.layer <= 12 ? 'in-layer' : ''}"><button type="button" data-i="${i}"${i === state.i ? ' aria-current="step"' : ''}><span>${s.name}</span></button></li>`);
  });
  list.innerHTML = items.join('');
  list.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => go(Number(b.dataset.i))));
}

let ticket = 0;
async function draw() {
  const mine = ++ticket;
  const rec = await load(state.prompt);
  if (mine !== ticket) return;
  const st = S[state.i], n = rec.tokens.length;
  if (st.layer >= 1 && st.layer <= 12) lRange.value = st.layer;
  lWrap.querySelector('output').textContent = st.layer >= 1 && st.layer <= 12 ? String(st.layer) : st.layer === 0 ? 'before layer 1' : 'after layer 12';
  back.disabled = state.i === 0; next.disabled = state.i === S.length - 1;
  const listHadFocus = !!(document.activeElement && list.contains(document.activeElement));
  drawList();
  const sh = shapeOf(st, n);
  const shape = st.key === 'pattern' ? `12 × ${n} × ${n}` : sh.length === 1 ? `${n}` : `${n} × ${fmtN(sh[1])}`;
  const p = st.params(C);
  const where = st.layer >= 1 && st.layer <= 12 ? `Layer ${st.layer}: ` : '';
  stage.innerHTML = `<p class="small-title">Step ${state.i + 1} of ${S.length}. ${where}${st.name}</p>
    <p class="what">${st.what}</p>
    <p>Shape: <b>${shape}</b>${st.key === 'tokens' ? ` (${n} tokens)` : ''}. Parameters in this step: <b>${fmtN(p)}</b>${st.key === 'logits' ? ' new (it reuses the 38,597,376 of the token embedding matrix)' : ''}.</p>`;
  const box = document.createElement('div');
  stage.append(box);
  drawTensor(box, rec, st);
  drawStrip();
  // Keep the current step in view in the list on small screens without moving the page.
  const curBtn = list.querySelector('[aria-current="step"]');
  if (listHadFocus && curBtn) curBtn.focus({ preventScroll: true });
}
draw();
let rt;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', draw);
