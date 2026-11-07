// Page 3.5: a gallery of the 144 attention heads of GPT-2 small, from precomputed data
// (tools/3-5/heads.py). Numbers quoted in the tasks are checked by tools/test-page-3-5.mjs.
import { segmented, cssVar } from '../lib/ui.js';
import { chart } from '../lib/chart.js';
import { loadJSON, unpackAttention, palette, drawPattern, sizedCanvas, showToken, escapeHTML } from '../lib/s3k-gpt2.js';

const FONT = '21.33px et-book, Georgia, serif';
const readout = document.getElementById('readout');

let data;
try {
  data = await loadJSON('../../data/3-5/heads.json', import.meta.url);
} catch (e) {
  readout.textContent = 'The attention data could not be loaded, so the tool cannot run.';
  throw e;
}
const T = data.meta.T;
const inputs = data.inputs.map((d) => ({ ...d, att: unpackAttention(d.att, d.n) }));
const S = data.scores;
// Head types, from the measured scores (thresholds stated on the page).
const TYPES = [
  { key: 'ind', name: 'Induction heads', test: (l, h) => S.ind[l][h] > 0.7, style: 'solid second' },
  { key: 'prev', name: 'Previous-token heads', test: (l, h) => S.prev[l][h] > 0.45, style: 'dashed second' },
  { key: 'dup', name: 'Duplicate-token heads', test: (l, h) => S.dup[l][h] > 0.4, style: 'dotted ink' },
];
const typeOf = (l, h) => TYPES.find((t) => t.test(l, h)) || null;

const LABELS = { random: 'Random tokens, repeated', sentence: 'A sentence', names: 'The Dursleys' };
const st = { input: 0, l: 0, h: 0, q: inputs[0].n - 1, labels: false };
const cur = () => inputs[st.input];

const ctl = document.getElementById('controls');
segmented(ctl, { name: 'Input', options: inputs.map((d, i) => [String(i), LABELS[d.key]]), value: '0',
  onChange: (k) => { st.input = Number(k); st.q = cur().n - 1; drawAll(); } });
segmented(ctl, { name: 'Head types', options: [['off', 'Not labeled'], ['on', 'Labeled']], value: 'off',
  onChange: (k) => { st.labels = k === 'on'; drawGrid(); drawTypes(); writeReadout(); } });

// ---- The grid of 144 thumbnails ----
const grid = document.getElementById('grid');
let G = null; // layout of the grid
function drawGrid() {
  const avail = grid.parentElement.clientWidth || 520;
  const W = Math.max(280, Math.min(544, avail));
  const left = 30, top = 26, gap = 2;
  const cell = Math.floor((W - left - gap * 11) / 12);
  const H = top + 12 * cell + 11 * gap + 4;
  const ctx = sizedCanvas(grid, W, H);
  G = { left, top, gap, cell, W, H };
  const ink = palette('--ink');
  ctx.font = FONT;
  ctx.fillStyle = cssVar('--quiet');
  ctx.textBaseline = 'alphabetic';
  for (let i = 0; i < 12; i++) {
    const p = left + i * (cell + gap) + cell / 2;
    if (cell >= 24 || i % 2 === 0) { ctx.textAlign = 'center'; ctx.fillText(String(i), p, top - 6); }
    const y = top + i * (cell + gap) + cell / 2 + 7;
    ctx.textAlign = 'right'; ctx.fillText(String(i), left - 6, y);
  }
  const att = cur().att;
  for (let l = 0; l < 12; l++) {
    for (let h = 0; h < 12; h++) {
      const x = left + h * (cell + gap), y = top + l * (cell + gap);
      drawPattern(ctx, att, l, h, x, y, cell, ink);
      ctx.strokeStyle = cssVar('--rule'); ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      const t = st.labels && typeOf(l, h);
      if (t) {
        const [dash, color] = t.style.split(' ');
        ctx.strokeStyle = cssVar(color === 'second' ? '--second' : '--ink');
        ctx.lineWidth = 2;
        ctx.setLineDash(dash === 'dashed' ? [4, 3] : dash === 'dotted' ? [1.5, 2.5] : []);
        ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
        ctx.setLineDash([]);
      }
    }
  }
  // The chosen head.
  const x = left + st.h * (cell + gap), y = top + st.l * (cell + gap);
  ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 2.5;
  ctx.strokeRect(x - 2, y - 2, cell + 4, cell + 4);
}
function pickHead(ev) {
  const r = grid.getBoundingClientRect();
  const px = ev.clientX - r.left, py = ev.clientY - r.top;
  const h = Math.floor((px - G.left) / (G.cell + G.gap)), l = Math.floor((py - G.top) / (G.cell + G.gap));
  if (h >= 0 && h < 12 && l >= 0 && l < 12) { st.l = l; st.h = h; drawGrid(); drawBig(); }
}
grid.addEventListener('click', pickHead);
grid.addEventListener('keydown', (e) => {
  const mv = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
  if (!mv) return;
  e.preventDefault();
  st.l = Math.max(0, Math.min(11, st.l + mv[0]));
  st.h = Math.max(0, Math.min(11, st.h + mv[1]));
  drawGrid(); drawBig();
});

function drawTypes() {
  const box = document.getElementById('types');
  box.hidden = !st.labels;
  if (!st.labels) return;
  const desc = { ind: 'outlined with a solid line', prev: 'outlined with a dashed line', dup: 'outlined with a dotted line' };
  box.innerHTML = TYPES.map((t) => {
    const heads = [];
    for (let l = 0; l < 12; l++) for (let h = 0; h < 12; h++) if (t.test(l, h)) heads.push(`${l}.${h}`);
    return `<p><span class="${t.style.includes('second') ? 'c-second' : ''}">${t.name}</span>, ${desc[t.key]}: ${heads.join(', ')}.</p>`;
  }).join('') + '<p style="color: var(--quiet)">Measured on 100 other random sequences of 20 tokens repeated twice; the thresholds are given below the tasks.</p>';
}

// ---- The chosen head, enlarged, and the input shaded by its weights ----
const big = document.getElementById('big');
const strip = document.getElementById('strip');
let B = null;
function drawBig() {
  const d = cur(), n = d.n;
  const avail = big.parentElement.clientWidth || 420;
  const size = Math.max(240, Math.min(420, avail));
  const ctx = sizedCanvas(big, size, size);
  const c = size / n;
  B = { size, c };
  drawPattern(ctx, d.att, st.l, st.h, 0, 0, size, palette('--ink'));
  ctx.strokeStyle = cssVar('--rule'); ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, size - 1, size - 1);
  if (d.key === 'random') { // the start of the second copy
    const b = (T + 1) * c;
    ctx.setLineDash([3, 3]); ctx.strokeStyle = cssVar('--quiet');
    ctx.beginPath(); ctx.moveTo(0, b); ctx.lineTo(size, b); ctx.moveTo(b, 0); ctx.lineTo(b, size); ctx.stroke(); ctx.setLineDash([]);
  }
  ctx.strokeStyle = cssVar('--accent'); ctx.lineWidth = 2;
  ctx.strokeRect(1, st.q * c, size - 2, Math.max(2, c));
  document.getElementById('big-title').textContent = `Head ${st.l}.${st.h}`;
  // The text, shaded by the weights from the chosen position.
  const ink = palette('--ink');
  const paper = cssVar('--paper'), inkc = cssVar('--ink');
  strip.innerHTML = d.tokens.map((t, j) => {
    const w = j <= st.q ? d.att(st.l, st.h, st.q, j) : 0;
    const bg = w > 0.004 ? ink(w) : 'transparent';
    const fg = Math.sqrt(w) > 0.55 ? paper : inkc;
    return `<span class="tk${j === st.q ? ' q' : ''}" data-j="${j}" style="background:${bg};color:${fg}" title="weight ${w.toFixed(2)}">${escapeHTML(showToken(t))}</span>`;
  }).join('');
  writeReadout();
}
function pickRow(ev) {
  const r = big.getBoundingClientRect();
  const q = Math.floor((ev.clientY - r.top) / B.c);
  if (q >= 0 && q < cur().n) { st.q = q; drawBig(); }
}
big.addEventListener('click', pickRow);
big.addEventListener('keydown', (e) => {
  const mv = { ArrowUp: -1, ArrowDown: 1 }[e.key];
  if (!mv) return;
  e.preventDefault();
  st.q = Math.max(0, Math.min(cur().n - 1, st.q + mv));
  drawBig();
});
strip.addEventListener('click', (e) => {
  const el = e.target.closest('.tk');
  if (el) { st.q = Number(el.dataset.j); drawBig(); }
});

// ---- Loss on each token ----
const lossBox = document.getElementById('loss');
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
function drawLoss() {
  const d = cur();
  lossBox.innerHTML = '';
  const c = chart(lossBox, { width: 760, height: 200, margin: { top: 30, right: 20, bottom: 52, left: 44 },
    x: { domain: [0.5, d.n - 0.5], ticks: 6, label: 'position' }, y: { domain: [0, 20], ticks: 4, label: 'loss' },
    title: 'Loss on each token of the input' });
  const bw = Math.max(1.5, c.x(1.5) - c.x(1) - 1);
  d.loss.forEach((v, i) => {
    if (v === null) return;
    c.g.append('rect').attr('x', c.x(i) - bw / 2).attr('width', bw).attr('y', c.y(Math.min(v, 20))).attr('height', c.y(0) - c.y(Math.min(v, 20)))
      .attr('fill', 'var(--ink)').attr('fill-opacity', 0.7);
  });
  if (d.key === 'random') c.vline('copy', T + 0.5, { label: 'second copy', labelY: -2 });
}

function writeReadout() {
  const d = cur();
  const row = d.tokens.map((_, j) => (j <= st.q ? d.att(st.l, st.h, st.q, j) : -1));
  const j = row.indexOf(Math.max(...row));
  const tk = (i) => `&ldquo;${escapeHTML(showToken(d.tokens[i]).trim() || d.tokens[i])}&rdquo;`;
  let s = `Head ${st.l}.${st.h}, from position ${st.q} (${tk(st.q)}): the largest weight, <b>${row[j].toFixed(2)}</b>, is on position ${j} (${tk(j)}).`;
  if (st.labels) s += ` Measured over 100 random repeated sequences, this head puts ${S.prev[st.l][st.h].toFixed(2)} of its weight on the previous token and ${S.ind[st.l][st.h].toFixed(2)} on the token after the earlier copy.`;
  const L = d.loss;
  if (d.key === 'random') {
    s += ` Average loss, not counting the first token of each copy: first copy <b>${mean(L.slice(2, T + 1)).toFixed(1)}</b>, second copy <b>${mean(L.slice(T + 2)).toFixed(2)}</b>.`;
  } else {
    s += ` Average loss over the text: <b>${mean(L.slice(1)).toFixed(2)}</b>.`;
  }
  readout.innerHTML = s;
}

function drawAll() { drawGrid(); drawBig(); drawLoss(); drawTypes(); figure(); }

// ---- Static figure: three heads on the random input ----
function figure() {
  const fig = document.getElementById('fig-heads');
  fig.innerHTML = '';
  const d = inputs[0];
  for (const [l, h, title] of [[4, 11, 'Head 4.11'], [5, 5, 'Head 5.5'], [1, 10, 'Head 1.10']]) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${title}</p>`;
    const cv = document.createElement('canvas');
    cv.setAttribute('aria-hidden', 'true');
    box.append(cv);
    fig.append(box);
    const size = Math.max(200, Math.min(260, box.clientWidth || 260));
    const ctx = sizedCanvas(cv, size, size);
    drawPattern(ctx, d.att, l, h, 0, 0, size, palette('--ink'));
    ctx.strokeStyle = cssVar('--rule'); ctx.strokeRect(0.5, 0.5, size - 1, size - 1);
  }
}

drawAll();
let rw;
window.addEventListener('resize', () => { clearTimeout(rw); rw = setTimeout(drawAll, 150); });
const mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
if (mq && mq.addEventListener) mq.addEventListener('change', drawAll);
