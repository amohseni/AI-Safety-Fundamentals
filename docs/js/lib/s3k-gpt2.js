// Viewers for precomputed GPT-2 small data (pages 3.5, 3.6, 6.6): loading, attention
// patterns packed as lower triangles of bytes, heat maps on canvas, and token display.
import { cssVar } from './ui.js';

export async function loadJSON(rel, base) {
  const r = await fetch(new URL(rel, base));
  if (!r.ok) throw new Error(`${rel}: ${r.status}`);
  return r.json();
}

// Attention for all 144 heads of one input, stored as 12 x 12 lower triangles of bytes
// (weight times 255, rounded). Row r of a triangle holds the weights from position r to
// positions 0..r. Returns att(layer, head, from, to) in [0, 1].
export function unpackAttention(b64, n) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const tri = (n * (n + 1)) / 2;
  if (bytes.length !== 144 * tri) throw new Error('attention data has the wrong size');
  const rowStart = (r) => (r * (r + 1)) / 2;
  const att = (l, h, from, to) => (to > from ? 0 : bytes[(l * 12 + h) * tri + rowStart(from) + to] / 255);
  att.n = n;
  return att;
}

// Mixes the paper color toward a token color by weight w in [0, 1], for canvas fills.
function parseColor(s) {
  s = s.trim();
  if (s.startsWith('#')) {
    const h = s.length === 4 ? s.slice(1).split('').map((c) => c + c).join('') : s.slice(1);
    return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  }
  const m = s.match(/\d+(\.\d+)?/g);
  return m ? m.slice(0, 3).map(Number) : [0, 0, 0];
}
export function palette(token = '--ink') {
  const paper = parseColor(cssVar('--paper') || '#fffff8');
  const ink = parseColor(cssVar(token) || '#111111');
  // A square root makes small weights visible without hiding the large ones.
  return (w) => {
    const t = Math.sqrt(Math.max(0, Math.min(1, w)));
    const c = paper.map((p, i) => Math.round(p + (ink[i] - p) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  };
}

// Draws one head's pattern as an n x n heat map into a square of side `size` at (x, y).
export function drawPattern(ctx, att, l, h, x, y, size, color) {
  const n = att.n;
  const c = size / n;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const w = att(l, h, i, j);
      if (w < 0.004) continue;
      ctx.fillStyle = color(w);
      ctx.fillRect(x + j * c, y + i * c, Math.ceil(c), Math.ceil(c));
    }
  }
}

// A canvas drawn at its true pixel width, sharp on high-density screens.
export function sizedCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return ctx;
}

// Token text for display: the end-of-text token is shown by name, newlines as a symbol.
export function showToken(t) {
  if (t === '<|endoftext|>') return '⟨start⟩';
  return t.replace(/\n/g, '⏎');
}

export function escapeHTML(s) {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
