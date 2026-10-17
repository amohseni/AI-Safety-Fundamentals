// Modular-addition runs: loading the precomputed training records and drawing them.
// The data were made by tools/modadd (see its README). Used by pages 5.4, 5.5, and 6.7.
import { cssVar } from './ui.js';

const BASE = new URL('../../data/modadd/', import.meta.url);
const cache = new Map();

export function loadJSON(name) {
  if (!cache.has(name)) {
    cache.set(name, fetch(new URL(name, BASE)).then((r) => {
      if (!r.ok) throw new Error(`Could not load ${name}`);
      return r.json();
    }));
  }
  return cache.get(name);
}

export const runFile = (wd, frac) => `run-wd${Number(wd).toFixed(1)}-f${Number(frac).toFixed(1)}.json`;
export const loadRun = (wd, frac) => loadJSON(runFile(wd, frac));
export const loadSeeds = () => loadJSON('seeds.json');
export const loadEmbeddings = () => loadJSON('seeds-embed.json');

// Base64 string to bytes.
export function bytes(b64) {
  const s = atob(b64), out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

// Packed bits (numpy.packbits order: first item in the high bit) to an array of 0/1.
export function bits(b64, n) {
  const B = bytes(b64), out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (B[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}

// Float32 little-endian embedding matrix, p rows of d values.
export function embedding(b64, p, d) {
  const B = bytes(b64);
  const f = new Float32Array(B.buffer, B.byteOffset, p * d);
  return Array.from({ length: p }, (_, a) => f.subarray(a * d, (a + 1) * d));
}

// Share of the squared embedding norm held by the n largest frequencies.
export function topShare(spectrum, n) {
  const sq = spectrum.map((v) => v * v);
  const total = sq.reduce((s, v) => s + v, 0);
  const top = sq.slice().sort((a, b) => b - a).slice(0, n).reduce((s, v) => s + v, 0);
  return total > 0 ? top / total : 0;
}

// Draws the p x p grid of pairs (a down, b across) on a canvas.
// Pairs the model answers correctly are filled: training pairs in --second, test pairs in --accent.
// Pairs answered wrongly are drawn in --faint. Returns the drawn cell size in CSS pixels.
export function drawPairGrid(canvas, p, correct, trainMask) {
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth;
  const cell = W / p;
  if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(W * dpr); }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, W);
  const colTrain = cssVar('--second'), colTest = cssVar('--accent');
  ctx.fillStyle = cssVar('--faint');
  ctx.fillRect(0, 0, W, W);
  for (const [want, col] of [[1, colTrain], [0, colTest]]) {
    ctx.fillStyle = col;
    ctx.beginPath();
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      const i = a * p + b;
      if (correct[i] && trainMask[i] === want) ctx.rect(b * cell, a * cell, cell + 0.02, cell + 0.02);
    }
    ctx.fill();
  }
  return cell;
}
