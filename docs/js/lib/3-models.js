// Pure computations behind the Section 3 tools (no DOM), shared by the pages and by tools/test-page-3-*.mjs.

// ---- Byte-pair encoding on a tiny corpus (page 3.1) ----
// Characters stand in for bytes so the merges stay readable. As in GPT-2, the text is first split into
// words that keep their leading space, and merges never cross a word boundary.
export function pretokenize(text) {
  return text.match(/ ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/gu) || [];
}

export class TinyBPE {
  constructor(text) {
    this.pieces = pretokenize(text);
    const counts = new Map();
    for (const p of this.pieces) counts.set(p, (counts.get(p) || 0) + 1);
    this.words = new Map([...counts.keys()].map((w) => [w, [...w]]));
    this.counts = counts;
    this.merges = [];
    this.base = new Set([...text]).size;
  }
  // Adjacent pairs with their counts, most frequent first; ties go to the pair that appears first in the text.
  pairs() {
    const n = new Map(), first = new Map();
    let order = 0;
    for (const p of this.pieces) {
      const t = this.words.get(p);
      for (let i = 0; i < t.length - 1; i++) {
        const k = t[i] + '\u0000' + t[i + 1];
        n.set(k, (n.get(k) || 0) + 1);
        if (!first.has(k)) first.set(k, order++);
      }
    }
    return [...n.entries()].map(([k, c]) => { const [a, b] = k.split('\u0000'); return { a, b, n: c, o: first.get(k) }; })
      .sort((x, y) => y.n - x.n || x.o - y.o);
  }
  merge() {
    const top = this.pairs()[0];
    if (!top) return null;
    const { a, b } = top;
    for (const [w, t] of this.words) {
      const out = [];
      for (let i = 0; i < t.length; i++) {
        if (i < t.length - 1 && t[i] === a && t[i + 1] === b) { out.push(a + b); i++; } else out.push(t[i]);
      }
      this.words.set(w, out);
    }
    this.merges.push(top);
    return top;
  }
  // The corpus as a list of tokens, in order.
  tokens() { return this.pieces.flatMap((p) => this.words.get(p)); }
  vocabSize() { return this.base + this.merges.length; }
}

// ---- Attention for one head with 2D queries and keys (page 3.4) ----
// q, k: arrays of [x, y]; returns scores (null where masked) and softmax weights, row i = position i.
export function attention(q, k, { causal = true, scale = true } = {}) {
  const n = q.length, d = q[0].length, s = scale ? 1 / Math.sqrt(d) : 1;
  const scores = [], weights = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) row.push(causal && j > i ? null : s * (q[i][0] * k[j][0] + q[i][1] * k[j][1]));
    const m = Math.max(...row.filter((v) => v !== null));
    const e = row.map((v) => (v === null ? 0 : Math.exp(v - m)));
    const z = e.reduce((a, b) => a + b, 0);
    scores.push(row);
    weights.push(e.map((v) => v / z));
  }
  return { scores, weights };
}
export function mix(w, values) {
  return values[0].map((_, c) => values.reduce((s, v, j) => s + w[j] * v[c], 0));
}

// ---- The Chinchilla loss fit (page 3.8) ----
// Hoffmann et al. (2022), Approach 3: L(N, D) = E + A / N^alpha + B / D^beta. The paper's text prints
// E = 1.69, alpha = 0.34, beta = 0.28; these are the unrounded values from its source, as reported by
// Besiroglu et al. (2024). They reproduce the paper's own figures (a = 0.46; 40B parameters at Gopher's budget).
export const FIT = { E: 1.6934, A: 406.4, B: 410.7, alpha: 0.3392, beta: 0.2849 };
export const loss = (N, D, f = FIT) => f.E + f.A / N ** f.alpha + f.B / D ** f.beta;
// Training tokens affordable with budget C for a model of N parameters, after serving S tokens
// (training costs about 6ND FLOPs; serving about 2N FLOPs per token).
export const tokensFor = (C, N, S = 0) => C / (6 * N) - S / 3;
// Closed-form compute-optimal size without serving cost: N_opt = G (C/6)^a, a = beta/(alpha+beta).
export function optimum(C, f = FIT) {
  const a = f.beta / (f.alpha + f.beta);
  const G = ((f.alpha * f.A) / (f.beta * f.B)) ** (1 / (f.alpha + f.beta));
  const N = G * (C / 6) ** a;
  return { N, D: C / 6 / N, a };
}
// Numerical optimum over log N, valid with or without serving cost.
export function optimumNumeric(C, S = 0, f = FIT, lo = 6, hi = 14, step = 0.001) {
  let best = { N: NaN, D: NaN, L: Infinity };
  for (let lg = lo; lg <= hi; lg += step) {
    const N = 10 ** lg, D = tokensFor(C, N, S);
    if (D <= 0) break;
    const L = loss(N, D, f);
    if (L < best.L) best = { N, D, L };
  }
  return best;
}
