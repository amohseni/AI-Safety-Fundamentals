// Word vectors for page 3.2: decoding, nearest neighbors by cosine similarity, analogies, and PCA.
// Pure computations (no DOM), shared by the page and by tools/test-page-3-2.mjs.

function b64bytes(s) {
  const bin = typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary');
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// Turns the stored file (signed bytes with one scale per word) into unit vectors.
export function decodeVectors(doc) {
  const n = doc.words.length, d = doc.dim;
  const q = new Int8Array(b64bytes(doc.q).buffer);
  const X = new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) { const v = q[i * d + j] * doc.scale[i]; X[i * d + j] = v; s += v * v; }
    const r = 1 / Math.sqrt(s);
    for (let j = 0; j < d; j++) X[i * d + j] *= r;
  }
  const index = new Map(doc.words.map((w, i) => [w, i]));
  return { n, d, X, words: doc.words, index, pca: doc.pca, umap: doc.umap };
}

export function row(E, i) { return E.X.subarray(i * E.d, (i + 1) * E.d); }

// Cosine similarity of every stored word with the vector v (v need not be unit length).
export function similarities(E, v) {
  let nv = 0;
  for (let j = 0; j < E.d; j++) nv += v[j] * v[j];
  nv = Math.sqrt(nv) || 1;
  const s = new Float32Array(E.n);
  for (let i = 0; i < E.n; i++) {
    let t = 0;
    const o = i * E.d;
    for (let j = 0; j < E.d; j++) t += E.X[o + j] * v[j];
    s[i] = t / nv;
  }
  return s;
}

// The k words with the highest cosine similarity to v, leaving out the indices in `exclude`.
export function nearest(E, v, k = 10, exclude = []) {
  const s = similarities(E, v);
  const ex = new Set(exclude);
  const order = [];
  for (let i = 0; i < E.n; i++) if (!ex.has(i)) order.push(i);
  order.sort((a, b) => s[b] - s[a]);
  return order.slice(0, k).map((i) => ({ i, word: E.words[i], sim: s[i] }));
}

// a - b + c with unit vectors (the "3CosAdd" rule of Mikolov et al.), top k answers.
// With exclude = true the three input words cannot be answers, as in the standard analogy test.
export function analogy(E, a, b, c, { k = 5, exclude = true } = {}) {
  const [ia, ib, ic] = [a, b, c].map((w) => E.index.get(w));
  if ([ia, ib, ic].some((i) => i === undefined)) return null;
  const v = new Float32Array(E.d);
  const A = row(E, ia), B = row(E, ib), C = row(E, ic);
  for (let j = 0; j < E.d; j++) v[j] = A[j] - B[j] + C[j];
  return nearest(E, v, k, exclude ? [ia, ib, ic] : []);
}

// First two principal components of the given rows (centered), by power iteration on the small Gram matrix.
export function pca2(E, ids) {
  const m = ids.length, d = E.d;
  const mean = new Float64Array(d);
  ids.forEach((i) => { const r = row(E, i); for (let j = 0; j < d; j++) mean[j] += r[j] / m; });
  const C = ids.map((i) => { const r = row(E, i); return Float64Array.from(r, (x, j) => x - mean[j]); });
  const G = C.map((a) => C.map((b) => a.reduce((s, x, j) => s + x * b[j], 0)));
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let u = Array.from({ length: m }, (_, i) => Math.cos(i + 1 + c));
    for (let it = 0; it < 500; it++) {
      let w = G.map((r) => r.reduce((s, x, j) => s + x * u[j], 0));
      comps.forEach((p) => { const dot = w.reduce((s, x, j) => s + x * p[j], 0); w = w.map((x, j) => x - dot * p[j]); });
      const nw = Math.hypot(...w);
      u = w.map((x) => x / nw);
    }
    comps.push(u);
  }
  // Coordinates of row i on component c: (C C^T u)_i / sqrt(eigenvalue) = sqrt(eigenvalue) * u_i.
  return comps.map((u) => {
    const Gu = G.map((r) => r.reduce((s, x, j) => s + x * u[j], 0));
    const lam = Gu.reduce((s, x, j) => s + x * u[j], 0);
    return u.map((x) => x * Math.sqrt(lam));
  });
}

// How many of the given neighbor indices are among the k points nearest to point i on a 2-D map P
// (Euclidean distance on the map, point i itself left out).
export function mapNeighbors(P, i, neighbors, k = neighbors.length) {
  const d = P.map((p, j) => [j, (p[0] - P[i][0]) ** 2 + (p[1] - P[i][1]) ** 2]).filter(([j]) => j !== i);
  d.sort((a, b) => a[1] - b[1]);
  const near = new Set(d.slice(0, k).map(([j]) => j));
  return neighbors.filter((j) => near.has(j)).length;
}
