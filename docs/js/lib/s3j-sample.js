// Next-token distributions for page 3.3, from GPT-2 small's outputs computed in advance.
// Pure computations (no DOM), shared by the page and by tools/test-page-3-3.mjs.
//
// Each stored node is a prefix (the prompt plus some generated tokens). For it the file holds GPT-2's 20 most
// likely next tokens, their log probabilities at temperature 1, and for each temperature T on the grid the
// number off_T = log sum_j exp((z_j - z_max) / T) over all 50,257 scores z_j. With these, the probability of
// a stored token at temperature T is exact: log p_i(T) = (z_i - z_max) / T - off_T.

function b64bytes(s) {
  const bin = typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary');
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function decodeTree(t, k, nTemps) {
  const parent = new Int32Array(b64bytes(t.parent).buffer);
  const tok = new Uint16Array(b64bytes(t.tok).buffer);
  const ids = new Uint16Array(b64bytes(t.ids).buffer);
  const nlp = new Uint16Array(b64bytes(t.nlp).buffer);
  const off = new Uint16Array(b64bytes(t.off).buffer);
  const n = t.n;
  const children = Array.from({ length: n }, () => new Map());
  for (let i = 1; i < n; i++) children[parent[i]].set(tok[i], i);
  const depth = new Int32Array(n);
  for (let i = 1; i < n; i++) depth[i] = depth[parent[i]] + 1;
  return { n, k, nTemps, parent, tok, ids, nlp, off, children, depth };
}

// The k stored tokens of a node at temperature T (T = 0 means greedy) and top-p p.
// Returns { rows, kept, nucleusInside } where each row is
//   { id, rank, p1, pT, keep, stored }: p1 = probability at T = 1, pT = probability at temperature T
//   (over all 50,257 tokens), keep = inside the top-p set, stored = the next step was computed in advance.
export function distribution(tree, node, temps, T, p) {
  const k = tree.k;
  const lp = Array.from({ length: k }, (_, j) => -tree.nlp[node * k + j] / 1000);
  let pT;
  if (T === 0) pT = lp.map((_, j) => (j === 0 ? 1 : 0));
  else {
    const ti = temps.indexOf(T);
    if (ti < 0) throw new Error(`temperature ${T} is not on the grid`);
    const offT = tree.off[node * tree.nTemps + ti] / 5000;
    pT = lp.map((x) => Math.exp((x - lp[0]) / T - offT));
  }
  let cum = 0, kept = 0;
  const rows = lp.map((x, j) => {
    const id = tree.ids[node * k + j];
    const keep = T === 0 ? j === 0 : cum < p;
    cum += pT[j];
    if (keep) kept++;
    return { id, rank: j, p1: Math.exp(x), pT: pT[j], keep, stored: tree.children[node].has(id) };
  });
  // If the k stored tokens hold less than p, the top-p set reaches past them.
  const nucleusInside = T === 0 || cum >= p;
  return { rows, kept, nucleusInside };
}

// Probability (at temperature T) held by the tokens that the sampler may choose here: inside the top-p set
// and with the next step stored.
export function storedMass(dist) {
  return dist.rows.filter((r) => r.keep && r.stored).reduce((s, r) => s + r.pT, 0);
}

// Draws the next token among the eligible stored tokens, in proportion to their probability at temperature T.
// u is a uniform random number in [0, 1). Returns the row, or null if no eligible token has a stored next step.
export function draw(dist, u) {
  const el = dist.rows.filter((r) => r.keep && r.stored);
  const tot = el.reduce((s, r) => s + r.pT, 0);
  if (!el.length || tot <= 0) return null;
  let acc = 0;
  for (const r of el) { acc += r.pT / tot; if (u < acc) return r; }
  return el[el.length - 1];
}
