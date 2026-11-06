// Probing tasks for page 6.3, built from docs/data/6-3/probe-data.json. No DOM.
// Three properties: the truth of a city statement, whether a token is a noun, and whether a
// token lies inside quotation marks. Each comes with a control task in the manner of Hewitt and
// Liang (2019): a random label fixed for each word type (for statements, for each city), so a
// probe can only fit it by memorizing which word (or city) it is looking at.
import { decodeF16, rows, rng } from './s6l-probe.js';

export const PROPERTIES = {
  truth: 'Truth of a city statement',
  noun: 'Noun',
  quote: 'Inside quotation marks',
};

function decodeI8(b64, clip) {
  const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
  const out = new Float32Array(bin.length);
  for (let i = 0; i < bin.length; i++) { const b = bin.charCodeAt(i); out[i] = ((b > 127 ? b - 256 : b) * clip) / 127; }
  return out;
}

export function decodeProbeData(json) {
  const k = json.k;
  const layers = json.layers;
  const X = {};
  for (const L of layers) {
    const d = json.data[L];
    X[L] = { truth: rows(decodeF16(d.cities), k), tok: rows(decodeI8(d.tokens, json.tokenClip), k) };
  }
  // ---- Truth: control label per city ----
  const r = rng(11);
  const C = json.cities;
  const cityCtl = new Map();
  C.forEach((c) => { if (!cityCtl.has(c.item)) cityCtl.set(c.item, r() < 0.5 ? 1 : 0); });
  // Half of the true and half of the false statements go to training, chosen at random, so a
  // city's two statements can fall on different sides of the split.
  const truthTrain = new Array(C.length).fill(false);
  for (const lab of [0, 1]) {
    const ids = C.map((c, i) => [r(), i, c.label]).filter((t) => t[2] === lab).sort((a, b) => a[0] - b[0]);
    ids.slice(0, ids.length / 2).forEach(([, i]) => { truthTrain[i] = true; });
  }
  // ---- Tokens: training on even-numbered passages, testing on odd ones ----
  const toks = json.tokens;
  function tokenTask(prop, seed) {
    const rr = rng(seed);
    // Within the training passages and within the test passages separately, keep all of the
    // rarer class and an equal random sample of the other, so both sets are balanced.
    let idx = [];
    for (const half of [0, 1]) {
      const pos = [], neg = [];
      toks.forEach((t, i) => { if (t.p % 2 === half) (t[prop] ? pos : neg).push(i); });
      const [few, many] = pos.length <= neg.length ? [pos, neg] : [neg, pos];
      const sample = many.map((i) => [rr(), i]).sort((a, b) => a[0] - b[0]).map(([, i]) => i).slice(0, few.length);
      idx = idx.concat(few, sample);
    }
    idx.sort((a, b) => a - b);
    // Control: a random label for each word type, 1 with probability one half (the classes are balanced).
    const types = new Map();
    const rc = rng(seed + 100);
    for (const i of idx) { const w = toks[i].w; if (!types.has(w)) types.set(w, rc() < 0.5 ? 1 : 0); }
    return {
      idx, y: idx.map((i) => toks[i][prop]), yc: idx.map((i) => types.get(toks[i].w)),
      train: idx.map((i) => toks[i].p % 2 === 0), words: idx.map((i) => toks[i].w),
    };
  }
  const tasks = {
    truth: { idx: C.map((_, i) => i), y: C.map((c) => c.label), yc: C.map((c) => cityCtl.get(c.item)), train: truthTrain, words: C.map((c) => String(c.item)) },
    noun: tokenTask('noun', 21),
    quote: tokenTask('quote', 31),
  };
  // Rows of activations for a property at a layer.
  const acts = (prop, L) => { const src = prop === 'truth' ? X[L].truth : X[L].tok; return tasks[prop].idx.map((i) => src[i]); };
  return { k, layers, tasks, acts, json };
}

// Splits a task's rows into training and test sets, with real and control labels.
export function split(task, A) {
  const s = { Xtr: [], ytr: [], yctr: [], Xte: [], yte: [], ycte: [] };
  task.idx.forEach((_, j) => {
    if (task.train[j]) { s.Xtr.push(A[j]); s.ytr.push(task.y[j]); s.yctr.push(task.yc[j]); } else { s.Xte.push(A[j]); s.yte.push(task.y[j]); s.ycte.push(task.yc[j]); }
  });
  return s;
}
