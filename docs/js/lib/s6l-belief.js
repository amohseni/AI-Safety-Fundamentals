// Probes for belief and their scorecard, page 7.2. No DOM; the Node tests use the same code.
// T is the decoded shared dataset (s6l-truth.js). Probes are trained on the training half of the
// city statements (even-numbered items) and scored on the held-out half and on other topics.
import { logistic, ccs } from './s6l-probe.js';

export const TRAIN_SETS = {
  supervised: [['plain', 'Plain'], ['neg', 'Negated'], ['mixed', 'Plain and negated']],
  ccs: [['plain', 'Plain'], ['alice', "With Alice's opinion"]],
};
export const SUP_SETTINGS = { l2: 0.01, steps: 300, lr: 0.05 };
export const CCS_SETTINGS = { steps: 400, lr: 0.01, restarts: 10, seed: 1 };

// The dataset lists each claim as three consecutive statements: plain, negated, paraphrased
// (tools/truth/make_statements.py). This finds the partner of a plain statement in another form.
function partnerIndex(T) {
  const off = { plain: 0, neg: 1, para: 2 };
  return (s, form) => { const j = s.i + off[form]; const t = T.S[j]; return t && t.form === form && t.item === s.item && t.topic === s.topic ? j : undefined; };
}

// Returns { credence(statement) -> number in [0, 1] or null, w (direction), basis, info }.
export function trainProbe(T, method, set) {
  const cityTrain = T.S.filter((s) => s.topic === 'cities' && T.isTrain(s));
  if (method === 'supervised') {
    const forms = set === 'mixed' ? ['plain', 'neg'] : [set];
    const tr = cityTrain.filter((s) => forms.includes(s.form));
    const p = logistic(tr.map((s) => T.X[s.i]), tr.map((s) => s.label), SUP_SETTINGS);
    return { method, set, w: p.w, basis: 'statements', credence: (s) => p.prob(T.X[s.i]), score: (s) => p.score(T.X[s.i]) };
  }
  if (set === 'plain') {
    const tr = cityTrain.filter((s) => s.form === 'plain');
    const p = ccs(tr.map((s) => T.pos[s.i]), tr.map((s) => T.neg[s.i]), CCS_SETTINGS);
    // CCS uses no labels; its sign is arbitrary. We set it so that it agrees with the training labels.
    let sign = 1;
    const agree = tr.filter((s) => (p.credence(T.pos[s.i], T.neg[s.i]) > 0.5 ? 1 : 0) === s.label).length;
    if (agree < tr.length / 2) sign = -1;
    const cred = (s) => { const c = p.credence(T.pos[s.i], T.neg[s.i]); return sign > 0 ? c : 1 - c; };
    return { method, set, w: Float32Array.from(p.w, (v) => sign * v), basis: 'pairs', credence: cred, loss: p.loss };
  }
  // Alice: the pairs are prefixed with Alice's opinion; stored for plain statements only.
  const tr = cityTrain.filter((s) => s.form === 'plain');
  const rowOf = (s) => T.aliceRow.get(s.i);
  const p = ccs(tr.map((s) => T.alicePos[rowOf(s)]), tr.map((s) => T.aliceNeg[rowOf(s)]), CCS_SETTINGS);
  let sign = 1;
  const agree = tr.filter((s) => (p.credence(T.alicePos[rowOf(s)], T.aliceNeg[rowOf(s)]) > 0.5 ? 1 : 0) === s.label).length;
  if (agree < tr.length / 2) sign = -1;
  const cred = (s) => { const r = rowOf(s); if (r === undefined) return null; const c = p.credence(T.alicePos[r], T.aliceNeg[r]); return sign > 0 ? c : 1 - c; };
  return { method, set, w: Float32Array.from(p.w, (v) => sign * v), basis: 'alice', credence: cred, loss: p.loss };
}

function acc(probe, list) {
  let n = 0, c = 0;
  for (const s of list) { const p = probe.credence(s); if (p === null) return null; n++; if ((p > 0.5 ? 1 : 0) === s.label) c++; }
  return n ? c / n : null;
}

export function scorecard(T, probe) {
  const partner = partnerIndex(T);
  const held = (topic, form) => T.S.filter((s) => s.topic === topic && s.form === form && !T.isTrain(s));
  const all = (topic, form) => T.S.filter((s) => s.topic === topic && s.form === form);
  const out = {
    truth: acc(probe, held('cities', 'plain')),
    neg: acc(probe, held('cities', 'neg')),
    spanish: acc(probe, all('spanish', 'plain')),
    elements: acc(probe, all('elements', 'plain')),
    spanishNeg: acc(probe, all('spanish', 'neg')),
    elementsNeg: acc(probe, all('elements', 'neg')),
    para: null, sum: null, alice: null,
  };
  // Paraphrase agreement and negation sums over held-out city items.
  const plain = held('cities', 'plain');
  let agree = 0, n = 0, dev = 0, nd = 0, sumMean = 0;
  for (const s of plain) {
    const a = probe.credence(s);
    const pi = partner(s, 'para'), ni = partner(s, 'neg');
    const cp = pi === undefined ? null : probe.credence(T.S[pi]);
    const cn = ni === undefined ? null : probe.credence(T.S[ni]);
    if (a !== null && cp !== null) { n++; if ((a > 0.5) === (cp > 0.5)) agree++; }
    if (a !== null && cn !== null) { nd++; dev += Math.abs(a + cn - 1); sumMean += a + cn; }
  }
  out.para = n ? agree / n : null;
  out.sum = nd ? dev / nd : null;
  out.sumMean = nd ? sumMean / nd : null;
  if (probe.set === 'alice') {
    const list = held('cities', 'plain');
    out.alice = list.filter((s) => (probe.credence(s) > 0.5 ? 1 : 0) === s.alice).length / list.length;
  }
  return out;
}

export function allProbeKeys() {
  return Object.entries(TRAIN_SETS).flatMap(([m, sets]) => sets.map(([k]) => `${m}-${k}`));
}
