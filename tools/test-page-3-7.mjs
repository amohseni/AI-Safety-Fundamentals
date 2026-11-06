// Claims on page 3.7, checked against the recorded GPT-2 forward passes. Run: node tools/test-page-3-7.mjs
import fs from 'node:fs';
import path from 'node:path';
import { steps, shapeOf, paramsThrough, tensorAt } from '../docs/js/lib/s3j-gpt2.js';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const dir = path.join(root, 'docs/data/3-7');
const index = JSON.parse(fs.readFileSync(path.join(dir, 'index.json')));
const C = index.counts;
const S = steps();
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const n = (v) => v.toLocaleString('en-US');

// Parameter counts (read from the model by tools/3-7/build_data.py).
check('total 124,439,808', C.total === 124439808, n(C.total));
check('token embedding matrix 50,257 x 768 = 38,597,376', C.wte === 50257 * 768 && C.wte === 38597376, n(C.wte));
check('position embedding 1,024 x 768 = 786,432', C.wpe === 1024 * 768, n(C.wpe));
const attn = C.qkv + C.proj, mlp = C.fc + C.fc2, layer = attn + mlp + 2 * C.ln;
check('attention per layer 2,362,368', attn === 2362368, n(attn));
check('MLP per layer 4,722,432', mlp === 4722432, n(mlp));
check('layer total 7,087,872; norms 3,072', layer === 7087872 && 2 * C.ln === 3072, n(layer));
check('MLP share of a layer is about two thirds', Math.abs(mlp / layer - 2 / 3) < 0.01, (mlp / layer).toFixed(4));
check('final normalization 1,536', C.lnf === 1536, n(C.lnf));
check('unembedding is tied to the token embedding (no new parameters)', C.tied === true, String(C.tied));
check('parts add up to the total', C.wte + C.wpe + 12 * layer + C.lnf === C.total, n(C.wte + C.wpe + 12 * layer + C.lnf));
check('the step list counts every parameter once', paramsThrough(S, S.length - 1, C) === C.total, n(paramsThrough(S, S.length - 1, C)));
check('127 steps', S.length === 127, S.length);
{
  const biggest = Math.max(C.wte, C.wpe, C.qkv, C.proj, C.fc, C.fc2);
  check('largest single matrix is the token embedding; an MLP matrix is about 2.4 million', biggest === C.wte && Math.abs(C.fc - 2.4e6) < 0.05e6, `${n(biggest)}; MLP ${n(C.fc)}`);
  check('token embedding is almost a third of the model', Math.abs(C.wte / C.total - 0.31) < 0.01, (C.wte / C.total).toFixed(3));
}

// Shapes: the first step wider than 768 is the query/key/value step of layer 1 (2,304 = 3 x 12 x 64).
{
  const first = S.findIndex((s) => s.width && s.width > 768);
  check('first wider step: layer 1 queries, keys, and values, 2,304', S[first].key === 'qkv' && S[first].layer === 1 && S[first].width === 3 * 12 * 64, `${S[first].layer} ${S[first].name} ${S[first].width}`);
  const order = S.filter((s) => s.width > 768).map((s) => s.width);
  check('then the MLP widens to 3,072, then the unembedding to 50,257', order[1] === 3072 && order[order.length - 1] === 50257, [...new Set(order)].join(', '));
}

// Every recorded prompt: the shapes in the files match the step list; the scores match the model's output.
for (const p of index.prompts) {
  const rec = JSON.parse(fs.readFileSync(path.join(dir, p.file)));
  const k = rec.tokens.length;
  let ok = true;
  S.forEach((s, i) => {
    if (s.key === 'tokens') return;
    const t = tensorAt(rec, S, i);
    const want = shapeOf(s, k);
    if (s.key === 'logits' || s.key === 'probs') { if (rec[s.key].shape[0] !== k || rec[s.key].shape[1] !== 50257) ok = false; return; }
    if (t.shape.join() !== want.join()) ok = false;
    const size = t.shape.reduce((a, b) => a * b, 1);
    if (t.values.length !== size) ok = false;
  });
  check(`${p.text}: stored shapes match, n = ${k}`, ok, ok ? 'all steps' : 'mismatch');
  check(`${p.text}: hand-computed scores match the model's output`, p.check < 1e-3, p.check.toExponential(1));
  // Every position gets a full distribution: top probabilities are a distribution's head (sum below 1, positive).
  const sums = rec.top.map((row) => row.reduce((s, r) => s + r[3], 0));
  check(`${p.text}: a distribution at every one of the ${k} positions`, rec.top.length === k && sums.every((v) => v > 0 && v <= 1.0001), sums.map((v) => v.toFixed(2)).join(' '));
  // Attention weights: rows sum to about 1 and masked cells are 0.
  const L = tensorAt(rec, S, S.findIndex((s) => s.key === 'pattern' && s.layer === 1));
  let rowErr = 0, masked = 0;
  for (let h = 0; h < 12; h++) for (let i = 0; i < k; i++) {
    let s = 0;
    for (let j = 0; j < k; j++) { const v = L.values[h * k * k + i * k + j] / 255; s += v; if (j > i) masked += v; }
    rowErr = Math.max(rowErr, Math.abs(s - 1));
  }
  check(`${p.text}: attention rows sum to 1 (within rounding), masked cells 0`, rowErr < 0.03 && masked === 0, `max row error ${rowErr.toFixed(3)}`);
}

console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
