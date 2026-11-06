// One GPT-2 small forward pass, recorded in advance, for page 3.7: the list of steps, their shapes and
// parameter counts, and decoding of the stored tensors. Pure computations (no DOM), shared with
// tools/test-page-3-7.mjs.

function b64bytes(s) {
  const bin = typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary');
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// Stored tensors are rounded to 15 levels: -7..7, in units of clip / 7 (values beyond clip are cut off).
export function decode4(t) {
  const bytes = b64bytes(t.q);
  const size = t.shape.reduce((a, b) => a * b, 1);
  const out = new Int8Array(size);
  for (let i = 0; i < size; i++) {
    const b = bytes[i >> 1];
    out[i] = ((i & 1) ? (b & 15) : (b >> 4)) - 8;
  }
  return out;
}
export function decode8(s) { return b64bytes(s); }

// The steps of the forward pass. `key` names the stored tensor; `params` is the parameter count of the
// operation (C holds the counts read from the model); `group` says where those parameters sit.
export const LAYER_STEPS = [
  { key: 'ln1', name: 'Normalize', what: 'Layer normalization: each position’s vector is shifted to mean 0, scaled to variance 1, then scaled and shifted by learned numbers.', width: 768, params: (C) => C.ln, group: 'norm' },
  { key: 'qkv', name: 'Queries, keys, and values', what: 'One matrix multiplication (plus a bias) gives each position its queries, keys, and values for all 12 heads: 3 × 12 × 64 = 2,304 numbers.', width: 2304, params: (C) => C.qkv, group: 'attn' },
  { key: 'pattern', name: 'Attention weights', what: 'Each head takes the dot products of queries with keys, divides by √64 = 8, masks later positions, and applies softmax (Section 3.4).', width: null, params: () => 0, group: 'attn' },
  { key: 'attn_out', name: 'Weighted average and output matrix', what: 'Each head averages the values with its weights; the 12 results (12 × 64 = 768 numbers) are multiplied by the output matrix.', width: 768, params: (C) => C.proj, group: 'attn' },
  { key: 'resid_mid', name: 'Add to the residual stream', what: 'The attention output is added to the residual stream.', width: 768, params: () => 0, group: null },
  { key: 'ln2', name: 'Normalize', what: 'Layer normalization again, with its own learned numbers.', width: 768, params: (C) => C.ln, group: 'norm' },
  { key: 'mlp_pre', name: 'MLP, first matrix', what: 'Each position’s vector is multiplied by a 768 × 3,072 matrix, plus a bias.', width: 3072, params: (C) => C.fc, group: 'mlp' },
  { key: 'mlp_post', name: 'MLP, activation function', what: 'GELU, applied to each of the 3,072 numbers separately.', width: 3072, params: () => 0, group: 'mlp' },
  { key: 'mlp_out', name: 'MLP, second matrix', what: 'A 3,072 × 768 matrix, plus a bias, brings each position back to 768 numbers.', width: 768, params: (C) => C.fc2, group: 'mlp' },
  { key: 'resid_post', name: 'Add to the residual stream', what: 'The MLP output is added to the residual stream.', width: 768, params: () => 0, group: null },
];

export function steps() {
  const S = [
    { key: 'tokens', name: 'Split the text into tokens', what: 'The tokenizer turns the text into token ids (Section 3.1). It has no parameters.', width: 0, params: () => 0, group: null, layer: 0 },
    { key: 'tok_emb', name: 'Token embeddings', what: 'Each id selects one row of the token embedding matrix, 50,257 × 768 (Section 3.2).', width: 768, params: (C) => C.wte, group: 'wte', layer: 0 },
    { key: 'pos_emb', name: 'Position embeddings', what: 'Each position selects one row of the position embedding matrix, 1,024 × 768.', width: 768, params: (C) => C.wpe, group: 'wpe', layer: 0 },
    { key: 'resid0', name: 'Add them', what: 'The two embeddings are added. The sum is the residual stream at the start.', width: 768, params: () => 0, group: null, layer: 0 },
  ];
  for (let L = 1; L <= 12; L++) LAYER_STEPS.forEach((s) => S.push({ ...s, layer: L }));
  S.push({ key: 'lnf', name: 'Normalize', what: 'A final layer normalization.', width: 768, params: (C) => C.lnf, group: 'norm', layer: 13 });
  S.push({ key: 'logits', name: 'Unembedding', what: 'Each position’s vector is multiplied by the transpose of the token embedding matrix, giving one score for each of the 50,257 tokens. GPT-2 reuses the token embedding matrix here, so this step adds no parameters.', width: 50257, params: () => 0, group: 'wte', layer: 13 });
  S.push({ key: 'probs', name: 'Softmax', what: 'Softmax turns each position’s 50,257 scores into probabilities for the next token.', width: 50257, params: () => 0, group: null, layer: 13 });
  return S;
}

// Shape of a step's output for a prompt of n tokens, as an array.
export function shapeOf(step, n) {
  if (step.key === 'tokens') return [n];
  if (step.key === 'pattern') return [12, n, n];
  return [n, step.width];
}

// Parameters of the operations from the start up to and including step i, counting the token
// embedding matrix once although it is used twice.
export function paramsThrough(S, i, C) {
  let s = 0;
  for (let j = 0; j <= i; j++) s += S[j].params(C);
  return s;
}

// The tensor stored for step i, for display: { values (Int8Array, -7..7, or Uint8Array 0..255), shape, info }.
export function tensorAt(rec, S, i) {
  const st = S[i];
  if (st.key === 'tokens') return null;
  if (st.layer >= 1 && st.layer <= 12) {
    const L = rec.layers[st.layer - 1];
    if (st.key === 'pattern') {
      const n = rec.tokens.length;
      return { kind: 'pattern', values: decode8(L.pattern), shape: [12, n, n] };
    }
    const t = L[st.key];
    return { kind: 'signed', values: decode4(t), shape: t.shape, info: t };
  }
  if (st.key === 'logits') return { kind: 'bins', values: decode8(rec.logits.q), shape: [rec.tokens.length, rec.logits.bins], info: rec.logits };
  if (st.key === 'probs') return { kind: 'bins', values: decode8(rec.probs.q), shape: [rec.tokens.length, rec.probs.bins], info: rec.probs };
  const t = rec[st.key];
  return { kind: 'signed', values: decode4(t), shape: t.shape, info: t };
}
