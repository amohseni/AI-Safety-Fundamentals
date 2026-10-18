// The trained modular-addition transformer of the main run (p = 59, weight decay 1, 30% of pairs),
// run in the browser for page 6.7. Parameters: docs/data/6-7/model.json, written by
// tools/modadd/export_params.py. Works in Node too (tools/test-page-6-7.mjs).
//
// The model (tools/modadd/modadd.py): input "a b =", token and position embeddings, one attention
// layer with 4 heads, a ReLU MLP of 512 neurons, the unembedding. No layer norm. The answer is read
// at the "=" position, so only that position's residual stream is computed after the attention.

const b64bytes = (s) => {
  if (typeof atob === 'function') {
    const t = atob(s), out = new Uint8Array(t.length);
    for (let i = 0; i < t.length; i++) out[i] = t.charCodeAt(i);
    return out;
  }
  return new Uint8Array(Buffer.from(s, 'base64'));
};

// float16 (IEEE half, little-endian) to float64.
function halfs(s) {
  const B = b64bytes(s), n = B.length / 2, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const h = B[2 * i] | (B[2 * i + 1] << 8);
    const sgn = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
    out[i] = e === 0 ? sgn * f * 2 ** -24 : e === 31 ? (f ? NaN : sgn * Infinity) : sgn * (1 + f / 1024) * 2 ** (e - 15);
  }
  return out;
}

export function unpackBits(s, n) {
  const B = b64bytes(s), out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = (B[i >> 3] >> (7 - (i & 7))) & 1;
  return out;
}

// Orthonormal real Fourier basis over 0..p-1 as rows: row 0 constant, rows 2k-1 and 2k cos and sin at k.
export function fourierBasis(p) {
  const rows = [new Float64Array(p).fill(1 / Math.sqrt(p))];
  for (let k = 1; k <= (p - 1) / 2; k++) {
    const c = new Float64Array(p), s = new Float64Array(p);
    for (let a = 0; a < p; a++) {
      c[a] = Math.cos((2 * Math.PI * k * a) / p) * Math.sqrt(2 / p);
      s[a] = Math.sin((2 * Math.PI * k * a) / p) * Math.sqrt(2 / p);
    }
    rows.push(c, s);
  }
  return rows;
}

export class ModAdd {
  constructor(json) {
    const { p, d_model: d, heads: H, d_head: dh, d_mlp: m } = json;
    Object.assign(this, { p, d, H, dh, m });
    const P = {};
    for (const [k, s] of Object.entries(json.params)) P[k] = halfs(s);
    this.P = P;
    this.F = fourierBasis(p);
    this.trainMask = unpackBits(json.train_mask, p * p);
    this.neuronFreq = Int32Array.from(json.neuron_freq);
    this.neuronShare = Float64Array.from(json.neuron_share);
    // W_out W_U: each neuron's direct effect on the 59 logits.
    const WoU = new Float64Array(m * p);
    for (let n = 0; n < m; n++) for (let j = 0; j < d; j++) {
      const w = P.W_out[n * d + j];
      if (w) for (let c = 0; c < p; c++) WoU[n * p + c] += w * P.W_U[j * p + c];
    }
    this.WoU = WoU;
    this.bU = this.matvecU(P.b_out); // b_out W_U
    this.setEmbedding(null);
  }

  matvecU(x) { // x (d) times W_U (d x p)
    const { d, p } = this, U = this.P.W_U, out = new Float64Array(p);
    for (let j = 0; j < d; j++) { const v = x[j]; if (v) for (let c = 0; c < p; c++) out[c] += v * U[j * p + c]; }
    return out;
  }

  // The number embeddings: rows 0..p-1 of W_E, or a replacement (array of p rows) for ablations.
  embedRow(a) { return this.WE ? this.WE[a] : this.P.W_E.subarray(a * this.d, (a + 1) * this.d); }

  // Precompute, for each number at positions 0 and 1 and for "=" at position 2, the residual stream
  // vector, its keys, and its values passed through W_O. Called again when the embedding changes.
  setEmbedding(rows) {
    this.WE = rows;
    const { p, d, H, dh, P } = this;
    const vec = (tok, pos) => {
      const x = new Float64Array(d), e = tok === p ? P.W_E.subarray(p * d, (p + 1) * d) : this.embedRow(tok);
      for (let j = 0; j < d; j++) x[j] = e[j] + P.W_pos[pos * d + j];
      return x;
    };
    const proj = (x, W, h) => { // x (d) times W[h] (d x dh)
      const out = new Float64Array(dh), off = h * d * dh;
      for (let j = 0; j < d; j++) { const v = x[j]; for (let e = 0; e < dh; e++) out[e] += v * W[off + j * dh + e]; }
      return out;
    };
    const ovo = (x, h) => { // value of x through head h, then W_O[h]: a d-vector
      const v = proj(x, P.W_V, h), out = new Float64Array(d), off = h * dh * d;
      for (let e = 0; e < dh; e++) for (let j = 0; j < d; j++) out[j] += v[e] * P.W_O[off + e * d + j];
      return out;
    };
    const prep = (tok, pos) => {
      const x = vec(tok, pos);
      const k = [], vo = [];
      for (let h = 0; h < H; h++) { k.push(proj(x, P.W_K, h)); vo.push(ovo(x, h)); }
      return { x, k, vo };
    };
    this.pos = [[], []];
    for (let a = 0; a < p; a++) { this.pos[0].push(prep(a, 0)); this.pos[1].push(prep(a, 1)); }
    this.eq = prep(p, 2);
    this.q = [];
    for (let h = 0; h < H; h++) this.q.push(proj(this.eq.x, P.W_Q, h));
  }

  // One forward pass on "a b =". `patch` copies activations of the run `clean` into this run:
  //   { heads: [h, ...] }  the outputs of these heads (after W_O)
  //   { mlp: true }        the MLP output
  //   { neurons: mask }    the activations of the neurons with mask[n] = 1
  // Returns the attention pattern, head outputs, MLP neuron activations, MLP output, and logits.
  run(a, b, patch = null, clean = null) {
    const { d, H, dh, m, p, P } = this;
    const src = [this.pos[0][a], this.pos[1][b], this.eq];
    const att = new Float64Array(H * 3), ho = new Float64Array(H * d);
    for (let h = 0; h < H; h++) {
      const s = src.map((t) => { let v = 0; for (let e = 0; e < dh; e++) v += this.q[h][e] * t.k[h][e]; return v / Math.sqrt(dh); });
      const mx = Math.max(...s), ex = s.map((v) => Math.exp(v - mx)), Z = ex[0] + ex[1] + ex[2];
      for (let i = 0; i < 3; i++) att[h * 3 + i] = ex[i] / Z;
      for (let i = 0; i < 3; i++) { const w = att[h * 3 + i], vo = src[i].vo[h]; for (let j = 0; j < d; j++) ho[h * d + j] += w * vo[j]; }
    }
    if (patch && patch.heads) for (const h of patch.heads) for (let j = 0; j < d; j++) ho[h * d + j] = clean.ho[h * d + j];
    const rmid = Float64Array.from(this.eq.x);
    for (let h = 0; h < H; h++) for (let j = 0; j < d; j++) rmid[j] += ho[h * d + j];
    const act = new Float64Array(m);
    for (let n = 0; n < m; n++) act[n] = P.b_in[n];
    for (let j = 0; j < d; j++) { const v = rmid[j]; for (let n = 0; n < m; n++) act[n] += v * P.W_in[j * m + n]; }
    for (let n = 0; n < m; n++) if (act[n] < 0) act[n] = 0;
    if (patch && patch.neurons) for (let n = 0; n < m; n++) if (patch.neurons[n]) act[n] = clean.act[n];
    let mo = new Float64Array(d);
    for (let j = 0; j < d; j++) mo[j] = P.b_out[j];
    for (let n = 0; n < m; n++) { const v = act[n]; if (v) for (let j = 0; j < d; j++) mo[j] += v * P.W_out[n * d + j]; }
    if (patch && patch.mlp) mo = Float64Array.from(clean.mo);
    const r = new Float64Array(d);
    for (let j = 0; j < d; j++) r[j] = rmid[j] + mo[j];
    return { att, ho, act, mo, logits: this.matvecU(r) };
  }

  // Activation patching: clean pair, corrupted pair, and what to patch from the clean run.
  // The logit difference is the logit of the clean answer minus the logit of the corrupted answer.
  // restored = (patched - corrupted) / (clean - corrupted): 0 means no effect, 1 means fully restored.
  patch([a, b], [a2, b2], patch) {
    const p = this.p, c = (a + b) % p, c2 = (a2 + b2) % p;
    const clean = this.run(a, b), corr = this.run(a2, b2), pt = this.run(a2, b2, patch, clean);
    const ld = (o) => o.logits[c] - o.logits[c2];
    const L = { clean: ld(clean), corrupted: ld(corr), patched: ld(pt) };
    return { c, c2, ld: L, restored: (L.patched - L.corrupted) / (L.clean - L.corrupted), logits: pt.logits, cleanRun: clean, corruptedRun: corr };
  }

  // Predictions for all pairs (row a*p + b) and accuracy on the test pairs, for the current embedding.
  // Uses that the logits depend on the MLP only through W_out W_U.
  accuracy() {
    const { p, d, H, m, P } = this;
    const Win = P.W_in;
    // Each precomputed d-vector passed through W_in, so the MLP input is a sum of known parts.
    const toIn = (x) => { const out = new Float64Array(m); for (let j = 0; j < d; j++) { const v = x[j]; for (let n = 0; n < m; n++) out[n] += v * Win[j * m + n]; } return out; };
    const toU = (x) => this.matvecU(x);
    const pre = (t) => ({ ...t, voIn: t.vo.map(toIn), voU: t.vo.map(toU) });
    const S0 = this.pos[0].map(pre), S1 = this.pos[1].map(pre), Q = pre(this.eq);
    const base = toIn(this.eq.x); for (let n = 0; n < m; n++) base[n] += P.b_in[n];
    const baseU = toU(this.eq.x); for (let c = 0; c < p; c++) baseU[c] += this.bU[c];
    const pred = new Int32Array(p * p);
    let right = 0, nTest = 0;
    const h3 = new Float64Array(3), u = new Float64Array(m), L = new Float64Array(p);
    for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) {
      const src = [S0[a], S1[b], Q];
      u.set(base); L.set(baseU);
      for (let h = 0; h < H; h++) {
        for (let i = 0; i < 3; i++) { let v = 0; const k = src[i].k[h], q = this.q[h]; for (let e = 0; e < k.length; e++) v += q[e] * k[e]; h3[i] = v / Math.sqrt(this.dh); }
        const mx = Math.max(h3[0], h3[1], h3[2]);
        let Z = 0; for (let i = 0; i < 3; i++) { h3[i] = Math.exp(h3[i] - mx); Z += h3[i]; }
        for (let i = 0; i < 3; i++) {
          const w = h3[i] / Z, vi = src[i].voIn[h], vu = src[i].voU[h];
          for (let n = 0; n < m; n++) u[n] += w * vi[n];
          for (let c = 0; c < p; c++) L[c] += w * vu[c];
        }
      }
      for (let n = 0; n < m; n++) { const v = u[n]; if (v > 0) { const off = n * p; for (let c = 0; c < p; c++) L[c] += v * this.WoU[off + c]; } }
      let best = 0; for (let c = 1; c < p; c++) if (L[c] > L[best]) best = c;
      const i = a * p + b;
      pred[i] = best;
      if (!this.trainMask[i]) { nTest++; if (best === (a + b) % p) right++; }
    }
    return { pred, testAcc: right / nTest };
  }

  // Number embeddings projected onto the constant and the kept frequencies (keep[k] true), as rows.
  ablated(keep) {
    const { p, d, F } = this;
    const rows = Array.from({ length: p }, () => new Float64Array(d));
    const use = (i) => i === 0 || keep[(i + 1) >> 1];
    for (let i = 0; i < p; i++) {
      if (!use(i)) continue;
      const coef = new Float64Array(d);
      for (let a = 0; a < p; a++) { const e = this.P.W_E.subarray(a * d, (a + 1) * d), f = F[i][a]; for (let j = 0; j < d; j++) coef[j] += f * e[j]; }
      for (let a = 0; a < p; a++) { const f = F[i][a]; for (let j = 0; j < d; j++) rows[a][j] += f * coef[j]; }
    }
    return rows;
  }

  // Size of each frequency k = 1..(p-1)/2 in the number embeddings, as in tools/modadd/modadd.py.
  spectrum() {
    const { p, d, F } = this, out = [];
    const sq = F.map((row) => {
      let s = 0;
      for (let j = 0; j < d; j++) { let c = 0; for (let a = 0; a < p; a++) c += row[a] * this.P.W_E[a * d + j]; s += c * c; }
      return s;
    });
    for (let k = 1; k <= (p - 1) / 2; k++) out.push(Math.sqrt(sq[2 * k - 1] + sq[2 * k]));
    return out;
  }

  // Each number's embedding, keeping only its component at frequency k. That component lies in the
  // plane spanned by the cos and sin coefficient vectors of frequency k; returns its coordinates in
  // an orthonormal basis of that plane, one point [x, y] per number. Lengths are in embedding units.
  circle(k) {
    const { p, d, F } = this;
    const comp = (row) => { const c = new Float64Array(d); for (let a = 0; a < p; a++) for (let j = 0; j < d; j++) c[j] += row[a] * this.P.W_E[a * d + j]; return c; };
    const cc = comp(F[2 * k - 1]), cs = comp(F[2 * k]);
    const dot = (x, y) => x.reduce((s, xi, j) => s + xi * y[j], 0);
    const u = Float64Array.from(cc), nu = Math.sqrt(dot(u, u)); u.forEach((x, j) => { u[j] = x / nu; });
    const v = Float64Array.from(cs), uv = dot(u, v); v.forEach((x, j) => { v[j] = x - uv * u[j]; });
    const nv = Math.sqrt(dot(v, v)); v.forEach((x, j) => { v[j] = x / nv; });
    // Component of number a at frequency k: F[2k-1][a] cc + F[2k][a] cs.
    const pu = [dot(cc, u), dot(cs, u)], pv = [dot(cc, v), dot(cs, v)];
    return Array.from({ length: p }, (_, a) => {
      const f1 = F[2 * k - 1][a], f2 = F[2 * k][a];
      return [f1 * pu[0] + f2 * pu[1], f1 * pv[0] + f2 * pv[1]];
    });
  }
}

export async function loadModel() {
  const r = await fetch(new URL('../../data/6-7/model.json', import.meta.url));
  if (!r.ok) throw new Error('Could not load the model parameters');
  return new ModAdd(await r.json());
}
