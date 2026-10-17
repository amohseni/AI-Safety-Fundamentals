// Page 4.3: the exact solution of the KL-regularized objective on a small set of responses.
// Pure JavaScript, no DOM.

// Eight constructed responses to one prompt. ref: the starting model's probability;
// proxy: the reward model's score; truth: the true reward. All numbers are made up for illustration.
export const RESPONSES = [
  { id: 'clear', label: 'Clear and correct, with an example', ref: 0.10, proxy: 0.80, truth: 0.90 },
  { id: 'dry', label: 'Correct but dry', ref: 0.20, proxy: 0.55, truth: 0.65 },
  { id: 'long', label: 'Correct but long-winded', ref: 0.12, proxy: 0.70, truth: 0.55 },
  { id: 'vague', label: 'Vague, partly correct', ref: 0.25, proxy: 0.35, truth: 0.35 },
  { id: 'off', label: 'Off topic', ref: 0.08, proxy: 0.05, truth: 0.05 },
  { id: 'refuse', label: 'Refuses to answer', ref: 0.08, proxy: 0.10, truth: 0.10 },
  { id: 'wrong', label: 'Wrong, stated confidently', ref: 0.14, proxy: 0.40, truth: 0.05 },
  { id: 'hack', label: 'Flattering, full of jargon, wrong', ref: 0.03, proxy: 1.00, truth: 0.10 },
];

// The responses in play, with the starting probabilities renormalized to sum to 1.
export function candidates(includeHack = true) {
  const rs = RESPONSES.filter((r) => includeHack || r.id !== 'hack');
  const z = rs.reduce((s, r) => s + r.ref, 0);
  return rs.map((r) => ({ ...r, ref: r.ref / z }));
}

// pi*(y) = pi_ref(y) exp(r(y)/beta) / Z, computed in log space so tiny beta does not overflow.
export function tilt(rs, beta) {
  const l = rs.map((r) => Math.log(r.ref) + r.proxy / beta);
  const m = Math.max(...l);
  const w = l.map((v) => Math.exp(v - m));
  const z = w.reduce((s, v) => s + v, 0);
  return w.map((v) => v / z);
}

export const expect = (pi, rs, key) => pi.reduce((s, p, i) => s + p * rs[i][key], 0);
export const kl = (pi, rs) => pi.reduce((s, p, i) => s + (p > 0 ? p * Math.log(p / rs[i].ref) : 0), 0);

// The objective itself, to check that pi* maximizes it: E[r] - beta KL(pi || pi_ref).
export const objective = (pi, rs, beta) => expect(pi, rs, 'proxy') - beta * kl(pi, rs);

// Expected proxy and true reward and KL along a grid of beta from hi down to lo (log spaced).
export function sweep(rs, lo = 0.01, hi = 10, n = 241) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const beta = Math.exp(Math.log(hi) + ((Math.log(lo) - Math.log(hi)) * i) / (n - 1));
    const pi = tilt(rs, beta);
    out.push({ beta, proxy: expect(pi, rs, 'proxy'), truth: expect(pi, rs, 'truth'), kl: kl(pi, rs) });
  }
  return out;
}
