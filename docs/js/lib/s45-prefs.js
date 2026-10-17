// Page 4.2: constructed responses with visible features, and a Bradley-Terry reward model
// fitted to pairwise choices. Pure JavaScript, no DOM.
import { rng } from './mlp.js';

// Four visible features of a response, each 0 or 1.
export const FEATURES = [
  { key: 'long', name: 'length', on: 'long', off: 'short' },
  { key: 'correct', name: 'correctness', on: 'correct', off: 'wrong' },
  { key: 'confident', name: 'confidence', on: 'confident', off: 'hedged' },
  { key: 'flattering', name: 'flattery', on: 'flattering', off: 'plain' },
];

// Twelve questions with a right answer, a wrong answer, and background sentences that are true
// whichever answer is given (used to make a response long).
export const PROMPTS = [
  { q: 'How many legs does a spider have?', right: 'eight', wrong: 'six', extra: 'Spiders are arachnids, the group that also includes scorpions and ticks. Insects are a separate group.' },
  { q: 'What is the chemical symbol for gold?', right: 'Au', wrong: 'Ag', extra: 'Chemical symbols have one or two letters, often taken from the element’s Latin name. Chemists everywhere use the same symbols.' },
  { q: 'Which planet is closest to the Sun?', right: 'Mercury', wrong: 'Venus', extra: 'Distances in the solar system are often given in astronomical units. One astronomical unit is the average distance from the Earth to the Sun.' },
  { q: 'At sea level, at what temperature in Celsius does water boil?', right: '100 degrees', wrong: '90 degrees', extra: 'The boiling point depends on air pressure, so it is lower high on a mountain. Cooking times change for the same reason.' },
  { q: 'Who wrote Pride and Prejudice?', right: 'Jane Austen', wrong: 'Charlotte Brontë', extra: 'The novel was first published in 1813. It has been adapted for film and television many times.' },
  { q: 'What is 7 times 8?', right: '56', wrong: '54', extra: 'Multiplication facts up to ten times ten are usually learned by heart in primary school. Written methods for larger products build on them.' },
  { q: 'What is the largest ocean on Earth?', right: 'the Pacific', wrong: 'the Atlantic', extra: 'Oceans cover about 71 percent of the Earth’s surface. Where one ocean ends and another begins is partly a matter of convention.' },
  { q: 'In what year did the Second World War end?', right: '1945', wrong: '1944', extra: 'The war in Europe and the war in the Pacific ended at different times. Each end is dated by a formal surrender.' },
  { q: 'Which gas do plants take from the air for photosynthesis?', right: 'carbon dioxide', wrong: 'oxygen', extra: 'Photosynthesis uses the energy of light to make sugars. In most plants it takes place mainly in the leaves.' },
  { q: 'How many sides does a hexagon have?', right: 'six', wrong: 'eight', extra: 'The names of polygons come from Greek number words. Hexagons appear in the cells of honeycombs.' },
  { q: 'What is the capital of Australia?', right: 'Canberra', wrong: 'Sydney', extra: 'The site of the capital was chosen as a compromise between Sydney and Melbourne. Parliament has met there since 1927.' },
  { q: 'What is the freezing point of water in Fahrenheit?', right: '32 degrees', wrong: '0 degrees', extra: 'The Fahrenheit scale is used mainly in the United States. Most other countries use the Celsius scale.' },
];

// The text of a response with features f = { long, correct, confident, flattering } to prompt k.
export function compose(k, f) {
  const P = PROMPTS[k];
  const ans = f.correct ? P.right : P.wrong;
  const parts = [];
  if (f.flattering) parts.push('What a great question!');
  parts.push(f.confident ? `The answer is ${ans}.` : `I think the answer is probably ${ans}, but I am not sure.`);
  if (f.long) parts.push(P.extra);
  return parts.join(' ');
}

export const vec = (f) => FEATURES.map((d) => (f[d.key] ? 1 : 0));
const randFeat = (r) => Object.fromEntries(FEATURES.map((d) => [d.key, r() < 0.5 ? 1 : 0]));
const same = (a, b) => FEATURES.every((d) => a[d.key] === b[d.key]);

// The reader's pairs: a fixed, seeded sequence, so every reader sees the same 20 pairs.
export function readerPairs(n = 20, seed = 7) {
  const r = rng(seed), out = [];
  for (let i = 0; i < n; i++) {
    let a, b;
    do { a = randFeat(r); b = randFeat(r); } while (same(a, b));
    out.push({ prompt: i % PROMPTS.length, a, b });
  }
  return out;
}

// A simulated labeler who judges by length alone: shown a long and a short answer, they pick
// the long one with probability 0.7. Pairs always differ in length; other features are random.
export function biasedLabels(n, seed, pLonger = 0.7) {
  const r = rng(seed), out = [];
  for (let i = 0; i < n; i++) {
    const a = randFeat(r), b = randFeat(r);
    a.long = r() < 0.5 ? 1 : 0; b.long = 1 - a.long;
    const pickLong = r() < pLonger;
    const aWins = pickLong ? a.long === 1 : a.long === 0;
    out.push({ prompt: Math.floor(r() * PROMPTS.length), a, b, aWins, simulated: true });
  }
  return out;
}

const sig = (z) => 1 / (1 + Math.exp(-z));

// Bradley-Terry fit: P(A preferred to B) = sigma(w . (x_A - x_B)). Maximizes the log-likelihood of
// the observed choices minus (lambda / 2) |w|^2, by Newton's method. The small penalty keeps the
// weights finite when the choices are perfectly consistent.
export function fitBT(labels, lambda = 0.5) {
  const d = FEATURES.length;
  let w = new Array(d).fill(0);
  for (let it = 0; it < 50; it++) {
    const g = w.map((v) => -lambda * v);
    const H = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? lambda : 0)));
    for (const L of labels) {
      const xa = vec(L.a), xb = vec(L.b);
      const z = xa.map((v, i) => v - xb[i]);
      const y = L.aWins ? 1 : 0;
      const p = sig(z.reduce((s, v, i) => s + v * w[i], 0));
      for (let i = 0; i < d; i++) {
        g[i] += (y - p) * z[i];
        for (let j = 0; j < d; j++) H[i][j] += p * (1 - p) * z[i] * z[j];
      }
    }
    const step = solve(H, g);
    w = w.map((v, i) => v + step[i]);
    if (Math.max(...step.map(Math.abs)) < 1e-10) break;
  }
  return w;
}

export const score = (w, f) => vec(f).reduce((s, v, i) => s + v * w[i], 0);
export const prefProb = (w, fa, fb) => sig(score(w, fa) - score(w, fb));

// Gaussian elimination with partial pivoting for the small Newton system.
function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}
