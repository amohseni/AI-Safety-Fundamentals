// Claims on page 7.1, tested on the page's own algebra (3 actions x 3 states = 9 atoms, 511 propositions).
// Run: node tools/test-page-7-1.mjs
import * as J from '../docs/js/lib/s7-jeffrey.js';
import { rng } from '../docs/js/lib/mlp.js';

let failed = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) failed++; };
const close = (x, y, tol = 1e-9) => Math.abs(x - y) <= tol * Math.max(1, Math.abs(x), Math.abs(y));

// The page's starting values.
const PA = [1 / 3, 1 / 3, 1 / 3];
const PS = [0.3, 0.5, 0.2];
const D0 = [[1, -0.3, -0.3], [-0.3, 1, -0.3], [0.1, 0.1, 0.8]];
const m0 = J.atoms(PA, PS, D0);
const props = J.allPropositions(9);
ok(props.length === 511, `the algebra has ${props.length} nonempty propositions`);

console.log('start: action values', J.actionValues(m0).map((x) => x.toFixed(3)).join(' '), 'choice', J.ACTIONS[J.argmax(J.actionValues(m0))]);
ok(J.argmax(J.actionValues(m0)) === 1, 'at the start the agent edits the renderer');

// Task 1: with credence 1 that the bug is in the parser, D(edit parser) = D(edit parser and bug in parser).
{
  const m = J.atoms(PA, [1, 0, 0], D0);
  ok(close(J.actionValues(m)[0], D0[0][0]), `certain of parser: D(edit parser) = ${J.actionValues(m)[0]} = D(edit parser ∧ parser) = ${D0[0][0]}`);
  ok(J.argmax(J.actionValues(m)) === 0, 'certain of parser: the agent edits the parser');
}

// Averaging (section 3.4.3) holds for any partition, for random pairs and their transforms.
function randomPair(r) {
  const ps = [r() + 0.05, r() + 0.05, r() + 0.05]; const t = ps[0] + ps[1] + ps[2];
  const D = [0, 1, 2].map(() => [0, 1, 2].map(() => 2 * r() - 1));
  return J.atoms(PA, ps.map((p) => p / t), D);
}
function randomConstants(r, m) {
  for (;;) {
    const k = { a: 4 * r() - 2, b: 2 * r() - 1, c: 3 * r() - 1.5 };
    k.d = J.normalizingD(m, k.c);
    if (J.check(m, k).ok) return k;
  }
}
function averagingHolds(m, r) {
  for (let t = 0; t < 300; t++) {
    const A = props[Math.floor(r() * props.length)];
    // random partition of A into up to 3 cells
    const cells = [[], [], []];
    A.forEach((k) => cells[Math.floor(r() * 3)].push(k));
    const S = cells.filter((c) => c.length);
    const pA = J.prob(m, A);
    const avg = S.reduce((s, c) => s + J.desir(m, c) * (J.prob(m, c) / pA), 0);
    if (!close(avg, J.desir(m, A), 1e-9)) return false;
  }
  return true;
}
{
  const r = rng(7);
  let allAvg = true, allOrder = true, allMobius = true, allProb = true, allChoice = true, allRank = true;
  for (let trial = 0; trial < 200; trial++) {
    const m = randomPair(r), k = randomConstants(r, m), mt = J.transform(m, k);
    allAvg = allAvg && averagingHolds(m, r) && averagingHolds(mt, r);
    allOrder = allOrder && J.sameOrdering(m, mt, props);
    // D'(A) = (a D(A) + b)/(c D(A) + d) for every proposition, not only atoms
    allMobius = allMobius && props.every((A) => close(J.desir(mt, A), (k.a * J.desir(m, A) + k.b) / (k.c * J.desir(m, A) + k.d), 1e-9));
    allProb = allProb && close(mt.P.reduce((s, p) => s + p, 0), 1, 1e-12) && mt.P.every((p) => p > 0);
    allChoice = allChoice && J.argmax(J.actionValues(m)) === J.argmax(J.actionValues(mt));
    allRank = allRank && J.ranking(J.actionValues(m)).join() === J.ranking(J.actionValues(mt)).join();
  }
  ok(allAvg, 'averaging over any partition holds for P, D and for the transformed P\', D\' (200 random pairs)');
  ok(allMobius, 'D\'(A) = (aD(A)+b)/(cD(A)+d) for all 511 propositions, computed from P\', D\' on atoms');
  ok(allOrder, 'the transformed pair orders all 511 propositions exactly as the original does');
  ok(allChoice && allRank, 'the chosen action and the full ranking of the three actions never change');
  ok(allProb, 'with d = 1 − c·D(T), P\' is a probability (sums to 1, all atoms positive)');
}

// Normalization: P' sums to c·D(T) + d. Rescaling it is the same as scaling a, b, c, d and leaves D' unchanged.
{
  const k = { a: 1.3, b: 0.2, c: 0.6, d: 1.4 };
  ok(J.check(m0, k).ok, 'constants (1.3, 0.2, 0.6, 1.4) are admissible on the start pair');
  const mt = J.transform(m0, k), DT = J.desir(m0, J.allAtoms(m0));
  const tot = mt.P.reduce((s, p) => s + p, 0);
  ok(close(tot, k.c * DT + k.d), `P'(T) = ${tot.toFixed(6)} = c·D(T) + d = ${(k.c * DT + k.d).toFixed(6)} (D(T) = ${DT.toFixed(4)})`);
  const mn = J.normalize(mt);
  const ks = { a: k.a / tot, b: k.b / tot, c: k.c / tot, d: k.d / tot };
  const ms = J.transform(m0, ks);
  ok(ms.P.every((p, i) => close(p, mn.P[i])) && ms.D.every((x, i) => close(x, mt.D[i])), 'renormalizing P\' equals dividing a, b, c, d by P\'(T); D\' is unchanged');
  ok(J.sameOrdering(m0, mn, props), 'the renormalized pair still orders all propositions the same way');
  // With the convention D(T) = 0, P' is a probability iff d = 1, and D'(T) = 0 iff b = 0.
  const shift = DT; const Dz = D0.map((row) => row.map((x) => x - shift)); const mz = J.atoms(PA, PS, Dz);
  ok(close(J.desir(mz, J.allAtoms(mz)), 0), 'shifting D so that D(T) = 0 is possible');
  const kz = { a: 1, b: 0.3, c: 0.5, d: 1 }; const mzt = J.transform(mz, kz);
  ok(close(mzt.P.reduce((s, p) => s + p, 0), 1), 'when D(T) = 0, d = 1 gives a probability whatever c is');
  ok(close(J.desir(mzt, J.allAtoms(mzt)), 0.3), 'when D(T) = 0, D\'(T) = b/d, so D\'(T) = 0 needs b = 0');
}

// The new credences depend on the action when c ≠ 0 (the old ones did not).
{
  const k = { a: 1, b: 0, c: 0.5 }; k.d = J.normalizingD(m0, k.c);
  const mt = J.transform(m0, k);
  const rows = [0, 1, 2].map((i) => J.conditional(mt, i));
  const rows0 = [0, 1, 2].map((i) => J.conditional(m0, i));
  ok(rows0.every((r) => r.every((p, j) => close(p, PS[j]))), 'original: P(state | action) = P(state) for every action');
  const spread = Math.max(...[0, 1, 2].map((j) => Math.max(...rows.map((r) => r[j])) - Math.min(...rows.map((r) => r[j]))));
  ok(spread > 0.05, `c = 0.5: P'(state | action) differs across actions by up to ${spread.toFixed(3)}`);
  console.log('   c=0.5 conditionals:', rows.map((r) => r.map((p) => p.toFixed(2)).join('/')).join('  '), 'd =', k.d.toFixed(3));
  console.log('   c=0.5 action values:', J.actionValues(mt).map((x) => x.toFixed(3)).join(' '));
  ok(J.argmax(J.actionValues(mt)) === J.argmax(J.actionValues(m0)), 'c = 0.5 at the start values: same choice');
}

// Fixing P (the probe reading) forces c = 0, leaving D' = (a/d) D + b/d, a positive affine map.
{
  const r = rng(3);
  let forced = true;
  for (let t = 0; t < 200; t++) {
    const m = randomPair(r);
    for (const c of [-0.8, -0.3, -0.05, -0.01, 0.01, 0.05, 0.3, 0.8]) {
      const d = J.normalizingD(m, c), k = { a: 1, b: 0, c, d };
      if (!J.check(m, k).ok) continue;
      const mt = J.transform(m, k);
      const dev = Math.max(...mt.P.map((p, i) => Math.abs(p - m.P[i])));
      if (!(dev > 1e-6)) forced = false;
    }
    const mt0 = J.transform(m, { a: 2, b: -0.4, c: 0, d: 1 });
    if (!mt0.P.every((p, i) => close(p, m.P[i]))) forced = false;
  }
  ok(forced, 'every c ≠ 0 changes P (200 random pairs, 8 values of c); c = 0, d = 1 leaves P unchanged');
  // Algebraic reason: P'(w) = P(w) for all atoms means c D(w) + d = 1 for all atoms; two distinct D values force c = 0, d = 1.
  const k = { a: 2, b: -0.4, c: 0, d: 1 }; const mt = J.transform(m0, k);
  ok(mt.D.every((x, i) => close(x, 2 * m0.D[i] - 0.4)), 'with c = 0, d = 1: D\' = aD + b, a rescaling and a shift of zero');
  ok(J.check(m0, { a: -1, b: 0, c: 0, d: 1 }).ok === false, 'a = −1 (with c = 0, d = 1) is rejected: ad − bc ≤ 0');
  const mr = J.transform(m0, { a: -1, b: 0, c: 0, d: 1 });
  ok(J.ranking(J.actionValues(mr)).join() === J.ranking(J.actionValues(m0)).reverse().join(), 'that rejected map would reverse the ranking of the actions');
}

// The page's slider ranges: a in [0.2, 3], b in [-1, 1], c in [-1.5, 1.5]; report how much of the c range is admissible at the start.
{
  let okc = 0, tot = 0;
  for (let c = -1.5; c <= 1.5001; c += 0.05) { tot++; if (J.check(m0, { a: 1, b: 0, c, d: J.normalizingD(m0, c) }).ok) okc++; }
  console.log(`   at the start values, ${okc} of ${tot} slider positions of c (a = 1, b = 0) are admissible`);
}

console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
