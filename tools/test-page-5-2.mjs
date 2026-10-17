// Numbers quoted on page 5.2. Run: node tools/test-page-5-2.mjs
import { learn, integrate, replicatorRHS, pgRHS, maxGap, curveGap, payoffs, runAndMeasure } from '../docs/js/lib/s45-selection.js';
let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) fail++; };
const X0 = [0.05, 0.15, 0.8], FIXED = { kind: 'fixed', pay: [0.9, 0.6, 0.3] }, T = 60, RUNS = 10;
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

// 0. Expected one-step change of Cross's rule equals step size times the replicator right-hand side.
{
  const x = [0.2, 0.3, 0.5], h = 0.01, p = payoffs(FIXED, x);
  const E = x.map((_, j) => x.reduce((s, xi, i) => s + xi * h * p[i] * ((i === j ? 1 : 0) - x[j]), 0));
  const R = replicatorRHS(FIXED, x).map((v) => h * v);
  check(Math.max(...E.map((v, j) => Math.abs(v - R[j]))) < 1e-15, "Cross's rule: expected step = h * replicator right-hand side");
}

// 1. How far learning paths stray from the replicator curve and from the policy-gradient curve,
//    as shapes (ignoring speed), for three step sizes. Paths subsampled every 0.2 units of time.
const G = {};
for (const rule of ['cross', 'exp', 'pg']) {
  const row = [];
  for (const h of [0.1, 0.01, 0.001]) {
    const m = runAndMeasure(rule, FIXED, X0, h, T, RUNS);
    G[rule + h] = [m.gRep, m.gPG];
    row.push(`h=${h}: from replicator curve ${m.gRep.toFixed(3)}${m.gPG !== null ? `, from PG curve ${m.gPG.toFixed(3)}` : ''}`);
  }
  console.log(`     ${rule} (the tool's readout, 10 runs): ${row.join('; ')}`);
}
check(G['cross0.001'][0] < G['cross0.01'][0] && G['cross0.01'][0] < G['cross0.1'][0], 'Cross: paths stray less from the replicator curve as the step size shrinks');
check(G['exp0.1'][0] < 1e-9 && G['exp0.001'][0] < 1e-9, 'exponential weights with fixed payoffs: exactly on the replicator curve at every step size');
check(G['pg0.001'][1] < G['pg0.01'][1] && G['pg0.01'][1] < G['pg0.1'][1] && G['pg0.001'][0] > 0.25, 'softmax policy gradient: approaches its own curve, stays away from the replicator curve');
const rep = integrate(replicatorRHS, FIXED, X0, T), pg = integrate(pgRHS, FIXED, X0, T * 3);
void maxGap;

// 2. The policy-gradient flow is a different curve from the replicator curve (compared as sets).
{
  const r = integrate(replicatorRHS, FIXED, X0, 200, 0.01, 0.01), q = integrate(pgRHS, FIXED, X0, 400, 0.01, 0.01);
  const d = Math.max(...q.filter((_, i) => i % 10 === 0).map((a) => Math.min(...r.map((b) => Math.hypot(...a.x.map((v, k) => v - b.x[k]))))));
  const tR = r.find((p) => p.x[0] > 0.9).t, tP = q.find((p) => p.x[0] > 0.9).t;
  check(d > 0.2, `PG curve departs from the replicator curve by up to ${d.toFixed(2)} (as curves, ignoring speed); time to x1 > 0.9: replicator ${tR.toFixed(1)}, PG ${tP.toFixed(1)}`);
  const peak2 = (c) => Math.max(...c.map((p) => p.x[1]));
  console.log(`     largest probability of action 2 along the way: replicator ${peak2(r).toFixed(2)}, PG ${peak2(q).toFixed(2)}`);
  console.log(`     at t = ${T}: replicator x = ${rep.at(-1).x.map((v) => v.toFixed(3))}, PG flow x = ${pg[Math.round(T / 0.05)].x.map((v) => v.toFixed(3))}`);
}

// 3. Nearly equal payoffs: selection between them is slow.
{
  const near = { kind: 'fixed', pay: [0.8, 0.78, 0.3] }, r = integrate(replicatorRHS, near, X0, T);
  const k = (t) => r[Math.round(t / 0.05)].x;
  console.log(`     near-equal payoffs from ${X0}: x at t = 10 ${k(10).map((v) => v.toFixed(3))}; t = 30 ${k(30).map((v) => v.toFixed(3))}; t = 60 ${k(60).map((v) => v.toFixed(3))}`);
  const ratio0 = X0[0] / X0[1], ratio60 = k(60)[0] / k(60)[1];
  check(Math.abs(ratio60 / ratio0 - Math.exp(0.02 * 60)) < 1e-3 * Math.exp(1.2), `x1/x2 grows exactly by exp(0.02 t): factor ${(ratio60 / ratio0).toFixed(3)} at t = 60 vs ${Math.exp(1.2).toFixed(3)}`);
}

// 4. Rock-paper-scissors: replicator orbits keep x1 x2 x3 constant; noisy learners drift outward.
{
  const rps = { kind: 'rps' }, x1 = [0.5, 0.3, 0.2], prod = (x) => x[0] * x[1] * x[2];
  const r = integrate(replicatorRHS, rps, x1, 120);
  const ps = r.map((p) => prod(p.x));
  check(Math.max(...ps) - Math.min(...ps) < 1e-6, `RPS replicator: x1 x2 x3 stays at ${prod(x1).toFixed(4)} (range ${(Math.max(...ps) - Math.min(...ps)).toExponential(1)})`);
  for (const h of [0.1, 0.01, 0.001]) {
    const ends = Array.from({ length: RUNS }, (_, s) => prod(learn('cross', rps, x1, h, 120, s + 1).at(-1).x));
    console.log(`     RPS Cross h=${h}: x1 x2 x3 at t = 120, 10 runs: median ${ends.sort((a, b) => a - b)[5].toFixed(4)}, min ${ends[0].toFixed(4)}, max ${ends.at(-1).toFixed(4)}`);
  }
}
process.exit(fail ? 1 : 0);
