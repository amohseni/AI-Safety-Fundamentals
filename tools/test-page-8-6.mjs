// Claims on page 8.6 (toy control protocol). Run: node tools/test-page-8-6.mjs
import * as C from '../docs/js/lib/s7-control.js';
import { rng } from '../docs/js/lib/mlp.js';

let failed = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) failed++; };
const pct = (x, d = 1) => (100 * x).toFixed(d) + '%';
const OFF = 99; // hand-off threshold that never triggers

// Normal distribution function against reference values.
{
  const ref = [[0, 0.5], [1, 0.8413447460685429], [2.054, 0.9800121541449581], [3, 0.9986501019683699], [-1.5, 0.0668072012688581], [5, 0.9999997133484281]];
  const worst = Math.max(...ref.map(([z, p]) => Math.abs(C.Phi(z) - p)));
  ok(worst < 1e-7, `Phi matches reference values (largest error ${worst.toExponential(1)})`);
  ok(Math.abs(C.quantileAbove(0.02) - 2.0537) < 1e-3, `threshold flagging 2% of normal tasks: ${C.quantileAbove(0.02).toFixed(3)}`);
}

// Dynamic program against Monte Carlo (200,000 runs).
{
  const r = rng(5);
  for (const p of [{ mu: 2, budget: 0.02, tA: 2.05, tD: OFF }, { mu: 2, budget: 0.02, tA: 1.5, tD: 1.0 }, { mu: 1, budget: 0.05, tA: 1.2, tD: 0.5 }]) {
    const e = C.evaluate(p), s = C.simulate(p, 40000, r);
    const se = Math.sqrt(e.safety * (1 - e.safety) / 40000);
    ok(Math.abs(e.safety - s.safety) < 4 * se + 1e-3 && Math.abs(e.usefulness - s.usefulness) < 2e-3,
      `exact vs simulated, mu ${p.mu} budget ${p.budget} tA ${p.tA} tD ${p.tD === OFF ? 'off' : p.tD}: safety ${e.safety.toFixed(4)} vs ${s.safety.toFixed(4)}, usefulness ${e.usefulness.toFixed(4)} vs ${s.usefulness.toFixed(4)}`);
  }
}

// Task 1: 2% budget, mu = 2, no hand-offs. Which audit threshold maximizes safety?
const sweep = (f, lo, hi, step) => { let best = null; for (let t = lo; t <= hi + 1e-9; t += step) { const v = f(t); if (!best || v > best.v) best = { t, v }; } return best; };
{
  const base = { mu: 2, budget: 0.02, tD: OFF };
  const best = sweep((t) => C.evaluate({ ...base, tA: t }).safety, -1, 5, 0.01);
  const q = C.quantileAbove(0.02);
  const atQ = C.evaluate({ ...base, tA: q });
  console.log(`   mu 2, 2% budget, no hand-off: best audit threshold ${best.t.toFixed(2)} (flags ${pct(1 - C.Phi(best.t))} of normal tasks), safety ${pct(best.v)}`);
  console.log(`   at the 2% threshold ${q.toFixed(2)}: safety ${pct(atQ.safety)}, expected audits ${atQ.audits.toFixed(1)} of ${atQ.B}`);
  // The page's slider moves in steps of 0.01, so the page quotes the threshold 2.05.
  const at205 = C.evaluate({ ...base, tA: 2.05 });
  ok(pct(at205.safety) === '48.0%', `audit threshold 2.05: safety ${pct(at205.safety)} (page: 48.0%)`);
  ok(Math.abs(best.t - 1.64) < 0.011 && pct(best.v) === '57.9%', `best audit threshold ${best.t.toFixed(2)}, safety ${pct(best.v)} (page: 1.64, 57.9%)`);
  for (const t of [0, 1, 1.5, 1.8, 2.0, 2.2, 2.5, 3, 4]) console.log(`     tA ${t}: safety ${pct(C.evaluate({ ...base, tA: t }).safety)}, flags ${pct(1 - C.Phi(t))}`);
  ok(1 - C.Phi(best.t) > 0.03 && best.v > atQ.safety + 0.05, 'the best audit threshold flags more than 2% of normal tasks (the budget runs out partway) and beats the 2% threshold');
  // Where does the budget run out, at the best threshold? Expected task index at which audits are exhausted.
  console.log(`   at the best threshold, ${pct(1 - C.Phi(best.t))} flagged: 20 audits last about ${(20 / (1 - C.Phi(best.t))).toFixed(0)} tasks`);
  ok(C.evaluate({ ...base, tA: 0 }).safety < 0.2 && C.evaluate({ ...base, tA: 4 }).safety < 0.1, 'very low and very high audit thresholds both give low safety');
  ok(C.evaluate({ ...base, tA: q }).usefulness > 0.999999, 'with no hand-offs, usefulness is 100%');
}

// Task 2: lowering the hand-off threshold raises safety and lowers usefulness.
{
  const tA = C.quantileAbove(0.02);
  let mono = true, prev = null;
  const rows = [];
  for (let tD = 3; tD >= -1; tD -= 0.25) {
    const e = C.evaluate({ mu: 2, budget: 0.02, tA, tD });
    if (prev && !(e.safety >= prev.safety - 1e-12 && e.usefulness <= prev.usefulness + 1e-12)) mono = false;
    prev = e; rows.push(`tD ${tD.toFixed(2)}: safety ${pct(e.safety)}, usefulness ${pct(e.usefulness)}`);
  }
  console.log('   mu 2, 2% budget, audit at the 2% threshold:\n     ' + rows.join('\n     '));
  ok(mono, 'as the hand-off threshold falls, safety never falls and usefulness never rises');
  const hand10 = C.evaluate({ mu: 2, budget: 0.02, tA, tD: C.quantileAbove(0.10) });
  const p2 = C.evaluate({ mu: 2, budget: 0.02, tA: 2.05, tD: 1.0 });
  ok(pct(p2.safety) === '75.1%' && pct(p2.usefulness) === '86.0%', `tA 2.05, hand-off 1.0: safety ${pct(p2.safety)}, usefulness ${pct(p2.usefulness)} (page: 75.1%, 86.0%)`);
  console.log(`   hand off the next 8% (tD at the 10% point, ${C.quantileAbove(0.10).toFixed(2)}): safety ${pct(hand10.safety)}, usefulness ${pct(hand10.usefulness)}`);
}

// Task 3: a poor monitor (mu = 0.5). What safety can still be reached, and at what usefulness?
{
  for (const mu of [0.5, 2]) {
    const tA = 2.05;
    let found = null;
    for (let tD = 5; tD >= -4; tD -= 0.01) { const e = C.evaluate({ mu, budget: 0.02, tA, tD }); if (e.safety >= 0.99) { found = { tD, ...e }; break; } }
    const none = C.evaluate({ mu, budget: 0.02, tA, tD: OFF });
    if (mu === 0.5) ok(found.tD.toFixed(2) === '-2.34' && pct(found.usefulness) === '2.8%', `mu 0.5: 99% safety first at hand-off ${found.tD.toFixed(2)}, usefulness ${pct(found.usefulness)} (page: -2.34, 2.8%)`);
    if (mu === 2) ok(pct(found.usefulness) === '29.9%', `mu 2: 99% safety costs usefulness down to ${pct(found.usefulness)} (page: 29.9%)`);
    console.log(`   mu ${mu}: no hand-off safety ${pct(none.safety)}; first reaches 99% safety at tD ${found.tD.toFixed(2)} with usefulness ${pct(found.usefulness)}`);
  }
  const all = C.evaluate({ mu: 0.5, budget: 0.02, tA: C.quantileAbove(0.02), tD: -6 });
  ok(all.safety > 0.9999 && all.usefulness < 0.03, `mu 0.5, hand off everything not audited: safety ${pct(all.safety, 2)}, usefulness ${pct(all.usefulness)}`);
  const muBest = sweep((t) => C.evaluate({ mu: 0.5, budget: 0.02, tA: t, tD: OFF }).safety, -1, 5, 0.01);
  ok(pct(muBest.v) === '15.0%', `mu 0.5, no hand-off: best audit threshold ${muBest.t.toFixed(2)} gives safety ${pct(muBest.v)} (page: 15.0%)`);
  ok(Math.abs((1000 + 1) / (5 + 1) - 166.8) < 0.1, 'expected position of the first of 5 random attacks among 1,000 tasks: (N + 1)/(K + 1) = 166.8');
  const mu0 = C.evaluate({ mu: 0, budget: 0.02, tA: C.quantileAbove(0.02), tD: OFF });
  console.log(`   mu 0 (useless monitor), no hand-off: safety ${pct(mu0.safety)}`);
}

// Budget: more audits raise safety (best threshold for each budget, no hand-offs).
{
  for (const b of [0.01, 0.02, 0.05, 0.1]) {
    const best = sweep((t) => C.evaluate({ mu: 2, budget: b, tA: t, tD: OFF }).safety, -1, 5, 0.02);
    console.log(`   mu 2, budget ${pct(b, 0)}: best safety ${pct(best.v)} at tA ${best.t.toFixed(2)}`);
  }
}

// Speed of one evaluation at the largest budget (the page recomputes on every control change).
{
  const t0 = performance.now(); for (let k = 0; k < 20; k++) C.evaluate({ mu: 2, budget: 0.1, tA: 1, tD: 0.5 }); const dt = (performance.now() - t0) / 20;
  console.log(`   one evaluation at a 10% budget takes ${dt.toFixed(1)} ms`);
}

console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
