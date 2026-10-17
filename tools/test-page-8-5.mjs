// Claims on page 8.5 (toy model of selection on behavior). Run: node tools/test-page-8-5.mjs
import * as S from '../docs/js/lib/s7-select.js';

let failed = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) failed++; };
const X0 = [1 / 3, 1 / 3, 1 / 3];
const pct = (x) => (100 * x).toFixed(1) + '%';

// Exact solution agrees with small Euler steps of the replicator equation.
{
  let worst = 0;
  for (const p of [{ q: 1, h: 0 }, { q: 0.8, h: 0.05 }, { q: 0.5, h: 0.2 }]) {
    const a = S.shares(X0, S.rewards(p), 40), b = S.euler(X0, p, 40);
    worst = Math.max(worst, ...a.map((v, i) => Math.abs(v - b[i])));
  }
  ok(worst < 1e-3, `closed form matches Euler integration (largest gap ${worst.toExponential(1)})`);
}

// Task 1: detection 1, no honeypots. Deceptive policies keep their share relative to aligned ones.
for (const T of [50, 200, 500]) {
  const x = S.shares(X0, S.rewards({ q: 1, h: 0 }), T);
  ok(Math.abs(x[2] - 0.5) < 1e-6 && Math.abs(x[0] - 0.5) < 1e-6, `q = 1, h = 0, ${T} rounds: aligned ${pct(x[0])}, open ${pct(x[1])}, deceptive ${pct(x[2])}`);
}
{
  const x = S.shares(X0, S.rewards({ q: 1, h: 0 }), 10);
  ok(x[1] < 1e-4, `q = 1, h = 0: open misaligned share after 10 rounds is ${x[1].toExponential(2)}`);
  const all = S.trajectory(X0, { q: 1, h: 0 }, 500);
  ok(all.every((x) => Math.abs(x[2] / x[0] - 1) < 1e-9), 'q = 1, h = 0: deceptive/aligned ratio is exactly 1 at every round');
}

// Lower detection: deceptive policies decline, at rate 1 - q.
for (const q of [0.99, 0.9]) {
  const x = S.shares(X0, S.rewards({ q, h: 0 }), 100);
  console.log(`   q = ${q}, h = 0, 100 rounds: deceptive ${pct(x[2])}`);
}

// Task 3: detection 0.9 (no honeypots) removes deceptive policies; 0.99 does not within 100 rounds.
{
  const a = S.shares(X0, S.rewards({ q: 0.9, h: 0 }), 100)[2], b = S.shares(X0, S.rewards({ q: 0.99, h: 0 }), 100)[2];
  ok(a < 0.0001 && Math.abs(100 * a - 0.0045) < 0.0005, `q = 0.9, 100 rounds: deceptive ${(100 * a).toFixed(4)}% (page: 0.005%)`);
  ok(Math.abs(b - 0.269) < 0.001, `q = 0.99, 100 rounds: deceptive ${(100 * b).toFixed(1)}%`);
}

// Task 2: honeypots. With q = 1 the ratio deceptive/aligned falls as e^{-h t}.
{
  const rows = [];
  for (const h of [0.01, 0.02, 0.05, 0.1]) {
    const row = [100, 200, 500].map((T) => S.shares(X0, S.rewards({ q: 1, h }), T)[2]);
    rows.push(`h = ${h}: ` + row.map(pct).join(' / '));
    const x = S.shares(X0, S.rewards({ q: 1, h }), 100);
    ok(Math.abs(x[2] / x[0] - Math.exp(-h * 100)) < 1e-9, `h = ${h}: deceptive/aligned after 100 rounds = e^{-${h}·100} = ${Math.exp(-h * 100).toFixed(3)}`);
  }
  console.log('   deceptive share after 100 / 200 / 500 rounds (q = 1):\n   ' + rows.join('\n   '));
  // Rounds needed to bring the deceptive share below 1%, from equal thirds, q = 1.
  for (const h of [0.01, 0.02, 0.05, 0.1]) {
    let t = 0; while (S.shares(X0, S.rewards({ q: 1, h }), t)[2] >= 0.01) t++;
    console.log(`   h = ${h}: deceptive share first below 1% at round ${t} (ln 99 / h = ${(Math.log(99) / h).toFixed(0)})`);
  }
  // The page's claim: with 1% honeypots and 100 rounds, about a quarter of the population is still deceptive.
  const x = S.shares(X0, S.rewards({ q: 1, h: 0.01 }), 100);
  ok(x[2] > 0.26 && x[2] < 0.28, `h = 0.01, 100 rounds: deceptive ${pct(x[2])}`);
  const y = S.shares(X0, S.rewards({ q: 1, h: 0.05 }), 100);
  ok(y[2] < 0.01, `h = 0.05, 100 rounds: deceptive ${pct(y[2])}`);
}

// Check question: more rounds on the same inputs do not change the deceptive share when q = 1, h = 0.
{
  const a = S.shares(X0, S.rewards({ q: 1, h: 0 }), 100), b = S.shares(X0, S.rewards({ q: 1, h: 0 }), 1000);
  ok(Math.abs(a[2] - b[2]) < 1e-9, 'q = 1, h = 0: 100 rounds and 1,000 rounds leave the same deceptive share');
}

console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
