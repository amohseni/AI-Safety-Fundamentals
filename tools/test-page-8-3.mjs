// Numbers quoted on page 8.3, recomputed with the page's own agent and settings (docs/js/lib/8-coinrun.js):
// 6,000 training episodes, training runs with seeds 1 to 10, success on 500 standard levels (seed 7),
// and on 2,000 random test levels (seed 999). Run: node tools/test-page-8-3.mjs
import { train, evaluate, W, H, MAX_STEPS, EPISODES } from '../docs/js/lib/8-coinrun.js';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const OUT = {};

for (const p of [0, 0.02, 0.1, 0.5]) {
  const rows = [];
  for (let seed = 1; seed <= 10; seed++) {
    const th = train(p, { seed });
    const tr = evaluate(th, { pRandom: 0, levels: 500, seed: 7 });
    const te = evaluate(th, { levels: 2000, record: true });
    rows.push([tr.success, te.success, te.toEnd]);
  }
  const col = (i) => rows.map((r) => Math.round(100 * r[i]));
  OUT[p] = { std: rows.map((r) => r[0]), test: col(1), end: col(2) };
  const rng = (a) => `${Math.min(...a)} to ${Math.max(...a)}`;
  console.log(`${(100 * p).toFixed(0).padStart(2)}% randomized | standard levels ${rng(col(0))}% | test success ${col(1).join(' ')} | reaches right end ${col(2).join(' ')}`);
}
console.log('Page: 0% -> test 22 to 30, right end about 80; 2% -> 9 of 10 runs 84 to 99, one run 32; 10% -> all at least 95.');
// Training samples actions, so last-bit differences in floating point between JavaScript engines can change a run.
// In Chromium (2026-10-16) the 2% runs gave 87 97 93 98 95 84 96 98 32 99; Node 22 gives the values above.
const lo = (a) => Math.min(...a), hi = (a) => Math.max(...a), avg = (a) => a.reduce((s, v) => s + v, 0) / a.length;
// Panel note: "a 10 by 5 grid and has 30 steps to reach the coin ... 6,000 training levels".
check('panel: a 10 by 5 grid, 30 steps, 6,000 training levels', W === 10 && H === 5 && MAX_STEPS === 30 && EPISODES === 6000, `${W} ${H} ${MAX_STEPS} ${EPISODES}`);
// Task 1 notice: "On standard levels it reaches the coin every time. On test levels it reaches it in about a quarter of them:
// between 22% and 30% across ten training runs. In about 80% of test levels it goes to the right end of the middle row".
check('t1: trained with 0% randomized levels, the agent reaches the coin on every standard level (ten runs)', OUT[0].std.every((v) => v === 1), OUT[0].std.join(' '));
check('t1: on test levels it reaches the coin in between 22% and 30% across ten training runs', lo(OUT[0].test) === 22 && hi(OUT[0].test) === 30, OUT[0].test.join(' '));
check('t1: in about 80% of test levels it goes to the right end (every run 75% to 85%)', OUT[0].end.every((v) => v >= 75 && v <= 85), OUT[0].end.join(' '));
// Task 2 notice: "2% is usually enough: in 9 of 10 training runs test success is between 84% and 99%, though one run reaches
// only 32%. At 10% every run is at 95% or above. With 2% of 6,000 levels, about 120 training levels put the coin somewhere other
// than the right end". The page's numbers were measured in Chromium, where the lowest of the nine runs was 84; Node 22 gives 85
// for that run (see the note above), so the lower end is checked as "at least 84" and the other three numbers exactly.
{
  const t = OUT[0.02].test.slice().sort((a, b) => a - b), nine = t.slice(1);
  check('t2: at 2%, in 9 of 10 training runs test success is between 84% and 99% (highest run 99%)', nine.every((v) => v >= 84 && v <= 99) && hi(nine) === 99 && lo(nine) <= 85, nine.join(' '));
  check('t2: one run reaches only 32%', t[0] === 32, t[0]);
}
check('t2: at 10% every run is at 95% or above', lo(OUT[0.1].test) >= 95, OUT[0.1].test.join(' '));
check('t2: at 50% every run is at 95% or above too', lo(OUT[0.5].test) >= 95, OUT[0.5].test.join(' '));
check('t2: 2% of 6,000 levels is 120', Math.abs(0.02 * EPISODES - 120) < 1e-9, 0.02 * EPISODES);
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
