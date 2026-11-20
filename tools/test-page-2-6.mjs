// Numbers behind page 2.6 (roll downhill). Run: node tools/test-page-2-6.mjs
import { SURFACES, makeRun, basin } from '../docs/js/lib/s2-optim.js';
let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const TOL = 1e-4; // "converged": loss within 0.0001 of the lowest value
function until(name, opts, max = 20000) {
  const S = SURFACES[name], r = makeRun(S, opts);
  for (let i = 0; i < max; i++) { r.step(); if (r.diverged) return `diverges (step ${r.t})`; if (r.losses[r.losses.length - 1] - S.min < TOL) return r.t; }
  return `not converged after ${max} (loss above lowest ${(r.losses[r.losses.length - 1] - S.min).toExponential(1)})`;
}
console.log('Narrow valley f = (x^2 + 100 y^2)/2, start (-2.5, 1): steps until within 1e-4 of the minimum');
for (const lr of [0.002, 0.005, 0.01, 0.015, 0.019, 0.0195, 0.02, 0.0204, 0.021]) console.log(`  lr ${lr}: GD ${until('valley', { kind: 'gd', lr })}; momentum ${until('valley', { kind: 'momentum', lr })}`);
for (const lr of [0.01, 0.05, 0.1, 0.3]) console.log(`  lr ${lr}: Adam ${until('valley', { kind: 'adam', lr })}`);
{ // At exactly lr = 0.02 the y coordinate flips sign each step without shrinking.
  const r = makeRun(SURFACES.valley, { kind: 'gd', lr: 0.02 }); for (let i = 0; i < 1000; i++) r.step();
  console.log(`  lr 0.02 after 1000 steps: y = ${r.p[1].toFixed(6)}, loss ${r.losses[r.losses.length - 1].toFixed(3)}`);
}
console.log('Round bowl f = (x^2 + y^2)/2 GD:', [0.1, 0.5, 1, 1.9, 2, 2.1].map((lr) => `${lr}: ${until('bowl', { kind: 'gd', lr })}`).join('; '));
console.log('Curved valley (Rosenbrock), start (-1.5, 2):');
for (const lr of [0.0005, 0.001, 0.002, 0.0025, 0.003]) console.log(`  lr ${lr}: GD ${until('rosenbrock', { kind: 'gd', lr })}; momentum ${until('rosenbrock', { kind: 'momentum', lr })}`);
for (const lr of [0.01, 0.03, 0.1]) console.log(`  lr ${lr}: Adam ${until('rosenbrock', { kind: 'adam', lr })}`);
const S = SURFACES.two, two = {};
console.log(`Two minima: narrow minimum value ${S.f(-1.2, 0).toFixed(3)}, wide minimum value ${S.f(1.4, 0).toFixed(3)}; start (-0.7, 0.5); 1000 steps per run`);
for (const [kind, lr] of [['gd', 0.1], ['gd', 0.05], ['momentum', 0.1], ['adam', 0.1]]) {
  const row = [];
  for (const noise of [0, 1, 2, 3]) {
    let first = 0, all = 0;
    for (let seed = 1; seed <= 100; seed++) { const r = makeRun(S, { kind, lr, noise, seed }); for (let i = 0; i < 1000; i++) r.step(); if (basin(r.p) === 'narrow') { all++; if (seed <= 10) first++; } }
    two[`${kind}|${lr}|${noise}`] = { first, all };
    row.push(`noise ${noise}: ${first}/10 (seeds 1-10), ${all}/100`);
  }
  console.log(`  ${kind} lr ${lr}, runs ending in the narrow minimum: ${row.join('; ')}`);
}
// Task 1 notice: "Just under 0.02. At 0.0195 the path reaches the bottom in 263 steps. At exactly 0.02 it bounces between
// the two walls of the valley forever without getting lower, and at 0.0204 it bounces further out each time and diverges."
const gd = (lr) => until('valley', { kind: 'gd', lr });
check('t1: plain gradient descent at 0.0195 reaches the bottom in 263 steps', gd(0.0195) === 263, gd(0.0195));
{
  const r = makeRun(SURFACES.valley, { kind: 'gd', lr: 0.02 }), ys = [];
  for (let i = 0; i < 1000; i++) { r.step(); ys.push(r.p[1]); }
  const flips = ys.every((y, i) => Math.abs(Math.abs(y) - 1) < 1e-9 && (i === 0 || Math.sign(y) !== Math.sign(ys[i - 1])));
  check('t1: at exactly 0.02 the path bounces between the two walls without getting lower (y alternates between -1 and 1 for 1,000 steps; never within 0.0001 of the bottom in 20,000)', flips && typeof gd(0.02) === 'string' && gd(0.02).startsWith('not converged'), `${ys.slice(0, 4).map((y) => y.toFixed(3)).join(' ')} ...; ${gd(0.02)}`);
}
check('t1: at 0.0204 the path diverges', String(gd(0.0204)).startsWith('diverges'), gd(0.0204));
check('t1: every tested learning rate below 0.02 reaches the bottom, every one above 0.02 diverges', [0.002, 0.005, 0.01, 0.015, 0.019, 0.0195].every((lr) => typeof gd(lr) === 'number') && [0.0204, 0.021].every((lr) => String(gd(lr)).startsWith('diverges')), 'see table above');
// Task 2: "Set the learning rate to 0.01. Plain gradient descent then takes 515 steps to get within 0.0001 of the bottom."
// Notice: "103 steps. ... Adam takes 631 steps at learning rate 0.01 and 96 at 0.1."
check('t2: the narrow valley opens at learning rate 0.01', SURFACES.valley.lr === 0.01, SURFACES.valley.lr);
check('t2: plain gradient descent at 0.01 takes 515 steps', gd(0.01) === 515, gd(0.01));
check('t2: momentum at 0.01 takes 103 steps', until('valley', { kind: 'momentum', lr: 0.01 }) === 103, until('valley', { kind: 'momentum', lr: 0.01 }));
check('t2: Adam takes 631 steps at 0.01 and 96 at 0.1', until('valley', { kind: 'adam', lr: 0.01 }) === 631 && until('valley', { kind: 'adam', lr: 0.1 }) === 96, `${until('valley', { kind: 'adam', lr: 0.01 })} ${until('valley', { kind: 'adam', lr: 0.1 })}`);
// Task 3: "from the default start, plain gradient descent at learning rate 0.1 with no noise ends in the narrow, deeper minimum".
// Notice: "Most end in the wide minimum, even though its loss is higher (-0.74 against -0.98). In 100 runs we measured, 15 ended
// in the narrow minimum. ... With noise 1, 89 of 100 runs stayed in the narrow minimum."
check('t3: the two-minimum surface opens at learning rate 0.1', S.lr === 0.1, S.lr);
check('t3: with no noise, gradient descent at 0.1 ends in the narrow minimum', two['gd|0.1|0'].all === 100, two['gd|0.1|0'].all);
check('t3: the loss is -0.74 at the wide minimum and -0.98 at the narrow one', S.f(1.4, 0).toFixed(2) === '-0.74' && S.f(-1.2, 0).toFixed(2) === '-0.98', `${S.f(1.4, 0)} ${S.f(-1.2, 0)}`);
check('t3: with noise 2, 15 of 100 runs end in the narrow minimum (most end in the wide one)', two['gd|0.1|2'].all === 15, two['gd|0.1|2'].all);
check('t3: with noise 1, 89 of 100 runs stay in the narrow minimum', two['gd|0.1|1'].all === 89, two['gd|0.1|1'].all);
check('t3: the first press of Ten runs with noise 2 (seeds 1 to 10) ends mostly in the wide minimum', two['gd|0.1|2'].first < 5, `${two['gd|0.1|2'].first} of 10 narrow`);
// Figure caption: 30 steps at 0.002, 0.019, 0.021; "At 0.021 each zig-zag is wider than the one before, and after 12 steps
// the path leaves the chart." The chart shows |x| <= 3, |y| <= 3 (page/2-6.js).
{
  const out = (lr) => { const r = makeRun(SURFACES.valley, { kind: 'gd', lr }); for (let i = 0; i < 30; i++) r.step(); return { k: r.path.findIndex((p) => Math.abs(p[0]) > 3 || Math.abs(p[1]) > 3), path: r.path }; };
  const a = out(0.002), b = out(0.019), c = out(0.021);
  check('figure: at 0.002 and 0.019 the 30-step path stays on the chart', a.k < 0 && b.k < 0, `${a.k} ${b.k}`);
  check('figure: at 0.019 the path zig-zags (y changes sign every step) and narrows', b.path.slice(1).every((p, i) => p[1] * b.path[i][1] < 0 && Math.abs(p[1]) < Math.abs(b.path[i][1])), b.path.slice(0, 4).map((p) => p[1].toFixed(2)).join(' '));
  check('figure: at 0.021 each zig-zag is wider than the one before, and the path leaves the chart after 12 steps', c.k === 12 && c.path.slice(1, 13).every((p, i) => p[1] * c.path[i][1] < 0 && Math.abs(p[1]) > Math.abs(c.path[i][1])), c.k);
}
// Theorem: gradient descent converges exactly when the learning rate is below 2 / (largest curvature): 0.02 on the valley
// (checked above), 2 on the round bowl.
check('theorem: on the round bowl (curvature 1) 1.9 converges and 2.1 diverges', typeof until('bowl', { kind: 'gd', lr: 1.9 }) === 'number' && String(until('bowl', { kind: 'gd', lr: 2.1 })).startsWith('diverges'), `${until('bowl', { kind: 'gd', lr: 1.9 })}; ${until('bowl', { kind: 'gd', lr: 2.1 })}`);
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
