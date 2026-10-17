// Numbers behind page 2.6 (roll downhill). Run: node tools/test-page-2-6.mjs
import { SURFACES, makeRun, basin } from '../docs/js/lib/s2-optim.js';
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
const S = SURFACES.two;
console.log(`Two minima: narrow minimum value ${S.f(-1.2, 0).toFixed(3)}, wide minimum value ${S.f(1.4, 0).toFixed(3)}; start (-0.7, 0.5); 1000 steps per run`);
for (const [kind, lr] of [['gd', 0.1], ['gd', 0.05], ['momentum', 0.1], ['adam', 0.1]]) {
  const row = [];
  for (const noise of [0, 1, 2, 3]) {
    let first = 0, all = 0;
    for (let seed = 1; seed <= 100; seed++) { const r = makeRun(S, { kind, lr, noise, seed }); for (let i = 0; i < 1000; i++) r.step(); if (basin(r.p) === 'narrow') { all++; if (seed <= 10) first++; } }
    row.push(`noise ${noise}: ${first}/10 (seeds 1-10), ${all}/100`);
  }
  console.log(`  ${kind} lr ${lr}, runs ending in the narrow minimum: ${row.join('; ')}`);
}
