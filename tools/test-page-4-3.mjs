// Numbers quoted on page 4.3. Run: node tools/test-page-4-3.mjs
import { candidates, tilt, expect, kl, objective, sweep } from '../docs/js/lib/s45-tilt.js';
import { rng } from '../docs/js/lib/mlp.js';
let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) fail++; };

// 1. pi* maximizes E[r] - beta KL: no random distribution does better (exactness check).
{
  const rs = candidates(true), r = rng(3);
  let worst = Infinity;
  for (const beta of [0.03, 0.1, 0.3, 1, 3]) {
    const best = objective(tilt(rs, beta), rs, beta);
    for (let i = 0; i < 20000; i++) {
      const w = rs.map(() => -Math.log(r())); const z = w.reduce((a, b) => a + b, 0);
      worst = Math.min(worst, best - objective(w.map((v) => v / z), rs, beta));
    }
    // The maximum value equals beta log Z, a closed form.
    const logZ = Math.log(rs.reduce((s, x) => s + x.ref * Math.exp(x.proxy / beta), 0));
    check(Math.abs(best - beta * logZ) < 1e-9, `beta ${beta}: objective at pi* equals beta log Z (${best.toFixed(6)})`);
  }
  check(worst >= -1e-12, `no random policy beats pi* (smallest margin ${worst.toExponential(2)})`);
}

// 2. With the hack response, the true reward rises then falls as beta decreases; report the peak.
{
  const rs = candidates(true), s = sweep(rs);
  const ip = s.reduce((b, v, i) => (v.truth > s[b].truth ? i : b), 0);
  const rises = s.slice(0, ip + 1).every((v, i, a) => i === 0 || v.truth >= a[i - 1].truth - 1e-12);
  const falls = s.slice(ip).every((v, i, a) => i === 0 || v.truth <= a[i - 1].truth + 1e-12);
  const proxyUp = s.every((v, i, a) => i === 0 || v.proxy >= a[i - 1].proxy - 1e-12);
  check(ip > 0 && ip < s.length - 1 && rises && falls, `with the hack: true reward rises then falls; peak at beta = ${s[ip].beta.toFixed(3)}, true reward ${s[ip].truth.toFixed(3)}, proxy ${s[ip].proxy.toFixed(3)}, KL ${s[ip].kl.toFixed(2)}`);
  check(proxyUp, 'expected proxy reward rises steadily as beta decreases');
  console.log(`     at beta = 10: true ${s[0].truth.toFixed(3)}, proxy ${s[0].proxy.toFixed(3)}; at beta = 0.01: true ${s.at(-1).truth.toFixed(3)}, proxy ${s.at(-1).proxy.toFixed(3)}, KL ${s.at(-1).kl.toFixed(2)}, hack prob ${tilt(rs, 0.01).at(-1).toFixed(3)}`);
  for (const b of [1, 0.3, 0.2, 0.1, 0.05]) { const pi = tilt(rs, b); console.log(`     beta ${b}: true ${expect(pi, rs, 'truth').toFixed(3)} proxy ${expect(pi, rs, 'proxy').toFixed(3)} KL ${kl(pi, rs).toFixed(2)} P(hack) ${pi.at(-1).toFixed(3)}`); }
}

// 3. Without the hack response, the true reward rises all the way.
{
  const rs = candidates(false), s = sweep(rs);
  const mono = s.every((v, i, a) => i === 0 || v.truth >= a[i - 1].truth - 1e-12);
  check(mono, `without the hack: true reward rises monotonically, from ${s[0].truth.toFixed(3)} to ${s.at(-1).truth.toFixed(3)}`);
}

// 4. Large beta returns the starting model.
{
  const rs = candidates(true), pi = tilt(rs, 1e6);
  check(Math.max(...pi.map((p, i) => Math.abs(p - rs[i].ref))) < 1e-5, 'beta -> infinity gives the starting distribution');
}
process.exit(fail ? 1 : 0);
