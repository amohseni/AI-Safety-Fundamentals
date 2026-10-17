// Toy model of selection on behavior (page 8.5). No DOM; also used by tools/test-page-8-5.mjs.
//
// Three types of policy: aligned, misaligned and open about it, misaligned and deceptive.
// Each training round shows the population evaluation inputs. A share h of them are honeypots:
// inputs built to look like deployment. A policy that behaves well on an input earns reward 1; one
// that behaves badly earns 0.
//   aligned:     always behaves well                       -> expected reward 1
//   open:        always behaves badly                      -> expected reward 0
//   deceptive:   behaves well when it recognizes training. On an ordinary training input it does so
//                with probability q; a honeypot looks like deployment, so there it behaves badly.
//                                                          -> expected reward q (1 - h)
// Selection follows the replicator equation of page 5.2, dx_i/dt = x_i (r_i - mean r). With rewards that
// do not depend on the population, its exact solution is x_i(t) = x_i(0) e^{r_i t} / sum_j x_j(0) e^{r_j t}.
// One training round is one unit of time.

export const TYPES = ['aligned', 'open', 'deceptive'];

export function rewards({ q, h }) {
  return [1, 0, q * (1 - h)];
}

export function shares(x0, r, t) {
  // Subtract the largest exponent before exponentiating, for numerical safety.
  const top = Math.max(...r.map((ri, i) => (x0[i] > 0 ? ri * t : -Infinity)));
  const w = x0.map((x, i) => (x > 0 ? x * Math.exp(r[i] * t - top) : 0));
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / s);
}

// Shares at every round 0..T.
export function trajectory(x0, params, T) {
  const r = rewards(params);
  return Array.from({ length: T + 1 }, (_, t) => shares(x0, r, t));
}

// Euler steps of the replicator equation with a small step, for checking the exact solution.
export function euler(x0, params, T, dt = 0.001) {
  const r = rewards(params);
  let x = x0.slice();
  for (let s = 0; s < Math.round(T / dt); s++) {
    const mean = x.reduce((a, xi, i) => a + xi * r[i], 0);
    x = x.map((xi, i) => xi + dt * xi * (r[i] - mean));
  }
  return x;
}
