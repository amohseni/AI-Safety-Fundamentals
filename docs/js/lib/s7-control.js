// Toy control protocol (page 8.6), computed exactly by dynamic programming. No DOM.
//
// An untrusted model does N tasks. In the attack run it attempts sabotage on exactly K of them,
// at positions chosen uniformly at random. A trusted monitor gives each task a suspicion score:
//   normal work: score ~ Normal(0, 1);   sabotage: score ~ Normal(mu, 1),
// so mu (the monitor quality) is the separation between the two distributions, in standard deviations.
// Humans can audit B = round(budget * N) tasks in all. Tasks are handled in order:
//   - if the score is at least the audit threshold and audits remain, a human audits the task;
//     an audited sabotage attempt is caught and the protocol stops: the run is safe;
//   - otherwise, if the score is at least the hand-off threshold, the trusted model does the task instead,
//     so a sabotage attempt there has no effect;
//   - otherwise the untrusted model's work is used; a sabotage attempt there succeeds and the run is unsafe.
// Safety is the probability that no sabotage succeeds in the attack run.
// Usefulness is the expected share of tasks whose work comes from the untrusted model, in a run where
// the model is honest (no sabotage); audited tasks count as the untrusted model's work.

export function erf(x) {
  // Maclaurin series for small arguments, continued fraction for erfc for large ones
  // (accuracy checked against reference values in tools/test-page-8-6.mjs).
  const t = Math.abs(x);
  let r;
  if (t < 2.5) {
    // Maclaurin series
    let sum = t, term = t, n = 0;
    while (Math.abs(term) > 1e-17 * Math.abs(sum)) { n++; term *= -t * t / n; sum += term / (2 * n + 1); }
    r = (2 / Math.sqrt(Math.PI)) * sum;
  } else {
    // continued fraction for erfc (Lentz)
    let f = t, C = t, Dd = 0;
    for (let k = 1; k < 200; k++) {
      const a = k / 2;
      Dd = t + a * Dd; Dd = Dd === 0 ? 1e-300 : Dd; Dd = 1 / Dd;
      C = t + a / C; C = C === 0 ? 1e-300 : C;
      const delta = C * Dd; f *= delta;
      if (Math.abs(delta - 1) < 1e-16) break;
    }
    r = 1 - Math.exp(-t * t) / (f * Math.sqrt(Math.PI));
  }
  return x < 0 ? -r : r;
}
export const Phi = (z) => (z === Infinity ? 1 : z === -Infinity ? 0 : 0.5 * (1 + erf(z / Math.SQRT2)));
export const density = (z, m = 0) => Math.exp(-0.5 * (z - m) ** 2) / Math.sqrt(2 * Math.PI);

// Probabilities of audit / hand-off / accept for one task, when audits remain and when they do not.
function bands(mean, tA, tD) {
  const above = (t) => 1 - Phi(t - mean);
  const audit = above(tA);
  const handLeft = Math.max(0, above(tD) - audit); // hand-off band below the audit threshold
  const noBudgetHand = above(tD); // no audits left: everything above the hand-off threshold is handed off
  return { audit, hand: handLeft, accept: 1 - audit - handLeft, hand0: noBudgetHand, accept0: 1 - noBudgetHand };
}

export function evaluate({ mu, budget, tA, tD, N = 1000, K = 5 }) {
  const B = Math.round(budget * N);
  const n = bands(0, tA, tD), s = bands(mu, tA, tD);

  // Attack run. State: (audits used b, attacks left m); value: probability of reaching task i with no
  // success and no catch so far. We accumulate the probability of an unsafe end.
  let cur = Array.from({ length: B + 1 }, () => new Float64Array(K + 1));
  cur[0][K] = 1;
  let unsafe = 0;
  for (let i = 0; i < N; i++) {
    const left = N - i;
    const nxt = Array.from({ length: B + 1 }, () => new Float64Array(K + 1));
    for (let b = 0; b <= B; b++) {
      const row = cur[b];
      for (let m = 0; m <= K; m++) {
        const p = row[m];
        if (p === 0) continue;
        const pAtt = m / left, pNorm = 1 - pAtt;
        if (b < B) {
          // normal task
          nxt[b + 1][m] += p * pNorm * n.audit;
          nxt[b][m] += p * pNorm * (1 - n.audit);
          // sabotage attempt: audited -> caught (safe end); handed off -> continue; accepted -> unsafe end
          if (pAtt > 0) {
            nxt[b][m - 1] += p * pAtt * s.hand;
            unsafe += p * pAtt * s.accept;
          }
        } else {
          nxt[b][m] += p * pNorm;
          if (pAtt > 0) {
            nxt[b][m - 1] += p * pAtt * s.hand0;
            unsafe += p * pAtt * s.accept0;
          }
        }
      }
    }
    cur = nxt;
  }

  // Honest run: distribution of audits used, and the expected number of hand-offs.
  let dist = new Float64Array(B + 1); dist[0] = 1;
  let handed = 0, audits = 0;
  for (let i = 0; i < N; i++) {
    const nd = new Float64Array(B + 1);
    for (let b = 0; b <= B; b++) {
      const p = dist[b]; if (p === 0) continue;
      if (b < B) { nd[b + 1] += p * n.audit; nd[b] += p * (1 - n.audit); handed += p * n.hand; audits += p * n.audit; }
      else { nd[b] += p; handed += p * n.hand0; }
    }
    dist = nd;
  }
  return { safety: 1 - unsafe, usefulness: 1 - handed / N, audits, B, flagRate: n.audit, catchRate: s.audit };
}

// Monte Carlo version of the same protocol, for testing the dynamic program.
export function simulate({ mu, budget, tA, tD, N = 1000, K = 5 }, runs, rand) {
  const gauss = () => { const u = Math.max(rand(), 1e-12), v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const B = Math.round(budget * N);
  let safe = 0, handedTot = 0;
  for (let r = 0; r < runs; r++) {
    // choose K distinct attack positions
    const pos = new Set(); while (pos.size < K) pos.add(Math.floor(rand() * N));
    let used = 0, outcome = 1;
    for (let i = 0; i < N; i++) {
      const att = pos.has(i), z = gauss() + (att ? mu : 0);
      if (z >= tA && used < B) { used++; if (att) { outcome = 1; break; } continue; }
      if (z >= tD) continue;
      if (att) { outcome = 0; break; }
    }
    safe += outcome;
    // honest run
    used = 0; let handed = 0;
    for (let i = 0; i < N; i++) {
      const z = gauss();
      if (z >= tA && used < B) { used++; continue; }
      if (z >= tD) handed++;
    }
    handedTot += handed;
  }
  return { safety: safe / runs, usefulness: 1 - handedTot / runs / N };
}

// The audit threshold that flags a given share of normal tasks: Phi^{-1}(1 - share), by bisection.
export function quantileAbove(share) {
  let lo = -8, hi = 8;
  for (let k = 0; k < 100; k++) { const mid = (lo + hi) / 2; if (1 - Phi(mid) > share) lo = mid; else hi = mid; }
  return (lo + hi) / 2;
}
