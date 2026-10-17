// Selection on a noisy proxy (page 8.1). Pure functions, no DOM, so Node tests can import them.
//
// Model: true value V ~ Normal(0, 1); proxy U = V + E, with E independent of V.
// Error families (s is the error scale):
//   { kind: 'normal', s }              E = s Z
//   { kind: 't', s, df }               E = s T, T a Student t variable with df degrees of freedom
//   { kind: 'mix', s, rate, big }      E = s Z with probability 1 - rate, s * big * Z with probability rate
// Selection: draw k options, keep the one with the highest proxy.
//
// selectionCurve(err) precomputes the distribution of U and E[V | U = u] on a grid,
// then curve.at(k) returns { v: E[V | selected], u: E[U | selected] } exactly up to quadrature error.

export function lgamma(x) {
  // Lanczos approximation (g = 7, n = 9).
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// Regularized incomplete beta I_x(a, b), continued fraction (Numerical Recipes).
function betacf(a, b, x) {
  const FPMIN = 1e-300;
  let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-14) break;
  }
  return h;
}
export function ibeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  if (x < (a + 1) / (a + b + 2)) return (Math.exp(lbt) * betacf(a, b, x)) / a;
  return 1 - (Math.exp(lbt) * betacf(b, a, 1 - x)) / b;
}

const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);
const lnorm = (x, sd) => -0.5 * (x / sd) ** 2 - Math.log(sd) - LOG_SQRT_2PI;

// Upper tail of the standard normal, accurate far into the tail (W. J. Cody style via erfc).
export function normSurv(x) {
  // erfc by a continued-fraction-free rational approximation (Numerical Recipes erfcc), rel. error < 1.2e-7.
  const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r / 2 : 1 - r / 2;
}

// Log density of the error at x.
export function errLogPdf(err, x) {
  if (err.kind === 'normal') return lnorm(x, err.s);
  if (err.kind === 't') {
    const nu = err.df, s = err.s;
    return lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI) - Math.log(s) - ((nu + 1) / 2) * Math.log1p((x / s) ** 2 / nu);
  }
  const a = Math.log1p(-err.rate) + lnorm(x, err.s), b = Math.log(err.rate) + lnorm(x, err.s * err.big);
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

// Survival function of the error, P(E > x), for x >= 0.
export function errSurv(err, x) {
  if (err.kind === 'normal') return normSurv(x / err.s);
  if (err.kind === 't') { const y = x / err.s, nu = err.df; return 0.5 * ibeta(nu / (nu + y * y), nu / 2, 0.5); }
  return (1 - err.rate) * normSurv(x / err.s) + err.rate * normSurv(x / (err.s * err.big));
}

// Random draw of the error.
export function errSample(err, rand, gauss) {
  if (err.kind === 'normal') return err.s * gauss();
  if (err.kind === 't') return (err.s * gauss()) / Math.sqrt(chi2(err.df, rand, gauss) / err.df);
  return err.s * (rand() < err.rate ? err.big : 1) * gauss();
}
function chi2(nu, rand, gauss) { return 2 * gammaSample(nu / 2, rand, gauss); }
function gammaSample(a, rand, gauss) {
  // Marsaglia and Tsang (2000); for a < 1 use the boost a+1 then multiply by U^(1/a).
  if (a < 1) return gammaSample(a + 1, rand, gauss) * Math.pow(Math.max(rand(), 1e-300), 1 / a);
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = gauss(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = rand();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// Precomputes the proxy's distribution and E[V | U = u] on a sinh-spaced grid of u.
export function selectionCurve(err, { n = 2401 } = {}) {
  // Upper end of the grid: far enough that P(E > top - 9) < 1e-15.
  let hi = 10;
  while (errSurv(err, hi) > 1e-15 && hi < 1e13) hi *= 2;
  const top = hi + 9;
  const A = Math.sqrt(1 + (err.kind === 'mix' ? err.s * err.s : err.s * err.s));
  const zmax = Math.asinh(top / A), dz = (2 * zmax) / (n - 1);
  const u = new Float64Array(n), w = new Float64Array(n), logf = new Float64Array(n), h = new Float64Array(n);
  // Grid of true values for the convolution with the standard normal.
  const vStep = err.kind === 't' ? Math.min(0.04, err.s / 8) : 0.04;
  const vs = []; for (let v = -9; v <= 9 + 1e-9; v += vStep) vs.push(v);
  const lphi = vs.map((v) => lnorm(v, 1));
  for (let j = 0; j < n; j++) {
    const z = -zmax + j * dz;
    u[j] = A * Math.sinh(z);
    w[j] = A * Math.cosh(z) * dz;
    if (err.kind === 'normal') {
      const s2 = 1 + err.s * err.s;
      logf[j] = lnorm(u[j], Math.sqrt(s2)); h[j] = u[j] / s2;
    } else if (err.kind === 'mix') {
      const comps = [[Math.log1p(-err.rate), err.s], [Math.log(err.rate), err.s * err.big]].map(([lw, sd]) => {
        const s2 = 1 + sd * sd; return [lw + lnorm(u[j], Math.sqrt(s2)), u[j] / s2];
      });
      const m = Math.max(comps[0][0], comps[1][0]);
      const e0 = Math.exp(comps[0][0] - m), e1 = Math.exp(comps[1][0] - m);
      logf[j] = m + Math.log(e0 + e1); h[j] = (e0 * comps[0][1] + e1 * comps[1][1]) / (e0 + e1);
    } else {
      // Log-sum-exp over the v grid, trapezoid weights.
      let m = -Infinity;
      const terms = new Float64Array(vs.length);
      for (let i = 0; i < vs.length; i++) { terms[i] = lphi[i] + errLogPdf(err, u[j] - vs[i]); if (terms[i] > m) m = terms[i]; }
      let sf = 0, sg = 0;
      for (let i = 0; i < vs.length; i++) { const e = Math.exp(terms[i] - m) * (i === 0 || i === vs.length - 1 ? 0.5 : 1); sf += e; sg += e * vs[i]; }
      logf[j] = m + Math.log(sf * vStep); h[j] = sg / sf;
    }
  }
  // Survival function of U, accumulated from the top so the far tail keeps its precision.
  const S = new Float64Array(n);
  S[n - 1] = 0; // beyond the top of the grid the probability is below 1e-15
  for (let j = n - 2; j >= 0; j--) S[j] = S[j + 1] + 0.5 * (Math.exp(logf[j]) * w[j] + Math.exp(logf[j + 1]) * w[j + 1]);
  const total = S[0];
  for (let j = 0; j < n; j++) S[j] = Math.min(1, S[j] / total); // normalize away quadrature error

  return {
    u, h, S,
    // Expected true value and expected proxy of the best of k options.
    at(k) {
      let ev = 0, eu = 0, prev = Math.exp(k * Math.log1p(-Math.min(S[0], 1 - 1e-16)));
      for (let j = 1; j < n; j++) {
        const G = S[j] <= 0 ? 1 : Math.exp(k * Math.log1p(-S[j]));
        const dG = G - prev;
        if (dG > 0) { ev += dG * 0.5 * (h[j] + h[j - 1]); eu += dG * 0.5 * (u[j] + u[j - 1]); }
        prev = G;
      }
      return { v: ev, u: eu };
    },
    // Quantile of U: the u with P(U > u) = q (linear interpolation in log q).
    upperQuantile(q) {
      let lo = 0, hi2 = n - 1;
      if (q >= S[0]) return u[0];
      if (q <= S[n - 2]) return u[n - 2];
      while (hi2 - lo > 1) { const mid = (lo + hi2) >> 1; if (S[mid] > q) lo = mid; else hi2 = mid; }
      const a = Math.log(S[lo]), b = Math.log(S[hi2]), f = (Math.log(q) - a) / (b - a);
      return u[lo] + f * (u[hi2] - u[lo]);
    },
    condMean(x) {
      let lo = 0, hi2 = n - 1;
      if (x <= u[0]) return h[0];
      if (x >= u[n - 1]) return h[n - 1];
      while (hi2 - lo > 1) { const mid = (lo + hi2) >> 1; if (u[mid] < x) lo = mid; else hi2 = mid; }
      const f = (x - u[lo]) / (u[hi2] - u[lo]);
      return h[lo] + f * (h[hi2] - h[lo]);
    },
  };
}

// Draws the winner of one best-of-k selection exactly: its proxy from the distribution of the maximum,
// then its true value from the conditional distribution given that proxy.
export function sampleWinner(err, curve, k, rand, gauss) {
  const r = Math.max(rand(), 1e-300);
  const q = -Math.expm1(Math.log(r) / k); // P(U > u) for the winner's quantile
  const u = curve.upperQuantile(q);
  // Posterior of V given U = u on a grid: phi(v) f_E(u - v).
  const step = err.kind === 't' ? Math.min(0.02, err.s / 10) : 0.01;
  const vs = [], lw = [];
  let m = -Infinity;
  for (let v = -7; v <= 7; v += step) { const l = lnorm(v, 1) + errLogPdf(err, u - v); vs.push(v); lw.push(l); if (l > m) m = l; }
  let tot = 0; const cw = lw.map((l) => (tot += Math.exp(l - m)));
  const t = rand() * tot;
  let i = 0; while (i < cw.length - 1 && cw[i] < t) i++;
  return { u, v: vs[i] + (rand() - 0.5) * step };
}

// Extremal Goodhart (second panel). True value of design x; the proxy is a straight line fitted by least squares
// to noisy measurements of the true value, taken only for designs x in [0, range].
export const trueValue = (x) => x - (x * x) / 8; // rises to its peak value 2 at x = 4, then falls
export function fitProxy(range, { n = 24, noise = 0.3, seed = 7 } = {}) {
  const rand = mulberry(seed);
  const xs = [], ys = [];
  for (let i = 0; i < n; i++) {
    const x = (range * (i + 0.5)) / n;
    xs.push(x); ys.push(trueValue(x) + noise * (2 * rand() - 1));
  }
  const mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
  let sxy = 0, sxx = 0;
  for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
  const slope = sxy / sxx, icpt = my - slope * mx;
  return { xs, ys, slope, icpt, f: (x) => icpt + slope * x };
}
// The optimizer picks the design with the highest proxy among x in [0, reach].
export function optimize(proxy, reach) {
  const x = proxy.slope >= 0 ? reach : 0;
  return { x, proxy: proxy.f(x), truth: trueValue(x) };
}
function mulberry(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
