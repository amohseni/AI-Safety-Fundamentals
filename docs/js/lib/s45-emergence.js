// Page 5.6: a smooth per-digit accuracy and the metrics computed from it.
// Pure JavaScript, no DOM. Model sizes and curves are made up; the functional form follows
// Schaeffer, Miranda, and Koyejo (2023): per-token loss falls as a power law in the number of
// parameters N, L(N) = (N / c)^(-alpha), and the probability of a correct token is exp(-L).

export const C = 1e7; // the size at which the per-digit loss is 1 (per-digit accuracy 0.37)
// Eleven models, from ten million to one trillion parameters, every half power of ten.
export const SIZES = Array.from({ length: 11 }, (_, i) => 10 ** (7 + i / 2));

export const perDigit = (N, alpha) => Math.exp(-((N / C) ** -alpha));

export function metric(kind, N, alpha, k) {
  const p = perDigit(N, alpha);
  if (kind === 'exact') return p ** k; // every digit right; digits independent
  if (kind === 'digit') return p; // expected fraction of digits right
  return k * Math.log(p); // log-likelihood of the whole k-digit answer (natural log)
}

// Width, in powers of ten of model size, of the rise of p^k from 10% to 90%.
// p^k = q exactly when N = C (k / -ln q)^(1/alpha), so the width does not depend on k.
export function riseWidth(alpha, lo = 0.1, hi = 0.9) {
  return Math.log10(Math.log(1 / lo) / Math.log(1 / hi)) / alpha;
}
export const sizeAt = (q, alpha, k) => C * (k / -Math.log(q)) ** (1 / alpha);
