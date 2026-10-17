// Numbers in scientific notation with a raised exponent at the same type size.
// HTML needs the page style `.exp { vertical-align: 0.45em; line-height: 0; }`.
const MINUS = '−';

// Returns [mantissa, exponent] with the mantissa rounded to `digits` decimals.
export function split(v, digits = 1) {
  let k = Math.floor(Math.log10(Math.abs(v)));
  let m = Number((v / 10 ** k).toFixed(digits));
  if (Math.abs(m) >= 10) { m /= 10; k += 1; }
  return [m, k];
}

// HTML string, e.g. "1.3 × 10<span class="exp">−9</span>". Plain decimals for moderate values.
export function sciHTML(v, { digits = 1, plainFrom = -2, plainTo = 3 } = {}) {
  if (!Number.isFinite(v)) return '∞';
  if (v === 0) return '0';
  const [m, k] = split(v, digits);
  if (k >= plainFrom && k <= plainTo) return Number(v.toPrecision(digits + 1)).toLocaleString();
  const e = String(k).replace('-', MINUS);
  return `${m === 1 ? '' : `${m.toFixed(digits)} × `}10<span class="exp">${e}</span>`;
}

// Plain text for screen readers, e.g. "1.3 times 10 to the power −9".
export function sciText(v, digits = 1) {
  if (!Number.isFinite(v) || v === 0) return String(v);
  const [m, k] = split(v, digits);
  return `${m.toFixed(digits)} times 10 to the power ${String(k).replace('-', MINUS)}`;
}

// Writes m × 10^k into an SVG <text> selection (d3), with the exponent raised.
export function setPow(t, v, { digits = 1, bare = false } = {}) {
  const [m, k] = split(v, digits);
  t.text(null);
  t.append('tspan').text(bare || m === 1 ? '10' : `${m.toFixed(digits)} × 10`);
  t.append('tspan').attr('dy', '-0.45em').text(String(k).replace('-', MINUS));
  t.append('tspan').attr('dy', '0.45em').text(' ');
}
