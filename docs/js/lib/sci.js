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

// ---- Subscript and superscript digits ----
// The site's faces (Computer Modern) have no glyphs for ₀ to ₉, ⁰, ⁴ to ⁹, or ⁻, so a browser would
// take them from another font. These helpers set such a digit in the same face instead: a subscript
// smaller and lowered, an exponent raised at the same size (as setPow does).
const SUBS = '₀₁₂₃₄₅₆₇₈₉', SUPS = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const kindOf = (c) => (SUBS.includes(c) ? 'sub' : SUPS.includes(c) || c === '⁻' ? 'sup' : '');
const plainOf = (c) => (c === '⁻' ? MINUS : SUBS.includes(c) ? String(SUBS.indexOf(c)) : SUPS.includes(c) ? String(SUPS.indexOf(c)) : c);

// Splits "w₁ x" into runs [text, kind] with kind '', 'sub', or 'sup'; script digits become plain digits.
// With sup false, superscript characters are left as they are (the faces do have ¹, ² and ³).
export function scriptRuns(s, sup = true) {
  const runs = [];
  for (const c of String(s)) {
    const k = !sup && kindOf(c) === 'sup' ? '' : kindOf(c);
    if (!k) { if (runs.length && !runs[runs.length - 1][1]) runs[runs.length - 1][0] += c; else runs.push([c, '']); continue; }
    if (runs.length && runs[runs.length - 1][1] === k) runs[runs.length - 1][0] += plainOf(c);
    else runs.push([plainOf(c), k]);
  }
  return runs;
}

// HTML string with subscript digits as <sub> elements, e.g. "weight on x<sub>1</sub>".
export function subHTML(s) {
  return scriptRuns(s, false).map(([t, k]) => (k === 'sub' ? `<sub style="font-size:0.75em;line-height:0">${t}</sub>` : t)).join('');
}

// Writes a label with script digits into an SVG <text> selection (d3) as tspans.
export function setScripts(t, s, sup = true) {
  const runs = scriptRuns(s, sup);
  if (runs.every(([, k]) => !k)) { if (t.text() !== String(s)) t.text(s); return t; }
  t.text(null);
  let back = null; // the dy that returns the next run to the baseline
  for (const [txt, k] of runs) {
    const ts = t.append('tspan').text(txt);
    if (k === 'sub') { ts.attr('font-size', '0.75em').attr('dy', '0.3em'); back = '-0.225em'; }
    else if (k === 'sup') { ts.attr('dy', '-0.45em'); back = '0.45em'; }
    else if (back) { ts.attr('dy', back); back = null; }
  }
  return t;
}

// Draws a label with subscript digits on a canvas at (x, y), honoring the context's textAlign and textBaseline.
export function fillScripts(ctx, s, x, y) {
  const runs = scriptRuns(s, false);
  const m = /(\d+(?:\.\d+)?)px/.exec(ctx.font);
  if (!m || runs.every(([, k]) => !k)) { ctx.fillText(runs.map((r) => r[0]).join(''), x, y); return; }
  const size = Number(m[1]), font = ctx.font, small = font.replace(m[0], `${0.75 * size}px`);
  const align = ctx.textAlign, base = ctx.textBaseline;
  // The alphabetic baseline that the current textBaseline puts the text on.
  const d0 = ctx.measureText('x').fontBoundingBoxDescent;
  ctx.textBaseline = 'alphabetic';
  const d1 = ctx.measureText('x').fontBoundingBoxDescent;
  const ya = y + (Number.isFinite(d0) && Number.isFinite(d1) ? d0 - d1 : 0);
  const parts = runs.map(([txt, k]) => { ctx.font = k ? small : font; return [txt, k, ctx.measureText(txt).width]; });
  const total = parts.reduce((a, p) => a + p[2], 0);
  let cx = align === 'right' || align === 'end' ? x - total : align === 'center' ? x - total / 2 : x;
  ctx.textAlign = 'left';
  for (const [txt, k, w] of parts) {
    ctx.font = k ? small : font;
    ctx.fillText(txt, cx, ya + (k === 'sub' ? 0.22 * size : k === 'sup' ? -0.4 * size : 0));
    cx += w;
  }
  ctx.font = font; ctx.textAlign = align; ctx.textBaseline = base;
}
