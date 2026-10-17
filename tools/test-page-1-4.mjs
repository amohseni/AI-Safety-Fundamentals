// Claims on page 1.4 about what the filters show on the shapes image.
// Run: node tools/test-page-1-4.mjs
import { shapes, conv3, PRESETS, sum } from '../docs/js/lib/conv.js';

const W = 240, H = 160;
const img = shapes(W, H);
let fails = 0;
const check = (ok, msg) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${msg}`); if (!ok) fails++; };
// Largest |response| in a box of output coordinates (output pixel (x, y) covers input x..x+2, y..y+2).
function maxIn(map, ow, x0, x1, y0, y1) {
  let m = 0;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) m = Math.max(m, Math.abs(map[y * ow + x]));
  return m;
}
const run = (k) => conv3(img, W, H, k);

// Task 1: vertical-edge filter.
{
  const { out, ow } = run(PRESETS.vertical.k);
  const full = maxIn(out, ow, 0, ow - 1, 0, H - 3);
  const sideL = maxIn(out, ow, 21, 25, 50, 95), sideR = maxIn(out, ow, 81, 85, 50, 95);
  const top = maxIn(out, ow, 32, 76, 40, 48), bottom = maxIn(out, ow, 32, 76, 100, 106);
  const diskSide = maxIn(out, ow, 98, 106, 70, 78), diskTop = maxIn(out, ow, 130, 136, 38, 46);
  const band = maxIn(out, ow, 185, 222, 40, 120);
  console.log({ full, sideL, sideR, top, bottom, diskSide, diskTop, band, bandFrac: band / full });
  check(sideL === full && sideR === full, 'square: vertical sides at full strength');
  check(top === 0 && bottom === 0, 'square: top and bottom vanish (exactly 0)');
  check(diskSide / full > 0.9 && diskTop / full < 0.35, 'disk strongest at left and right, weak at top');
  check(band / full === 0.75, 'band at three-quarters strength');
}
// Task 2: an edge filter gives zero on constant regions; blur keeps brightness.
{
  const { out, ow } = run(PRESETS.vertical.k);
  check(out[(70 - 1) * ow + (50 - 1)] === 0 && out[10 * ow + 10] === 0, 'vertical-edge filter: zero inside the square and in the background');
  check(Math.abs(sum(PRESETS.blur.k) - 1) < 1e-12, 'blur numbers add up to 1');
  const b = run(PRESETS.blur.k);
  check(Math.abs(b.out[(70 - 1) * b.ow + (50 - 1)] - 220) < 1e-9, 'blur keeps the brightness of a constant region');
}
// Task 3: candidate diagonal filters on the band.
{
  const A = [2, 1, 0, 1, 0, -1, 0, -1, -2], C = [0, 1, 2, -1, 0, 1, -2, -1, 0];
  const a = run(A), c = run(C);
  const fullA = maxIn(a.out, a.ow, 0, a.ow - 1, 0, H - 3);
  const bandA = maxIn(a.out, a.ow, 185, 222, 40, 120), bandC = maxIn(c.out, c.ow, 185, 222, 40, 120);
  const squareC = maxIn(c.out, c.ow, 21, 85, 40, 106);
  console.log({ fullA, bandA, bandC, ratioCA: bandC / bandA, squareC });
  check(bandA === fullA, 'filter 1 (2 1 0; 1 0 -1; 0 -1 -2) responds at full strength on the band');
  check(bandC === 0, 'filter 3 gives exactly zero along the band edges');
}
process.exitCode = fails ? 1 : 0;
