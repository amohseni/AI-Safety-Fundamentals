// Page 6.2: find the direction.
import { segmented, button } from '../lib/ui.js';
import { decodeTruth, TOPICS } from '../lib/s6l-truth.js';
import { decodeF16, rows, massMean, dot, norm } from '../lib/s6l-probe.js';
import { scatterArrow, mirrorHist } from '../lib/s6l-plot.js';
import { planeBasis, weekdayPlane, DAYS, bestSplit } from '../lib/s6l-geometry.js';

const d3 = window.d3;
const [truthJson, wdJson] = await Promise.all([
  fetch(new URL('../../data/truth/statements.json', import.meta.url)).then((r) => r.json()),
  fetch(new URL('../../data/6-2/weekdays.json', import.meta.url)).then((r) => r.json()),
]);
const T = decodeTruth(truthJson);
const plain = T.S.filter((s) => s.form === 'plain');
const plainIdx = plain.map((s) => s.i);
// Center the coordinates on the average plain statement, so the arrow starts at their center.
{
  const k = T.k, mu = new Float32Array(k);
  plainIdx.forEach((i) => { for (let j = 0; j < k; j++) mu[j] += T.X[i][j] / plainIdx.length; });
  T.X = T.X.map((x) => Float32Array.from(x, (v, j) => v - mu[j]));
}
// Plot scale: a high percentile of the coordinates, so a few outlying statements do not shrink the rest.
const robustScale = (vals) => { const a = vals.map(Math.abs).sort((p, q) => p - q); return a[Math.floor(0.98 * (a.length - 1))] * 1.15; };
const pct = (v) => `${Math.round(v * 100)}%`;
const deg = (r) => Math.round((((r * 180) / Math.PI) % 360 + 360) % 360);
const unit = (v) => { const n = norm(v); return Float32Array.from(v, (t) => t / n); };
const label = (k) => TOPICS[k][0].toUpperCase() + TOPICS[k].slice(1);

// Difference of class means for each topic, from its training half (even-numbered items).
const MM = {};
for (const t of Object.keys(TOPICS)) {
  const tr = plain.filter((s) => s.topic === t && T.isTrain(s));
  MM[t] = unit(massMean(tr.map((s) => T.X[s.i]), tr.map((s) => s.label)).w);
}

// ---- Panel 1: statements ----
const state = { plane: 'pca', topic: 'cities', source: 'cities', seed: 1, angle: (25 * Math.PI) / 180 };
let basis, scale;
const ctl = document.getElementById('st-controls');
const sc = scatterArrow(document.getElementById('st-plot'), {
  label: 'True and false statements in a plane of the activation space, with a direction arrow you can turn',
  onArrow: (a) => { state.angle = a; draw(); },
});
const hist = mirrorHist(document.getElementById('st-hist'), { label: 'Dot products of the statements with the arrow direction: true statements above the line, false below' });
const out = document.getElementById('st-readout');

const planeCtl = segmented(ctl, { name: 'Plane', options: [['pca', 'Principal directions'], ['random', 'Random'], ['mm', 'Difference of means']], value: state.plane,
  onChange: (k) => { state.plane = k; rebuild(); } });
const randBtn = button(ctl, 'New random plane', () => { state.seed++; state.plane = 'random'; planeCtl.set('random'); rebuild(); });
segmented(ctl, { name: 'Statements shown', options: Object.keys(TOPICS).map((k) => [k, label(k)]), value: state.topic,
  onChange: (k) => { state.topic = k; draw(); } });
segmented(ctl, { name: 'Difference of means from', options: Object.keys(TOPICS).map((k) => [k, label(k)]), value: state.source,
  onChange: (k) => { state.source = k; if (state.plane === 'mm') rebuild(); else draw(); } });
button(ctl, 'Point the arrow along the difference of means', () => {
  const m = MM[state.source];
  state.angle = Math.atan2(dot(m, basis.e2), dot(m, basis.e1));
  draw();
});

function rebuild() {
  basis = planeBasis(T.X, plainIdx, state.plane, { mm: MM[state.source], seed: state.seed });
  randBtn.disabled = state.plane !== 'random';
  draw();
}

function draw() {
  const shown = plain.filter((s) => s.topic === state.topic);
  // The plot is centered on the shown topic's average; dot products are unchanged up to a shift.
  const xy = shown.map((s) => [dot(T.X[s.i], basis.e1), dot(T.X[s.i], basis.e2)]);
  const mx = xy.reduce((a, p) => a + p[0], 0) / xy.length, my = xy.reduce((a, p) => a + p[1], 0) / xy.length;
  scale = robustScale(xy.flatMap((p) => [p[0] - mx, p[1] - my]));
  sc.setPoints(shown.map((s, k) => ({ x: xy[k][0] - mx, y: xy[k][1] - my, kind: s.label ? 'a' : 'b' })), { fixedScale: scale });
  sc.setAngle(state.angle);
  const u = basis.e1.map((v, j) => Math.cos(state.angle) * v + Math.sin(state.angle) * basis.e2[j]);
  const c = Math.cos(state.angle) * mx + Math.sin(state.angle) * my;
  const pa = shown.filter((s) => s.label).map((s) => dot(T.X[s.i], u) - c);
  const pb = shown.filter((s) => !s.label).map((s) => dot(T.X[s.i], u) - c);
  const best = bestSplit(pa, pb);
  hist.set(pa, pb, best.t, [-scale, scale]);
  const ang = Math.round((Math.acos(Math.max(-1, Math.min(1, dot(u, MM[state.source])))) * 180) / Math.PI);
  out.innerHTML = `Showing the ${shown.length} statements about ${TOPICS[state.topic]}. Along the arrow, the best threshold sorts <b>${pct(best.acc)}</b> of them correctly, ` +
    `with the true statements toward the ${best.sign > 0 ? 'arrowhead' : 'tail'}. The arrow is at ${ang}&deg; to the difference of means from ${TOPICS[state.source]}, measured in all 32 dimensions.`;
}
rebuild();

// Day names sit outside each day's cluster, along the line from the center.
function dayLabels(mp, s2) {
  return mp.map((p, d) => { const r = Math.hypot(p.x, p.y) || 1, f = 1 + (0.3 * s2) / r; return { x: p.x * f, y: p.y * f, text: DAYS[d].slice(0, 3), color: 'ink' }; });
}

// ---- Panel 2: weekdays ----
(function weekdays() {
  const X = rows(decodeF16(wdJson.X), wdJson.k);
  const { e1, e2, means, center } = weekdayPlane(X, wdJson.day);
  let angle = 0;
  const plot = scatterArrow(document.getElementById('wd-plot'), {
    label: 'Activations at the names of the days of the week, in the plane of their two principal directions, with a direction arrow',
    onArrow: (a) => { angle = a; drawW(); },
  });
  const pts = X.map((x) => { const c = center(x); return { x: dot(c, e1), y: dot(c, e2), kind: 'q' }; });
  const s2 = Math.max(...pts.map((p) => Math.max(Math.abs(p.x), Math.abs(p.y)))) * 1.05;
  plot.setPoints(pts, { fixedScale: s2 * 1.25 }); // room for the day names outside the circle
  const mp = means.map((m) => ({ x: dot(m, e1), y: dot(m, e2) }));
  // Day names sit just outside each day's cluster.
  plot.setLabels(dayLabels(mp, s2));
  const strip = document.getElementById('wd-strip');
  const W = Math.max(260, Math.min(520, strip.clientWidth || 520));
  const svg = d3.select(strip).append('svg').attr('class', 'chart').attr('width', W).attr('height', 120).attr('viewBox', `0 0 ${W} 120`)
    .attr('role', 'img').attr('aria-label', 'Where each day falls along the arrow');
  svg.append('line').attr('x1', 10).attr('x2', W - 10).attr('y1', 60).attr('y2', 60).attr('stroke', 'var(--rule)');
  const ticks = svg.selectAll('g.t').data(DAYS).join('g').attr('class', 't');
  ticks.append('line').attr('y1', 52).attr('y2', 68).attr('stroke', 'var(--ink)');
  ticks.append('text').attr('text-anchor', 'middle').attr('class', 'label-ink');
  const x = d3.scaleLinear().domain([-s2, s2]).range([24, W - 24]);
  const wout = document.getElementById('wd-readout');
  function drawW() {
    plot.setAngle(angle);
    const proj = mp.map((p) => p.x * Math.cos(angle) + p.y * Math.sin(angle));
    const order = proj.map((v, d) => [v, d]).sort((a, b) => a[0] - b[0]);
    // Each label goes in the first of four rows where it does not overlap the previous label.
    const rowsY = [42, 88, 18, 112], last = [-1e9, -1e9, -1e9, -1e9], rowOf = new Map();
    order.forEach(([v, d]) => { const px = x(v); let r = rowsY.findIndex((_, k) => px - last[k] > 44); if (r < 0) r = 0; last[r] = px; rowOf.set(d, rowsY[r]); });
    ticks.attr('transform', (_, d) => `translate(${x(proj[d])},0)`);
    ticks.select('text').attr('y', (_, d) => rowOf.get(d)).text((d) => d.slice(0, 3));
    let close = null;
    for (let a = 0; a < 7; a++) for (let b = a + 1; b < 7; b++) {
      if (Math.min(b - a, 7 - (b - a)) < 2) continue;
      const dd = Math.abs(proj[a] - proj[b]);
      if (!close || dd < close.dd) close = { a, b, dd };
    }
    wout.innerHTML = `Along the arrow, from tail to head: ${order.map(([, d]) => DAYS[d]).join(', ')}. ` +
      `The closest two days that are not next to each other in the week: <b>${DAYS[close.a]}</b> and <b>${DAYS[close.b]}</b>.`;
  }
  drawW();
  window.__s62 = { setWeekAngle: (a) => { angle = a; drawW(); } };
})();

// ---- Static figure: the statements in the difference-of-means plane, and the weekday circle ----
(function figure() {
  const fig = document.getElementById('fig-dir');
  const box1 = document.createElement('div'), box2 = document.createElement('div');
  box1.innerHTML = '<p class="small-title">City statements, layer 13</p>';
  box2.innerHTML = '<p class="small-title">Days of the week, layer 13</p>';
  fig.append(box1, box2);
  const b = planeBasis(T.X, plainIdx, 'mm', { mm: MM.cities });
  const cities = plain.filter((s) => s.topic === 'cities');
  const cxy = cities.map((s) => [dot(T.X[s.i], b.e1), dot(T.X[s.i], b.e2)]);
  const m1 = [0, 1].map((j) => cxy.reduce((a, p) => a + p[j], 0) / cxy.length);
  const s1 = robustScale(cxy.flatMap((p) => [p[0] - m1[0], p[1] - m1[1]]));
  const p1 = scatterArrow(box1, { height: 300, maxWidth: 340, interactive: false, label: 'True and false city statements with the difference of class means drawn as an arrow' });
  p1.setPoints(cities.map((s, k) => ({ x: cxy[k][0] - m1[0], y: cxy[k][1] - m1[1], kind: s.label ? 'a' : 'b' })), { fixedScale: s1 }).setAngle(0);
  const X = rows(decodeF16(wdJson.X), wdJson.k);
  const { e1, e2, means, center } = weekdayPlane(X, wdJson.day);
  const pts = X.map((x) => { const c = center(x); return { x: dot(c, e1), y: dot(c, e2), kind: 'q' }; });
  const s2 = Math.max(...pts.map((p) => Math.max(Math.abs(p.x), Math.abs(p.y)))) * 1.05;
  const p2 = scatterArrow(box2, { height: 300, maxWidth: 340, arrow: false, interactive: false, label: 'The seven weekday names arranged on a circle in weekly order' });
  p2.setPoints(pts, { fixedScale: s2 * 1.25 });
  p2.setLabels(dayLabels(means.map((m) => ({ x: dot(m, e1), y: dot(m, e2) })), s2));
})();
