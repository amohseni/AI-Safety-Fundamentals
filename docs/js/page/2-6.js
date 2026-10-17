// Page 2.6: roll downhill.
import { SURFACES, makeRun, basin } from '../lib/s2-optim.js';
import { chart } from '../lib/chart.js';
import { slider, segmented, button, reducedMotion } from '../lib/ui.js';
import { bands } from '../lib/s2-contour.js';

const d3 = window.d3;
const TOL = 1e-4;
const MAX = { bowl: 3000, valley: 5000, rosenbrock: 20000, two: 1000 };
const OPT = { gd: 'plain gradient descent', momentum: 'momentum', adam: 'Adam' };
const sig3 = (v) => Number(v.toPrecision(3));
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const pow10 = (d) => { const e = Math.round(Math.log10(d)); return e % 2 !== 0 ? '' : e === 0 ? '1' : `10${String(e).split('').map((c) => SUP[c]).join('')}`; };

// Contour map of a surface inside a chart made by chart().
function contourMap(c, S) {
  const levels = S.levels === 'log' ? d3.range(-2, 3.01, 0.25).map((k) => 10 ** k) : d3.range(0.02, 1.6, 0.06);
  const g = c.g.insert('g', ':first-child');
  // Bands of minus the value, so each region is "below a level" and stacking makes low ground darker.
  bands(g, { f: (x, y) => -(S.f(x, y) - S.min), xd: S.xd, yd: S.yd, iw: c.iw, ih: c.ih, n: 160,
    thresholds: levels.map((v) => -v).reverse(), opacity: 0.45 / levels.length });
  return g;
}
function squareChart(el, S, extra = {}) {
  const m = { top: 30, right: 20, bottom: 50, left: 46 };
  const w = Math.max(280, Math.min(extra.width || 440, el.clientWidth || 440));
  return chart(el, { width: w, height: w - m.left - m.right + m.top + m.bottom, margin: m,
    x: { domain: S.xd, label: 'x', ticks: 5 }, y: { domain: S.yd, label: 'y', ticks: 5 }, title: extra.title || `Map of the ${S.name.toLowerCase()} loss` });
}

// ---- Static figure: three learning rates on the narrow valley, as small multiples ----
(function figure() {
  const S = SURFACES.valley;
  const fig = document.getElementById('fig-paths');
  for (const lr of [0.002, 0.019, 0.021]) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">Learning rate ${lr}</p>`;
    fig.append(box);
    const m = { top: 10, right: 12, bottom: 34, left: 34 }, w = 250;
    const c = chart(box, { width: w, height: w - m.left - m.right + m.top + m.bottom, margin: m, x: { domain: S.xd, ticks: 3 }, y: { domain: S.yd, ticks: 3 },
      title: `Gradient descent path at learning rate ${lr} on the narrow valley` });
    contourMap(c, S);
    const r = makeRun(S, { kind: 'gd', lr });
    for (let i = 0; i < 30; i++) r.step();
    const k = r.path.findIndex((p) => Math.abs(p[0]) > 3 || Math.abs(p[1]) > 3);
    c.line('p', k < 0 ? r.path : r.path.slice(0, k + 1), { color: 'accent', width: 1.3 });
    c.point('s', S.start[0], S.start[1], { color: 'ink', r: 3 });
  }
})();

// ---- The tool ----
const state = { surface: 'valley', kind: 'gd', lr: SURFACES.valley.lr, noise: 0, speed: 5, start: SURFACES.valley.start.slice(), seed: 1, batchSeed: 1 };
let run = null, prev = null, raf = null, playing = false, surfC, lossC, many = [];
const ctl = document.getElementById('controls');
const out = document.getElementById('readout');

segmented(ctl, { name: 'Surface', options: Object.entries(SURFACES).map(([k, s]) => [k, s.name]), value: state.surface, onChange: (k) => setSurface(k) });
segmented(ctl, { name: 'Optimizer', options: [['gd', 'Plain'], ['momentum', 'Momentum'], ['adam', 'Adam']], value: state.kind,
  onChange: (k) => { state.kind = k; reset(true); } });
const lrS = slider(ctl, { name: 'Learning rate', min: -4, max: 0, step: 0.01, value: Math.log10(state.lr), format: (v) => String(sig3(10 ** v)),
  onInput: (v) => { state.lr = sig3(10 ** v); reset(true); } });
// Arrow keys on the learning-rate slider move by one hundredth of a decade; Page Up and Page Down by a tenth.
const noiseS = slider(ctl, { name: 'Noise added to each gradient', min: 0, max: 3, step: 0.5, value: 0, format: (v) => v.toFixed(1),
  onInput: (v) => { state.noise = v; reset(true); } });
slider(ctl, { name: 'Speed (steps per frame)', min: 1, max: 50, step: 1, value: state.speed, format: (v) => String(v), onInput: (v) => { state.speed = v; } });
const btns = document.createElement('div');
btns.className = 'seg';
ctl.append(btns);
const runBtn = button(btns, 'Run', () => (playing ? pause() : play()), 'primary');
button(btns, 'Step', () => { pause(); stepOnce(); draw(); });
button(btns, 'Reset', () => reset(true));
const tenBtn = button(btns, 'Ten runs', tenRuns);

function setSurface(k) {
  state.surface = k;
  const S = SURFACES[k];
  state.lr = S.lr; lrS.set(Math.log10(S.lr));
  state.start = S.start.slice();
  prev = null;
  build();
}

function build() {
  pause();
  const S = SURFACES[state.surface];
  for (const id of ['surface-chart', 'loss-chart']) document.getElementById(id).innerHTML = '';
  surfC = squareChart(document.getElementById('surface-chart'), S);
  contourMap(surfC, S);
  surfC.many = surfC.g.append('g');
  surfC.svg.attr('tabindex', 0).attr('aria-label', `${surfC.svg.attr('aria-label')}. Press to set the starting point; arrow keys move it.`);
  surfC.svg.on('pointerdown', (ev) => {
    const [mx, my] = d3.pointer(ev, surfC.g.node());
    state.start = [Math.round(surfC.x.invert(mx) * 100) / 100, Math.round(surfC.y.invert(my) * 100) / 100]
      .map((v, i) => Math.max([S.xd, S.yd][i][0], Math.min([S.xd, S.yd][i][1], v)));
    reset(true);
  });
  surfC.svg.on('keydown', (ev) => {
    const d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[ev.key];
    if (!d) return;
    ev.preventDefault();
    state.start = [Math.round((state.start[0] + d[0]) * 100) / 100, Math.round((state.start[1] + d[1]) * 100) / 100];
    reset(true);
  });
  const lw = document.getElementById('loss-chart').clientWidth || 440;
  lossC = chart(document.getElementById('loss-chart'), { width: Math.min(440, lw), height: 220, margin: { top: 30, right: 84, bottom: 50, left: 58 },
    x: { domain: [0, 100], label: 'step', ticks: 3 }, y: { domain: [1e-5, 1e3], log: true, ticks: 3, format: pow10, label: 'loss above lowest' }, title: 'Loss above its lowest value, against step' });
  tenBtn.disabled = state.surface !== 'two';
  reset(false);
}

function reset(keepPrev) {
  pause();
  if (keepPrev && run && run.t > 0) prev = run;
  many = [];
  run = makeRun(SURFACES[state.surface], { kind: state.kind, lr: state.lr, noise: state.noise, seed: state.seed++, start: state.start });
  draw();
}

function stepOnce() { if (run.t < MAX[state.surface] && !run.diverged) run.step(); }
function done() { return run.diverged || run.t >= MAX[state.surface] || (state.surface !== 'two' && state.noise === 0 && run.losses[run.losses.length - 1] - SURFACES[state.surface].min < 1e-9); }
function play() {
  if (done()) reset(true);
  playing = true; runBtn.textContent = 'Pause';
  if (reducedMotion()) { while (!done()) stepOnce(); pause(); draw(); return; }
  const tick = () => {
    for (let i = 0; i < state.speed && !done(); i++) stepOnce();
    draw();
    if (done()) { pause(); return; }
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}
function pause() { playing = false; cancelAnimationFrame(raf); runBtn.textContent = 'Run'; }

function thin(path, n = 1500) { if (path.length <= n) return path; const k = Math.ceil(path.length / n); return path.filter((_, i) => i % k === 0 || i === path.length - 1); }
const clip = (p, S) => [Math.max(S.xd[0] - 1, Math.min(S.xd[1] + 1, p[0])), Math.max(S.yd[0] - 1, Math.min(S.yd[1] + 1, p[1]))];
function firstWithin(r, S) { const i = r.losses.findIndex((L) => L - S.min < TOL); return i; }

const fmtL = (L) => (!Number.isFinite(L) ? 'infinite' : Math.abs(L) > 1e4 || (Math.abs(L) < 1e-3 && L !== 0) ? L.toExponential(2).replace(/e\+?(-?\d+)/, ' × 10<sup>$1</sup>') : L.toFixed(4));

function draw() {
  const S = SURFACES[state.surface];
  if (prev) surfC.line('prev', thin(prev.path).map((p) => clip(p, S)), { color: 'quiet', width: 1.2, dash: '3 3' }); else surfC.remove('prev');
  surfC.line('path', thin(run.path).map((p) => clip(p, S)), { color: 'accent', width: 1.6 });
  surfC.point('start', state.start[0], state.start[1], { color: 'ink', r: 3.5, label: 'start', dx: 7, dy: -7 });
  const cur = clip(run.p, S);
  surfC.point('cur', cur[0], cur[1], { color: 'accent', r: 4.5 });
  surfC.many.selectAll('circle').data(many).join('circle').attr('cx', (p) => surfC.x(p[0])).attr('cy', (p) => surfC.y(p[1])).attr('r', 4)
    .attr('fill', 'none').attr('stroke', 'var(--accent)').attr('stroke-width', 1.5);
  // Loss chart
  const tmax = Math.max(100, run.t, prev ? Math.min(prev.t, MAX[state.surface]) : 0);
  lossC.setX([0, tmax]);
  const series = (r) => thin(r.losses.map((L, i) => [i, Math.max(1e-6, Math.min(1e5, L - S.min))]), 800);
  if (prev) lossC.line('prev', series(prev), { color: 'quiet', width: 1.2, dash: '3 3', label: 'previous' }); else lossC.remove('prev');
  lossC.line('cur', series(run), { color: 'accent', width: 1.6, label: 'this run' });
  if (prev) lossC.spreadLabels(['prev', 'cur'], 18);
  // Readout
  const L = run.losses[run.losses.length - 1];
  let s = `${OPT[state.kind][0].toUpperCase() + OPT[state.kind].slice(1)}, learning rate ${state.lr}, noise ${state.noise}. Step <b>${run.t.toLocaleString()}</b>; loss ${fmtL(L)}.`;
  if (run.diverged) s += ` <b>Diverged</b> at step ${run.t}.`;
  else {
    const i = firstWithin(run, S);
    if (state.surface !== 'two' && i >= 0) s += ` First within 0.0001 of the lowest value at step <b>${i}</b>.`;
    if (state.surface === 'two' && run.t >= MAX.two) s += ` Ended in the basin of the ${basin(run.p)} minimum (the one that noise-free descent from the end point reaches).`;
    if (run.t >= MAX[state.surface] && state.surface !== 'two' && i < 0) s += ` Stopped at ${MAX[state.surface].toLocaleString()} steps without getting within 0.0001 of the lowest value.`;
  }
  if (many.length) {
    const narrow = many.filter((p) => basin(p) === 'narrow').length;
    s += ` Ten runs with noise ${state.noise}: <b>${narrow}</b> ended in the basin of the narrow minimum and <b>${many.length - narrow}</b> in the basin of the wide one (circles mark the end points).`;
  }
  out.innerHTML = s;
}

// Ten noisy runs from the current start, each for 1,000 steps; their end points are drawn as circles.
function tenRuns() {
  pause();
  const S = SURFACES.two;
  many = [];
  for (let k = 0; k < 10; k++) {
    const r = makeRun(S, { kind: state.kind, lr: state.lr, noise: state.noise, seed: state.batchSeed++, start: state.start });
    for (let i = 0; i < MAX.two; i++) r.step();
    many.push(r.p.slice());
    if (k === 9) { if (run && run.t > 0) prev = run; run = r; }
  }
  draw();
}

let rT;
window.addEventListener('resize', () => { clearTimeout(rT); rT = setTimeout(build, 200); });
build();
