// Page 5.1: build any function.
import { chart } from '../lib/chart.js';
import { slider, segmented, button } from '../lib/ui.js';
import { TARGETS, relu, handNet, rms, fitEven, deepSaw, sawtooth } from '../lib/s45-approx.js';

const d3 = window.d3;
const XS = d3.range(0, 1.0001, 0.0025);

// A target the reader draws: 101 values, linearly interpolated.
const drawn = Array.from({ length: 101 }, (_, i) => 0.5 + 0.3 * Math.sin(4 * Math.PI * i / 100) * (i / 100));
const drawnF = (x) => { const u = Math.min(100, Math.max(0, x * 100)); const i = Math.min(99, Math.floor(u)), t = u - i; return drawn[i] * (1 - t) + drawn[i + 1] * t; };
const TNAMES = [['bump', 'Bump'], ['smooth', 'Smooth'], ['jagged', 'Jagged'], ['drawn', 'Draw your own']];
const targetFn = (k) => (k === 'drawn' ? drawnF : TARGETS[k]);
const curve = (f) => XS.map((x) => [x, f(x)]);
const drawnListeners = [];

// Lets the reader draw a target by dragging on a chart.
function enableDrawing(c, active, onChange) {
  let last = null;
  const at = (ev) => {
    const [mx, my] = d3.pointer(ev, c.g.node());
    const x = Math.min(1, Math.max(0, c.x.invert(mx))), y = Math.min(1.2, Math.max(-0.2, c.y.invert(my)));
    const i = Math.round(x * 100);
    if (last !== null) { const [i0, y0] = last; const lo = Math.min(i0, i), hi = Math.max(i0, i); for (let j = lo; j <= hi; j++) drawn[j] = hi === lo ? y : y0 + ((y - y0) * (j - i0)) / (i - i0); } else drawn[i] = y;
    last = [i, y];
    onChange();
  };
  c.svg.on('pointerdown.draw', (ev) => { if (!active()) return; last = null; at(ev); c.svg.on('pointermove.draw', at); })
    .on('pointerup.draw pointerleave.draw', () => { last = null; c.svg.on('pointermove.draw', null); drawnListeners.forEach((f) => f()); });
}

// ---- Static figure: smooth target with 4, 8, 32 units ----
(function figure() {
  const fig = document.getElementById('fig-approx');
  for (const m of [4, 8, 32]) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${m} units</p>`;
    fig.append(box);
    const c = chart(box, { width: 300, height: 220, margin: { top: 14, right: 16, bottom: 36, left: 40 },
      x: { domain: [0, 1], ticks: 2 }, y: { domain: [-0.2, 1], ticks: 3 }, title: `Fit with ${m} units` });
    const fit = fitEven(TARGETS.smooth, m);
    fit.units.forEach((u, k) => c.line(`u${k}`, curve((x) => u.a * relu(x - u.c)), { color: 'quiet', width: 0.6 }));
    c.line('t', curve(TARGETS.smooth), { color: 'ink', width: 2.5 });
    c.line('f', curve(fit.f), { color: 'accent', width: 1.4 });
  }
})();

// ---- Panel 1: by hand ----
(function byHand() {
  const st = { target: 'bump', b: 0, units: [], sel: -1 };
  const ctl = document.getElementById('h-controls'), out = document.getElementById('h-readout');
  const c = chart(document.getElementById('h-chart'), { height: 320, margin: { top: 34, right: 30, bottom: 52, left: 52 },
    x: { domain: [0, 1], label: 'x' }, y: { domain: [-0.3, 1.1], label: 'output' }, title: 'Target and the sum of the units' });
  segmented(ctl, { name: 'Target', options: TNAMES, value: st.target, onChange: (k) => { st.target = k; draw(); } });
  const addRow = document.createElement('div'); addRow.className = 'seg'; ctl.append(addRow);
  button(addRow, 'Add a unit', () => { st.units.push({ c: 0.5, a: 1 }); st.sel = st.units.length - 1; sync(); draw(); }, 'primary');
  button(addRow, 'Remove it', () => { if (st.sel < 0) return; st.units.splice(st.sel, 1); st.sel = st.units.length - 1; sync(); draw(); });
  button(addRow, 'Clear', () => { st.units = []; st.sel = -1; st.b = 0; bS.set(0); sync(); draw(); });
  const selWrap = document.createElement('div'); ctl.append(selWrap);
  const cS = slider(ctl, { name: 'Kink of the chosen unit', min: 0, max: 1, step: 0.01, value: 0.5, format: (v) => v.toFixed(2),
    onInput: (v) => { if (st.sel >= 0) { st.units[st.sel].c = v; draw(); } } });
  const aS = slider(ctl, { name: 'Slope of the chosen unit', min: -10, max: 10, step: 0.1, value: 1, format: (v) => v.toFixed(1),
    onInput: (v) => { if (st.sel >= 0) { st.units[st.sel].a = v; draw(); } } });
  const bS = slider(ctl, { name: 'Constant b', min: -1, max: 1, step: 0.01, value: 0, format: (v) => v.toFixed(2),
    onInput: (v) => { st.b = v; draw(); } });
  function sync() {
    selWrap.innerHTML = '';
    if (st.units.length) {
      segmented(selWrap, { name: 'Chosen unit', options: st.units.map((u, k) => [String(k), String(k + 1)]), value: String(st.sel),
        onChange: (k) => { st.sel = Number(k); sync(); draw(); } });
    }
    const u = st.units[st.sel];
    cS.el.disabled = aS.el.disabled = !u;
    if (u) { cS.set(u.c); aS.set(u.a); }
  }
  // Drag a unit's kink along the axis.
  let dragging = -1;
  c.svg.on('pointerdown.kink', (ev) => {
    if (st.target === 'drawn') return;
    const [mx] = d3.pointer(ev, c.g.node());
    let best = -1, bd = 14;
    st.units.forEach((u, k) => { const d = Math.abs(c.x(u.c) - mx); if (d < bd) { bd = d; best = k; } });
    if (best < 0) return;
    dragging = best; st.sel = best; sync();
    c.svg.on('pointermove.kink', (e) => {
      const [x] = d3.pointer(e, c.g.node());
      st.units[dragging].c = Math.round(Math.min(1, Math.max(0, c.x.invert(x))) * 100) / 100;
      cS.set(st.units[dragging].c); draw();
    });
  }).on('pointerup.kink pointerleave.kink', () => { dragging = -1; c.svg.on('pointermove.kink', null); });
  enableDrawing(c, () => st.target === 'drawn', () => draw());
  drawnListeners.push(() => { if (st.target === 'drawn') draw(); });

  let shown = 0;
  function draw() {
    const T = targetFn(st.target), f = handNet(st.b, st.units);
    for (let k = 0; k < shown; k++) { c.remove(`u${k}`); c.remove(`h${k}`); }
    st.units.forEach((u, k) => {
      c.line(`u${k}`, curve((x) => u.a * relu(x - u.c)), { color: 'quiet', width: k === st.sel ? 1.6 : 0.8, dash: '4 3' });
      c.point(`h${k}`, u.c, -0.3, { color: k === st.sel ? 'accent' : 'quiet', r: 6, label: String(k + 1), dx: 0, dy: -10, anchor: 'middle' });
    });
    shown = st.units.length;
    c.line('t', curve(T), { color: 'ink', width: 2.5 });
    c.line('f', curve(f), { color: 'accent', width: 1.6 });
    const e = rms(f, T);
    out.innerHTML = `Units: <b>${st.units.length}</b>. Error: <b>${e.toFixed(3)}</b>.` +
      (st.units.length ? ` Unit ${st.sel + 1}: kink ${st.units[st.sel].c.toFixed(2)}, slope ${st.units[st.sel].a.toFixed(1)}.` : ' Press Add a unit.');
  }
  sync(); draw();
})();

// ---- Panel 2: fit ----
(function fit() {
  const st = { target: 'jagged', m: 8 };
  const ctl = document.getElementById('f-controls'), out = document.getElementById('f-readout');
  const c = chart(document.getElementById('f-chart'), { height: 300, margin: { top: 34, right: 30, bottom: 52, left: 52 },
    x: { domain: [0, 1], label: 'x' }, y: { domain: [-0.3, 1.1], label: 'output' }, title: 'Target and the fitted network' });
  const e = chart(document.getElementById('f-err'), { height: 280, margin: { top: 34, right: 60, bottom: 52, left: 64 },
    x: { domain: [1, 128], log: true, ticks: 4, format: '~g', label: 'number of units m' }, y: { domain: [1e-4, 0.5], log: true, ticks: 3, format: '~g', label: 'error' },
    title: 'Error against the number of units' });
  segmented(ctl, { name: 'Target', options: TNAMES, value: st.target, onChange: (k) => { st.target = k; full(); } });
  slider(ctl, { name: 'Number of units m', min: 1, max: 128, step: 1, value: st.m, format: (v) => String(v),
    onInput: (v) => { st.m = v; draw(); } });
  enableDrawing(c, () => st.target === 'drawn', () => draw());
  drawnListeners.push(() => { if (st.target === 'drawn') full(); });
  let errs = [];
  function full() {
    const T = targetFn(st.target);
    errs = d3.range(1, 129).map((m) => [m, Math.max(1e-4, rms(fitEven(T, m).f, T))]);
    e.line('err', errs, { color: 'ink', width: 1.2 });
    e.line('thr', [[1, 0.01], [128, 0.01]], { color: 'quiet', dash: '3 3', label: '0.01' });
    draw();
  }
  function draw() {
    const T = targetFn(st.target), F = fitEven(T, st.m);
    c.line('t', curve(T), { color: 'ink', width: 2.5 });
    c.line('f', curve(F.f), { color: 'accent', width: 1.6 });
    const er = rms(F.f, T);
    e.point('m', st.m, Math.max(1e-4, er), { color: 'accent', r: 5, label: null });
    const first = errs.find((p) => p[1] < 0.01);
    out.innerHTML = `With <span class="c-accent">\\(m\\) = <b>${st.m}</b></span> units the error is <b>${er < 0.1 ? er.toFixed(4) : er.toFixed(3)}</b>.` +
      (first ? ` For this target the error first falls below 0.01 at ${first[0]} units.` : ' For this target the error stays above 0.01 up to 128 units.');
    if (window.renderMathInElement) window.renderMathInElement(out, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
  }
  full();
})();

// ---- Panel 3: depth ----
(function depth() {
  const st = { k: 2 };
  const ctl = document.getElementById('d-controls'), out = document.getElementById('d-readout');
  const c = chart(document.getElementById('d-chart'), { height: 300, margin: { top: 34, right: 30, bottom: 52, left: 52 },
    x: { domain: [0, 1], label: 'x' }, y: { domain: [-0.2, 1.2], label: 'output' }, title: 'Sawtooth, deep network, and shallow network' });
  slider(ctl, { name: 'Hidden layers of the deep network k', min: 1, max: 6, step: 1, value: st.k, format: (v) => String(v),
    onInput: (v) => { st.k = v; draw(); } });
  const fine = d3.range(0, 1.00001, 1 / 1024);
  function draw() {
    const k = st.k, T = sawtooth(k), D = deepSaw(k), S = fitEven(T, 2 * k);
    c.line('t', fine.map((x) => [x, T(x)]), { color: 'ink', width: 4 });
    c.line('d', fine.map((x) => [x, D(x)]), { color: 'accent', width: 1.6 });
    c.line('s', fine.map((x) => [x, S.f(x)]), { color: 'second', width: 1.6 });
    const es = rms(S.f, T), ed = rms(D, T);
    const piecesS = 2 * k + 1;
    out.innerHTML = `Both networks have <b>${2 * k}</b> units. The sawtooth has ${2 ** k} straight pieces. ` +
      `<span class="c-accent">Deep network</span> (${k} layer${k > 1 ? 's' : ''} of 2): error <b>${ed.toFixed(3)}</b>. ` +
      `<span class="c-second">Shallow network</span> (1 layer of ${2 * k}): error <b>${es.toFixed(3)}</b>, at most ${piecesS} pieces. ` +
      `To match the sawtooth exactly, one hidden layer needs at least ${2 ** k - 1} unit${k > 1 ? 's' : ''}.`;
  }
  draw();
})();
