// Page 5.4: grok it yourself. Precomputed runs (tools/modadd), scrubbed in the browser.
import { chart } from '../lib/chart.js';
import { slider, segmented, button, reducedMotion } from '../lib/ui.js';
import { loadRun, bits, topShare, drawPairGrid } from '../lib/modadd.js';

const d3 = window.d3;
const TOP = 5; // the concentration measure: share of the embedding in its 5 largest frequencies
const pct = (v) => `${Math.round(v * 100)}%`;
const stepText = (s) => s.toLocaleString('en-US');

// Accuracy at a step, read from the dense curve (the last recorded value at or before it).
function at(curve, key, step) {
  let j = 0;
  while (j + 1 < curve.step.length && curve.step[j + 1] <= step) j++;
  return curve[key][j];
}

// ---- Static figure: the main run's accuracy curves on a log axis ----
async function figure() {
  const run = await loadRun(1.0, 0.3);
  const box = document.getElementById('fig-grok');
  const nar = (box.clientWidth || 620) < 440;
  const c = chart(box, { width: 620, height: 260, margin: { top: 30, right: nar ? 84 : 185, bottom: 52, left: 52 },
    x: { domain: [1, run.steps_total], log: true, ticks: nar ? 3 : 4, format: '~s', label: nar ? 'step (log scale)' : 'training step (log scale)' },
    y: { domain: [0, 1], ticks: 3, format: '.0%', label: 'accuracy' },
    title: 'Training and test accuracy of the main run against training step' });
  const pts = (k) => run.curve.step.map((s, i) => [Math.max(s, 1), run.curve[k][i]]);
  c.line('train', pts('train_acc'), { color: 'second', label: nar ? 'training' : 'training pairs' });
  c.line('test', pts('test_acc'), { color: 'accent', label: nar ? 'test' : 'test pairs' });
  c.line('restricted', run.ckpt.step.map((s, i) => [Math.max(s, 1), run.ckpt.restricted_test_acc[i]]),
    { color: 'ink', width: 1.2, dash: '4 3', label: nar ? 'key part' : 'key-frequency part' });
  c.spreadLabels(['train', 'test', 'restricted'], 18);
}

// ---- The tool ----
const state = { wd: 1.0, frac: 0.3, i: 0 };
let run = null, mask = null, correct = [], token = 0;
const out = document.getElementById('readout');
const gridCanvas = document.getElementById('pairs');

const ctl = document.getElementById('controls');
segmented(ctl, { name: 'Weight decay λ', options: [['0', '0'], ['0.3', '0.3'], ['1', '1']], value: '1',
  onChange: (k) => { state.wd = Number(k); load(); } });
segmented(ctl, { name: 'Training pairs', options: [['0.3', '30%'], ['0.5', '50%'], ['0.7', '70%']], value: '0.3',
  onChange: (k) => { state.frac = Number(k); load(); } });
const stepS = slider(ctl, { name: 'Training step', min: 0, max: 1, step: 1, value: 0,
  format: (i) => (run ? stepText(run.ckpt.step[i]) : '…'),
  onInput: (i) => { stop(); state.i = i; draw(); } });
const playBtn = button(ctl, 'Play through training', () => (raf ? stop() : play()), 'primary');

// Accuracy chart; dragging on it sets the step.
const narrow = (document.getElementById('acc-chart').clientWidth || 560) < 440;
const LBL = narrow ? { train: 'training', test: 'test', share: 'share' } : { train: 'training pairs', test: 'test pairs', share: 'frequency share' };
const acc = chart(document.getElementById('acc-chart'), {
  height: 300, margin: { top: 24, right: narrow ? 84 : 140, bottom: 52, left: 56 },
  x: { domain: [1, 20000], log: true, ticks: narrow ? 3 : 4, format: '~s', label: narrow ? 'step (log scale)' : 'training step (log scale)' },
  y: { domain: [0, 1], ticks: 3, format: '.0%' },
  title: 'Accuracy on training and test pairs, and the share of the embedding in its five largest frequencies, against training step' });
const setFromPointer = (ev) => {
  if (!run) return;
  const [mx] = d3.pointer(ev, acc.g.node());
  const s = acc.x.invert(Math.max(0, Math.min(acc.iw, mx)));
  const steps = run.ckpt.step;
  let best = 0;
  for (let j = 0; j < steps.length; j++) if (Math.abs(Math.log(Math.max(steps[j], 1)) - Math.log(s)) < Math.abs(Math.log(Math.max(steps[best], 1)) - Math.log(s))) best = j;
  stop(); state.i = best; stepS.set(best); draw();
};
acc.svg.on('pointerdown', (ev) => { setFromPointer(ev); acc.svg.on('pointermove', setFromPointer); })
  .on('pointerup pointerleave', () => acc.svg.on('pointermove', null));

// Frequency chart: one stem per frequency k = 1, ..., (p-1)/2.
let freq = null;
function makeFreqChart(p, yMax) {
  const box = document.getElementById('freq-chart');
  box.innerHTML = '';
  const K = (p - 1) / 2;
  freq = chart(box, { width: 420, height: 240, margin: { top: 34, right: 16, bottom: 52, left: 44 },
    x: { domain: [0, K + 1], ticks: 4, label: 'frequency k' }, y: { domain: [0, yMax], ticks: 3, label: 'size in the embedding' },
    title: 'Size of each frequency in the embedding at the chosen step' });
  freq.K = K;
}

async function load() {
  stop();
  const my = ++token;
  out.textContent = 'Loading the run…';
  const r = await loadRun(state.wd, state.frac);
  if (my !== token) return;
  run = r;
  mask = bits(run.train_mask, run.p * run.p);
  correct = run.ckpt.correct.map((b) => bits(b, run.p * run.p));
  run.share = run.ckpt.spectrum.map((s) => topShare(s, TOP));
  stepS.el.max = run.ckpt.step.length - 1;
  state.i = Math.min(state.i, run.ckpt.step.length - 1);
  stepS.set(state.i);
  acc.setX([1, run.steps_total]);
  const pts = (k) => run.curve.step.map((s, i) => [Math.max(s, 1), run.curve[k][i]]);
  acc.line('train', pts('train_acc'), { color: 'second', label: LBL.train });
  acc.line('test', pts('test_acc'), { color: 'accent', label: LBL.test });
  acc.line('share', run.ckpt.step.map((s, i) => [Math.max(s, 1), run.share[i]]), { color: 'ink', width: 1.2, dash: '4 3', label: LBL.share });
  acc.spreadLabels(['train', 'test', 'share'], 22);
  makeFreqChart(run.p, Math.max(...run.ckpt.spectrum.flat()) * 1.05);
  draw();
}

function draw() {
  if (!run) return;
  const i = state.i, s = run.ckpt.step[i];
  acc.vline('now', Math.max(s, 1), { color: 'quiet', label: null });
  drawPairGrid(gridCanvas, run.p, correct[i], mask);
  const spec = run.ckpt.spectrum[i];
  const order = spec.map((v, k) => k).sort((a, b) => spec[b] - spec[a]);
  for (let k = 1; k <= freq.K; k++) {
    const v = spec[k - 1];
    freq.line(`k${k}`, [[k, 0], [k, v]], { color: 'ink', width: 3 });
  }
  // Label the largest frequencies with their numbers.
  // Only frequencies that stand out (more than twice the median size) get a number.
  const median = spec.slice().sort((a, b) => a - b)[Math.floor(spec.length / 2)];
  for (let r = 0; r < 3; r++) {
    const k = order[r] + 1, big = spec[k - 1] > 2 * median;
    freq.text(`t${r}`, k, spec[k - 1] + freq.y.domain()[1] * 0.04, big ? String(k) : '', { color: 'ink', anchor: 'middle' });
  }
  const tr = at(run.curve, 'train_acc', s), te = at(run.curve, 'test_acc', s);
  out.innerHTML = `Step <b>${stepText(s)}</b> of ${stepText(run.steps_total)}. The model answers <span class="c-second">${pct(tr)} of the training pairs</span> and <span class="c-accent">${pct(te)} of the test pairs</span> correctly. The five largest frequencies hold ${pct(run.share[i])} of the embedding (an even spread would give ${pct(TOP / freq.K)}); the three largest are ${order.slice(0, 3).map((k) => k + 1).join(', ')}.`;
}

let raf = null;
function play() {
  if (!run) return;
  if (state.i >= run.ckpt.step.length - 1) state.i = 0;
  if (reducedMotion()) { state.i = run.ckpt.step.length - 1; stepS.set(state.i); draw(); return; }
  playBtn.textContent = 'Pause';
  let last = 0;
  const tick = (now) => {
    if (now - last > 90) {
      last = now;
      state.i++;
      stepS.set(state.i);
      draw();
      if (state.i >= run.ckpt.step.length - 1) { stop(); return; }
    }
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}
function stop() { cancelAnimationFrame(raf); raf = null; playBtn.textContent = 'Play through training'; }

let rt;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 100); });

load();
figure();
