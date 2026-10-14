// Page 2.4: watch the plane bend.
import { MLP, datasets } from '../lib/mlp.js';
import { PlaneView, stageNames } from '../lib/plane.js';
import { slider, segmented, button, reducedMotion } from '../lib/ui.js';

const DATA = { xor: datasets.xor(), circles: datasets.circles(), spirals: datasets.spirals() };
// Starting points per dataset, ordered so the first ones train well (see tools/test-mlp.mjs).
const SEEDS = { xor: [1, 2, 4, 5, 6, 7], circles: [1, 2, 3, 4, 5], spirals: [1, 3, 2, 4, 5, 6, 7, 8] };
const HIDDEN = 2;
const LAST = 2 * HIDDEN;
const MAX_STEPS = 6000;

// ---- Static figure: XOR through a width-2 network, as small multiples ----
(function figure() {
  const fig = document.getElementById('fig-xor');
  const net = new MLP([2, 2, 2, 1], { seed: 1 });
  const data = DATA.xor;
  for (let i = 0; i < 1500; i++) net.step(data.X, data.Y);
  const panels = [[0, 'Input'], [1, 'Layer 1, linear map'], [2, 'Layer 1, activation'], [4, "Layer 2, with the output's line"]];
  for (const [t, title] of panels) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${title}</p>`;
    const c = document.createElement('canvas');
    c.className = 'mini';
    c.setAttribute('aria-hidden', 'true');
    box.append(c);
    fig.append(box);
    requestAnimationFrame(() => new PlaneView(c, { dim: 2, interactive: false, gridLines: 9 }).draw(net, data, t, { showBoundary: true }));
  }
})();

// ---- The tool ----
const state = { dataset: 'xor', width: 2, act: 'tanh', seedIx: 0, t: 0, training: false, steps: 0 };
let net, raf = null;
const canvas = document.getElementById('plane');
const view = new PlaneView(canvas, { dim: 2 });
const names = stageNames(HIDDEN);
const out = document.getElementById('readout');

function reset() {
  stop();
  net = new MLP([2, state.width, state.width, 1], { seed: SEEDS[state.dataset][state.seedIx % SEEDS[state.dataset].length], act: state.act, lr: 0.03 });
  state.steps = 0;
  view.setDim(state.width);
  draw();
}

function draw(smooth = false) {
  const data = DATA[state.dataset];
  view.draw(net, data, state.t, { shadeInput: true, showBoundary: true, smooth });
  const acc = net.accuracy(data.X, data.Y), loss = net.loss(data.X, data.Y);
  const k = Math.round(state.t);
  const where = Math.abs(state.t - k) < 1e-3 ? names[k] : `between ${names[Math.floor(state.t)]} and ${names[Math.ceil(state.t)]}`;
  out.innerHTML = `Training step <b>${state.steps.toLocaleString()}</b>. Accuracy on the training points: <b>${Math.round(acc * 100)}%</b>; loss ${loss.toFixed(3)}. Showing: ${where}.` +
    (state.steps >= MAX_STEPS ? ' Training stopped at 6,000 steps.' : '') +
    (state.width === 3 ? ' Drag the picture to turn it.' : '');
}

function loop() {
  const data = DATA[state.dataset];
  for (let i = 0; i < 10 && state.steps < MAX_STEPS; i++) { net.step(data.X, data.Y); state.steps++; }
  draw(true);
  if (state.steps >= MAX_STEPS) { stop(); return; }
  raf = requestAnimationFrame(loop);
}
function start() { if (state.training) return; state.training = true; trainBtn.textContent = 'Pause'; raf = requestAnimationFrame(loop); }
function stop() { state.training = false; cancelAnimationFrame(raf); if (trainBtn) trainBtn.textContent = 'Train'; }

const ctl = document.getElementById('controls');
segmented(ctl, { name: 'Dataset', options: [['xor', 'XOR'], ['circles', 'Circles'], ['spirals', 'Spirals']], value: state.dataset,
  onChange: (k) => { state.dataset = k; state.seedIx = 0; reset(); } });
segmented(ctl, { name: 'Width (units per hidden layer)', options: [['2', '2'], ['3', '3']], value: '2',
  onChange: (k) => { state.width = Number(k); reset(); } });
segmented(ctl, { name: 'Activation function', options: [['tanh', 'tanh'], ['relu', 'ReLU']], value: 'tanh',
  onChange: (k) => { state.act = k; reset(); } });
const btnRow = document.createElement('div');
btnRow.className = 'seg';
ctl.append(btnRow);
const trainBtn = button(btnRow, 'Train', () => (state.training ? stop() : start()), 'primary');
button(btnRow, 'New start', () => { state.seedIx++; reset(); });
const stageS = slider(ctl, { name: 'Step through the network', min: 0, max: LAST, step: 0.01, value: 0,
  format: (v) => (Math.abs(v - Math.round(v)) < 1e-3 ? names[Math.round(v)] : '…'),
  onInput: (v) => { state.t = v; if (!state.training) draw(false); } });
let playRaf = null;
button(ctl, 'Play through the layers', () => {
  cancelAnimationFrame(playRaf);
  if (reducedMotion()) { state.t = LAST; stageS.set(LAST); draw(); return; }
  const t0 = performance.now(), dur = 6000;
  const tick = (now) => {
    const f = Math.min(1, (now - t0) / dur);
    state.t = Math.round(f * LAST * 100) / 100; stageS.set(state.t);
    if (!state.training) draw(true);
    if (f < 1) playRaf = requestAnimationFrame(tick);
  };
  playRaf = requestAnimationFrame(tick);
});
// Arrow and page keys move the step slider one whole step at a time.
stageS.el.addEventListener('keydown', (e) => {
  const up = ['ArrowRight', 'ArrowUp', 'PageUp'].includes(e.key), down = ['ArrowLeft', 'ArrowDown', 'PageDown'].includes(e.key);
  if (!up && !down) return;
  e.preventDefault();
  state.t = up ? Math.min(LAST, Math.floor(state.t + 1e-6) + 1) : Math.max(0, Math.ceil(state.t - 1e-6) - 1);
  stageS.set(state.t);
  if (!state.training) draw(false);
});
view.onRotate = () => { if (!state.training) draw(false); };
window.addEventListener('resize', () => draw(false));

reset();
