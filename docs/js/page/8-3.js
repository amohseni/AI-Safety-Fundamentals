// Page 8.3: coin or right end?
import { segmented, button, cssVar } from '../lib/ui.js';
import { fitCanvas } from '../lib/gridworld.js';
import { rng } from '../lib/mlp.js';
import { W, H, START, END, MOVES, NA, features, probs, randomCoin, episode, makeTrainer, evaluate, EPISODES } from '../lib/8-coinrun.js';

const SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const st = { pct: 0, seedIx: 0, theta: null, trainer: null, exampleSeed: 1, eval: null, train: null };
const out = document.getElementById('coin-readout');
const exCanvas = document.getElementById('coin-example');
const heatCanvas = document.getElementById('coin-heat');
let job = null;

const ctl = document.getElementById('coin-controls');
segmented(ctl, { name: 'Training levels with the coin placed at random', options: [['0', '0%'], ['0.02', '2%'], ['0.1', '10%'], ['0.5', '50%']], value: '0',
  onChange: (k) => { st.pct = Number(k); st.seedIx = 0; startTraining(); } });
button(ctl, 'New training run', () => { st.seedIx = (st.seedIx + 1) % SEEDS.length; startTraining(); });
button(ctl, 'New test level', () => { st.exampleSeed++; drawExample(); });

function startTraining() {
  clearTimeout(job);
  st.trainer = makeTrainer(st.pct, { seed: SEEDS[st.seedIx] });
  st.theta = null; st.eval = null;
  const slice = () => {
    const done = st.trainer.run(600);
    if (!done) { out.textContent = `Training on ${EPISODES.toLocaleString('en-US')} levels: ${Math.round((100 * st.trainer.done) / EPISODES)}%`; job = setTimeout(slice, 0); return; }
    st.theta = st.trainer.theta;
    st.train = evaluate(st.theta, { pRandom: 0, levels: 500, seed: 7 });
    st.eval = evaluate(st.theta, { levels: 2000, record: true });
    drawAll();
  };
  out.textContent = 'Training…';
  job = setTimeout(slice, 0);
}

function layout(canvas) {
  const avail = Math.min(470, canvas.parentElement.clientWidth || 470);
  const c = Math.floor((avail - 2) / W);
  const ctx = fitCanvas(canvas, c * W + 2, c * H + 2);
  return { ctx, c, cx: (x) => 1 + (x + 0.5) * c, cy: (y) => 1 + (y + 0.5) * c };
}
const colors = () => ({ ink: cssVar('--ink'), quiet: cssVar('--quiet'), rule: cssVar('--rule'), accent: cssVar('--accent'), second: cssVar('--second'), faint: cssVar('--faint') });

export function drawLevel(canvas, { coin, path, heat = null }) {
  const { ctx, c, cx, cy } = layout(canvas);
  const col = colors();
  ctx.clearRect(0, 0, c * W + 2, c * H + 2);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (heat) { ctx.fillStyle = col.accent; ctx.globalAlpha = Math.min(1, heat[y * W + x]) * 0.85; ctx.fillRect(1 + x * c, 1 + y * c, c, c); ctx.globalAlpha = 1; }
    ctx.strokeStyle = col.rule; ctx.lineWidth = 1; ctx.strokeRect(1.5 + x * c, 1.5 + y * c, c - 1, c - 1);
  }
  // The right end of the middle row: a short bar on the right wall.
  ctx.fillStyle = col.ink; ctx.fillRect(1 + W * c - 3, 1 + END[1] * c + c * 0.15, 3, c * 0.7);
  // Start.
  ctx.strokeStyle = col.quiet; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx(START[0]), cy(START[1]), c * 0.22, 0, Math.PI * 2); ctx.stroke();
  if (coin) { ctx.fillStyle = col.second; ctx.beginPath(); ctx.arc(cx(coin[0]), cy(coin[1]), c * 0.26, 0, Math.PI * 2); ctx.fill(); }
  if (path) {
    ctx.strokeStyle = col.accent; ctx.lineWidth = 2.5; ctx.beginPath();
    path.forEach(([x, y], i) => (i ? ctx.lineTo(cx(x), cy(y)) : ctx.moveTo(cx(x), cy(y))));
    ctx.stroke();
    const [lx, ly] = path[path.length - 1];
    ctx.fillStyle = col.accent; ctx.beginPath(); ctx.arc(cx(lx), cy(ly), c * 0.16, 0, Math.PI * 2); ctx.fill();
  }
}

function drawExample() {
  if (!st.theta) return;
  const r = rng(500 + st.exampleSeed);
  const coin = randomCoin(r);
  const ep = episode(st.theta, coin, r, { record: true });
  drawLevel(exCanvas, { coin, path: ep.path });
  st.example = { coin, got: ep.got, steps: ep.steps };
  readout();
}

function drawAll() {
  const v = st.eval.visits, n = 2000;
  drawLevel(heatCanvas, { heat: Array.from(v, (k) => k / n) });
  drawExample();
}

function readout() {
  const pc = (x) => `${Math.round(100 * x)}%`;
  const ex = st.example;
  out.innerHTML = `Training run ${st.seedIx + 1} of ${SEEDS.length}, with ${pc(st.pct)} of training levels randomized. ` +
    `On standard levels the agent reaches the coin in <b>${pc(st.train.success)}</b> of 500. ` +
    `On 2,000 test levels it reaches the coin in <span class="c-accent"><b>${pc(st.eval.success)}</b></span> and visits the right end of the middle row in ${pc(st.eval.toEnd)}. ` +
    (ex ? `In the test level shown, the coin is at column ${ex.coin[0] + 1}, row ${ex.coin[1] + 1}, and the agent ${ex.got ? `reaches it in ${ex.steps} steps` : 'does not reach it in 30 steps'}.` : '');
}

let resizeT = null;
window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(() => { if (st.eval) drawAll(); figure(); }, 100); });
startTraining();

// ---- Static figure: one agent trained with the coin always at the right end, on a training and a test level ----
let figTheta = null;
function greedyPath(theta, coin) {
  let [x, y] = START; const path = [[x, y]], xs = new Float64Array(features(0, 0, 0, 0).length), p = new Float64Array(NA);
  for (let t = 0; t < 30; t++) {
    features(x, y, coin[0], coin[1], xs); probs(theta, xs, p);
    let a = 0; for (let b = 1; b < NA; b++) if (p[b] > p[a]) a = b;
    const nx = x + MOVES[a][0], ny = y + MOVES[a][1];
    if (nx >= 0 && nx < W && ny >= 0 && ny < H) { x = nx; y = ny; }
    path.push([x, y]);
    if (x === coin[0] && y === coin[1]) break;
  }
  return path;
}
function figure() {
  const fig = document.getElementById('fig-coin');
  if (!figTheta) {
    const tr = makeTrainer(0, { seed: 1 }); tr.run(EPISODES); figTheta = tr.theta;
    for (const t of ['Training level', 'Test level']) {
      const box = document.createElement('div');
      box.innerHTML = `<p class="small-title">${t}</p><canvas class="level" aria-hidden="true"></canvas>`;
      fig.append(box);
    }
  }
  const cans = fig.querySelectorAll('canvas');
  const trainCoin = END.slice(), testCoin = [4, 0];
  drawLevel(cans[0], { coin: trainCoin, path: greedyPath(figTheta, trainCoin) });
  drawLevel(cans[1], { coin: testCoin, path: greedyPath(figTheta, testCoin) });
}
setTimeout(figure, 50);
