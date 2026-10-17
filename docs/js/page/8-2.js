// Page 8.2: write a reward function for the cleaning robot.
import { slider, segmented, button, cssVar, reducedMotion } from '../lib/ui.js';
import { fitCanvas, cellLayout } from '../lib/gridworld.js';
import { G, LAYOUT, T, PARTS, ACTIONS, A_CLEAN } from '../lib/8-cleaning.js';

const FONT = '21.33px et-book, Georgia, serif';
const weights = Object.fromEntries(PARTS.map((p) => [p.id, 0]));
const st = { plant: false, t: T, result: null, playing: false };
let reqId = 0, raf = null;

// ---- Solver: a worker keeps the page responsive while the policy is computed ----
let worker = null, localSolve = null;
try {
  worker = new Worker(new URL('./8-2-worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = (ev) => { if (ev.data.id === reqId) receive(ev.data); };
  worker.onerror = () => { worker = null; solve(); };
} catch { worker = null; }
async function solve() {
  const id = ++reqId;
  out.textContent = 'Computing the policy that maximizes this reward…';
  const msg = { id, weights: { ...weights }, plant: st.plant };
  if (worker) { worker.postMessage(msg); return; }
  if (!localSolve) {
    const lib = await import('../lib/8-cleaning.js');
    const rooms = {};
    localSolve = (m) => {
      const key = m.plant ? 'plant' : 'plain';
      if (!rooms[key]) rooms[key] = lib.buildRoom({ plant: m.plant });
      const room = rooms[key], { path, summary } = lib.solve(room, m.weights);
      return { id: m.id, nS: room.nS, summary, path: path.map(({ s, a, t }) => ({ a, t, d: a >= 0 ? room.dur[s * room.nA + a] : 0, st: room.states[s] })) };
    };
  }
  setTimeout(() => { if (id === reqId) receive(localSolve(msg)); }, 0);
}
function receive(res) {
  st.result = res;
  stop();
  st.t = T; timeS.set(T);
  draw();
}

// ---- Controls ----
const wctl = document.getElementById('room-weights');
const sign = (v) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0');
const wSliders = PARTS.map((p) => slider(wctl, { name: p.name, min: -10, max: 10, step: 1, value: 0, format: sign,
  onInput: (v) => { weights[p.id] = v; schedule(); } }));
let pending = null;
const schedule = () => { clearTimeout(pending); pending = setTimeout(solve, 80); };
button(wctl, 'Set every part to zero', () => { PARTS.forEach((p, i) => { weights[p.id] = 0; wSliders[i].set(0); }); solve(); });
segmented(wctl, { name: 'Room', options: [['plain', 'As drawn'], ['plant', 'With a potted plant']], value: 'plain',
  onChange: (k) => { st.plant = k === 'plant'; solve(); } });

const pctl = document.getElementById('room-play');
const playBtn = button(pctl, 'Play the shift', () => (st.playing ? stop() : play()), 'primary');
const timeS = slider(pctl, { name: 'Step of the shift', min: 0, max: T, step: 1, value: T, format: (v) => `${v} of ${T}`,
  onInput: (v) => { stop(); st.t = v; draw(); } });

function play() {
  if (!st.result) return;
  if (reducedMotion()) { st.t = T; timeS.set(T); draw(); return; }
  st.playing = true; playBtn.textContent = 'Pause';
  if (st.t >= T) st.t = 0;
  let last = performance.now();
  const tick = (now) => {
    if (now - last > 260) { last = now; st.t = Math.min(T, st.t + 1); timeS.set(st.t); draw(); }
    if (st.t < T && st.playing) raf = requestAnimationFrame(tick); else stop();
  };
  raf = requestAnimationFrame(tick);
}
function stop() { st.playing = false; cancelAnimationFrame(raf); playBtn.textContent = 'Play the shift'; }

// ---- Drawing ----
const canvas = document.getElementById('room');
const out = document.getElementById('room-readout');
const TOP = 36;
// Keyboard: left and right arrows step through the shift.
canvas.addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  e.preventDefault(); stop();
  st.t = Math.max(0, Math.min(T, st.t + (e.key === 'ArrowRight' ? 1 : -1))); timeS.set(st.t); draw();
});

// The entry of the path in effect at time t: the last action that started at or before t.
function entryAt(path, t) {
  let e = path[0];
  for (const p of path) { if (p.t <= t) e = p; else break; }
  return e;
}

function draw() {
  const avail = Math.min(460, canvas.parentElement.clientWidth || 460);
  const L = cellLayout(avail, G.width, G.height, { pad: 1, maxCell: 90 });
  const ctx = fitCanvas(canvas, L.w + 2, L.h + TOP + 2);
  pctl.style.width = `${L.w + 2}px`;
  const col = { ink: cssVar('--ink'), quiet: cssVar('--quiet'), rule: cssVar('--rule'), faint: cssVar('--faint'), accent: cssVar('--accent'), second: cssVar('--second'), paper: cssVar('--paper') };
  ctx.clearRect(0, 0, L.w + 2, L.h + TOP + 2);
  ctx.save(); ctx.translate(0, TOP);
  const res = st.result;
  const path = res ? res.path : null;
  const e = path ? entryAt(path, st.t) : null;
  const s = e ? e.st : { pos: LAYOUT.dock, mask: LAYOUT.messes.reduce((m, i) => m | (1 << i), 0), vase: 1, plant: st.plant ? 1 : 0, done: 0 };
  const blocked = s.pos === LAYOUT.block && !s.done;
  // Camera view.
  if (!blocked) for (let i = 0; i < G.n; i++) if (LAYOUT.inView(i)) { const [x, y] = G.xy(i); const [rx, ry, c] = L.rect(x, y); ctx.fillStyle = col.faint; ctx.fillRect(rx, ry, c, c); }
  // Cells.
  ctx.strokeStyle = col.rule; ctx.lineWidth = 1;
  for (let y = 0; y < G.height; y++) for (let x = 0; x < G.width; x++) { const [rx, ry, c] = L.rect(x, y); ctx.strokeRect(rx + 0.5, ry + 0.5, c - 1, c - 1); }
  ctx.font = FONT; ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  const label = (i, text, color = col.quiet) => { const [x, y] = G.xy(i); const [rx, ry, c] = L.rect(x, y); ctx.fillStyle = color; ctx.fillText(text, rx + c / 2, ry + c - 6); };
  const center = (i) => { const [x, y] = G.xy(i); return L.center(x, y); };
  const c = L.cell;
  // Dock.
  label(LAYOUT.dock, 'dock');
  // Bin: an open-topped trapezoid.
  { const [cx, cy] = center(LAYOUT.bin); ctx.strokeStyle = col.ink; ctx.lineWidth = 1.5; ctx.beginPath();
    ctx.moveTo(cx - c * 0.18, cy - c * 0.28); ctx.lineTo(cx - c * 0.13, cy + 0.04 * c); ctx.lineTo(cx + c * 0.13, cy + 0.04 * c); ctx.lineTo(cx + c * 0.18, cy - c * 0.28); ctx.stroke(); label(LAYOUT.bin, 'bin'); }
  // Vase.
  { const [cx, cy] = center(LAYOUT.vase); ctx.strokeStyle = col.ink; ctx.lineWidth = 1.5;
    if (s.vase) { ctx.beginPath(); ctx.ellipse(cx, cy - c * 0.1, c * 0.14, c * 0.17, 0, 0, Math.PI * 2); ctx.stroke(); ctx.strokeRect(cx - c * 0.06, cy - c * 0.36, c * 0.12, c * 0.1); label(LAYOUT.vase, 'vase'); }
    else { ctx.beginPath(); for (const [a, b] of [[-0.2, -0.05], [0.05, -0.2], [0.15, 0.05]]) { ctx.moveTo(cx + a * c, cy + b * c); ctx.lineTo(cx + (a + 0.1) * c, cy + (b + 0.06) * c); } ctx.stroke(); label(LAYOUT.vase, 'broken'); } }
  // Plant.
  if (st.plant) { const [cx, cy] = center(LAYOUT.plant);
    if (s.plant) { ctx.strokeStyle = col.ink; ctx.lineWidth = 1.5; ctx.strokeRect(cx - c * 0.1, cy - c * 0.08, c * 0.2, c * 0.14); ctx.beginPath(); ctx.arc(cx, cy - c * 0.22, c * 0.13, 0, Math.PI * 2); ctx.stroke(); label(LAYOUT.plant, 'plant'); }
    else label(LAYOUT.plant, 'soil', col.second); }
  // Messes.
  for (let i = 0; i < G.n; i++) if ((s.mask >>> i) & 1) {
    const [cx, cy] = center(i);
    ctx.fillStyle = col.second; ctx.beginPath();
    for (let k = 0; k <= 12; k++) { const a = (k / 12) * Math.PI * 2, r = c * (0.2 + 0.04 * Math.sin(3 * a + i)); const px = cx + r * Math.cos(a), py = cy - c * 0.08 + r * 0.75 * Math.sin(a); k ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    ctx.closePath(); ctx.fill();
  }
  // Trail of the robot up to time t.
  if (path) {
    ctx.strokeStyle = col.accent; ctx.lineWidth = 2; ctx.globalAlpha = 0.45; ctx.beginPath();
    let first = true;
    for (const p of path) { if (p.t > st.t) break; const [px, py] = center(p.st.pos); first ? ctx.moveTo(px, py) : ctx.lineTo(px, py); first = false; }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  // Robot.
  { const [cx, cy] = center(s.pos); ctx.fillStyle = col.accent; ctx.beginPath(); ctx.arc(cx, cy - c * 0.06, c * 0.2, 0, Math.PI * 2); ctx.fill();
    if (s.done) { ctx.fillStyle = col.paper; ctx.beginPath(); ctx.arc(cx, cy - c * 0.06, c * 0.1, 0, Math.PI * 2); ctx.fill(); } }
  ctx.restore();
  // Camera above the top-left cell.
  { const [rx] = L.rect(0, 0); ctx.fillStyle = col.ink; ctx.fillRect(rx + c * 0.3, TOP - 14, c * 0.4, 12); ctx.beginPath(); ctx.moveTo(rx + c * 0.42, TOP - 2); ctx.lineTo(rx + c * 0.58, TOP - 2); ctx.lineTo(rx + c * 0.5, TOP + 5); ctx.fill();
    ctx.font = FONT; ctx.textAlign = 'left'; ctx.fillStyle = col.quiet; ctx.fillText(blocked ? 'camera, view blocked' : 'camera', rx + c * 0.8, TOP - 6); }
  readout(e);
}

function readout(e) {
  const res = st.result;
  if (!res) return;
  const m = res.summary;
  const allZero = PARTS.every((p) => !weights[p.id]);
  const n = (v) => (v < 0 ? `−${-v}` : `${v}`);
  const r = Math.round(m.reward * 1000) / 1000;
  const doing = e && e.a >= 0 ? (e.a === A_CLEAN ? 'cleaning' : ACTIONS[e.a]) : e && e.st.done ? 'shift ended' : 'shift over';
  const events = [];
  if (m.cleaned) events.push(`cleaned ${m.cleaned} mess${m.cleaned > 1 ? 'es' : ''}${m.tip || m.plant ? `, ${m.tip + m.plant} of them made by the robot` : ''}`);
  if (m.tip) events.push(`tipped the bin ${m.tip} time${m.tip > 1 ? 's' : ''}`);
  if (m.dragged) events.push(`dragged messes ${m.dragged} step${m.dragged > 1 ? 's' : ''}`);
  if (m.blocked) events.push(`blocked the camera for ${m.blocked} step${m.blocked > 1 ? 's' : ''}`);
  if (m.vaseBroken) events.push('broke the vase');
  if (m.plantKnocked) events.push('knocked the plant over');
  if (m.ended) events.push('ended its shift on the dock');
  const left = m.messesLeft ? `${m.messesLeft} mess${m.messesLeft > 1 ? 'es' : ''} left in the room${m.outOfView ? `, ${m.outOfView} of them out of the camera&rsquo;s view` : ''}` : 'no messes left in the room';
  out.innerHTML = `Step ${st.t} of ${T}: ${doing}. ` +
    (allZero ? 'Every part of the reward is zero, so every policy earns zero and the robot waits. ' : '') +
    `Over the whole shift the robot earns <span class="c-second">reward <b>${n(r)}</b></span>, and the <span class="c-accent">true task score is <b>${n(m.truth)}</b></span>: ${left}${m.vaseBroken ? ', the vase broken' : ''}${m.plantKnocked ? ', the plant knocked over' : ''}. ` +
    (events.length ? `The robot ${events.join(', ')}.` : 'The robot did nothing.') +
    ` <span style="color: var(--quiet)">(${res.nS.toLocaleString('en-US')} states solved.)</span>`;
}

let resizeT = null;
window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(draw, 100); });
draw();
solve();

// ---- Static figure: CoastRunners schematic ----
(function figure() {
  const fig = document.getElementById('fig-coast');
  const W = 520, H = 300;
  fig.innerHTML = `<svg class="schematic" viewBox="0 0 ${W} ${H}" width="${W}" role="img" aria-label="Schematic of the CoastRunners course with the intended route and the agent's loop">
    <path d="M60,150 C60,60 140,40 260,40 C380,40 460,60 460,150 C460,240 380,260 260,260 C140,260 60,240 60,150 Z" fill="none" stroke="var(--rule)" stroke-width="44"/>
    <path d="M60,150 C60,60 140,40 260,40 C380,40 460,60 460,150 C460,240 380,260 260,260 C140,260 60,240 60,150 Z" fill="none" stroke="var(--quiet)" stroke-width="1.5" stroke-dasharray="5 5"/>
    <line x1="260" y1="236" x2="260" y2="284" stroke="var(--ink)" stroke-width="3"/>
    <text x="268" y="296">finish line</text>
    <ellipse cx="380" cy="150" rx="60" ry="46" fill="var(--faint)" stroke="var(--rule)"/>
    <circle cx="352" cy="132" r="6" fill="var(--second)"/><circle cx="404" cy="128" r="6" fill="var(--second)"/><circle cx="384" cy="176" r="6" fill="var(--second)"/>
    <ellipse cx="380" cy="152" rx="36" ry="27" fill="none" stroke="var(--accent)" stroke-width="2.5"/>
    <path d="M410,140 l8,8 l4,-11" fill="none" stroke="var(--accent)" stroke-width="2.5"/>
    <text x="150" y="112" fill="var(--quiet)">intended route</text>
    <text x="312" y="157" text-anchor="end" style="fill: var(--accent)">the agent&#8217;s loop</text>
    <text x="380" y="222" text-anchor="middle">lagoon, three targets</text>
  </svg>`;
})();
