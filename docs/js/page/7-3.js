// Page 7.3: many means toward one end, and interventions on a goal-reaching network.
import { segmented, slider, button, cssVar, reducedMotion } from '../lib/ui.js';
import { fitCanvas, cellLayout, drawGrid, grid } from '../lib/gridworld.js';
import * as M from '../lib/s73-maze.js';
import * as P from '../lib/s73-policy.js';

const G = grid(M.W, M.H);
const MAX_STEPS = 40;
const DEFAULT_SUB = M.cell(1, 6);
const colors = () => ({ ink: cssVar('--ink'), quiet: cssVar('--quiet'), rule: cssVar('--rule'), accent: cssVar('--accent'), second: cssVar('--second'), paper: cssVar('--paper') });
const f2 = (v) => v.toFixed(2);
let net = null;

// ---- Drawing one environment on a canvas ----
// opts: { env, cssWidth, maxCell, paths: [{ cells, color, offset }], keyShown, agent, goal, sub, decoded, start }
function drawMaze(canvas, o) {
  const L = cellLayout(o.cssWidth, M.W, M.H, { pad: 1, maxCell: o.maxCell });
  const ctx = fitCanvas(canvas, L.w + 2, L.cssHeight);
  const col = colors(), c = L.cell;
  ctx.clearRect(0, 0, L.w + 2, L.cssHeight);
  const B = M.blocked(o.env);
  const door = (x, y) => M.isDoor(M.cell(x, y));
  drawGrid(ctx, L, G, { wall: (x, y) => B[M.cell(x, y)] && door(x, y) < 0, colors: col });
  // Doors: closed and locked doors are light gray; a locked door has a keyhole.
  M.DOORS.forEach(([x, y], i) => {
    const s = o.env.doors[i];
    if (s === 0) return;
    const [rx, ry] = L.rect(x, y);
    ctx.fillStyle = col.rule; ctx.fillRect(rx, ry, c, c);
    if (s === 2) {
      const [cx, cy] = L.center(x, y);
      ctx.fillStyle = col.ink;
      ctx.beginPath(); ctx.arc(cx, cy - c * 0.08, c * 0.11, 0, 2 * Math.PI); ctx.fill();
      ctx.fillRect(cx - c * 0.045, cy - c * 0.05, c * 0.09, c * 0.25);
    }
  });
  // Key: a ring with a stem.
  if (o.env.key >= 0 && o.keyShown !== false) {
    const [cx, cy] = L.center(...M.xy(o.env.key));
    ctx.strokeStyle = col.ink; ctx.lineWidth = Math.max(1.5, c * 0.07);
    ctx.beginPath(); ctx.arc(cx - c * 0.14, cy, c * 0.13, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - c * 0.01, cy); ctx.lineTo(cx + c * 0.3, cy); ctx.moveTo(cx + c * 0.2, cy); ctx.lineTo(cx + c * 0.2, cy + c * 0.14); ctx.stroke();
  }
  // Start: an open circle. Goal: a filled square. Substitute goal: an outlined square.
  const start = o.start ?? M.START;
  {
    const [cx, cy] = L.center(...M.xy(start));
    ctx.strokeStyle = col.quiet; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(cx, cy, c * 0.24, 0, 2 * Math.PI); ctx.stroke();
  }
  const sq = (cell, fill, color, size) => {
    const [cx, cy] = L.center(...M.xy(cell)), h = c * size;
    if (fill) { ctx.fillStyle = color; ctx.fillRect(cx - h, cy - h, 2 * h, 2 * h); }
    else { ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.strokeRect(cx - h, cy - h, 2 * h, 2 * h); }
  };
  sq(o.goal ?? M.GOAL, true, col.ink, 0.2);
  if (o.sub >= 0) sq(o.sub, false, col.accent, 0.3);
  for (const p of o.paths || []) {
    if (!p.cells || p.cells.length < 1) continue;
    ctx.strokeStyle = col[p.color]; ctx.lineWidth = p.width || 2.5; ctx.lineJoin = 'round';
    ctx.beginPath();
    p.cells.forEach((cell, i) => { const [x, y] = L.center(...M.xy(cell)); const d = p.offset || 0; i ? ctx.lineTo(x + d, y + d) : ctx.moveTo(x + d, y + d); });
    ctx.stroke();
  }
  if (o.decoded >= 0) {
    const [cx, cy] = L.center(...M.xy(o.decoded));
    ctx.strokeStyle = col.second; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(cx, cy, c * 0.44, 0, 2 * Math.PI); ctx.stroke();
  }
  if (o.agent >= 0) {
    const [cx, cy] = L.center(...M.xy(o.agent));
    ctx.fillStyle = col.accent;
    ctx.beginPath(); ctx.arc(cx, cy, c * 0.17, 0, 2 * Math.PI); ctx.fill();
  }
  return L;
}

const avail = (el, cap) => Math.min(cap, el.clientWidth || cap);

// ---- Panel 1: many means, one end ----
const means = { ran: false, results: null };
const meansStage = document.getElementById('means-stage');
const meansOut = document.getElementById('means-readout');
const multi = document.createElement('div');
multi.className = 'multi';
meansStage.append(multi);
const minis = M.ENVIRONMENTS.map((E) => {
  const box = document.createElement('div');
  const canvas = document.createElement('canvas');
  canvas.className = 'maze';
  canvas.setAttribute('role', 'img');
  const name = document.createElement('p');
  name.className = 'env-name';
  name.textContent = E.name;
  const res = document.createElement('p');
  res.className = 'res';
  box.append(canvas, name, res);
  multi.append(box);
  return { E, canvas, res };
});
function drawMeans() {
  minis.forEach((m, i) => {
    const r = means.ran ? means.results[i] : null;
    drawMaze(m.canvas, {
      env: m.E.env, cssWidth: 9 * 15 + 2, maxCell: 15,
      paths: r ? [{ cells: r.net.path, color: 'accent', offset: -2 }, { cells: r.direct.path, color: 'second', offset: 2 }] : [],
    });
    m.res.innerHTML = r
      ? `<span class="c-accent">Network</span>: ${r.net.reached ? `${r.net.steps} steps` : 'did not arrive'}. <span class="c-second">Direct</span>: ${r.direct.reached ? `${r.direct.steps} steps` : 'blocked'}.`
      : '&nbsp;';
    m.canvas.setAttribute('aria-label', `${m.E.name}.` + (r ? ` The network ${r.net.reached ? `reaches the goal in ${r.net.steps} steps` : 'does not reach the goal'}; the direct controller ${r.direct.reached ? `reaches it in ${r.direct.steps} steps` : 'is blocked'}.` : ''));
  });
  if (!means.ran) { meansOut.textContent = net ? 'Press Run in all six to see both paths.' : 'Loading the network…'; return; }
  const ok = means.results.filter((r) => r.net.reached), dok = means.results.filter((r) => r.direct.reached);
  const n1 = new Set(ok.map((r) => r.net.path.join())).size, n2 = new Set(dok.map((r) => r.direct.path.join())).size;
  meansOut.innerHTML = `The <span class="c-accent">network</span> reached the goal in ${ok.length} of 6 environments, by ${n1} different path${n1 === 1 ? '' : 's'}. The <span class="c-second">direct controller</span> reached it in ${dok.length} of 6, by ${n2} path${n2 === 1 ? '' : 's'}.`;
}
{
  const ctl = document.getElementById('means-controls');
  button(ctl, 'Run in all six', () => {
    if (!net) return;
    means.results = M.ENVIRONMENTS.map((E) => ({
      net: M.run(E.env, M.START, M.GOAL, P.policy(net, E.env, M.GOAL), MAX_STEPS),
      direct: M.run(E.env, M.START, M.GOAL, M.directPolicy(M.GOAL), MAX_STEPS),
    }));
    means.ran = true; drawMeans();
  }, 'primary');
  button(ctl, 'Clear', () => { means.ran = false; drawMeans(); });
}

// ---- Panel 2: inside the network ----
const st = { envIx: 0, goalMode: 'as-is', belief: 'seen', sub: DEFAULT_SUB, t: 0, run: null };
const BELIEFS = { seen: null, upper: [0, 1], lower: [1, 0] };
const insideStage = document.getElementById('inside-stage');
const insideOut = document.getElementById('inside-readout');
const big = document.createElement('canvas');
big.className = 'maze';
big.tabIndex = 0;
insideStage.append(big);
let layout = null, timer = null;
const ctlIn = document.getElementById('inside-controls');
const envSeg = segmented(ctlIn, { name: 'Environment', options: M.ENVIRONMENTS.map((E, i) => [String(i), E.short]), value: '0', onChange: (k) => { st.envIx = Number(k); fixSub(); recompute(); } });
const goalSeg = segmented(ctlIn, { name: 'Goal representation', options: [['as-is', 'As computed'], ['remove', 'Removed'], ['swap', 'Swapped']], value: 'as-is', onChange: (k) => { st.goalMode = k; recompute(); } });
const beliefSeg = segmented(ctlIn, { name: 'Belief about the doors', options: [['seen', 'As seen'], ['upper', 'Upper open, lower closed'], ['lower', 'Upper closed, lower open']], value: 'seen', onChange: (k) => { st.belief = k; recompute(); } });
const stepSl = slider(ctlIn, { name: 'Step', min: 0, max: MAX_STEPS, step: 1, value: 0, format: (v) => `${v} of ${st.run ? st.run.steps : 0}`, onInput: (v) => { stop(); st.t = Math.min(v, st.run ? st.run.steps : 0); stepSl.set(st.t); drawInside(); } });
const playBtn = button(ctlIn, 'Play', () => play());
button(ctlIn, 'Reset', () => {
  stop();
  Object.assign(st, { envIx: 0, goalMode: 'as-is', belief: 'seen', sub: DEFAULT_SUB });
  envSeg.set('0'); goalSeg.set('as-is'); beliefSeg.set('seen');
  recompute();
});

const env = () => M.ENVIRONMENTS[st.envIx].env;
const validSub = (c) => M.freeCells(env()).includes(c) && c !== M.START && c !== M.GOAL;
function fixSub() { if (!validSub(st.sub)) st.sub = validSub(DEFAULT_SUB) ? DEFAULT_SUB : M.freeCells(env()).find(validSub); }
const cfg = () => ({ goalMode: st.goalMode, goal2: st.goalMode === 'swap' ? st.sub : -1, doorBelief: BELIEFS[st.belief] });

function recompute() {
  stop();
  if (!net) return;
  const target = st.goalMode === 'swap' ? st.sub : M.GOAL;
  st.run = M.run(env(), M.START, target, P.policy(net, env(), M.GOAL, cfg()), MAX_STEPS);
  st.t = st.run.steps;
  stepSl.el.max = st.run.steps;
  stepSl.set(st.t);
  drawInside();
}

function where(cell) {
  if (cell === M.GOAL) return 'the real goal';
  if (st.goalMode === 'swap' && cell === st.sub) return 'the outlined cell';
  if (cell === M.START) return 'the start';
  const [x, y] = M.xy(cell);
  return `another cell (column ${x + 1}, row ${y + 1})`;
}

function drawInside() {
  if (!net || !st.run) return;
  const r = st.run, t = Math.min(st.t, r.steps);
  const pos = r.path[t], hasKey = r.keys[t];
  const { h } = P.compute(net, env(), { pos, hasKey }, M.GOAL, cfg());
  const goalP = P.readGoal(net, h), decoded = P.argmax(goalP);
  const doorP = P.readDoors(net, h);
  layout = drawMaze(big, {
    env: env(), cssWidth: avail(insideStage, 9 * 48 + 2), maxCell: 48, keyShown: !hasKey,
    paths: [{ cells: r.path.slice(0, t + 1), color: 'accent' }],
    agent: pos, sub: st.goalMode === 'swap' ? st.sub : -1, decoded,
  });
  const E = M.ENVIRONMENTS[st.envIx];
  const target = st.goalMode === 'swap' ? 'the outlined cell' : 'its goal';
  let outcome;
  if (r.reached) outcome = `The agent reached ${target} in ${r.steps} steps${r.keys[r.steps] && E.env.key >= 0 ? ', picking up the key on the way' : ''}; the shortest route takes ${M.distances(E.env, st.goalMode === 'swap' ? st.sub : M.GOAL)[M.START]}.`;
  else {
    const last = new Set(r.path.slice(-10));
    outcome = `The agent did not reach ${target} in ${MAX_STEPS} steps. ` + (r.stuck ? 'It ended pushing against a wall or a door.' : `Over the last 10 steps it moved among ${last.size} cells.`);
  }
  const doorText = doorP.map((p, i) => { const k = P.argmax(p); return `${M.DOOR_NAMES[i]} ${M.DOOR_STATES[k]} (${f2(p[k])})`; }).join(' and ');
  insideOut.innerHTML = `${outcome} At step ${t}, the goal probe&rsquo;s most likely goal is <span class="c-second">${where(decoded)}</span> (probability ${f2(goalP[decoded])}), and the door probes&rsquo; most likely states are ${doorText}.`;
  big.setAttribute('aria-label', `${E.name}. ${outcome}` + (st.goalMode === 'swap' ? ' Use the arrow keys to move the outlined substitute goal.' : ''));
}

function stop() { if (timer) { clearInterval(timer); timer = null; } playBtn.textContent = 'Play'; }
function play() {
  if (!st.run) return;
  if (timer) { stop(); return; }
  if (reducedMotion()) { st.t = st.run.steps; stepSl.set(st.t); drawInside(); return; }
  st.t = 0; stepSl.set(0); drawInside();
  playBtn.textContent = 'Pause';
  timer = setInterval(() => {
    if (st.t >= st.run.steps) { stop(); return; }
    st.t++; stepSl.set(st.t); drawInside();
  }, 160);
}

// Moving the substitute goal: tap a cell, or use the arrow keys on the focused grid.
function moveSub(c) {
  if (st.goalMode !== 'swap' || !validSub(c)) return false;
  st.sub = c; recompute(); return true;
}
big.addEventListener('click', (ev) => {
  if (!layout) return;
  const rect = big.getBoundingClientRect();
  const x = Math.floor((ev.clientX - rect.left - layout.ox) / layout.cell), y = Math.floor((ev.clientY - rect.top - layout.oy) / layout.cell);
  if (G.inside(x, y)) moveSub(M.cell(x, y));
});
big.addEventListener('keydown', (ev) => {
  const k = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3 }[ev.key];
  if (k === undefined || st.goalMode !== 'swap') return;
  ev.preventDefault();
  let [x, y] = M.xy(st.sub);
  for (;;) {
    x += M.MOVES[k][0]; y += M.MOVES[k][1];
    if (!G.inside(x, y)) return;
    if (moveSub(M.cell(x, y))) return;
  }
});

// ---- Static figure ----
function meansSVG(title, meansList, endLines, note) {
  const NS = 'http://www.w3.org/2000/svg';
  const rowH = 26, top = 54, n = meansList.length;
  const H = top + Math.max(n, 2) * rowH + (note ? 30 : 6), Wd = 320;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'means'); svg.setAttribute('width', Wd); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${Wd} ${H}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', `${title}: ${meansList.join(', ')}, toward ${endLines.join(' ')}`);
  const el = (tag, a, text) => { const e = document.createElementNS(NS, tag); Object.entries(a).forEach(([k, v]) => e.setAttribute(k, v)); if (text) e.textContent = text; svg.append(e); return e; };
  const defs = el('defs', {});
  const mk = document.createElementNS(NS, 'marker');
  const id = `ah-${Math.random().toString(36).slice(2)}`;
  Object.entries({ id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 8, markerHeight: 8, markerUnits: 'userSpaceOnUse', orient: 'auto' }).forEach(([k, v]) => mk.setAttribute(k, v));
  const tip = document.createElementNS(NS, 'path'); tip.setAttribute('d', 'M0,0 L10,5 L0,10 z'); tip.setAttribute('fill', 'var(--quiet)');
  mk.append(tip); defs.append(mk);
  el('text', { x: 0, y: 22 }, title);
  const endY = top + ((Math.max(n, 2) - 1) * rowH) / 2;
  meansList.forEach((m, i) => {
    const y = top + (n === 1 ? (rowH / 2) : i * rowH);
    el('text', { x: 0, y: y + 5, class: 'quiet' }, m);
    el('line', { x1: 160, y1: y, x2: 212, y2: n === 1 ? y : endY, 'marker-end': `url(#${id})` });
  });
  endLines.forEach((s, i) => el('text', { x: 220, y: (n === 1 ? top + rowH / 2 : endY) + 5 + (i - (endLines.length - 1) / 2) * 22 }, s));
  if (note) el('text', { x: 0, y: H - 8, class: 'quiet' }, note);
  return svg;
}
function figure() {
  const topRow = document.getElementById('fig-top');
  topRow.innerHTML = '';
  for (const [title, list, end, note] of [
    ['A thermostat', ['turn the heater on'], ['room at its', 'set point'], 'in any room'],
    ['A coding agent', ['read the failing test', 'search the code', 'edit the parser', 'install a package', 'run the tests again'], ['the tests', 'pass'], 'in any repository'],
  ]) { const d = document.createElement('div'); d.append(meansSVG(title, list, end, note)); topRow.append(d); }
  const bottom = document.getElementById('fig-bottom');
  bottom.innerHTML = '';
  if (!net) return;
  const E = M.ENVIRONMENTS[0];
  const before = M.run(E.env, M.START, M.GOAL, P.policy(net, E.env, M.GOAL), MAX_STEPS);
  const after = M.run(E.env, M.START, DEFAULT_SUB, P.policy(net, E.env, M.GOAL, { goalMode: 'swap', goal2: DEFAULT_SUB }), MAX_STEPS);
  for (const [title, r, sub] of [['Before', before, -1], ['After the swap', after, DEFAULT_SUB]]) {
    const d = document.createElement('div');
    d.innerHTML = `<p class="small-title">${title}</p>`;
    const cv = document.createElement('canvas');
    cv.className = 'maze'; cv.setAttribute('role', 'img');
    cv.setAttribute('aria-label', `${title}: the network's path ends at ${r.path.at(-1) === M.GOAL ? 'its goal' : 'the outlined cell'} after ${r.steps} steps`);
    d.append(cv); bottom.append(d);
    const fc = (bottom.clientWidth || 400) < 400 ? 16 : 20;
    drawMaze(cv, { env: E.env, cssWidth: 9 * fc + 2, maxCell: fc, paths: [{ cells: r.path, color: 'accent' }], sub, agent: r.path.at(-1) });
  }
}

// ---- Start ----
drawMeans();
figure();
insideOut.textContent = 'Loading the network…';
fetch('../data/7-3/policy.json').then((res) => res.json()).then((json) => {
  net = P.load(json);
  drawMeans(); figure(); recompute();
}).catch(() => {
  const msg = 'The network&rsquo;s parameters could not be loaded. Reload the page to try again.';
  meansOut.innerHTML = msg; insideOut.innerHTML = msg;
});

let resizeT = null;
window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(() => { drawMeans(); figure(); drawInside(); }, 150); });
window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => { drawMeans(); figure(); drawInside(); });
