// Page 6.7: activation patching on the trained modular-addition transformer, run live.
// Numbers quoted in the tasks are checked by tools/test-page-6-7.mjs.
import { chart } from '../lib/chart.js';
import { slider, segmented, button } from '../lib/ui.js';
import { ModAdd } from '../lib/s67-modadd.js';

const d3 = window.d3;
const p = 59;
const pctTxt = (v) => `${Math.round(v * 100)}%`;

let json;
try {
  const r = await fetch(new URL('../../data/6-7/model.json', import.meta.url));
  if (!r.ok) throw new Error(String(r.status));
  json = await r.json();
} catch (e) {
  document.getElementById('patch-readout').textContent = 'The model parameters could not be loaded, so the tool cannot run.';
  throw e;
}
const M = new ModAdd(json);  // used for patching; its embedding never changes
const MF = new ModAdd(json); // used for the frequency ablations

// ---- Components that can be patched ----
const groupMask = (k) => Uint8Array.from(M.neuronFreq, (f) => (f === k ? 1 : 0));
const COMPS = [
  { id: 'h0', label: 'head 0', name: 'head 0', patch: { heads: [0] } },
  { id: 'h1', label: 'head 1', name: 'head 1', patch: { heads: [1] } },
  { id: 'h2', label: 'head 2', name: 'head 2', patch: { heads: [2] } },
  { id: 'h3', label: 'head 3', name: 'head 3', patch: { heads: [3] } },
  { id: 'hall', label: 'all heads', name: 'all four heads', patch: { heads: [0, 1, 2, 3] } },
  { id: 'mlp', label: 'MLP output', name: 'the MLP output', patch: { mlp: true } },
  ...[1, 12, 27].map((k) => ({ id: `f${k}`, label: `neurons ${k}`, name: `Neurons ${k} (${M.neuronFreq.filter((f) => f === k).length} neurons)`, patch: { neurons: groupMask(k) } })),
];
const byId = Object.fromEntries(COMPS.map((c) => [c.id, c]));

// Restored share for each component, for one clean pair and one corrupted pair.
function restoredFor(clean, corrupted, ids) {
  const [a, b] = clean, [a2, b2] = corrupted, c = (a + b) % p, c2 = (a2 + b2) % p;
  const cl = M.run(a, b), co = M.run(a2, b2);
  const ld = (o) => o.logits[c] - o.logits[c2];
  const L0 = ld(co), L1 = ld(cl);
  const out = {};
  for (const id of ids) {
    const L = ld(M.run(a2, b2, byId[id].patch, cl));
    out[id] = { restored: (L - L0) / (L1 - L0), patched: L };
  }
  return { c, c2, clean: L1, corrupted: L0, out };
}

// ---- Panel 1: patching ----
const st = { a: 12, b: 30, which: 'a', bad: 20, current: null };
const tried = new Set(); // components patched at least once
const corrupted = () => (st.which === 'a' ? [st.bad, st.b] : [st.a, st.bad]);
const same = () => st.bad === (st.which === 'a' ? st.a : st.b);

const pc = document.getElementById('patch-controls');
const aS = slider(pc, { name: 'Clean a', min: 0, max: 58, step: 1, value: st.a, onInput: (v) => { st.a = v; update(); } });
const bS = slider(pc, { name: 'Clean b', min: 0, max: 58, step: 1, value: st.b, onInput: (v) => { st.b = v; update(); } });
segmented(pc, { name: 'Input to corrupt', options: [['a', 'a'], ['b', 'b']], value: st.which,
  onChange: (k) => { st.which = k; update(); } });
const badS = slider(pc, { name: 'Corrupted value', min: 0, max: 58, step: 1, value: st.bad, onInput: (v) => { st.bad = v; update(); } });
const compWrap = document.createElement('div');
compWrap.className = 'ctl';
compWrap.setAttribute('role', 'group');
compWrap.setAttribute('aria-label', 'Patch from the clean run');
compWrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Patch from the clean run</span></span>';
const compBtns = document.createElement('div');
compBtns.className = 'comp-buttons';
compWrap.append(compBtns);
pc.append(compWrap);
const buttons = COMPS.map((c) => {
  const btn = button(compBtns, c.label[0].toUpperCase() + c.label.slice(1), () => { st.current = c.id; tried.add(c.id); update(); });
  btn.setAttribute('aria-pressed', 'false');
  return btn;
});
const mapBtn = button(pc, 'Show the full patching map', () => { mapShown = true; document.getElementById('map').hidden = false; mapBtn.hidden = true; drawMap(true); });
mapBtn.disabled = true;
button(pc, 'Clear the bars', () => { tried.clear(); st.current = null; update(); });
let mapShown = false;

const chartBox = document.getElementById('patch-chart');
const out1 = document.getElementById('patch-readout');
let svg1 = null;

function drawBars(res) {
  const W = Math.max(280, Math.min(560, chartBox.clientWidth || 560));
  const rowH = 30, top = 34, H = top + rowH * COMPS.length + 8;
  const labelW = 132, valW = 64;
  if (!svg1 || Number(svg1.attr('width')) !== W) {
    chartBox.innerHTML = '';
    svg1 = d3.select(chartBox).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
      .attr('role', 'img').attr('aria-label', 'Share of the logit difference restored by each component patched so far');
  }
  svg1.selectAll('*').remove();
  const x = d3.scaleLinear().domain([-0.5, 1.5]).range([labelW, W - valW]);
  for (const [v, t] of [[0, '0%'], [1, '100%']]) {
    svg1.append('line').attr('x1', x(v)).attr('x2', x(v)).attr('y1', top - 6).attr('y2', H - 4)
      .attr('stroke', 'var(--rule)').attr('stroke-dasharray', v ? '3 3' : null);
    svg1.append('text').attr('x', x(v)).attr('y', top - 12).attr('text-anchor', 'middle').text(t);
  }
  COMPS.forEach((c, i) => {
    const y = top + i * rowH;
    const r = res && tried.has(c.id) ? res.out[c.id] : null;
    const isCur = c.id === st.current;
    svg1.append('text').attr('x', 0).attr('y', y + rowH / 2 + 7).attr('class', isCur ? 'label-accent' : r ? 'label-ink' : '').text(c.label);
    if (!r) return;
    const v = Math.max(-0.5, Math.min(1.5, r.restored));
    svg1.append('rect').attr('x', Math.min(x(0), x(v))).attr('width', Math.abs(x(v) - x(0))).attr('y', y + 6).attr('height', rowH - 12)
      .attr('fill', isCur ? 'var(--accent)' : 'var(--quiet)').attr('fill-opacity', isCur ? 0.9 : 0.45);
    svg1.append('text').attr('x', W - valW + 6).attr('y', y + rowH / 2 + 7).attr('class', isCur ? 'label-accent' : 'label-ink').text(pctTxt(r.restored));
  });
}

function update() {
  buttons.forEach((btn, i) => btn.setAttribute('aria-pressed', String(COMPS[i].id === st.current)));
  mapBtn.disabled = tried.size < 3;
  mapBtn.textContent = tried.size < 3 ? `Show the full patching map (after ${3 - tried.size} more patch${3 - tried.size === 1 ? '' : 'es'})` : 'Show the full patching map';
  const [a2, b2] = corrupted();
  const sum = `Clean: ${st.a} + ${st.b} = ${(st.a + st.b) % p}. Corrupted: ${a2} + ${b2} = ${(a2 + b2) % p}.`;
  if (same()) {
    drawBars(null);
    out1.innerHTML = `${sum} The corrupted input equals the clean input; choose another corrupted value.`;
    if (mapShown) drawMap(false);
    return;
  }
  const res = restoredFor([st.a, st.b], [a2, b2], [...tried]);
  drawBars(res);
  let txt = `${sum} Logit difference (logit for ${res.c} minus logit for ${res.c2}): clean <b>${res.clean.toFixed(1)}</b>, corrupted <b>${res.corrupted.toFixed(1)}</b>.`;
  if (st.current) {
    const r = res.out[st.current];
    txt += ` After patching ${byId[st.current].name}: <span class="c-accent">${r.patched.toFixed(1)}</span>, so <span class="c-accent">${pctTxt(r.restored)}</span> restored.`;
  } else txt += ' Choose a component to patch.';
  out1.innerHTML = txt;
  if (mapShown) drawMap(false);
}

// ---- The full patching map ----
const mapBox = document.getElementById('map-cells');
let mapKey = null, mapData = null;
function drawMap(force) {
  const key = `${st.a},${st.b},${st.which}`;
  if (force || key !== mapKey) {
    mapKey = key;
    mapData = [];
    for (let v = 0; v < p; v++) {
      const cv = st.which === 'a' ? [v, st.b] : [st.a, v];
      mapData.push(v === (st.which === 'a' ? st.a : st.b) ? null : restoredFor([st.a, st.b], cv, COMPS.map((c) => c.id)).out);
    }
  }
  const W = Math.max(280, Math.min(720, mapBox.clientWidth || 720));
  const labelW = 132, rowH = 26, top = 6, H = top + rowH * COMPS.length + 58;
  const cw = (W - labelW - 8) / p;
  mapBox.innerHTML = '';
  const svg = d3.select(mapBox).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
    .attr('role', 'img').attr('aria-label', `Patching map for the clean pair ${st.a} + ${st.b}, corrupting ${st.which}`);
  COMPS.forEach((c, i) => {
    const y = top + i * rowH;
    svg.append('text').attr('x', 0).attr('y', y + rowH / 2 + 7).text(c.label);
    for (let v = 0; v < p; v++) {
      const cell = mapData[v];
      if (!cell) continue;
      const r = Math.max(0, Math.min(1, cell[c.id].restored));
      if (r > 0) svg.append('rect').attr('x', labelW + v * cw).attr('y', y + 1).attr('width', cw + 0.3).attr('height', rowH - 2)
        .attr('fill', 'var(--accent)').attr('fill-opacity', r);
    }
  });
  const yb = top + rowH * COMPS.length;
  if (!same()) svg.append('rect').attr('x', labelW + st.bad * cw - 1).attr('y', top - 1).attr('width', cw + 2).attr('height', rowH * COMPS.length + 2)
    .attr('fill', 'none').attr('stroke', 'var(--ink)').attr('stroke-width', 1.5);
  for (const v of [0, 20, 40, 58]) svg.append('text').attr('x', labelW + (v + 0.5) * cw).attr('y', yb + 22).attr('text-anchor', 'middle').text(v);
  svg.append('text').attr('x', W).attr('y', yb + 50).attr('text-anchor', 'end').text(`corrupted value of ${st.which}`);
  // Tap a column to choose that corrupted value.
  svg.append('rect').attr('x', labelW).attr('y', top).attr('width', cw * p).attr('height', rowH * COMPS.length).attr('fill', 'transparent')
    .style('cursor', 'pointer').on('click', (ev) => {
      const [mx] = d3.pointer(ev, svg.node());
      const v = Math.max(0, Math.min(p - 1, Math.floor((mx - labelW) / cw)));
      st.bad = v; badS.set(v); update();
    });
}

window.addEventListener('resize', () => { svg1 = null; update(); if (mapShown) drawMap(false); drawFreq(); });
update();

// ---- Panel 2: frequencies ----
const spec = M.spectrum();
const top3 = spec.map((v, i) => [v, i + 1]).sort((x, y) => y[0] - x[0]).slice(0, 3).map((t) => t[1]);
const fs = { sel: 5, keep: Object.fromEntries(Array.from({ length: 29 }, (_, i) => [i + 1, true])) };
const fc = document.getElementById('freq-controls');
const out2 = document.getElementById('freq-readout');
const selS = slider(fc, { name: 'Selected frequency k', min: 1, max: 29, step: 1, value: fs.sel, onInput: (v) => { fs.sel = v; drawFreq(); } });
const togBtn = button(fc, '', () => { fs.keep[fs.sel] = !fs.keep[fs.sel]; drawFreq(); recompute(); });
button(fc, 'Keep only the three largest', () => { for (let k = 1; k <= 29; k++) fs.keep[k] = top3.includes(k); drawFreq(); recompute(); });
button(fc, 'Put back all', () => { for (let k = 1; k <= 29; k++) fs.keep[k] = true; drawFreq(); recompute(); });

const specBox = document.getElementById('spec-chart'), circBox = document.getElementById('circle');
let acc = 1, computing = false, job = 0;
const kept = () => Object.values(fs.keep).filter(Boolean).length;

function drawFreq() {
  togBtn.textContent = fs.keep[fs.sel] ? `Remove frequency ${fs.sel}` : `Put back frequency ${fs.sel}`;
  // Spectrum
  specBox.innerHTML = '';
  const c = chart(specBox, { width: 560, height: 250, margin: { top: 34, right: 16, bottom: 52, left: 50 },
    x: { domain: [0.5, 29.5], ticks: 6, label: 'frequency k' }, y: { domain: [0, 6.4], ticks: 3, label: 'size in the embedding' },
    title: 'Size of each frequency in the number embeddings' });
  const bw = Math.max(3, (c.x(2) - c.x(1)) * 0.6);
  for (let k = 1; k <= 29; k++) {
    const on = fs.keep[k], isSel = k === fs.sel;
    const col = isSel ? 'var(--accent)' : 'var(--ink)';
    const r = c.g.append('rect').attr('x', c.x(k) - bw / 2).attr('width', bw).attr('y', c.y(spec[k - 1])).attr('height', Math.max(0.5, c.ih - c.y(spec[k - 1])));
    if (on) r.attr('fill', col).attr('fill-opacity', isSel ? 1 : 0.75);
    else r.attr('fill', 'none').attr('stroke', col).attr('stroke-dasharray', '2 2');
    if (spec[k - 1] > 2) c.g.append('text').attr('x', c.x(k)).attr('y', c.y(spec[k - 1]) - 6).attr('text-anchor', 'middle').attr('class', isSel ? 'label-accent' : 'label-ink').text(k);
  }
  if (spec[fs.sel - 1] <= 2) c.g.append('text').attr('x', c.x(fs.sel)).attr('y', c.y(spec[fs.sel - 1]) - 6).attr('text-anchor', 'middle').attr('class', 'label-accent').text(fs.sel);
  c.g.append('rect').attr('x', 0).attr('y', 0).attr('width', c.iw).attr('height', c.ih).attr('fill', 'transparent').style('cursor', 'pointer')
    .on('click', (ev) => { const [mx] = d3.pointer(ev, c.g.node()); fs.sel = Math.max(1, Math.min(29, Math.round(c.x.invert(mx)))); selS.set(fs.sel); drawFreq(); });
  // Circle
  circBox.innerHTML = '';
  const S = Math.max(260, Math.min(360, circBox.clientWidth || 360));
  const pts = M.circle(fs.sel);
  const R = 0.85, sc = d3.scaleLinear().domain([-R, R]).range([20, S - 20]);
  const svg = d3.select(circBox).append('svg').attr('class', 'chart').attr('width', S).attr('height', S + 30).attr('viewBox', `0 0 ${S} ${S + 30}`)
    .attr('role', 'img').attr('aria-label', `The numbers' embeddings at frequency ${fs.sel}`);
  svg.append('text').attr('x', 0).attr('y', 20).attr('class', 'label-accent').text(`frequency ${fs.sel}${fs.keep[fs.sel] ? '' : ' (removed)'}`);
  const g = svg.append('g').attr('transform', 'translate(0,30)');
  pts.forEach(([px, py], a) => g.append('circle').attr('cx', sc(px)).attr('cy', sc(-py)).attr('r', 3).attr('fill', 'var(--ink)'));
  for (const a of [0, 1, 2]) {
    const [px, py] = pts[a], rr = Math.hypot(px, py);
    if (rr < 0.3) continue;
    const ux = px / rr, uy = py / rr;
    g.append('text').attr('x', sc(px + ux * 0.1)).attr('y', sc(-(py + uy * 0.1)) + 7).attr('text-anchor', 'middle').attr('class', 'label-accent').text(a);
  }
  const maxR = Math.max(...pts.map(([px, py]) => Math.hypot(px, py)));
  const desc = maxR > 0.3 ? `At frequency ${fs.sel} the numbers lie on a circle; the labels show that 1 is a step of ${fs.sel}/59 of a turn from 0, and 2 another step on.` : `At frequency ${fs.sel} every number&rsquo;s part is near the center: the embedding hardly uses this frequency.`;
  out2.innerHTML = `${desc} Frequencies kept: <b>${kept()}</b> of 29. Test accuracy: <b>${computing ? 'computing…' : (acc * 100).toFixed(1) + '%'}</b>.`;
}

function recompute() {
  const my = ++job;
  computing = true;
  drawFreq();
  setTimeout(() => {
    if (my !== job) return;
    const all = Object.values(fs.keep).every(Boolean);
    MF.setEmbedding(all ? null : MF.ablated(fs.keep));
    acc = MF.accuracy().testAcc;
    computing = false;
    drawFreq();
  }, 30);
}
drawFreq();

// ---- Static figure: how the model adds 12 and 30 ----
(function figure() {
  const fig = document.getElementById('fig-algo');
  const add = (title) => { const box = document.createElement('div'); box.innerHTML = `<p class="small-title">${title}</p>`; fig.append(box); return box; };
  // 1. The circle at frequency 12.
  {
    const box = add('Angles add, at frequency 12');
    const S = Math.max(240, Math.min(300, box.clientWidth || 300));
    const svg = d3.select(box).append('svg').attr('class', 'chart').attr('width', S).attr('height', S).attr('viewBox', `0 0 ${S} ${S}`).attr('aria-hidden', 'true');
    const cx = S / 2, cy = S / 2, rad = S / 2 - 44, w = (2 * Math.PI * 12) / p;
    svg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', rad).attr('fill', 'none').attr('stroke', 'var(--rule)');
    for (let a = 0; a < p; a++) svg.append('circle').attr('cx', cx + rad * Math.cos(w * a)).attr('cy', cy - rad * Math.sin(w * a)).attr('r', 2).attr('fill', 'var(--quiet)');
    for (const [a, col] of [[12, 'accent'], [30, 'second'], [42, 'ink']]) {
      const ang = w * a, x = cx + rad * Math.cos(ang), y = cy - rad * Math.sin(ang);
      svg.append('line').attr('x1', cx).attr('y1', cy).attr('x2', x).attr('y2', y).attr('stroke', `var(--${col})`).attr('stroke-width', 1.6);
      svg.append('circle').attr('cx', x).attr('cy', y).attr('r', 4).attr('fill', `var(--${col})`);
      svg.append('text').attr('x', cx + (rad + 22) * Math.cos(ang)).attr('y', cy - (rad + 22) * Math.sin(ang) + 7).attr('text-anchor', 'middle').attr('class', `label-${col}`).text(a);
    }
  }
  // 2. The three cosines and their sum.
  const cs = d3.range(p);
  {
    const box = add('Scores cos(w(42 − c))');
    const c = chart(box, { width: 320, height: 240, margin: { top: 30, right: 16, bottom: 48, left: 40 }, x: { domain: [0, 58], ticks: 3, label: 'candidate c' }, y: { domain: [-3, 3.4], ticks: 3 }, title: 'Cosine scores at three frequencies and their sum' });
    for (const k of [1, 12, 27]) c.line(`k${k}`, cs.map((v) => [v, Math.cos((2 * Math.PI * k * (42 - v)) / p)]), { color: 'quiet', width: 1 });
    c.line('sum', cs.map((v) => [v, [1, 12, 27].reduce((s, k) => s + Math.cos((2 * Math.PI * k * (42 - v)) / p), 0)]), { color: 'ink', width: 1.8 });
    c.point('peak', 42, 3, { color: 'ink', r: 3, label: 'sum, 42', anchor: 'end', dx: -6, dy: 6 });
  }
  // 3. The model's logits.
  {
    const box = add('The model’s logits for 12 + 30');
    const L = Array.from(M.run(12, 30).logits);
    const c = chart(box, { width: 320, height: 240, margin: { top: 30, right: 16, bottom: 48, left: 48 }, x: { domain: [0, 58], ticks: 3, label: 'candidate c' }, y: { domain: [Math.floor(d3.min(L) / 10) * 10, Math.ceil(d3.max(L) / 10) * 10 + 5], ticks: 4 }, title: 'Logits of the trained model for 12 + 30' });
    c.line('L', cs.map((v) => [v, L[v]]), { color: 'ink', width: 1.6 });
    c.point('peak', 42, L[42], { color: 'ink', r: 3, label: '42', anchor: 'end', dx: -6, dy: 6 });
  }
})();
