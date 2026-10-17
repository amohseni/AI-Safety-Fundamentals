// Page 8.4: random goals, common moves; and an incomplete proxy.
import { chart } from '../lib/chart.js';
import { slider, segmented, button, fmt } from '../lib/ui.js';
import { defaultGraph, SHUTDOWN, sampleRewards, analyze, allocate, SHAPES, DISTS } from '../lib/8-power.js';

const d3 = window.d3;
const pct = (v) => `${Math.round(100 * v)}%`;
const num = (d) => (v) => { const t = v.toFixed(d); return /^-0\.?0*$/.test(t) ? t.slice(1) : t.replace('-', '−'); };

// Draws a graph with first-move shares and power into an SVG. Returns the node selection for wiring events.
function drawGraph(svgSel, graph, result, { width, height = width < 450 ? 420 : 340, selected = null, interactive = false }) {
  const R = 15;
  svgSel.attr('viewBox', `0 0 ${width} ${height}`).attr('width', width).attr('height', height);
  svgSel.selectAll('*').remove();
  const defs = svgSel.append('defs');
  for (const [id, color] of [['arrow-q', 'var(--quiet)'], ['arrow-a', 'var(--accent)']]) {
    defs.append('marker').attr('id', id).attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 10).attr('markerHeight', 10).attr('markerUnits', 'userSpaceOnUse').attr('orient', 'auto-start-reverse')
      .append('path').attr('d', 'M0,0 L10,5 L0,10 z').attr('fill', color);
  }
  const padX = 40, padY = 44;
  const px = (n) => padX + n.x * (width - 2 * padX), py = (n) => padY + n.y * (height - 2 * padY);
  const byId = new Map(graph.nodes.map((n) => [n.id, n]));
  const ix = result.ix;
    // Edges: pairs with arrows both ways are drawn as one line with two heads.
  const set = new Set(graph.edges.map(([a, b]) => `${a}>${b}`));
  const done = new Set();
  const eg = svgSel.append('g');
  for (const [a, b] of graph.edges) {
    if (done.has(`${a}>${b}`) || !byId.has(a) || !byId.has(b) || a === b) continue;
    const both = set.has(`${b}>${a}`);
    done.add(`${a}>${b}`); if (both) done.add(`${b}>${a}`);
    const A = byId.get(a), B = byId.get(b);
    const x1 = px(A), y1 = py(A), x2 = px(B), y2 = py(B), len = Math.hypot(x2 - x1, y2 - y1) || 1;
    const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
    const fromStart = a === 'start' && !both;
    const share = fromStart ? result.share.get(ix.get(b)) || 0 : 0;
    eg.append('line').attr('x1', x1 + ux * (R + 2)).attr('y1', y1 + uy * (R + 2)).attr('x2', x2 - ux * (R + 3)).attr('y2', y2 - uy * (R + 3))
      .attr('stroke', fromStart ? 'var(--accent)' : 'var(--rule)').attr('stroke-width', fromStart ? 1.5 + 14 * share : 1.5)
      .attr('marker-end', `url(#${fromStart ? 'arrow-a' : 'arrow-q'})`).attr('marker-start', both ? 'url(#arrow-q)' : null);
    if (fromStart) {
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, nx = -uy, ny = ux;
      // Label on the upper side of a level arrow, and on the left side of a steep one.
      const steep = Math.abs(uy) > 0.45;
      const side = steep ? (nx < 0 ? 1 : -1) : (ny > 0 ? -1 : 1) * (width < 450 ? -1 : 1); // below a level arrow on narrow screens
      const below = !steep && width < 450;
      eg.append('text').attr('class', 'share').attr('x', mx + side * nx * 14).attr('y', my + side * ny * 14 + (steep ? 5 : below ? 12 : -2))
        .attr('text-anchor', steep ? 'end' : 'middle').text(pct(share));
    }
  }
  const ng = svgSel.append('g');
  const nodes = ng.selectAll('g.node').data(graph.nodes).join('g').attr('class', (n) => `node${n.id === selected ? ' selected' : ''}`)
    .attr('transform', (n) => `translate(${px(n)},${py(n)})`);
  if (interactive) nodes.attr('tabindex', 0).attr('role', 'button').attr('aria-label', (n) => `State ${n.name || n.id}, power ${num(2)(result.power[ix.get(n.id)])}`);
  nodes.append('circle').attr('r', R);
  nodes.append('text').attr('class', 'power').attr('text-anchor', (n) => (n.label === 'right' ? 'start' : 'middle'))
    .attr('x', (n) => (n.label === 'right' ? R + 6 : 0)).attr('y', (n) => (n.label === 'right' ? 25 : R + 18)).text((n) => num(2)(result.power[ix.get(n.id)]));
  const lab = nodes.filter((n) => n.name).append('text').text((n) => n.name);
  lab.attr('text-anchor', (n) => (n.label === 'right' ? 'start' : n.label === 'aboveleft' || n.label === 'left' ? 'end' : 'middle'))
    .attr('x', (n) => (n.label === 'right' ? R + 6 : n.label === 'aboveleft' ? -4 : n.label === 'left' ? -R - 5 : 0))
    .attr('y', (n) => (n.label === 'right' ? 0 : n.label === 'left' ? 5 : -R - 7));
  return nodes;
}

// ---- Panel 1: where optimal agents go ----
(function graphPanel() {
  const st = { graph: defaultGraph(), shutdown: false, dist: 'uniform', gamma: 0.9, seed: 1, selected: null, linking: false, nextId: 1 };
  const ctl = document.getElementById('graph-controls');
  const out = document.getElementById('graph-readout');
  const stage = document.getElementById('graph-stage');
  const svg = d3.select(stage).append('svg').attr('class', 'graph').attr('role', 'group').attr('aria-label', 'Graph of states; arrows out of the start show the share of reward functions for which each first move is optimal');
  let msg = '';

  segmented(ctl, { name: 'Reward distribution', options: Object.entries(DISTS).map(([k, d]) => [k, d.name]), value: st.dist, onChange: (k) => { st.dist = k; draw(); } });
  slider(ctl, { name: 'Discount factor γ', min: 0.05, max: 0.99, step: 0.01, value: st.gamma, format: fmt(2), onInput: (v) => { st.gamma = v; draw(); } });
  segmented(ctl, { name: 'Shutdown state', options: [['no', 'None'], ['yes', 'Add one']], value: 'no', onChange: (k) => { st.shutdown = k === 'yes'; if (!st.shutdown && st.selected === 'off') st.selected = null; draw(); } });
  button(ctl, 'Draw 1,000 new reward functions', () => { st.seed++; draw(); });
  const edit = document.createElement('div'); edit.className = 'ctl';
  edit.innerHTML = '<span class="ctl-head"><span class="ctl-name">Edit the graph</span></span>';
  const row = document.createElement('div'); row.className = 'edit-row'; edit.append(row); ctl.append(edit);
  const addBtn = button(row, 'Add a state after the selected one', () => addState());
  const linkBtn = button(row, 'Draw or erase an arrow from the selected state', () => { if (!st.selected) return; st.linking = true; msg = 'Now select the state the arrow should point to.'; draw(); });
  const delBtn = button(row, 'Remove the selected state', () => removeState());
  button(row, 'Restore the original graph', () => { st.graph = defaultGraph(); st.selected = null; st.linking = false; msg = ''; draw(); });

  const graphNow = () => (st.shutdown ? { nodes: [...st.graph.nodes, SHUTDOWN], edges: [...st.graph.edges, ['start', 'off']] } : st.graph);

  function addState() {
    if (!st.selected || st.selected === 'off') return;
    const g = st.graph, from = g.nodes.find((n) => n.id === st.selected);
    const others = graphNow().nodes;
    let best = null, bestD = -1;
    for (let k = 0; k < 24; k++) {
      const a = (k / 24) * 2 * Math.PI;
      for (const rr of [0.18, 0.28]) {
        const x = Math.min(0.97, Math.max(0.03, from.x + rr * Math.cos(a))), y = Math.min(0.97, Math.max(0.03, from.y + rr * 1.6 * Math.sin(a)));
        const d = Math.min(...others.map((n) => Math.hypot(n.x - x, (n.y - y) / 1.6)));
        if (d > bestD) { bestD = d; best = { x, y }; }
      }
    }
    const id = `n${st.nextId++}`;
    g.nodes.push({ id, name: '', ...best });
    g.edges.push([st.selected, id]);
    st.selected = id; msg = 'Added a state, with an arrow to it from the selected state.';
    draw();
  }
  function removeState() {
    const id = st.selected;
    if (!id || id === 'start') { msg = 'The start cannot be removed.'; draw(); return; }
    if (id === 'off') { msg = 'Use the shutdown control to remove the shutdown state.'; draw(); return; }
    st.graph.nodes = st.graph.nodes.filter((n) => n.id !== id);
    st.graph.edges = st.graph.edges.filter(([a, b]) => a !== id && b !== id);
    st.selected = null; msg = 'Removed the state and its arrows.';
    draw();
  }
  function pick(id) {
    if (st.linking && st.selected && id !== st.selected) {
      if (st.selected === 'off' || id === 'off') { msg = 'The shutdown state keeps its single arrow from the start.'; st.linking = false; draw(); return; }
      const g = st.graph, k = g.edges.findIndex(([a, b]) => a === st.selected && b === id);
      if (k >= 0) { g.edges.splice(k, 1); msg = 'Erased the arrow.'; } else { g.edges.push([st.selected, id]); msg = 'Drew the arrow.'; }
      st.linking = false;
    } else { st.selected = st.selected === id ? null : id; st.linking = false; msg = ''; }
    draw();
  }

  function draw() {
    const g = graphNow();
    const samples = sampleRewards(st.dist, g.nodes.length, 1000, st.seed);
    const res = analyze(g, samples, st.gamma);
    const width = Math.max(280, Math.min(540, stage.clientWidth || 540));
    const nodes = drawGraph(svg, g, res, { width, selected: st.selected, interactive: true });
    nodes.on('click', (ev, n) => { st.focusId = null; pick(n.id); })
      .on('keydown', (ev, n) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); st.focusId = n.id; pick(n.id); } });
    // The drawing is rebuilt on every change; keep keyboard focus on the state the reader was on.
    if (st.focusId) { const el = nodes.filter((n) => n.id === st.focusId).node(); if (el) el.focus(); }
    addBtn.disabled = !st.selected || st.selected === 'off';
    linkBtn.disabled = !st.selected;
    delBtn.disabled = !st.selected || st.selected === 'start';
    const name = (i) => { const n = g.nodes[i]; return n.id === 'start' ? 'staying at the start' : n.id === 'off' ? 'shutdown' : n.name ? `the ${n.name}` : 'an unnamed state'; };
    const moves = [...res.share].sort((a, b) => b[1] - a[1]).map(([t, v]) => `${name(t)} <b>${pct(v)}</b>`);
    const off = st.shutdown ? res.share.get(res.ix.get('off')) : null;
    out.innerHTML = `Optimal first move from the start, over 1,000 ${DISTS[st.dist].name.toLowerCase()} reward functions with γ = ${st.gamma.toFixed(2)}: ${moves.join(', ')}. ` +
      (off !== null ? `Moving to shutdown is optimal for ${pct(off)}, so ${pct(1 - off)} of reward functions make avoiding it optimal. ` : '') +
      (st.selected ? `Selected: ${name(res.ix.get(st.selected)).replace('staying at ', '')}. ` : '') + msg;
  }
  let resizeT = null;
  window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(draw, 100); });
  draw();

  // Static figure: the original graph, uniform rewards, gamma 0.9.
  const fig = document.getElementById('fig-hub');
  const fsvg = d3.select(fig).append('svg').attr('class', 'graph').attr('role', 'img').attr('aria-label', 'The original graph with first-move shares');
  const drawFig = () => {
    const g0 = defaultGraph();
    const r0 = analyze(g0, sampleRewards('uniform', g0.nodes.length, 1000, 1), 0.9);
    drawGraph(fsvg, g0, r0, { width: Math.max(280, Math.min(540, fig.clientWidth || 540)) });
  };
  drawFig();
  window.addEventListener('resize', () => setTimeout(drawFig, 120));
})();

// ---- Panel 2: optimizing an incomplete proxy ----
(function allocPanel() {
  const st = { L: 4, J: 3, shape: 'diminishing', floor: -3 };
  const ctl = document.getElementById('alloc-controls');
  const out = document.getElementById('alloc-readout');
  const LS = slider(ctl, { name: 'Attributes the person cares about, L', min: 2, max: 8, step: 1, value: st.L, format: fmt(0),
    onInput: (v) => { st.L = v; if (st.J > v - 1) { st.J = v - 1; JS.set(st.J); } JS.el.max = v - 1; draw(); } });
  const JS = slider(ctl, { name: 'Attributes in the proxy, J', min: 1, max: st.L - 1, step: 1, value: st.J, format: fmt(0),
    onInput: (v) => { st.J = Math.min(v, st.L - 1); draw(); } });
  segmented(ctl, { name: 'The person’s utility for each attribute', options: Object.entries(SHAPES).map(([k, s]) => [k, s.name]), value: st.shape,
    onChange: (k) => { st.shape = k; draw(); } });
  slider(ctl, { name: 'Floor: the lowest any attribute can go', min: -6, max: -1, step: 0.5, value: st.floor, format: num(1),
    onInput: (v) => { st.floor = v; draw(); } });

  let cx = null, cu = null;
  const mk = () => {
    document.getElementById('alloc-x').innerHTML = ''; document.getElementById('alloc-u').innerHTML = '';
    cx = chart(document.getElementById('alloc-x'), { height: 260, margin: { top: 34, right: 112, bottom: 52, left: 52 },
      x: { domain: [0, 100], label: 'optimization step' }, y: { domain: [-6, 6], label: 'attribute value' }, title: 'Attribute values over the optimization' });
    cu = chart(document.getElementById('alloc-u'), { height: 260, margin: { top: 34, right: 112, bottom: 52, left: 52 },
      x: { domain: [0, 100], label: 'optimization step' }, y: { domain: [-20, 5], label: 'utility' }, title: 'True utility and proxy over the optimization' });
  };
  mk();
  let ids = [];
  function draw() {
    const tr = allocate(st);
    const n = tr.length - 1;
    cx.setX([0, n]); cu.setX([0, n]);
    const xs = tr.map((s) => s.x).flat();
    cx.setY([Math.min(st.floor - 0.5, d3.min(xs)), Math.max(1, d3.max(xs) + 0.5)]);
    ids.forEach((id) => cx.remove(id)); ids = [];
    for (let i = 0; i < st.L; i++) {
      const inProxy = i < st.J, id = `x${i}`;
      const showLabel = i === 0 || i === st.J;
      cx.line(id, tr.map((s, t) => [t, s.x[i]]), { color: inProxy ? 'second' : 'accent', label: showLabel ? (inProxy ? 'in the proxy' : 'left out') : null });
      ids.push(id);
    }
    cx.line('floorline', [[0, st.floor], [n, st.floor]], { color: 'quiet', dash: '3 3', width: 1, label: 'floor', dy: 16 });
    const us = tr.map((s) => s.true), ps = tr.map((s) => s.proxy);
    cu.setY([Math.min(d3.min(us), d3.min(ps), 0) * 1.05 - 0.5, Math.max(d3.max(us), d3.max(ps)) * 1.1 + 0.5]);
    cu.line('true', tr.map((s, t) => [t, s.true]), { color: 'accent', label: 'true utility' });
    cu.line('proxy', tr.map((s, t) => [t, s.proxy]), { color: 'second', label: 'proxy' });
    cu.spreadLabels(['true', 'proxy'], 18);
    cx.spreadLabels(ids, 18);
    let peak = -Infinity, pt = 0; us.forEach((u, t) => { if (u > peak) { peak = u; pt = t; } });
    const end = tr[n];
    const omitted = end.x.slice(st.J);
    out.innerHTML = `After ${n} steps the agent stops. The ${st.L - st.J === 1 ? 'attribute' : `${st.L - st.J} attributes`} left out of the proxy ${st.L - st.J === 1 ? 'is' : 'are'} at <span class="c-accent"><b>${omitted.map(num(2)).join(', ')}</b></span>; ` +
      `the ${st.J === 1 ? 'attribute' : `${st.J} attributes`} in the proxy ${st.J === 1 ? 'is' : 'are each'} at <span class="c-second">${num(2)(end.x[0])}</span>. ` +
      `The person&rsquo;s <span class="c-accent">true utility</span> peaked at ${num(2)(peak)} at step ${pt} and ends at <b>${num(2)(end.true)}</b>; the <span class="c-second">proxy</span> ends at ${num(2)(end.proxy)}.`;
  }
  draw();
  let resizeT = null;
  window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(() => { mk(); ids = []; draw(); }, 120); });
})();
