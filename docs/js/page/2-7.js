// Page 2.7: gradients by hand.
import { GRAPHS, forward, backward, chainPaths, nudge } from '../lib/s2-graph.js';
import { slider, segmented, button } from '../lib/ui.js';

const d3 = window.d3;
const minus = (str) => str.replace('-', '−');
const f4 = (v) => minus((Math.abs(v) < 5e-5 ? 0 : v).toFixed(4));
const f3 = (v) => minus((Math.abs(v) < 5e-4 ? 0 : v).toFixed(3));
const f4p = (v) => (v < -5e-5 ? `(${f4(v)})` : f4(v)); // in products and sums
const nodeOf = (g, id) => g.nodes.find((n) => n.id === id);
const dname = (g, id) => `∂L/∂${nodeOf(g, id).name}`;

// Draws a graph into `el` at the element's true width. reveal: {fwd: number of computed nodes shown, bwd: number of backward steps done,
// active: id of the node whose backward step is shown, selected: parameter id}.
function render(el, g, val, back, reveal, onPick) {
  el.innerHTML = '';
  const W = Math.max(300, Math.min(820, el.clientWidth || 820));
  const horiz = W >= 680;
  const lanes = g.nodes.map((n) => n.pos[1]), lmin = Math.min(...lanes), lmax = Math.max(...lanes);
  const pad = horiz ? 52 : 50;
  const colW = horiz ? (W - 2 * pad) / 6 : (W - 2 * pad) / Math.max(1, lmax - lmin);
  const rowH = horiz ? 84 : 96;
  const H = horiz ? (lmax - lmin) * rowH + 110 : 6 * rowH + 90;
  const at = (n) => (horiz ? [pad + n.pos[0] * colW, 46 + (n.pos[1] - lmin) * rowH] : [pad + (n.pos[1] - lmin) * colW, 30 + n.pos[0] * rowH]);
  const svg = d3.select(el).append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .attr('role', 'img').attr('aria-label', `Computation graph: ${g.title}`);
  const computed = g.nodes.filter((n) => n.op !== 'input');
  const shownVal = new Set([...g.nodes.filter((n) => n.op === 'input').map((n) => n.id), ...computed.slice(0, reveal.fwd).map((n) => n.id)]);
  // Gradients known after the first `bwd` backward steps: a node's derivative is complete once all its users have sent theirs.
  const done = new Set(back.steps.slice(0, reveal.bwd).map((s) => s.node));
  const L = g.nodes[g.nodes.length - 1].id;
  const gradShown = (id) => reveal.bwd > 0 && (id === L || g.nodes.filter((n) => n.in && n.in.includes(id)).every((u) => done.has(u.id)));
  const activeStep = back.steps.find((s) => s.node === reveal.active);
  // Nodes
  const widths = {};
  for (const n of g.nodes) {
    const [x, y] = at(n);
    const gEl = svg.append('g').attr('transform', `translate(${x},${y})`);
    const label = n.op === 'input' ? n.name : `${n.name} = ${n.def}`;
    const sel = reveal.selected === n.id;
    gEl.append('text').attr('text-anchor', 'middle').attr('class', n.param ? 'label-ink' : '').attr('font-style', n.param ? 'italic' : null)
      .attr('text-decoration', sel ? 'underline' : null).text(label);
    widths[n.id] = gEl.node().getBBox().width || 8 * label.length;
    if (shownVal.has(n.id)) gEl.append('text').attr('text-anchor', 'middle').attr('y', 22).attr('class', 'label-ink').text(f3(val[n.id]));
    if (gradShown(n.id)) gEl.append('text').attr('text-anchor', 'middle').attr('y', 44).attr('class', 'label-accent').text(f3(back.grad[n.id]));
    else if (reveal.bwd > 0 && back.parts[n.id].some((p) => done.has(p.from))) gEl.append('text').attr('text-anchor', 'middle').attr('y', 44).text('…');
    if (n.param && onPick) {
      gEl.style('cursor', 'pointer').on('click', () => onPick(n.id));
      gEl.insert('rect', ':first-child').attr('x', -34).attr('y', -22).attr('width', 68).attr('height', 72).attr('fill', 'transparent');
    }
  }
  // Edges, drawn after the nodes so their ends can be placed at the measured edges of the labels.
  const edgeLayer = svg.insert('g', ':first-child');
  for (const n of g.nodes) for (const k of n.in || []) {
    const src = nodeOf(g, k);
    const [x1, y1] = at(src), [x2, y2] = at(n);
    const hot = activeStep && activeStep.node === n.id;
    const w1 = widths[src.id] / 2 + 6, w2 = widths[n.id] / 2 + 6;
    const s = horiz ? [x1 + w1, y1 - 6, x2 - w2, y2 - 6] : [x1, y1 + 52, x2, y2 - 22];
    edgeLayer.append('line').attr('x1', s[0]).attr('y1', s[1]).attr('x2', s[2]).attr('y2', s[3])
      .attr('stroke', hot ? 'var(--accent)' : 'var(--rule)').attr('stroke-width', hot ? 2 : 1.2);
  }
}

// ---- Static figure ----
(function figure() {
  const g = GRAPHS.one, inp = Object.fromEntries(g.nodes.filter((n) => n.op === 'input').map((n) => [n.id, n.value]));
  const val = forward(g, inp), back = backward(g, val);
  const el = document.getElementById('fig-graph');
  const draw = () => render(el, g, val, back, { fwd: 99, bwd: 99 });
  draw();
  let t; window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(draw, 200); });
})();

// ---- The tool ----
const state = { graph: 'one', fwd: 0, bwd: 0, active: null, selected: null, inputs: {}, nudged: null, msg: '' };
for (const [k, g] of Object.entries(GRAPHS)) state.inputs[k] = Object.fromEntries(g.nodes.filter((n) => n.op === 'input').map((n) => [n.id, n.value]));
const ctl = document.getElementById('controls');
const stage = document.getElementById('graph');
const out = document.getElementById('readout');
const inputsBox = document.createElement('div');
inputsBox.style.display = 'contents';
let paramSeg = null, paramBox = document.createElement('div');
paramBox.style.display = 'contents';

segmented(ctl, { name: 'Graph', options: [['one', 'One path'], ['two', 'A value used twice']], value: state.graph,
  onChange: (k) => { state.graph = k; state.fwd = 0; state.bwd = 0; state.active = null; state.selected = null; state.nudged = null; state.msg = ''; build(); } });
const btns = document.createElement('div'); btns.className = 'seg'; ctl.append(btns);
button(btns, 'Forward', () => fwdStep(), 'primary');
button(btns, 'Backward', () => bwdStep(), 'primary');
const btns2 = document.createElement('div'); btns2.className = 'seg'; ctl.append(btns2);
button(btns2, 'Show all', () => { const g = GRAPHS[state.graph]; state.fwd = g.nodes.length; state.bwd = g.nodes.length; state.active = null; state.msg = 'All values and derivatives are shown.'; draw(); });
button(btns2, 'Clear', () => { state.fwd = 0; state.bwd = 0; state.active = null; state.nudged = null; state.msg = ''; draw(); });
ctl.append(paramBox);
ctl.append(inputsBox);

function build() {
  const g = GRAPHS[state.graph], inp = state.inputs[state.graph];
  inputsBox.innerHTML = '';
  for (const n of g.nodes.filter((n) => n.op === 'input')) {
    slider(inputsBox, { name: `${n.param ? 'Parameter' : 'Input'} ${n.name}`, min: -3, max: 3, step: 0.01, value: inp[n.id], format: (v) => minus(v.toFixed(2)),
      onInput: (v) => { inp[n.id] = v; state.nudged = null; draw(); } });
  }
  paramBox.innerHTML = '';
  const params = g.nodes.filter((n) => n.param);
  paramSeg = segmented(paramBox, { name: 'Parameter to follow', options: params.map((n) => [n.id, n.name]), value: state.selected, onChange: (k) => pick(k) });
  const nb = document.createElement('div'); nb.className = 'seg'; paramBox.append(nb);
  button(nb, 'Nudge test', () => { if (!state.selected) pick(params[0].id); state.nudged = state.selected; draw(); });
  draw();
}
function pick(id) { state.selected = id; paramSeg.set(id); draw(); }

function fwdStep() {
  const g = GRAPHS[state.graph], computed = g.nodes.filter((n) => n.op !== 'input');
  if (state.fwd >= computed.length) { state.msg = 'The forward pass is complete: every value is known.'; draw(); return; }
  const n = computed[state.fwd];
  state.fwd++; state.active = null;
  const val = forward(g, state.inputs[state.graph]);
  const args = n.in.map((k) => f3(val[k]));
  const expr = { mul: `${args[0]} × ${args[1]}`, add: `${args[0]} + ${args[1]}`, sub: `${args[0]} − ${args[1]}`, sigmoid: `σ(${args[0]})`, square: `${args[0]}²` }[n.op];
  state.msg = `Forward: ${n.name} = ${n.def} = ${expr} = <b>${f3(val[n.id])}</b>.`;
  draw();
}
function bwdStep() {
  const g = GRAPHS[state.graph], computed = g.nodes.filter((n) => n.op !== 'input');
  if (state.fwd < computed.length) { state.fwd = computed.length; state.msg = 'The backward pass needs every value from the forward pass, so the forward pass was completed first. Press Backward again.'; draw(); return; }
  const val = forward(g, state.inputs[state.graph]), back = backward(g, val);
  if (state.bwd >= back.steps.length) { state.msg = 'The backward pass is complete: every derivative is known.'; state.active = null; draw(); return; }
  const st = back.steps[state.bwd];
  state.bwd++; state.active = st.node;
  const n = nodeOf(g, st.node);
  const lines = st.sends.map((s) => {
    const parts = back.parts[s.to];
    const complete = g.nodes.filter((u) => u.in && u.in.includes(s.to)).every((u) => back.steps.slice(0, state.bwd).some((q) => q.node === u.id));
    const sumNote = parts.length > 1 && complete ? ` Two derivatives arrive at ${nodeOf(g, s.to).name}, and they add: ${dname(g, s.to)} = ${parts.map((p) => f4p(p.value)).join(' + ')} = <b>${f4(back.grad[s.to])}</b>.` : '';
    return `${n.name} sends to ${nodeOf(g, s.to).name}: local derivative ∂${n.name}/∂${nodeOf(g, s.to).name} = ${f4(s.local)}, times incoming ${dname(g, n.id)} = ${f4(s.upstream)}, gives ${f4(s.contrib)}.${sumNote}`;
  });
  state.msg = `Backward step ${state.bwd} of ${back.steps.length}. ` + lines.join(' ');
  draw();
}

function draw() {
  const g = GRAPHS[state.graph], inp = state.inputs[state.graph];
  const val = forward(g, inp), back = backward(g, val);
  render(stage, g, val, back, { fwd: state.fwd, bwd: state.bwd, active: state.active, selected: state.selected }, pick);
  let s = state.msg ? `<p>${state.msg}</p>` : '<p>Press Forward to compute the values one node at a time.</p>';
  if (state.selected) {
    const p = nodeOf(g, state.selected), paths = chainPaths(g, val, p.id);
    const term = (path) => path.factors.slice().reverse().map((f) => f4p(f.local)).join(' × ');
    const sym = (path) => path.factors.slice().reverse().map((f) => `∂${f.node.name}/∂${nodeOf(g, f.input).name}`).join(' × ');
    s += `<p>Chain rule for ${p.name}${paths.length > 1 ? `, summed over its ${paths.length} paths to the loss` : ''}: ${dname(g, p.id)} = ` +
      paths.map((q) => sym(q)).join(' + ') + ' = ' + paths.map((q) => term(q)).join(' + ') + ` = <b>${f4(paths.reduce((a, q) => a + q.product, 0))}</b>.</p>`;
  }
  if (state.nudged) {
    const p = nodeOf(g, state.nudged), t = nudge(g, inp, p.id);
    s += `<p>Nudge test for ${p.name}: raising it by 0.001 changes the loss from ${f4(t.base)} to ${minus(t.up.toFixed(6))}, and lowering it by 0.001 changes the loss to ${minus(t.down.toFixed(6))}. ` +
      `Change in loss divided by the total nudge, 0.002: <b>${f4(t.slope)}</b>. Backpropagation: <b>${f4(back.grad[p.id])}</b>.</p>`;
  }
  out.innerHTML = s;
}

let rT;
window.addEventListener('resize', () => { clearTimeout(rT); rT = setTimeout(draw, 200); });
build();
