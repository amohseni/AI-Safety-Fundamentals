// Page 4.2: be the labeler.
import { chart } from '../lib/chart.js';
import { segmented, button } from '../lib/ui.js';
import { FEATURES, PROMPTS, compose, readerPairs, biasedLabels, fitBT, score, prefProb } from '../lib/s45-prefs.js';

const d3 = window.d3;
const PAIRS = readerPairs();
const state = { mine: [], batches: 0 };
const simulated = () => Array.from({ length: state.batches }, (_, k) => biasedLabels(20, 1001 + k)).flat();
const labels = () => state.mine.concat(simulated());
let w = [0, 0, 0, 0];
const listeners = [];
const refit = () => { w = labels().length ? fitBT(labels()) : [0, 0, 0, 0]; listeners.forEach((f) => f()); };

// ---- Static figure: the Bradley-Terry curve ----
(function figure() {
  const box = document.createElement('div');
  document.getElementById('fig-bt').append(box);
  const c = chart(box, { width: 520, height: 260, margin: { top: 30, right: 30, bottom: 56, left: 52 },
    x: { domain: [-5, 5], ticks: 5, label: 'r(A) − r(B)' }, y: { domain: [0, 1], ticks: 2, label: 'P(A preferred)' }, title: 'Bradley-Terry curve' });
  c.line('s', d3.range(-5, 5.001, 0.05).map((z) => [z, 1 / (1 + Math.exp(-z))]), { color: 'ink' });
  for (const z of [0, 1, 3]) {
    const p = 1 / (1 + Math.exp(-z));
    c.point(`p${z}`, z, p, { color: 'accent', r: 4, label: p.toFixed(2), dx: z === 3 ? -8 : 8, dy: z === 3 ? -10 : 16, anchor: z === 3 ? 'end' : 'start' });
  }
})();

// ---- Panel 1: label the pairs ----
(function labeler() {
  const q = document.getElementById('question'), pairEl = document.getElementById('pair');
  const btns = document.getElementById('label-buttons'), prog = document.getElementById('progress');
  const choose = (aWins) => {
    const i = state.mine.length;
    if (i >= PAIRS.length) return;
    state.mine.push({ ...PAIRS[i], aWins });
    show(); refit();
  };
  const bA = button(btns, 'A is better', () => choose(true), 'primary');
  const bB = button(btns, 'B is better', () => choose(false), 'primary');
  button(btns, 'Undo', () => { state.mine.pop(); show(); refit(); });
  button(btns, 'Start over', () => { state.mine = []; show(); refit(); });
  function show() {
    const i = state.mine.length;
    if (i >= PAIRS.length) {
      q.textContent = 'All 20 pairs are labeled.';
      pairEl.innerHTML = '';
      bA.disabled = bB.disabled = true;
    } else {
      const P = PAIRS[i];
      q.innerHTML = `<em>Question:</em> ${PROMPTS[P.prompt].q}`;
      pairEl.innerHTML = `<div><p class="who">Response A</p><p>${compose(P.prompt, P.a)}</p></div><div><p class="who">Response B</p><p>${compose(P.prompt, P.b)}</p></div>`;
      bA.disabled = bB.disabled = false;
    }
    prog.textContent = `Pairs labeled: ${i} of ${PAIRS.length}.`;
  }
  show();
})();

// ---- Panel 2: the weights ----
(function weights() {
  const ctl = document.getElementById('w-controls'), el = document.getElementById('weights'), out = document.getElementById('w-readout');
  button(ctl, 'Add 20 labels from the biased labeler', () => { state.batches++; refit(); });
  button(ctl, 'Remove the simulated labels', () => { state.batches = 0; refit(); });
  const W = Math.max(280, Math.min(520, el.clientWidth || 520)), rowH = 40, top = 10, left = 118, right = 50;
  const H = top + rowH * FEATURES.length + 34;
  const svg = d3.select(el).append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .attr('role', 'img').attr('aria-label', 'Fitted weight on each feature');
  const x = d3.scaleLinear().domain([-3, 3]).range([left, W - right]);
  svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${top + rowH * FEATURES.length})`).call(d3.axisBottom(x).ticks(6).tickSizeOuter(0));
  svg.append('line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', top).attr('y2', top + rowH * FEATURES.length).attr('stroke', 'var(--rule)');
  const g = svg.selectAll('g.f').data(FEATURES).join('g').attr('class', 'f').attr('transform', (d, i) => `translate(0,${top + i * rowH + rowH / 2})`);
  g.append('text').attr('x', 0).attr('y', 6).text((d) => d.name);
  g.append('line').attr('class', 'bar').attr('x1', x(0)).attr('y1', 0).attr('y2', 0).attr('stroke', 'var(--accent)').attr('stroke-width', 3);
  g.append('circle').attr('r', 5).attr('fill', 'var(--accent)');
  g.append('text').attr('class', 'v label-accent').attr('y', 6);
  const draw = () => {
    const cl = (v) => Math.max(-3, Math.min(3, v));
    g.select('line.bar').attr('x2', (d, i) => x(cl(w[i])));
    g.select('circle').attr('cx', (d, i) => x(cl(w[i])));
    g.select('text.v').attr('x', (d, i) => x(cl(w[i])) + (w[i] >= 0 ? 10 : -10)).attr('text-anchor', (d, i) => (w[i] >= 0 ? 'start' : 'end'))
      .text((d, i) => w[i].toFixed(2));
    const n = state.mine.length, m = state.batches * 20;
    out.textContent = n + m === 0 ? 'No labels yet. Label some pairs above.'
      : `Fitted to ${n} of your labels${m ? ` and ${m} simulated labels` : ''}.`;
  };
  listeners.push(draw);
  draw();
})();

// ---- Panel 3: score two responses ----
(function composer() {
  const el = document.getElementById('composer'), out = document.getElementById('s-readout');
  const resp = {
    A: { long: 1, correct: 0, confident: 1, flattering: 0 },
    B: { long: 0, correct: 1, confident: 0, flattering: 0 },
  };
  let prompt = 0;
  const texts = {};
  for (const k of ['A', 'B']) {
    const col = document.createElement('div');
    col.innerHTML = `<p class="who">Response ${k}</p>`;
    el.append(col);
    for (const f of FEATURES) {
      segmented(col, { name: `${k}: ${f.name}`, options: [['1', f.on], ['0', f.off]], value: String(resp[k][f.key]),
        onChange: (v) => { resp[k][f.key] = Number(v); draw(); } });
    }
    texts[k] = document.createElement('p');
    col.append(texts[k]);
  }
  const row = document.createElement('div');
  row.className = 'seg';
  el.after(row);
  button(row, 'Another question', () => { prompt = (prompt + 1) % PROMPTS.length; draw(); });
  // Hide the repeated "A:" or "B:" in the visible control names; keep them for screen readers.
  el.querySelectorAll('.ctl-name').forEach((s) => { s.textContent = s.textContent.slice(3); });
  function draw() {
    for (const k of ['A', 'B']) texts[k].innerHTML = `<em>${PROMPTS[prompt].q}</em> ${compose(prompt, resp[k])}`;
    const rA = score(w, resp.A), rB = score(w, resp.B), p = prefProb(w, resp.A, resp.B);
    out.innerHTML = `Scores: response A <b>${rA.toFixed(2)}</b>, response B <b>${rB.toFixed(2)}</b>. ` +
      `The reward model predicts that a labeler prefers A with probability <b>${p.toFixed(2)}</b>.`;
  }
  listeners.push(draw);
  draw();
})();
