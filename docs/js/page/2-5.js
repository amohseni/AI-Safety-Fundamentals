// Page 2.5: the forecaster's game.
import { chart } from '../lib/chart.js';
import { slider, segmented, button, fmt, reducedMotion } from '../lib/ui.js';
import { rng } from '../lib/mlp.js';

const d3 = window.d3;

// Loss of report q when the outcome is y (1 = event happens).
const RULES = {
  log: { name: 'Log', loss: (q, y) => -Math.log(y ? q : 1 - q), yMax: 2.5, manyMax: 1.5 },
  brier: { name: 'Brier', loss: (q, y) => (y - q) ** 2, yMax: 1, manyMax: 0.5 },
  linear: { name: 'Linear', loss: (q, y) => 1 - (y ? q : 1 - q), yMax: 1, manyMax: 0.8 },
};
const lc = (rule) => (rule === 'brier' ? 'Brier' : RULES[rule].name.toLowerCase());
const expected = (rule, p, q) => p * RULES[rule].loss(q, 1) + (1 - p) * RULES[rule].loss(q, 0);
const curve = (rule, p) => d3.range(0.005, 0.9951, 0.005).map((q) => [q, expected(rule, p, q)]);
function argmin(rule, p) {
  let best = 0, bq = 0;
  for (let q = 0; q <= 1.0000001; q += 0.001) {
    const v = expected(rule, p, Math.min(q, 1));
    if (q === 0 || v < best) { best = v; bq = Math.min(q, 1); }
  }
  return [bq, best];
}

const state = { rule: 'log', p: 0.7, q: 0.85 };
const listeners = [];
const changed = () => listeners.forEach((f) => f());

// ---- Static figure: three small multiples at p = 0.7 ----
(function figure() {
  const fig = document.getElementById('fig-rules');
  for (const rule of ['log', 'brier', 'linear']) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${RULES[rule].name} rule</p>`;
    fig.append(box);
    const pts = curve(rule, 0.7);
    const c = chart(box, { width: 300, height: 220, margin: { top: 16, right: 20, bottom: 36, left: 46 },
      x: { domain: [0, 1], ticks: 3 }, y: { domain: [0, 1.2], ticks: 3 }, title: `${RULES[rule].name} rule, expected loss` });
    c.line('c', pts, { color: 'ink' });
    const [bq, bv] = argmin(rule, 0.7);
    c.point('m', bq, bv, { color: 'accent', label: `lowest at ${bq.toFixed(2)}`, anchor: bq > 0.8 ? 'end' : 'middle', dx: 0, dy: 18 });
  }
})();

// ---- Panel 1: one forecast ----
(function one() {
  const ctl = document.getElementById('one-controls');
  segmented(ctl, { name: 'Scoring rule', options: Object.entries(RULES).map(([k, r]) => [k, r.name]), value: state.rule,
    onChange: (k) => { state.rule = k; changed(); } });
  const pS = slider(ctl, { name: 'True probability p', min: 0.05, max: 0.95, step: 0.01, value: state.p, format: fmt(2),
    onInput: (v) => { state.p = v; changed(); } });
  const qS = slider(ctl, { name: 'Your report q', min: 0, max: 1, step: 0.01, value: state.q, format: fmt(2),
    onInput: (v) => { state.q = v; changed(); } });
  ctl.dataset.ready = '1';

  const c = chart(document.getElementById('one-chart'), {
    x: { domain: [0, 1], label: 'report q' }, y: { domain: [0, 2.5], label: 'expected loss' }, title: 'Expected loss against the report',
  });
  // Drag on the chart to set q.
  const setFromPointer = (ev) => {
    const [mx] = d3.pointer(ev, c.g.node());
    state.q = Math.round(Math.min(1, Math.max(0, c.x.invert(mx))) * 100) / 100;
    qS.set(state.q); changed();
  };
  c.svg.on('pointerdown', (ev) => { setFromPointer(ev); c.svg.on('pointermove', setFromPointer); })
    .on('pointerup pointerleave', () => c.svg.on('pointermove', null));

  const out = document.getElementById('one-readout');
  const draw = () => {
    const { rule, p, q } = state;
    c.setY([0, RULES[rule].yMax]);
    c.line('curve', curve(rule, p), { color: 'ink', label: `${RULES[rule].name} rule` });
    c.vline('p', p, { color: 'quiet', label: 'true p' });
    const [bq, bv] = argmin(rule, p);
    c.point('best', bq, bv, { color: 'quiet', r: 3, label: null });
    const v = expected(rule, p, q);
    c.point('q', q, v, { color: 'accent', r: 5 });
    const flat = Math.abs(expected(rule, p, 0) - expected(rule, p, 1)) < 1e-9 && rule === 'linear';
    out.innerHTML = flat ? `With \\(p\\) = 0.50 under the linear rule, every report gives the same expected loss, 0.500.` : `With \\(p\\) = ${p.toFixed(2)} and your report <span class="c-accent">\\(q\\) = ${q.toFixed(2)}</span>, your expected ${lc(rule)} loss is <b>${fmt(3)(v)}</b>. The lowest possible is ${bv.toFixed(3)}, at \\(q\\) = ${bq.toFixed(2)}.`;
    if (window.renderMathInElement) window.renderMathInElement(out, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false });
  };
  listeners.push(draw);
  draw();
})();

// ---- Panel 2: many forecasts ----
(function many() {
  const N = 1000;
  const ctl = document.getElementById('many-controls');
  const out = document.getElementById('many-readout');
  const c = chart(document.getElementById('many-chart'), {
    x: { domain: [1, N], label: 'number of events' }, y: { domain: [0, 1.5], label: 'average loss so far' }, title: 'Average loss of three forecasters',
  });
  let seed = 11, n = 0, outcomes = [], raf = null;
  const strategies = () => {
    const p = state.p, likelyYes = p >= 0.5;
    return [
      { id: 'honest', label: 'honest', q: p, color: 'accent' },
      { id: 'over', label: 'overconfident', q: likelyYes ? 0.95 : 0.05, color: 'second' },
      { id: 'under', label: 'underconfident', q: 0.5 + 0.4 * (p - 0.5), color: 'quiet' },
    ];
  };
  const draw = () => {
    c.setY([0, RULES[state.rule].manyMax]);
    const S = strategies();
    if (n === 0) {
      S.forEach((s) => c.remove(s.id));
      out.textContent = `Press Run. Same events for all three forecasters; rule: ${lc(state.rule)}, p = ${state.p.toFixed(2)}.`;
      return;
    }
    const finals = [];
    S.forEach((s, k) => {
      let sum = 0; const pts = [];
      for (let i = 0; i < n; i++) { sum += RULES[state.rule].loss(s.q, outcomes[i]); pts.push([i + 1, sum / (i + 1)]); }
      const step = Math.max(1, Math.floor(n / 300));
      c.line(s.id, pts.filter((_, i) => i % step === 0 || i === n - 1), { color: s.color, label: s.label });
      finals.push([s.label, sum / n, s.q]);
    });
    c.spreadLabels(S.map((s) => s.id));
    finals.sort((a, b) => a[1] - b[1]);
    out.innerHTML = `After ${n.toLocaleString()} events, average ${lc(state.rule)} loss: ` +
      finals.map(([l, v, q]) => `${l} (reports ${q.toFixed(2)}) <b>${v.toFixed(3)}</b>`).join(', ') + '.';
  };
  const run = () => {
    cancelAnimationFrame(raf);
    const r = rng(seed++);
    outcomes = Array.from({ length: N }, () => (r() < state.p ? 1 : 0));
    n = 0;
    if (reducedMotion()) { n = N; draw(); return; }
    const tick = () => { n = Math.min(N, n + 12); draw(); if (n < N) raf = requestAnimationFrame(tick); };
    tick();
  };
  button(ctl, 'Run 1,000 events', run, 'primary');
  button(ctl, 'Clear', () => { cancelAnimationFrame(raf); n = 0; draw(); });
  listeners.push(() => { cancelAnimationFrame(raf); n = 0; draw(); });
  draw();
})();

// ---- Panel 3: next tokens ----
(function tokens() {
  const tokens = ['Paris', 'a', 'the', 'located', 'also'];
  const truth = [0.62, 0.12, 0.10, 0.09, 0.07];
  let model = [0.2, 0.2, 0.2, 0.2, 0.2];
  const ctl = document.getElementById('tok-controls');
  const out = document.getElementById('tok-readout');
  const W = Math.max(280, Math.min(520, document.getElementById('tok-chart').clientWidth || 520)), H = 300, m = { top: 34, right: 16, bottom: 40, left: 52 };
  const svg = d3.select('#tok-chart').append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .attr('role', 'group').attr('aria-label', 'Model probabilities for five next tokens; each bar is adjustable');
  const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  const iw = W - m.left - m.right, ih = H - m.top - m.bottom;
  const x = d3.scaleBand().domain(tokens).range([0, iw]).padding(0.35);
  const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
  g.append('g').attr('class', 'axis').call(d3.axisLeft(y).ticks(4).tickSizeOuter(0));
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`).call(d3.axisBottom(x).tickSizeOuter(0));
  g.append('text').attr('x', -m.left + 4).attr('y', -16).text('probability');

  const setBar = (i, v) => {
    v = Math.min(1, Math.max(0, v));
    const rest = 1 - model[i];
    const newRest = 1 - v;
    model = model.map((q, j) => (j === i ? v : rest > 1e-9 ? (q * newRest) / rest : newRest / (tokens.length - 1)));
    draw();
  };
  const bars = g.selectAll('rect.model').data(tokens).join('rect').attr('class', 'model')
    .attr('x', (t) => x(t)).attr('width', x.bandwidth()).attr('fill', 'var(--accent)').attr('fill-opacity', 0.8)
    .attr('tabindex', 0).attr('role', 'slider').attr('aria-valuemin', 0).attr('aria-valuemax', 1)
    .attr('aria-label', (t) => `Model probability for ${t}`)
    .on('keydown', (ev, t) => {
      const i = tokens.indexOf(t), d = ev.shiftKey ? 0.1 : 0.01;
      if (ev.key === 'ArrowUp' || ev.key === 'ArrowRight') { setBar(i, model[i] + d); ev.preventDefault(); }
      if (ev.key === 'ArrowDown' || ev.key === 'ArrowLeft') { setBar(i, model[i] - d); ev.preventDefault(); }
    });
  const outlines = g.selectAll('rect.truth').data(tokens).join('rect').attr('class', 'truth')
    .attr('x', (t) => x(t) - 3).attr('width', x.bandwidth() + 6).attr('fill', 'none').attr('stroke', 'var(--ink)').attr('stroke-width', 1)
    .attr('y', (t, i) => y(truth[i])).attr('height', (t, i) => ih - y(truth[i])).attr('pointer-events', 'none');
  const vals = g.selectAll('text.val').data(tokens).join('text').attr('class', 'val label-accent').attr('text-anchor', 'middle');

  // Drag anywhere in a token's column to set its height.
  let dragging = null;
  const pick = (ev) => {
    const [mx, my] = d3.pointer(ev, g.node());
    if (dragging === null) dragging = tokens.findIndex((t) => mx >= x(t) - 6 && mx <= x(t) + x.bandwidth() + 6);
    if (dragging >= 0) setBar(dragging, y.invert(my));
  };
  svg.on('pointerdown', (ev) => { dragging = null; pick(ev); svg.on('pointermove', pick); })
    .on('pointerup pointerleave', () => { dragging = null; svg.on('pointermove', null); });

  button(ctl, 'Match the true probabilities', () => { model = truth.slice(); draw(); });
  button(ctl, 'Spread evenly', () => { model = tokens.map(() => 1 / tokens.length); draw(); });

  function draw() {
    bars.attr('y', (t, i) => y(model[i])).attr('height', (t, i) => ih - y(model[i])).attr('aria-valuenow', (t, i) => model[i].toFixed(2));
    vals.attr('x', (t) => x(t) + x.bandwidth() / 2).attr('y', (t, i) => y(Math.max(model[i], truth[i])) - 6).text((t, i) => model[i].toFixed(2));
    const H = -truth.reduce((s, p, i) => s + p * Math.log(model[i]), 0);
    const Hmin = -truth.reduce((s, p) => s + p * Math.log(p), 0);
    out.innerHTML = Number.isFinite(H)
      ? `Expected loss: <b>${H.toFixed(3)}</b>. The lowest possible, reached when the filled bars match the outlines, is ${Hmin.toFixed(3)}. The extra ${(H - Hmin).toFixed(3)} measures how far the model&rsquo;s probabilities are from the true ones; it is called the KL divergence.`
      : `Expected loss: <b>infinite</b>. A token that sometimes comes next has probability zero, and each time it occurs the loss is −log 0.`;
  }
  draw();
})();
