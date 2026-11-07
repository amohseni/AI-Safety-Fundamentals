// Page 6.3: train a probe.
import { chart } from '../lib/chart.js';
import { slider, segmented, button } from '../lib/ui.js';
import { decodeProbeData, split, PROPERTIES } from '../lib/s6l-tasks.js';
import { logistic, mlp, accuracy } from '../lib/s6l-probe.js';
import { PROBE_DEFAULTS } from '../lib/s6l-probe-settings.js';

const [dataJson, ablJson] = await Promise.all([
  fetch(new URL('../../data/6-3/probe-data.json', import.meta.url)).then((r) => r.json()),
  fetch(new URL('../../data/6-3/ablation.json', import.meta.url)).then((r) => r.json()),
]);
const D = decodeProbeData(dataJson);
const pct = (v) => `${Math.round(v * 100)}%`;
const L2 = [0, 0.0001, 0.001, 0.01, 0.1, 1];
const state = { prop: 'truth', layer: 12, kind: 'linear', l2ix: 3 };
const results = {}; // key -> { L: { tr, te, ctr, cte } }
const key = () => `${state.prop}|${state.kind}|${state.l2ix}`;
const settings = () => (state.kind === 'linear'
  ? { ...PROBE_DEFAULTS.linear, l2: L2[state.l2ix] }
  : { ...PROBE_DEFAULTS.mlp, l2: L2[state.l2ix] });

function train(prop, L, kind, opts) {
  const s = split(D.tasks[prop], D.acts(prop, L));
  const fit = kind === 'linear' ? logistic : mlp;
  const p = fit(s.Xtr, s.ytr, opts), c = fit(s.Xtr, s.yctr, opts);
  return { tr: accuracy(p, s.Xtr, s.ytr), te: accuracy(p, s.Xte, s.yte), ctr: accuracy(c, s.Xtr, s.yctr), cte: accuracy(c, s.Xte, s.ycte) };
}

// ---- Controls ----
const ctl = document.getElementById('controls');
const out = document.getElementById('readout');
const ablOut = document.getElementById('ablation');
segmented(ctl, { name: 'Property', options: Object.entries(PROPERTIES), value: state.prop, onChange: (k) => { state.prop = k; changed(); } });
slider(ctl, { name: 'Layer', min: 0, max: 24, step: 2, value: state.layer, format: (v) => `${v} of 24`, onInput: (v) => { state.layer = v; changed(); } });
segmented(ctl, { name: 'Probe', options: [['linear', 'Linear'], ['mlp', 'Two-layer']], value: state.kind,
  onChange: (k) => { state.kind = k; state.l2ix = L2.indexOf(PROBE_DEFAULTS[k].l2); regS.set(state.l2ix); changed(); } });
const regS = slider(ctl, { name: 'Regularization strength', min: 0, max: L2.length - 1, step: 1, value: state.l2ix, format: (i) => String(L2[i]), onInput: (i) => { state.l2ix = i; changed(); } });
const row = document.createElement('div'); row.className = 'seg'; ctl.append(row);
const allBtn = button(row, 'Train at every layer', () => trainAll());
const ablBtn = button(ctl, "Remove the probe's direction", () => showAblation());

// ---- Accuracy by layer chart ----
const c = chart(document.getElementById('by-layer'), {
  x: { domain: [0, 24], label: 'layer', ticks: 6 }, y: { domain: [0.4, 1], label: 'test accuracy', ticks: 4, format: '.0%' },
  title: 'Test accuracy of the probe and of the control probe at each layer', margin: { top: 34, right: 96, bottom: 52, left: 60 },
});

function drawChart() {
  const r = results[key()] || {};
  const Ls = D.layers.filter((L) => r[L]);
  c.line('real', Ls.map((L) => [L, r[L].te]), { color: 'accent', label: Ls.length ? 'property' : null });
  c.line('ctl', Ls.map((L) => [L, r[L].cte]), { color: 'second', dash: '5 4', label: Ls.length ? 'control' : null });
  Ls.forEach((L) => { c.point(`p${L}`, L, r[L].te, { color: 'accent', r: 3 }); c.point(`c${L}`, L, r[L].cte, { color: 'second', r: 3 }); });
  D.layers.filter((L) => !r[L]).forEach((L) => { c.remove(`p${L}`); c.remove(`c${L}`); });
  c.vline('now', state.layer, { color: 'quiet' });
  c.spreadLabels(['real', 'ctl'], 18);
}

let busy = 0;
function changed() {
  ablOut.innerHTML = '';
  ablBtn.disabled = state.kind !== 'linear';
  const k = key(), L = state.layer;
  results[k] = results[k] || {};
  if (!results[k][L]) {
    out.innerHTML = 'Training&hellip;';
    const ticket = ++busy;
    setTimeout(() => {
      if (ticket !== busy) return;
      results[k][L] = train(state.prop, L, state.kind, settings());
      report();
    }, 30);
  } else report();
  drawChart();
}

function report() {
  const r = results[key()][state.layer];
  if (!r) return;
  out.innerHTML = `${PROPERTIES[state.prop]}, layer ${state.layer}, ${state.kind === 'linear' ? 'linear' : 'two-layer'} probe. ` +
    `On the property: training accuracy ${pct(r.tr)}, <span class="c-accent">test accuracy <b>${pct(r.te)}</b></span>. ` +
    `On the control task: training ${pct(r.ctr)}, <span class="c-second">test <b>${pct(r.cte)}</b></span>. ` +
    `Selectivity, the difference between the two test accuracies: <b>${Math.round((r.te - r.cte) * 100)} points</b>.`;
  drawChart();
}

function trainAll() {
  const k = key(), opts = settings(), prop = state.prop, kind = state.kind;
  results[k] = results[k] || {};
  const todo = D.layers.filter((L) => !results[k][L]);
  allBtn.disabled = true;
  const step = () => {
    if (!todo.length || key() !== k) { allBtn.disabled = false; drawChart(); return; }
    const L = todo.shift();
    results[k][L] = train(prop, L, kind, opts);
    out.innerHTML = `Trained layer ${L}; ${todo.length} to go.`;
    drawChart();
    setTimeout(step, 0);
  };
  step();
  const finish = setInterval(() => { if (!todo.length) { clearInterval(finish); if (key() === k) report(); } }, 200);
}

// ---- Ablation (computed in advance with the model) ----
function showAblation() {
  const L = state.layer, b = ablJson.base, p = ablJson.probe[L], r = ablJson.random[L];
  if (!p || p[state.prop] === null) {
    ablOut.innerHTML = 'Removal was computed for layers 2 to 24, after a transformer layer; choose one of those.';
    return;
  }
  const f = (v) => v.toFixed(3);
  if (state.prop === 'truth') {
    ablOut.innerHTML = `Asked whether each of 60 held-out city statements is true, the model answers correctly ${pct(b.judge[0])} of the time. ` +
      `With the default linear probe&rsquo;s direction removed after layer ${L}: <b>${pct(p.truth[0])}</b>. With a random direction removed instead: ${pct(r.truth[0])}.`;
  } else {
    const j = state.prop === 'noun' ? 1 : 2;
    const what = state.prop === 'noun' ? 'when the next word is a noun' : 'when the next token is a closing quotation mark';
    ablOut.innerHTML = `The model&rsquo;s average next-token loss on the held-out passages ${what} is ${f(b.loss[j])} (over all tokens, ${f(b.loss[0])}). ` +
      `With the default linear probe&rsquo;s direction removed after layer ${L}: <b>${f(p[state.prop][j])}</b> (all tokens ${f(p[state.prop][0])}). ` +
      `With a random direction removed instead: ${f(r[state.prop][j])} (all tokens ${f(r[state.prop][0])}).`;
  }
}

changed();

// ---- Static figure: accuracy by layer for the three properties, linear probes, default settings ----
(function figure() {
  const fig = document.getElementById('fig-layers');
  const boxes = Object.entries(PROPERTIES).map(([prop, name]) => {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${name}</p>`;
    fig.append(box);
    return [prop, box];
  });
  let i = 0;
  const next = () => {
    if (i >= boxes.length) return;
    const [prop, box] = boxes[i++];
    const pts = D.layers.map((L) => [L, train(prop, L, 'linear', PROBE_DEFAULTS.linear)]);
    const cc = chart(box, { width: 300, height: 220, margin: { top: 16, right: 88, bottom: 40, left: 52 },
      x: { domain: [0, 24], ticks: 3 }, y: { domain: [0.4, 1], ticks: 3, format: '.0%' }, title: `${PROPERTIES[prop]}: test accuracy by layer` });
    cc.line('r', pts.map(([L, r]) => [L, r.te]), { color: 'accent', label: 'property' });
    cc.line('c', pts.map(([L, r]) => [L, r.cte]), { color: 'second', dash: '5 4', label: 'control' });
    cc.spreadLabels(['r', 'c'], 18);
    setTimeout(next, 0);
  };
  setTimeout(next, 200);
})();
