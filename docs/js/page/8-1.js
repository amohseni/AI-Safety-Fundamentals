// Page 8.1: select on the proxy.
import { chart } from '../lib/chart.js';
import { slider, segmented, button, fmt } from '../lib/ui.js';
import { rng, gaussian } from '../lib/mlp.js';
import { selectionCurve, sampleWinner, errSample, trueValue, fitProxy, optimize } from '../lib/8-selection.js';

const d3 = window.d3;
const MIX = { rate: 0.001, big: 30 };
const errOf = (st) => (st.kind === 'normal' ? { kind: 'normal', s: st.s }
  : st.kind === 't' ? { kind: 't', s: st.s, df: st.df } : { kind: 'mix', s: st.s, ...MIX });
const kOf = (lk) => Math.round(10 ** lk);
const kText = (k) => k.toLocaleString('en-US');
// Labels for a log10 axis.
const logLabel = (v) => ({ 0: '1', 2: '100', 3: '1,000', 4: '10,000', 6: '1,000,000' }[Math.round(v)] ?? '');
// Numbers with a true minus sign and no negative zero.
const num = (d) => (v) => { const t = v.toFixed(d); return /^-0\.?0*$/.test(t) ? t.slice(1) : t.replace('-', '\u2212'); };

// ---- Panel 1: selecting on error ----
(function regressional() {
  const st = { kind: 'normal', df: 3, s: 1, lk: 3 };
  let curve = null, curveKey = '', drawSeed = 1, pending = null;
  const ctl = document.getElementById('reg-controls');
  const out = document.getElementById('reg-readout');

  segmented(ctl, { name: 'Error distribution', options: [['normal', 'Normal'], ['t', 'Student’s t'], ['mix', 'Rare huge errors']], value: st.kind,
    onChange: (k) => { st.kind = k; dfWrap.style.display = k === 't' ? '' : 'none'; update(); } });
  const dfS = slider(ctl, { name: 'Degrees of freedom', min: 1.5, max: 30, step: 0.5, value: st.df, format: fmt(1),
    onInput: (v) => { st.df = v; update(); } });
  const dfWrap = dfS.el.closest('.ctl');
  dfWrap.style.display = 'none';
  slider(ctl, { name: 'Error size s', min: 0.25, max: 3, step: 0.05, value: st.s, format: fmt(2),
    onInput: (v) => { st.s = v; update(); } });
  slider(ctl, { name: 'Selection: best of k', min: 0, max: 6, step: 0.05, value: st.lk, format: (v) => kText(kOf(v)),
    onInput: (v) => { st.lk = v; drawSel(); } });
  button(ctl, 'Draw the winner again', () => { drawSeed++; drawSel(); });

  const cc = chart(document.getElementById('reg-curve'), {
    height: 300, margin: { top: 34, right: 96, bottom: 52, left: 48 },
    x: { domain: [0, 6], ticks: 3, label: 'options compared, k' }, y: { domain: [-0.5, 8], label: 'expected value of the winner' },
    title: 'Expected proxy score and true value of the winner against the number of options compared',
  });
  cc.g.select('.axis').call((g) => g.selectAll('.tick').remove());
  const xTicks = () => { cc.g.select('.axis').call(d3.axisBottom(cc.x).tickValues(cc.iw < 330 ? [0, 3, 6] : [0, 2, 4, 6]).tickFormat(logLabel).tickSizeOuter(0)); };
  xTicks();

  const sc = chart(document.getElementById('reg-scatter'), {
    height: 300, margin: { top: 34, right: 24, bottom: 52, left: 48 },
    x: { domain: [-4, 6], ticks: 5, label: 'proxy score U' }, y: { domain: [-4, 4], ticks: 4, label: 'true value V' },
    title: 'Two thousand options and the winner of one best-of-k draw',
  });
  const dots = sc.g.append('g');
  const win = sc.g.append('circle').attr('class', 'winner').attr('r', 5);
  const winLabel = sc.g.append('text').attr('class', 'label-ink');

  let cloud = [];
  function makeCloud() {
    const r = rng(41), g = () => gaussian(r), err = errOf(st);
    cloud = Array.from({ length: 2000 }, () => { const v = g(); return [v + errSample(err, r, g), v]; });
  }

  function update() {
    const key = JSON.stringify(errOf(st));
    if (key === curveKey) { drawSel(); return; }
    // Student's t needs a numerical convolution (about 0.2 s); wait until the slider rests.
    clearTimeout(pending);
    const go = () => { curve = selectionCurve(errOf(st)); curveKey = key; makeCloud(); drawCurve(); drawSel(); };
    if (st.kind === 't') { out.textContent = 'Computing…'; pending = setTimeout(go, 120); } else go();
  }

  function drawCurve() {
    const ks = [...new Set(d3.range(0, 6.0001, 0.05).map(kOf))];
    const pts = ks.map((k) => [Math.log10(k), curve.at(k)]);
    const vMax = d3.max(pts, (p) => p[1].v), vMin = d3.min(pts, (p) => p[1].v);
    const top = Math.max(3, Math.ceil(Math.min(8, d3.max(pts, (p) => p[1].u)) + 0.5), Math.ceil(vMax * 1.3));
    cc.setY([Math.min(-0.5, Math.floor(vMin)), top]);
    xTicks();
    cc.line('u', pts.map(([x, e]) => [x, e.u]), { color: 'second', label: 'proxy' });
    cc.line('v', pts.map(([x, e]) => [x, e.v]), { color: 'accent', label: 'true value' });
    cc.spreadLabels(['u', 'v'], 20);
  }

  function drawSel() {
    if (!curve) return;
    const k = kOf(st.lk), e = curve.at(k);
    const lx = Math.log10(k);
    cc.vline('k', lx, { color: 'quiet' });
    cc.point('pu', lx, e.u, { color: 'second', r: 4 });
    cc.point('pv', lx, e.v, { color: 'accent', r: 4 });
    const r = rng(1000 + drawSeed);
    const wv = sampleWinner(errOf(st), curve, k, r, () => gaussian(r));
    // Scatter: x range covers the cloud and the winner.
    const xs = cloud.map((p) => p[0]);
    const lo = Math.min(-4, d3.quantile(xs.slice().sort(d3.ascending), 0.001)), hi = Math.max(6, d3.max(xs), wv.u);
    sc.setX([lo, hi * 1.04]);
    dots.selectAll('circle').data(cloud).join('circle').attr('class', 'dot').attr('r', 1.6)
      .attr('cx', (p) => sc.x(p[0])).attr('cy', (p) => sc.y(Math.max(-4, Math.min(4, p[1]))));
    const cx = sc.x(wv.u), cy = sc.y(Math.max(-4, Math.min(4, wv.v)));
    win.attr('cx', cx).attr('cy', cy).raise();
    const right = cx > sc.iw * 0.6;
    winLabel.attr('x', right ? cx - 9 : cx + 9).attr('y', cy - 9).attr('text-anchor', right ? 'end' : 'start').text(`best of ${kText(k)}`).raise();
    const err = errOf(st);
    let corr;
    if (err.kind === 'normal') corr = 1 / Math.sqrt(1 + err.s ** 2);
    else if (err.kind === 'mix') corr = 1 / Math.sqrt(1 + err.s ** 2 * (1 - MIX.rate + MIX.rate * MIX.big ** 2));
    else corr = err.df > 2 ? 1 / Math.sqrt(1 + (err.s ** 2 * err.df) / (err.df - 2)) : null;
    out.innerHTML = `Best of ${kText(k)}: expected <span class="c-second">proxy score ${num(2)(e.u)}</span>, expected <span class="c-accent">true value <b>${num(2)(e.v)}</b></span>. ` +
      `In this draw the winner has proxy score ${num(2)(wv.u)} and true value ${num(2)(wv.v)}. ` +
      (corr === null ? 'With 2 or fewer degrees of freedom the error has infinite variance, so the correlation between proxy and true value is undefined.'
        : `The correlation between proxy and true value is ${corr.toFixed(2)}.`);
  }
  update();
})();

// ---- Panel 2: pushing past the data ----
(function extremal() {
  const st = { range: 3, reach: 12 };
  const ctl = document.getElementById('ext-controls');
  const out = document.getElementById('ext-readout');
  slider(ctl, { name: 'End of the data used to fit the proxy', min: 1, max: 12, step: 0.5, value: st.range, format: fmt(1),
    onInput: (v) => { st.range = v; draw(); } });
  slider(ctl, { name: 'Search limit of the optimizer', min: 0, max: 16, step: 0.5, value: st.reach, format: fmt(1),
    onInput: (v) => { st.reach = v; draw(); } });
  const c = chart(document.getElementById('ext-chart'), {
    height: 340, margin: { top: 34, right: 70, bottom: 52, left: 48 },
    x: { domain: [0, 16], label: 'design x' }, y: { domain: [-16, 10], label: 'value' }, title: 'True value and fitted proxy against the design',
  });
  const data = c.g.append('g');
  const xs = d3.range(0, 16.001, 0.1);
  c.line('truth', xs.map((x) => [x, trueValue(x)]), { color: 'accent' });
  c.text('truth-label', 10.2, -5.6, 'true value', { color: 'accent', anchor: 'end' });
  function draw() {
    const p = fitProxy(st.range);
    c.line('proxy', xs.map((x) => [x, p.f(x)]), { color: 'second', label: 'proxy', labelAt: 'end' });
    data.selectAll('circle').data(p.xs.map((x, i) => [x, p.ys[i]])).join('circle').attr('class', 'datum').attr('r', 2.5)
      .attr('cx', (d) => c.x(d[0])).attr('cy', (d) => c.y(d[1]));
    c.vline('range', st.range, { color: 'quiet', label: 'end of data', labelY: 0 });
    c.vline('reach', st.reach, { color: 'ink', dash: null });
    c.text('reach-label', st.reach, -14.5, 'search limit ', { color: 'ink', anchor: st.reach > 3 ? 'end' : 'start' });
    const o = optimize(p, st.reach);
    c.point('pp', o.x, o.proxy, { color: 'second', r: 4 });
    c.point('pt', o.x, o.truth, { color: 'accent', r: 5 });
    // Best design inside the fitted range, for comparison.
    const bestIn = Math.min(4, st.range);
    out.innerHTML = `The optimizer picks design <b>${num(1)(o.x)}</b>, where the <span class="c-second">proxy score is ${num(1)(o.proxy)}</span> and the <span class="c-accent">true value is <b>${num(1)(o.truth)}</b></span>. ` +
      `The best design between 0 and ${st.range.toFixed(1)} is ${bestIn.toFixed(1)}, with true value ${num(1)(trueValue(bestIn))}; the best of all is 4, with true value 2.0.`;
  }
  draw();
})();

// ---- Static figure: top 1% by proxy, normal and heavy-tailed errors ----
(function figure() {
  const fig = document.getElementById('fig-tails');
  const r = rng(3), g = () => gaussian(r);
  for (const [name, err, ids] of [['Normal errors', { kind: 'normal', s: 1 }, ['cap-n-u', 'cap-n-v']], ['Student’s t errors, 3 degrees of freedom', { kind: 't', s: 1, df: 3 }, ['cap-t-u', 'cap-t-v']]]) {
    const pts = Array.from({ length: 10000 }, () => { const v = g(); return [v + errSample(err, r, g), v]; });
    const cut = pts.map((p) => p[0]).sort(d3.descending)[99];
    const top = pts.filter((p) => p[0] >= cut);
    document.getElementById(ids[0]).textContent = d3.mean(top, (p) => p[0]).toFixed(1);
    document.getElementById(ids[1]).textContent = d3.mean(top, (p) => p[1]).toFixed(1);
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${name}</p>`;
    fig.append(box);
    const xMax = Math.ceil(d3.max(pts, (p) => p[0]));
    const c = chart(box, { width: 340, height: 280, margin: { top: 34, right: 16, bottom: 56, left: 40 },
      x: { domain: [Math.floor(d3.min(pts, (p) => p[0])), xMax], ticks: 4, label: 'proxy score' }, y: { domain: [-4, 4], ticks: 4, label: 'true value' }, title: name });
    c.g.append('g').selectAll('circle').data(pts).join('circle').attr('class', (p) => (p[0] >= cut ? 'dot-top' : 'dot')).attr('r', (p) => (p[0] >= cut ? 2.2 : 1.2))
      .attr('cx', (p) => c.x(p[0])).attr('cy', (p) => c.y(Math.max(-4, Math.min(4, p[1]))));
  }
})();
