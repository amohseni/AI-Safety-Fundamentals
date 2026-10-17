// Page 1.5: compute over time (Epoch AI data, precomputed in data/compute.json).
import { chart } from '../lib/chart.js';
import { segmented, button } from '../lib/ui.js';
import { sciHTML, sciText, setPow } from '../lib/sci.js';

const d3 = window.d3;
const toYear = (s) => { const d = new Date(s + 'T00:00:00Z'); const y = d.getUTCFullYear(); return y + (d - Date.UTC(y, 0, 1)) / (365.25 * 864e5); };
const LABELED = new Set(['AlexNet', 'Transformer', 'GPT-2', 'GPT-3', 'GPT-4']);
const DOMAIN_NAMES = { language: 'language', vision: 'vision', games: 'games', other: 'other' };

// Least-squares line of log10(FLOP) on year.
export function fit(pts) {
  const n = pts.length, mx = d3.mean(pts, (p) => p.year), my = d3.mean(pts, (p) => Math.log10(p.flop));
  const b = d3.sum(pts, (p) => (p.year - mx) * (Math.log10(p.flop) - my)) / d3.sum(pts, (p) => (p.year - mx) ** 2);
  return { slope: b, at: (x) => 10 ** (my + b * (x - mx)), n };
}

const state = { scale: 'log', domain: 'all', trend: false, sel: null, prev: null };
let ALL = [];

function fmtParams(p) {
  if (p.parameters == null) return 'not published';
  const v = p.parameters;
  if (v >= 1e9) return `${Number((v / 1e9).toPrecision(3))} billion`;
  if (v >= 1e6) return `${Number((v / 1e6).toPrecision(3))} million`;
  return v.toLocaleString();
}
function ratioText(r) {
  if (r >= 100) return Math.round(r).toLocaleString();
  if (r >= 10) return r.toFixed(0);
  return r.toFixed(1);
}

// Draws the chart into `box`. Returns an object with the point selection for interaction.
function drawChart(box, { scale, pts, trend, interactive, sel }) {
  box.innerHTML = '';
  const log = scale === 'log';
  const yMax = d3.max(ALL, (p) => p.flop);
  const c = chart(box, {
    width: 640, height: 380, margin: { top: 44, right: 24, bottom: 52, left: 70 },
    x: { domain: [2012, 2025], ticks: (box.clientWidth || 640) < 440 ? 3 : 5, format: 'd', label: 'publication year' },
    y: log ? { domain: [1e15, 1e27], log: true } : { domain: [0, yMax * 1.08] },
    title: `Training compute of ${pts.length} AI models by publication date, ${log ? 'log' : 'linear'} scale`,
  });
  // Tick labels as powers of ten (log) or multiples of 10^25 (linear).
  const yAxisG = c.g.selectAll('.axis').filter((_, i) => i === 1);
  if (log) {
    yAxisG.call(d3.axisLeft(c.y).tickValues(d3.range(15, 28, 3).map((k) => 10 ** k)).tickFormat(() => '').tickSizeOuter(0));
    yAxisG.selectAll('.tick text').each(function (v) { setPow(d3.select(this), v, { bare: true }); });
  } else {
    yAxisG.call(d3.axisLeft(c.y).ticks(4).tickFormat(() => '').tickSizeOuter(0));
    yAxisG.selectAll('.tick text').each(function (v) { if (v === 0) d3.select(this).text('0'); else setPow(d3.select(this), v, { digits: 0 }); });
  }
  c.g.append('text').attr('x', -66).attr('y', -28).text('training compute (FLOP)');
  let f = null;
  if (trend && pts.length >= 6) {
    f = fit(pts);
    const xs = d3.range(2012, 2025.01, 0.1);
    c.line('trend', xs.map((x) => [x, f.at(x)]), { color: 'ink', width: 1, dash: '4 3' });
  }
  const g = c.g.append('g');
  const circles = g.selectAll('circle').data(pts, (p) => p.name).join('circle')
    .attr('class', interactive ? 'pt' : null)
    .attr('cx', (p) => c.x(p.year)).attr('cy', (p) => c.y(Math.max(p.flop, c.y.domain()[0])))
    .attr('r', (p) => (sel && p.name === sel.name ? 6.5 : 4.5))
    .attr('fill', (p) => (sel && p.name === sel.name ? 'var(--accent)' : 'var(--quiet)'))
    .attr('fill-opacity', (p) => (sel && p.name === sel.name ? 1 : 0.75));
  if (interactive) {
    circles.attr('tabindex', 0).attr('role', 'button').attr('aria-label', (p) => `${p.label}, ${p.date.slice(0, 4)}`);
  }
  // Labels on a few landmark points, and on the selected point.
  const labelSet = pts.filter((p) => LABELED.has(p.label) || (sel && p.name === sel.name));
  g.selectAll('text').data(labelSet).join('text')
    .attr('class', (p) => (sel && p.name === sel.name ? 'label-accent' : 'label-ink'))
    .attr('x', (p) => c.x(p.year) + (p.year > 2018 ? -9 : 9)).attr('y', (p) => c.y(Math.max(p.flop, c.y.domain()[0])) + (p.year > 2018 ? -6 : 16))
    .attr('text-anchor', (p) => (p.year > 2018 ? 'end' : 'start'))
    .text((p) => p.label);
  return { c, circles, f };
}

function figure() {
  const box = document.getElementById('fig-compute');
  const pts = ALL;
  drawChart(box, { scale: 'log', pts, trend: true, interactive: false, sel: null });
}

function tool() {
  const ctl = document.getElementById('controls');
  const stage = document.getElementById('chart');
  const out = document.getElementById('readout');
  const trendOut = document.getElementById('trend');
  const visible = () => ALL.filter((p) => state.domain === 'all' || p.domain === state.domain);

  segmented(ctl, { name: 'Vertical axis', options: [['log', 'Log'], ['linear', 'Linear']], value: state.scale,
    onChange: (k) => { state.scale = k; draw(); } });
  segmented(ctl, { name: 'Domain', options: [['all', 'All'], ['language', 'Language'], ['vision', 'Vision'], ['games', 'Games']], value: state.domain,
    onChange: (k) => { state.domain = k; if (state.sel && !visible().includes(state.sel)) { state.sel = null; } draw(); } });
  segmented(ctl, { name: 'Fitted trend', options: [['off', 'Off'], ['on', 'On']], value: 'off',
    onChange: (k) => { state.trend = k === 'on'; draw(); } });
  const nav = document.createElement('div');
  nav.className = 'seg';
  ctl.append(nav);
  const step = (d) => {
    const v = visible();
    const i = state.sel ? v.indexOf(state.sel) : -1;
    select(v[(i + d + v.length) % v.length] || null);
  };
  button(nav, 'Previous model', () => step(-1));
  button(nav, 'Next model', () => step(1));

  let refocus = false;
  function select(p) {
    if (p && p !== state.sel) { state.prev = state.sel; state.sel = p; }
    draw();
  }

  function draw() {
    const pts = visible();
    const { circles, f } = drawChart(stage, { scale: state.scale, pts, trend: state.trend, interactive: true, sel: state.sel });
    circles.on('click', (ev, p) => select(p))
      .on('keydown', (ev, p) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); refocus = true; select(p); } });
    // Keep keyboard focus on the selected point after a redraw.
    if (refocus && state.sel) { refocus = false; circles.filter((q) => q.name === state.sel.name).node()?.focus(); }

    if (!state.trend) trendOut.innerHTML = 'Fitted trend: off.';
    else if (!f) trendOut.innerHTML = `Only ${pts.length} models in this domain, too few to fit a trend.`;
    else {
      const perYear = 10 ** f.slope, months = 12 * Math.log10(2) / f.slope;
      trendOut.innerHTML = `Fitted trend for the ${pts.length} models shown: training compute grows about <b>${perYear.toFixed(1)} times per year</b>, a doubling about every ${months.toFixed(1)} months.`;
    }
    const p = state.sel;
    if (!p) { out.innerHTML = 'No model selected. Tap a point, or use Previous model and Next model.'; return; }
    let cmp = '';
    if (state.prev && state.prev !== p) {
      const r = p.flop / state.prev.flop;
      cmp = ` That is ${r >= 1 ? `<b>${ratioText(r)} times</b>` : `<b>1/${ratioText(1 / r)}</b> of`} the training compute of ${state.prev.label}, the model you selected before.`;
    }
    out.innerHTML = `<span class="c-accent">${p.label}</span> (${DOMAIN_NAMES[p.domain]}), published ${p.date}. Parameters: ${fmtParams(p)}. Training compute: <b>${sciHTML(p.flop, { digits: 2 })}</b> FLOP.${cmp}` +
      `${p.note ? ' ' + p.note : ''}${p.parametersNote ? ' ' + p.parametersNote : ''} <a href="${p.source}">Source</a>.`;
    stage.querySelector('svg').setAttribute('aria-label', `Training compute by year; selected: ${p.label}, ${sciText(p.flop, 2)} FLOP`);
  }
  draw();
  let rt;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
}

fetch('../data/compute.json').then((r) => r.json()).then((data) => {
  ALL = data.points.filter((p) => p.selected).map((p) => ({ ...p, year: toYear(p.date) }));
  figure();
  tool();
}).catch((e) => {
  document.getElementById('readout').textContent = 'The data file could not be loaded.';
  console.error(e);
});
