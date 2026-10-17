// Page 1.1: the timeline. Events from data/timeline.json, training compute from data/compute.json.
import { segmented, button } from '../lib/ui.js';
import { sections } from '../pages.js';
import { setPow } from '../lib/sci.js';

const d3 = window.d3;
const X0 = 1940, X1 = 2026;
const ROW = 27; // height of one label row, px
const ROWS = { ideas: 6, compute: 3, funding: 4 };
const COMPUTE_H = 160;

// "1958-07-08" -> fractional year; a year alone is placed at mid-year.
function toYear(s) {
  const [y, m, d] = s.split('-').map(Number);
  if (!m) return y + 0.5;
  const start = Date.UTC(y, 0, 1), t = Date.UTC(y, m - 1, d || 15);
  return y + (t - start) / (365.25 * 864e5);
}
function dateText(s) {
  const [y, m, d] = s.split('-').map(Number);
  if (!m) return String(y);
  const month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][m - 1];
  return d ? `${month} ${d}, ${y}` : `${month} ${y}`;
}
const pageIndex = new Map(sections.flatMap((s) => s.pages.map((p) => [p.id, p])));
function pageLink(id) {
  const p = pageIndex.get(id);
  if (!p) return '';
  return p.href ? `More on page <a href="../${p.href}">${id} ${p.title}</a>.` : `Page ${id}, ${p.title}, will explain it.`;
}

// Least-squares slope of log10(FLOP) on year.
function slope(pts) {
  const mx = d3.mean(pts, (p) => p.year), my = d3.mean(pts, (p) => p.lf);
  const b = d3.sum(pts, (p) => (p.year - mx) * (p.lf - my)) / d3.sum(pts, (p) => (p.year - mx) ** 2);
  return { b, mx, my };
}

Promise.all([fetch('../data/timeline.json').then((r) => r.json()), fetch('../data/compute.json').then((r) => r.json())])
  .then(([tl, comp]) => { staticFigure(tl); tool(tl, comp); })
  .catch((e) => { document.getElementById('card').textContent = 'The timeline data could not be loaded.'; console.error(e); });

// ---- Static figure: vertical axis, 12 milestones, winters shaded ----
const KEY = ['McCulloch and Pitts', 'Dartmouth workshop', 'Perceptron', 'Minsky and Papert', 'Backpropagation popularized', 'LeCun: zip codes', 'LSTM', 'AlexNet', 'Transformer', 'GPT-3', 'ChatGPT', 'o1'];
function staticFigure(tl) {
  const box = document.getElementById('fig-static');
  const ev = KEY.map((k) => tl.events.find((e) => e.label === k)).filter(Boolean);
  const W = Math.max(300, Math.min(560, box.clientWidth || 560)), H = 620;
  const m = { top: 14, bottom: 14, left: 48 };
  const y = d3.scaleLinear().domain([1940, 2026]).range([m.top, H - m.bottom]);
  const svg = d3.select(box).append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
    .attr('role', 'img').attr('aria-label', 'Timeline from 1940 to 2025: ' + ev.map((e) => `${e.date.slice(0, 4)} ${e.label}`).join(', ') + '; AI winters about 1974 to 1980 and 1987 to 1993.');
  tl.winters.forEach((w) => {
    svg.append('rect').attr('x', m.left).attr('width', W - m.left).attr('y', y(w.from)).attr('height', y(w.to) - y(w.from)).attr('fill', 'var(--faint)');
  });
  svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.left},0)`).call(d3.axisLeft(y).tickValues(d3.range(1940, 2030, 10)).tickFormat(String).tickSizeOuter(0));
  // Place labels top to bottom, pushing each down if it would overlap the one above.
  // A short label keeps each line inside a phone-width figure.
  const SHORT = { 'Backpropagation popularized': 'Backpropagation' };
  const boxes = [];
  let last = -Infinity;
  ev.forEach((e) => {
    const yy = y(toYear(e.date));
    const ly = Math.max(yy + 6, last + 22);
    last = ly;
    svg.append('circle').attr('cx', m.left).attr('cy', yy).attr('r', 4).attr('fill', 'var(--ink)');
    svg.append('line').attr('x1', m.left + 4).attr('x2', m.left + 14).attr('y1', yy).attr('y2', ly - 6).attr('stroke', 'var(--rule)');
    const t = svg.append('text').attr('x', m.left + 18).attr('y', ly).attr('class', 'label-ink').text(`${e.date.slice(0, 4)} ${SHORT[e.label] || e.label}`);
    boxes.push({ x0: m.left + 18, x1: m.left + 18 + t.node().getComputedTextLength(), y0: ly - 16, y1: ly + 4 });
  });
  // Winter labels go at the right edge, inside the band if no event label is in the way,
  // otherwise just below or above the band.
  tl.winters.forEach((w) => {
    const t = svg.append('text').attr('x', W - 4).attr('text-anchor', 'end').attr('font-style', 'italic').text(w.label);
    const wl = t.node().getComputedTextLength();
    const free = (base) => !boxes.some((b) => b.x1 > W - 4 - wl - 6 && b.y1 > base - 16 && b.y0 < base + 4);
    const cands = [y((w.from + w.to) / 2) + 6, y(w.to) + 20, y(w.from) - 6];
    const base = cands.find(free) ?? cands[0];
    t.attr('y', base);
    boxes.push({ x0: W - 4 - wl, x1: W - 4, y0: base - 16, y1: base + 4 });
  });
}

// ---- The tool ----
function tool(tl, comp) {
  const LANES = tl.lanes; // [[key, name]]
  const CATS = tl.categories;
  const events = tl.events.map((e, i) => ({ ...e, i, year: toYear(e.date) }));
  const winters = tl.winters.map((w) => ({ ...w, winter: true }));
  const cpts = comp.points.map((p) => ({ ...p, year: toYear(p.date), lf: Math.log10(p.flop) }));
  const state = { cats: new Set(CATS.map((c) => c[0])), compute: false, sel: null, prev: null, view: 'record' };

  const bar = document.getElementById('bar');
  const holder = document.getElementById('timeline');
  const card = document.getElementById('card');

  // Category filter chips.
  const chips = document.createElement('div');
  chips.className = 'ctl';
  chips.setAttribute('role', 'group');
  chips.setAttribute('aria-label', 'Kinds of event to show');
  chips.innerHTML = '<span class="ctl-head"><span class="ctl-name">Show</span></span>';
  const chipSeg = document.createElement('span');
  chipSeg.className = 'seg';
  chips.append(chipSeg);
  CATS.forEach(([k, name]) => {
    const b = button(chipSeg, name, () => {
      if (state.cats.has(k)) state.cats.delete(k); else state.cats.add(k);
      b.setAttribute('aria-pressed', String(state.cats.has(k)));
      render();
    });
    b.setAttribute('aria-pressed', 'true');
  });
  bar.append(chips);
  segmented(bar, { name: 'Training compute in the computing lane', options: [['off', 'Off'], ['on', 'On']], value: 'off',
    onChange: (k) => { state.compute = k === 'on'; build(); } });
  const zoomCtl = document.createElement('div');
  zoomCtl.className = 'ctl';
  zoomCtl.innerHTML = '<span class="ctl-head"><span class="ctl-name">Zoom</span></span>';
  const zs = document.createElement('span');
  zs.className = 'seg';
  zoomCtl.append(zs);
  bar.append(zoomCtl);
  const rangeCtl = document.createElement('div');
  rangeCtl.className = 'ctl';
  rangeCtl.innerHTML = '<span class="ctl-head"><span class="ctl-name">Show years</span></span>';
  const rs = document.createElement('span');
  rs.className = 'seg';
  rangeCtl.append(rs);
  bar.append(rangeCtl);
  const navCtl = document.createElement('div');
  navCtl.className = 'seg';
  bar.append(navCtl);

  // Rates for task 2, shown under the timeline when compute is on.
  const rates = document.createElement('p');
  rates.className = 'readout';
  holder.after(rates);
  const pre = cpts.filter((p) => p.year < 2010), post = cpts.filter((p) => p.year >= 2010);
  const rateText = () => {
    if (!state.compute) { rates.innerHTML = ''; return; }
    const f = (pts) => { const b = slope(pts).b; return `${(10 ** b).toFixed(1)} times per year (a doubling every ${(12 * Math.log10(2) / b).toFixed(0)} months)`; };
    rates.innerHTML = `Training compute, fitted by the dashed lines: ${pre.length} models from 1950 to 2009 grew about ${f(pre)}; ${post.length} notable models from 2012 to 2024 about ${f(post)}. Data from Epoch AI.`;
  };

  let svg, x0, x, zoom, W, gContent, laneTop = {}, H;
  let lastRange = null;
  const visible = () => events.filter((e) => state.cats.has(e.cat));

  // Measures label widths once, with the real font.
  const widths = new Map();
  function measure(svgSel) {
    const t = svgSel.append('text').attr('class', 'label-ink').attr('visibility', 'hidden');
    events.forEach((e) => { t.text(e.label); widths.set(e.label, t.node().getComputedTextLength()); });
    t.remove();
  }

  function build() {
    holder.innerHTML = '';
    W = Math.max(300, holder.clientWidth || 600);
    const top = 30;
    let yy = top;
    LANES.forEach(([k]) => {
      laneTop[k] = yy;
      yy += 28 + (k === 'compute' && state.compute ? COMPUTE_H : 0) + ROWS[k] * ROW + 10;
    });
    H = yy + 48;
    x0 = d3.scaleLinear().domain([X0, X1]).range([8, W - 8]);
    x = x0.copy();
    svg = d3.select(holder).append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`).attr('width', W).attr('height', H)
      .attr('tabindex', 0).attr('role', 'group').attr('aria-label', 'Timeline of AI from 1940 to 2025 in three lanes. Arrow keys move it; plus and minus zoom.');
    const clipId = 'tl-clip';
    svg.append('defs').append('clipPath').attr('id', clipId).append('rect').attr('x', 0).attr('y', 0).attr('width', W).attr('height', H);
    gContent = svg.append('g').attr('clip-path', `url(#${clipId})`);
    measure(svg);
    zoom = d3.zoom().scaleExtent([1, 24]).translateExtent([[x0(X0), 0], [x0(X1), H]]).extent([[0, 0], [W, H]])
      .filter((ev) => !ev.button && ev.type !== 'dblclick')
      .on('zoom', (ev) => { x = ev.transform.rescaleX(x0); render(); });
    svg.call(zoom);
    svg.on('keydown', (ev) => {
      const k = ev.key;
      if (k === 'ArrowLeft' || k === 'ArrowRight') { svg.call(zoom.translateBy, (k === 'ArrowLeft' ? 1 : -1) * W * 0.1 / d3.zoomTransform(svg.node()).k, 0); ev.preventDefault(); }
      if (k === '+' || k === '=') { svg.call(zoom.scaleBy, 1.5); ev.preventDefault(); }
      if (k === '-' || k === '_') { svg.call(zoom.scaleBy, 1 / 1.5); ev.preventDefault(); }
    });
    if (lastRange) showRange(lastRange[0], lastRange[1]); else render();
    rateText();
  }

  function render() {
    if (!svg) return;
    const g = gContent;
    g.selectAll('*').remove();
    // Winters, shaded across all lanes.
    winters.forEach((w, wi) => {
      const wg = g.append('g').attr('class', 'winter').attr('tabindex', 0).attr('role', 'button').attr('aria-label', `The ${w.label}, about ${w.from} to ${w.to}`);
      wg.append('rect').attr('x', x(w.from)).attr('width', x(w.to) - x(w.from)).attr('y', 0).attr('height', H).attr('fill', 'var(--faint)');
      wg.on('click', () => select(w)).on('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); select(w); } });
    });
    // Winter labels after all the shading, so neither band covers the other's label.
    winters.forEach((w, wi) => {
      g.append('text').attr('x', x(w.from) + 4).attr('y', H - 30 + 24 * wi).attr('font-style', 'italic').attr('pointer-events', 'none').text(w.label);
    });
    // Year axis.
    const span = x.domain()[1] - x.domain()[0];
    const step = W / span * 10 > 60 ? (W / span * 5 > 60 ? (W / span > 60 ? 1 : 5) : 10) : 20;
    g.append('g').attr('class', 'axis').attr('transform', 'translate(0,22)')
      .call(d3.axisTop(x).tickValues(d3.range(Math.ceil(x.domain()[0] / step) * step, x.domain()[1] + 0.01, step).filter((v) => x(v) > 20 && x(v) < W - 20)).tickFormat(String).tickSizeOuter(0));
    // Lanes.
    LANES.forEach(([k, name]) => {
      const t = laneTop[k];
      g.append('line').attr('x1', 0).attr('x2', W).attr('y1', t + 4).attr('y2', t + 4).attr('stroke', 'var(--rule)');
      g.append('text').attr('x', 4).attr('y', t + 24).attr('font-style', 'italic').text(name);
      let rowsTop = t + 32;
      if (k === 'compute' && state.compute) { drawCompute(g, t + 30); rowsTop += COMPUTE_H; }
      // Pack labels into rows, left to right.
      const ev = visible().filter((e) => e.lane === k).sort((a, b) => a.year - b.year);
      const ends = Array(ROWS[k]).fill(-Infinity);
      ev.forEach((e) => {
        const px = x(e.year), w = widths.get(e.label) + 12;
        // The label goes right of the dot, or left of it near the right edge.
        const left = px + w > W - 4;
        const s0 = left ? px - w : px - 8, e0 = left ? px + 8 : px + w;
        const fits = !left || s0 >= 0;
        const r = fits ? ends.findIndex((end) => end < s0 - 2) : -1;
        // Events with no room for a label sit as dots on the lane's top rule.
        const labeled = r >= 0;
        if (labeled) ends[r] = e0;
        const cy = labeled ? rowsTop + r * ROW + 8 : t + 4;
        const eg = g.append('g').attr('class', 'ev').attr('tabindex', 0).attr('role', 'button')
          .attr('aria-label', `${e.date.slice(0, 4)}: ${e.title}`).attr('aria-pressed', String(state.sel === e));
        const on = state.sel === e;
        eg.append('circle').attr('cx', px).attr('cy', cy).attr('r', on ? 6.5 : labeled ? 5 : 4)
          .attr('fill', on ? 'var(--accent)' : e.lore ? 'var(--paper)' : 'var(--ink)').attr('stroke', on ? 'var(--accent)' : 'var(--ink)').attr('stroke-width', 1.5);
        if (labeled) eg.append('text').attr('x', left ? px - 9 : px + 9).attr('y', cy + 6).attr('text-anchor', left ? 'end' : 'start')
          .attr('class', on ? 'label-accent' : 'label-ink').text(e.label);
        // A wider invisible target for touch.
        eg.append('rect').attr('x', labeled && left ? px - w : px - 10).attr('y', cy - 12).attr('width', labeled ? w + 10 : 20).attr('height', 24).attr('fill', 'transparent');
        eg.on('click', () => select(e)).on('keydown', (evk) => { if (evk.key === 'Enter' || evk.key === ' ') { evk.preventDefault(); select(e); } });
      });
    });
  }

  function drawCompute(g, top) {
    const y = d3.scaleLinear().domain([0, 27]).range([top + COMPUTE_H - 10, top + 40]);
    const ax = g.append('g').attr('class', 'axis').attr('transform', 'translate(4,0)')
      .call(d3.axisRight(y).tickValues([10, 20]).tickFormat(() => '').tickSize(0).tickPadding(0));
    ax.select('.domain').remove();
    ax.selectAll('.tick text').each(function (v) { setPow(d3.select(this), 10 ** v, { bare: true }); });
    g.append('text').attr('x', W - 4).attr('y', top + 22).attr('text-anchor', 'end').text('training compute, FLOP');
    g.selectAll('circle.cp').data(cpts).join('circle').attr('class', 'cp')
      .attr('cx', (p) => x(p.year)).attr('cy', (p) => y(p.lf)).attr('r', 2.6).attr('fill', 'var(--quiet)');
    const pre = cpts.filter((p) => p.year < 2010), post = cpts.filter((p) => p.year >= 2010);
    [[pre, 1950, 2010], [post, 2012, 2025]].forEach(([pts, a, b]) => {
      const s = slope(pts);
      const at = (yr) => s.my + s.b * (yr - s.mx);
      g.append('line').attr('x1', x(a)).attr('x2', x(b)).attr('y1', y(at(a))).attr('y2', y(at(b)))
        .attr('stroke', 'var(--ink)').attr('stroke-dasharray', '4 3');
    });
  }

  function select(e) {
    if (e !== state.sel) { state.prev = state.sel; state.sel = e; state.view = 'record'; }
    render();
    showCard();
    // Keep keyboard focus on the selected event.
    const node = [...holder.querySelectorAll('.ev')].find((n) => n.getAttribute('aria-pressed') === 'true');
    if (node && document.activeElement && (document.activeElement.closest && document.activeElement.closest('#timeline'))) node.focus();
  }

  function showCard() {
    const e = state.sel;
    if (!e) { card.innerHTML = '<p class="meta">No event selected.</p>'; return; }
    if (e.winter) {
      card.innerHTML = `<p class="meta">About ${e.from} to ${e.to}</p><h3>The ${e.label}</h3><p>${e.text}</p><p class="meta">${pageLink(e.page)} <a href="${e.source}">Source</a>.</p>`;
      return;
    }
    const lane = LANES.find((l) => l[0] === e.lane)[1], cat = CATS.find((c) => c[0] === e.cat)[1];
    let gap = '';
    if (state.prev && !state.prev.winter && state.prev !== e) {
      const dy = Number(e.date.slice(0, 4)) - Number(state.prev.date.slice(0, 4));
      const yrs = Math.abs(dy);
      gap = `<p class="meta">${yrs === 0 ? 'Same year as' : `${yrs} year${yrs === 1 ? '' : 's'} ${dy > 0 ? 'after' : 'before'}`} the card you opened before, ${state.prev.label} (${state.prev.date.slice(0, 4)}).</p>`;
    }
    const sources = `<a href="${e.source}">Source</a>${e.source2 ? `, <a href="${e.source2}">second source</a>` : ''}.`;
    card.innerHTML = `<p class="meta">${dateText(e.date)} &middot; ${lane} &middot; ${cat}</p><h3>${e.title}</h3>` +
      `<p><i>What happened.</i> ${e.what}</p><p><i>Why it mattered.</i> ${e.why}</p>` +
      (e.lore ? `<div class="lr" id="lr"></div><p id="lr-text"></p>` : '') +
      gap + `<p class="meta">${e.page ? pageLink(e.page) + ' ' : ''}${sources}</p>`;
    if (e.lore) {
      const lr = card.querySelector('#lr');
      const txt = card.querySelector('#lr-text');
      const set = () => { txt.innerHTML = state.view === 'lore' ? `<i>The lore.</i> ${e.lore}` : `<i>The record.</i> ${e.record}`; };
      segmented(lr, { name: 'This event has a well-known story', options: [['lore', 'Lore'], ['record', 'Record']], value: state.view,
        onChange: (k) => { state.view = k; set(); } });
      set();
    }
  }

  // Zoom buttons and year ranges.
  function showRange(a, b) {
    lastRange = [a, b];
    const k = (X1 - X0) / (b - a);
    const t = d3.zoomIdentity.translate(x0(X0) - k * x0(a), 0).scale(k);
    svg.call(zoom.transform, t);
  }
  button(zs, 'In', () => { lastRange = null; svg.call(zoom.scaleBy, 1.6); });
  button(zs, 'Out', () => { lastRange = null; svg.call(zoom.scaleBy, 1 / 1.6); });
  [[1940, 2026, 'All'], [1940, 1992, '1940–1992'], [1980, 2012, '1980–2012'], [2008, 2026, '2008–2025']].forEach(([a, b, label]) => {
    button(rs, label, () => showRange(a, b));
  });
  const stepSel = (d) => {
    const v = visible().sort((a, b) => a.year - b.year);
    if (!v.length) return;
    let i = state.sel && !state.sel.winter ? v.indexOf(state.sel) : -1;
    i = i < 0 ? (d > 0 ? 0 : v.length - 1) : (i + d + v.length) % v.length;
    const e = v[i];
    // Bring the event into view if it is outside the current range.
    const [a, b] = x.domain();
    if (e.year < a || e.year > b) { const half = (b - a) / 2; showRange(Math.max(X0, e.year - half), Math.min(X1, e.year + half)); }
    select(e);
  };
  button(navCtl, 'Previous event', () => stepSel(-1));
  button(navCtl, 'Next event', () => stepSel(1));

  build();
  showCard();
  let rt;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(build, 150); });
}
