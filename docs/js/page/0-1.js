// Page 0.1: the site map.
import { sections } from '../pages.js';
import { store } from '../site.js';
import { segmented, button } from '../lib/ui.js';
import { shortestPath, prerequisites, usedBy } from '../lib/s7-graph.js';

const NS = 'http://www.w3.org/2000/svg';
const svgEl = (parent, tag, attrs = {}, text) => {
  const e = document.createElementNS(NS, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
  if (text !== undefined) e.textContent = text;
  parent.append(e);
  return e;
};
const SHORT = { 0: 'Start here', 1: 'History', 2: 'Networks', 3: 'Transformers', 4: 'Training', 5: 'What networks can do', 6: 'Interpretability', 7: 'Beliefs and goals', 8: 'AI safety', 9: 'Open problems' };
const EXSHORT = { 'The plane classifier': 'Plane classifier', 'The modular-addition transformer': 'Modular addition', 'GPT-2 small': 'GPT-2 small', 'The gridworld': 'Gridworld' };

const pages = sections.flatMap((s) => s.pages.map((p) => ({ ...p, n: s.n, sectionTitle: s.title })));
const byId = new Map(pages.map((p) => [p.id, p]));
const done = (id) => { try { return !!store.get(`done:${id}`, false); } catch { return false; } };

let data;
try {
  const res = await fetch('../data/sitemap.json');
  data = await res.json();
} catch {
  document.getElementById('map-stage').textContent = 'The site map data could not be loaded.';
}
if (data) { build(); threads(); }

function build() {
  const q = new Map(data.nodes.map((n) => [n.id, n]));
  const edges = data.edges;
  const state = { sel: null, path: null, example: 'none', anc: false };

  // ---- controls ----
  const ctl = document.getElementById('controls');
  const pair = document.createElement('div');
  pair.className = 'ctl';
  pair.innerHTML = '<span class="ctl-head"><span class="ctl-name">Shortest path</span></span>';
  const row = document.createElement('div'); row.className = 'pair';
  const mkSelect = (label, value) => {
    const l = document.createElement('label'); l.textContent = label;
    const s = document.createElement('select');
    pages.filter((p) => p.id !== '0.1').forEach((p) => { const o = document.createElement('option'); o.value = p.id; o.textContent = p.id; s.append(o); });
    s.value = value; l.append(s); row.append(l); return s;
  };
  const from = mkSelect('From', '2.5'), to = mkSelect('To', '8.1');
  pair.append(row);
  ctl.append(pair);
  const pathRow = document.createElement('div'); pathRow.className = 'seg'; ctl.append(pathRow);
  button(pathRow, 'Show the path', () => { state.path = shortestPath(edges, from.value, to.value) || []; state.anc = false; draw(); }, 'primary');
  button(pathRow, 'Clear', () => { state.path = null; state.sel = null; state.anc = false; draw(); });
  segmented(ctl, { name: 'Running example', options: [['none', 'None'], ...data.examples.map((e, i) => [String(i), EXSHORT[e.name] || e.name])], value: 'none', onChange: (k) => { state.example = k; draw(); } });
  const ancBtn = button(ctl, 'Show everything it depends on', () => { if (state.sel) { state.anc = !state.anc; draw(); } });

  // ---- layout ----
  const stage = document.getElementById('map-stage');
  const W = Math.max(300, Math.min(820, stage.clientWidth || 600));
  const wide = W >= 640;
  const labelW = wide ? 250 : 0;
  const maxN = Math.max(...sections.map((s) => s.pages.length));
  const gap = Math.min(62, (W - labelW - 40) / (maxN - 1 || 1));
  const rowH = wide ? 62 : 86;
  const top = wide ? 20 : 34;
  const pos = new Map();
  sections.forEach((s, r) => s.pages.forEach((p, k) => pos.set(p.id, [labelW + 20 + k * gap, top + r * rowH])));
  const H = top + sections.length * rowH;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'map'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.setAttribute('height', H);
  svg.setAttribute('role', 'group'); svg.setAttribute('aria-label', 'Site map: pages by section, with lines for the ideas each page uses');
  stage.append(svg);
  const edgeLayer = svgEl(svg, 'g');
  const edgeEls = edges.map((e) => {
    const [x1, y1] = pos.get(e.from), [x2, y2] = pos.get(e.to);
    const dy = (y2 - y1) * 0.5;
    const d = y1 === y2 ? `M${x1},${y1} Q${(x1 + x2) / 2},${y1 - 26} ${x2},${y2}` : `M${x1},${y1} C${x1},${y1 + dy} ${x2},${y2 - dy} ${x2},${y2}`;
    return { e, el: svgEl(edgeLayer, 'path', { class: 'edge', d }) };
  });
  sections.forEach((s, r) => {
    const y = top + r * rowH;
    if (wide) svgEl(svg, 'text', { x: 0, y: y + 6, class: 'quiet' }, `${s.n} ${SHORT[s.n]}`);
    else svgEl(svg, 'text', { x: 6, y: y - 16, class: 'quiet' }, `${s.n} ${SHORT[s.n]}`);
  });
  const nodeEls = new Map();
  pages.forEach((p) => {
    const [x, y] = pos.get(p.id);
    const g = svgEl(svg, 'g', { class: 'node', tabindex: '-1', role: 'button', transform: `translate(${x},${y})` });
    g.setAttribute('aria-label', `${p.id} ${p.title}${p.href ? '' : ', not written yet'}`);
    svgEl(g, 'circle', { class: 'ring', r: 13 });
    svgEl(g, 'circle', { class: 'dot', r: 7 });
    svgEl(g, 'text', { x: 0, y: wide ? 30 : 30, 'text-anchor': 'middle' }, p.id);
    svgEl(g, 'rect', { x: -gap / 2, y: -16, width: gap, height: 50, fill: 'transparent' });
    g.addEventListener('click', () => select(p.id));
    g.addEventListener('keydown', (ev) => key(ev, p.id));
    nodeEls.set(p.id, g);
  });
  let focusId = '0.1';
  nodeEls.get(focusId).setAttribute('tabindex', '0');
  function focusNode(id) { nodeEls.get(focusId).setAttribute('tabindex', '-1'); focusId = id; const g = nodeEls.get(id); g.setAttribute('tabindex', '0'); g.focus(); }
  function key(ev, id) {
    const p = byId.get(id), s = sections.findIndex((x) => x.n === p.n), k = sections[s].pages.findIndex((x) => x.id === id);
    let next = null;
    if (ev.key === 'ArrowRight') next = sections[s].pages[k + 1];
    if (ev.key === 'ArrowLeft') next = sections[s].pages[k - 1];
    if (ev.key === 'ArrowDown' && sections[s + 1]) next = sections[s + 1].pages[Math.min(k, sections[s + 1].pages.length - 1)];
    if (ev.key === 'ArrowUp' && sections[s - 1]) next = sections[s - 1].pages[Math.min(k, sections[s - 1].pages.length - 1)];
    if (next) { ev.preventDefault(); focusNode(next.id); return; }
    if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); select(id); }
  }
  function select(id) { state.sel = id; state.anc = false; if (focusId !== id) { nodeEls.get(focusId).setAttribute('tabindex', '-1'); focusId = id; nodeEls.get(id).setAttribute('tabindex', '0'); } draw(); }

  const ancestors = (id) => { const seen = new Set(); const walk = (v) => prerequisites(edges, v).forEach((u) => { if (!seen.has(u)) { seen.add(u); walk(u); } }); walk(id); return seen; };
  const out = document.getElementById('readout');
  const link = (id) => { const p = byId.get(id); return p.href ? `<a href="../${p.href}">${id}</a>` : id; };

  function draw() {
    const sel = state.sel, pre = sel ? new Set(prerequisites(edges, sel)) : new Set(), post = sel ? new Set(usedBy(edges, sel)) : new Set();
    const anc = sel && state.anc ? ancestors(sel) : new Set();
    const ex = state.example === 'none' ? new Set() : new Set(data.examples[Number(state.example)].pages);
    const path = state.path || [];
    const onPathEdge = new Set(path.slice(1).map((v, i) => `${path[i]}>${v}`));
    edgeEls.forEach(({ e, el }) => {
      let cls = 'edge';
      if (onPathEdge.has(`${e.from}>${e.to}`)) cls += ' path';
      else if (sel && e.to === sel) cls += ' pre';
      else if (sel && e.from === sel) cls += ' post';
      else if (sel || path.length) cls += ' dim';
      el.setAttribute('class', cls);
    });
    // draw highlighted edges last so they sit on top
    edgeEls.filter(({ el }) => /pre|post|path/.test(el.getAttribute('class'))).forEach(({ el }) => edgeLayer.append(el));
    pages.forEach((p) => {
      const cls = ['node'];
      if (!p.href) cls.push('planned');
      if (done(p.id)) cls.push('done');
      if (ex.has(p.id)) cls.push('ex');
      if (anc.has(p.id)) cls.push('anc');
      if (p.id === sel) cls.push('sel');
      if (path.includes(p.id)) cls.push('onpath');
      if (pre.has(p.id)) cls.push('pre');
      if (post.has(p.id)) cls.push('post');
      nodeEls.get(p.id).setAttribute('class', cls.join(' '));
    });
    ancBtn.disabled = !sel;
    ancBtn.textContent = state.anc ? 'Hide what it depends on' : 'Show everything it depends on';
    const parts = [];
    if (state.path) parts.push(path.length ? `Shortest path from ${from.value} to ${to.value}: <b>${path.join(' → ')}</b>, with ${path.length - 2} page${path.length === 3 ? '' : 's'} in between.` : `No chain of pages leads from ${from.value} to ${to.value}.`);
    if (state.example !== 'none') { const e = data.examples[Number(state.example)]; parts.push(`${e.name}: ${e.what} Used on ${e.pages.join(', ')}.`); }
    if (sel) {
      const p = byId.get(sel), n = q.get(sel);
      parts.push(`<b>${sel} ${p.title}</b>${p.core ? ' (core path)' : ''}. ${n.question}` +
        (n.takeaway ? ` <em>Takeaway:</em> ${n.takeaway}` : '') +
        ` <span class="c-accent">Builds on:</span> ${pre.size ? [...pre].sort((a, b) => parseFloat(a) - parseFloat(b)).map(link).join(', ') : 'no earlier page'}.` +
        ` <span class="c-second">Built on by:</span> ${post.size ? [...post].sort((a, b) => parseFloat(a) - parseFloat(b)).map(link).join(', ') : 'no later page'}.` +
        (state.anc ? ` It depends, directly or through other pages, on ${anc.size} page${anc.size === 1 ? '' : 's'}: ${[...anc].sort((a, b) => parseFloat(a) - parseFloat(b)).join(', ')}.` : '') +
        (p.href ? ` <a href="../${p.href}">Open page ${sel}</a>.` : ' Not written yet.'));
    }
    if (!parts.length) parts.push('Select a page to see its question and links.');
    out.innerHTML = parts.join(' ');
  }
  draw();
}

// ---- Static figure: running examples as threads through the sections ----
function threads() {
  const fig = document.getElementById('fig-threads');
  const W = Math.max(300, Math.min(620, fig.clientWidth || 620));
  const ex = data.examples;
  const colW = Math.min(90, (W - 170) / ex.length);
  const x0 = W - colW * ex.length + colW / 2;
  const headH = 190, rowH = 34;
  const secs = sections.filter((s) => s.n >= 1);
  const H = headH + secs.length * rowH + 10;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'threads'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('width', W); svg.setAttribute('height', H);
  svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', 'Which sections use each running example');
  secs.forEach((s, r) => {
    const y = headH + r * rowH + rowH / 2;
    svgEl(svg, 'text', { x: 0, y: y + 6, class: 'quiet' }, `${s.n} ${SHORT[s.n]}`);
  });
  ex.forEach((e, i) => {
    const x = x0 + i * colW;
    svgEl(svg, 'text', { x: 0, y: 0, transform: `translate(${x + 6},${headH - 10}) rotate(-90)`, 'text-anchor': 'start' }, EXSHORT[e.name] || e.name);
    const rows = secs.map((s, r) => (e.pages.some((id) => Number(id.split('.')[0]) === s.n) ? r : null)).filter((r) => r !== null);
    const yOf = (r) => headH + r * rowH + rowH / 2;
    for (let k = 1; k < rows.length; k++) {
      const gapRows = rows[k] - rows[k - 1] > 1;
      svgEl(svg, 'line', { class: gapRows ? 'thread gap' : 'thread', x1: x, x2: x, y1: yOf(rows[k - 1]), y2: yOf(rows[k]) });
    }
    rows.forEach((r) => svgEl(svg, 'circle', { cx: x, cy: yOf(r), r: 5 }));
  });
  fig.append(svg);
}
