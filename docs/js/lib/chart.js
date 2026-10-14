// Chart kit: quiet axes, no frame or grid, labels placed on the lines.
// Uses the global d3 loaded from vendor/d3.min.js.
// Colors are named by role: 'accent', 'second', 'ink', 'quiet'.
const d3 = window.d3;
const COLOR = { accent: 'var(--accent)', second: 'var(--second)', ink: 'var(--ink)', quiet: 'var(--quiet)', rule: 'var(--rule)' };

export function chart(container, opts) {
  const { height = 280, margin = { top: 28, right: 92, bottom: 40, left: 52 }, x: xo, y: yo, title } = opts;
  // Draw at the container's real width (capped), so text stays at its set size on phones.
  const avail = container.clientWidth || opts.width || 560;
  const width = Math.max(280, Math.min(opts.width || 560, avail));
  const svg = d3.select(container).append('svg')
    .attr('class', 'chart').attr('viewBox', `0 0 ${width} ${height}`).attr('width', '100%')
    .attr('role', 'img').attr('aria-label', title || '');
  const iw = width - margin.left - margin.right, ih = height - margin.top - margin.bottom;
  const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
  const x = (xo.log ? d3.scaleLog() : d3.scaleLinear()).domain(xo.domain).range([0, iw]);
  const y = (yo.log ? d3.scaleLog() : d3.scaleLinear()).domain(yo.domain).range([ih, 0]);
  const xAxis = g.append('g').attr('class', 'axis').attr('transform', `translate(0,${ih})`);
  const yAxis = g.append('g').attr('class', 'axis');
  const drawAxes = () => {
    xAxis.call(d3.axisBottom(x).ticks(xo.ticks || 5, xo.format).tickSizeOuter(0));
    yAxis.call(d3.axisLeft(y).ticks(yo.ticks || 4, yo.format).tickSizeOuter(0));
  };
  drawAxes();
  if (xo.label) g.append('text').attr('x', iw).attr('y', ih + 34).attr('text-anchor', 'end').text(xo.label);
  if (yo.label) g.append('text').attr('x', -margin.left + 4).attr('y', -14).attr('text-anchor', 'start').text(yo.label);
  const clipId = `clip-${Math.random().toString(36).slice(2)}`;
  svg.append('defs').append('clipPath').attr('id', clipId).append('rect').attr('x', -6).attr('y', -6).attr('width', iw + 12).attr('height', ih + 12);
  const layer = g.append('g').attr('clip-path', `url(#${clipId})`);
  const labels = g.append('g');
  const items = new Map();
  const get = (id, tag) => {
    if (!items.has(id)) items.set(id, (tag === 'text' ? labels : layer).append(tag));
    return items.get(id);
  };

  const api = {
    svg, g, x, y, iw, ih,
    setY(domain) { y.domain(domain); drawAxes(); },
    setX(domain) { x.domain(domain); drawAxes(); },
    line(id, points, { color = 'ink', width: w = 1.6, dash = null, label = null, labelAt = 'end', dy = 0 } = {}) {
      const finite = points.filter((p) => Number.isFinite(p[1]));
      get(id, 'path').attr('fill', 'none').attr('stroke', COLOR[color]).attr('stroke-width', w)
        .attr('stroke-dasharray', dash).attr('d', d3.line().x((p) => x(p[0])).y((p) => y(Math.min(p[1], y.domain()[1] * 4)))(finite));
      if (label) {
        const inside = finite.filter((q) => q[1] <= y.domain()[1]);
        const p = labelAt === 'end' ? inside[inside.length - 1] : inside[Math.round(labelAt * (inside.length - 1))];
        get(id + ':label', 'text').attr('x', x(p[0]) + 6).attr('y', y(Math.min(Math.max(p[1], y.domain()[0]), y.domain()[1])) + 4 + dy)
          .attr('class', `label-${color}`).text(label);
      }
      return api;
    },
    vline(id, xv, { color = 'quiet', dash = '3 3', label = null, labelY = 0 } = {}) {
      get(id, 'line').attr('x1', x(xv)).attr('x2', x(xv)).attr('y1', 0).attr('y2', ih)
        .attr('stroke', COLOR[color]).attr('stroke-width', 1).attr('stroke-dasharray', dash);
      if (label) get(id + ':label', 'text').attr('x', x(xv) + 4).attr('y', labelY + 10).attr('class', `label-${color}`).text(label);
      return api;
    },
    point(id, xv, yv, { color = 'accent', r = 4, label = null, dx = 7, dy = -7, anchor = 'start' } = {}) {
      const show = Number.isFinite(yv);
      const yy = show ? y(Math.min(Math.max(yv, y.domain()[0]), y.domain()[1])) : 0;
      get(id, 'circle').attr('cx', x(xv)).attr('cy', yy).attr('r', r).attr('fill', COLOR[color]).attr('opacity', show ? 1 : 0);
      if (label !== null) get(id + ':label', 'text').attr('x', x(xv) + dx).attr('y', yy + dy).attr('text-anchor', anchor).attr('class', `label-${color}`).text(label);
      return api;
    },
    text(id, xv, yv, str, { color = 'quiet', anchor = 'start' } = {}) {
      get(id, 'text').attr('x', x(xv)).attr('y', y(yv)).attr('text-anchor', anchor).attr('class', `label-${color}`).text(str);
      return api;
    },
    // Moves end labels apart vertically so none overlap (gap in pixels).
    spreadLabels(ids, gap = 15) {
      const els = ids.map((id) => items.get(id + ':label')).filter(Boolean);
      const ys = els.map((e) => ({ e, y: Number(e.attr('y')) })).sort((a, b) => a.y - b.y);
      for (let i = 1; i < ys.length; i++) if (ys[i].y - ys[i - 1].y < gap) ys[i].y = ys[i - 1].y + gap;
      ys.forEach(({ e, y }) => e.attr('y', y));
      return api;
    },
    remove(id) { [id, id + ':label'].forEach((k) => { if (items.has(k)) { items.get(k).remove(); items.delete(k); } }); },
  };
  return api;
}
