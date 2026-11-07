// Plots shared by pages 6.2 and 7.2: a scatter of activations in a plane with a draggable
// direction arrow, and a mirrored histogram of the projections onto that direction.
// Drawn with the global d3 at the container's true pixel width; colors by role tokens only.
const d3 = window.d3;

// Scatter with an arrow from the origin. Points: { x, y, kind } with kind 'a' (filled, accent),
// 'b' (open circle, second color) or 'q' (small, quiet). onArrow(angle) is called while dragging.
// Options: arrow (draw the arrow at all), interactive (the arrow can be dragged and turned with keys).
export function scatterArrow(container, { height = 360, maxWidth = 520, label = '', onArrow = null, arrowColor = 'ink', arrow = true, interactive = true } = {}) {
  const avail = container.clientWidth || maxWidth;
  const W = Math.max(260, Math.min(maxWidth, avail)), H = Math.min(height, W);
  const svg = d3.select(container).append('svg').attr('class', 'chart').attr('width', W).attr('height', H)
    .attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img').attr('aria-label', label);
  const cx = W / 2, cy = H / 2, R = Math.min(W, H) / 2 - 18;
  const gPts = svg.append('g'), gLab = svg.append('g'), gArrow = svg.append('g');
  let scale = 1, angle = 0, pts = [];
  const line = gArrow.append('line').attr('stroke', `var(--${arrowColor})`).attr('stroke-width', 2);
  const tail = gArrow.append('line').attr('stroke', `var(--${arrowColor})`).attr('stroke-width', 1).attr('stroke-dasharray', '3 4');
  const head = gArrow.append('path').attr('fill', `var(--${arrowColor})`);
  const knob = gArrow.append('circle').attr('r', 9).attr('fill', 'var(--paper)').attr('stroke', `var(--${arrowColor})`).attr('stroke-width', 2)
    .attr('tabindex', 0).attr('role', 'slider').attr('aria-label', 'Direction of the arrow, in degrees').attr('aria-valuemin', 0).attr('aria-valuemax', 359)
    .style('cursor', 'grab');
  const drawArrow = () => {
    const L = R * 0.92, ex = cx + L * Math.cos(angle), ey = cy - L * Math.sin(angle);
    line.attr('x1', cx).attr('y1', cy).attr('x2', ex).attr('y2', ey);
    tail.attr('x1', cx).attr('y1', cy).attr('x2', cx - L * Math.cos(angle)).attr('y2', cy + L * Math.sin(angle));
    const a = -angle, s = 11;
    head.attr('d', `M${ex + s * Math.cos(a)},${ey + s * Math.sin(a)} L${ex + s * 0.8 * Math.cos(a + 2.2)},${ey + s * 0.8 * Math.sin(a + 2.2)} L${ex + s * 0.8 * Math.cos(a - 2.2)},${ey + s * 0.8 * Math.sin(a - 2.2)} Z`);
    knob.attr('cx', ex).attr('cy', ey).attr('aria-valuenow', Math.round(((angle * 180 / Math.PI) % 360 + 360) % 360));
  };
  const setFromPointer = (ev) => {
    const [mx, my] = d3.pointer(ev, svg.node());
    angle = Math.atan2(cy - my, mx - cx);
    drawArrow();
    onArrow && onArrow(angle);
  };
  if (!arrow) gArrow.attr('display', 'none');
  if (!interactive) knob.remove();
  let dragging = false;
  if (interactive) svg.on('pointerdown', (ev) => { dragging = true; svg.node().setPointerCapture && svg.node().setPointerCapture(ev.pointerId); setFromPointer(ev); })
    .on('pointermove', (ev) => { if (dragging) setFromPointer(ev); })
    .on('pointerup pointercancel', () => { dragging = false; });
  if (interactive) knob.on('keydown', (ev) => {
    const step = (ev.shiftKey ? 15 : 3) * Math.PI / 180;
    if (['ArrowLeft', 'ArrowDown'].includes(ev.key)) { angle -= step; } else if (['ArrowRight', 'ArrowUp'].includes(ev.key)) { angle += step; } else return;
    ev.preventDefault(); drawArrow(); onArrow && onArrow(angle);
  });
  const api = {
    svg, W, H,
    setPoints(p, { fixedScale = null } = {}) {
      pts = p;
      const m = fixedScale || d3.max(p, (q) => Math.max(Math.abs(q.x), Math.abs(q.y))) || 1;
      scale = R / m;
      const sel = gPts.selectAll('circle').data(pts);
      sel.join('circle')
        .attr('cx', (q) => cx + Math.max(-R - 10, Math.min(R + 10, q.x * scale))).attr('cy', (q) => cy - Math.max(-R - 10, Math.min(R + 10, q.y * scale)))
        .attr('r', (q) => (q.kind === 'q' ? 2.5 : 3.6))
        .attr('fill', (q) => (q.kind === 'a' ? 'var(--accent)' : q.kind === 'q' ? 'var(--quiet)' : 'none'))
        .attr('fill-opacity', (q) => (q.kind === 'a' ? 0.75 : 0.6))
        .attr('stroke', (q) => (q.kind === 'b' ? 'var(--second)' : 'none')).attr('stroke-width', 1.2);
      return api;
    },
    // Text labels placed in data coordinates: [{ x, y, text, color }].
    setLabels(labels) {
      gLab.selectAll('text').data(labels).join('text')
        .attr('x', (l) => cx + l.x * scale).attr('y', (l) => cy - l.y * scale + 6).attr('text-anchor', 'middle')
        .attr('class', (l) => `label-${l.color || 'ink'}`).text((l) => l.text);
      return api;
    },
    setAngle(a) { angle = a; drawArrow(); return api; },
    get angle() { return angle; },
  };
  drawArrow();
  return api;
}

// Mirrored histogram: values of group A above the axis (accent), group B below (second).
export function mirrorHist(container, { height = 190, maxWidth = 520, labelA = 'true', labelB = 'false', bins = 30, label = '' } = {}) {
  const avail = container.clientWidth || maxWidth;
  const W = Math.max(260, Math.min(maxWidth, avail)), H = height;
  const m = { left: 8, right: 8, top: 26, bottom: 26 };
  const svg = d3.select(container).append('svg').attr('class', 'chart').attr('width', W).attr('height', H)
    .attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img').attr('aria-label', label);
  const g = svg.append('g');
  const mid = m.top + (H - m.top - m.bottom) / 2;
  const axis = svg.append('line').attr('x1', m.left).attr('x2', W - m.right).attr('y1', mid).attr('y2', mid).attr('stroke', 'var(--rule)');
  const thr = svg.append('line').attr('y1', m.top - 6).attr('y2', H - m.bottom + 6).attr('stroke', 'var(--ink)').attr('stroke-dasharray', '3 3');
  const tA = svg.append('text').attr('class', 'label-accent').attr('y', 18);
  const tB = svg.append('text').attr('class', 'label-second').attr('y', H - 6);
  const api = {
    // a, b: arrays of numbers; t: threshold (or null); dom: [lo, hi] (optional)
    set(a, b, t = null, dom = null) {
      const all = a.concat(b);
      const lo = dom ? dom[0] : d3.min(all), hi = dom ? dom[1] : d3.max(all);
      const x = d3.scaleLinear().domain([lo, hi]).range([m.left, W - m.right]);
      const edges = d3.range(bins + 1).map((i) => lo + (i * (hi - lo)) / bins);
      const count = (v) => { const c = new Array(bins).fill(0); v.forEach((u) => { c[Math.min(bins - 1, Math.max(0, Math.floor(((u - lo) / (hi - lo)) * bins)))]++; }); return c; };
      const ca = count(a), cb = count(b);
      const ymax = Math.max(1, d3.max(ca), d3.max(cb));
      const hh = (H - m.top - m.bottom) / 2 - 2;
      const bw = Math.max(1, x(edges[1]) - x(edges[0]) - 1);
      g.selectAll('rect.a').data(ca).join('rect').attr('class', 'a').attr('x', (_, i) => x(edges[i])).attr('width', bw)
        .attr('y', (c) => mid - (c / ymax) * hh).attr('height', (c) => (c / ymax) * hh).attr('fill', 'var(--accent)').attr('fill-opacity', 0.8);
      g.selectAll('rect.b').data(cb).join('rect').attr('class', 'b').attr('x', (_, i) => x(edges[i])).attr('width', bw)
        .attr('y', mid).attr('height', (c) => (c / ymax) * hh).attr('fill', 'var(--second)').attr('fill-opacity', 0.8);
      tA.attr('x', m.left).text(labelA);
      tB.attr('x', m.left).text(labelB);
      if (t === null || !Number.isFinite(t)) thr.attr('opacity', 0);
      else thr.attr('opacity', 1).attr('x1', x(t)).attr('x2', x(t));
      return api;
    },
  };
  return api;
}

