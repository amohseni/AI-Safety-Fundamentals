// Simplex plot: the triangle of probabilities over three actions, with a flow field and paths.
// Used on page 5.2 (and meant for reuse, e.g. 8.5). Uses the global d3; colors by role.
const d3 = window.d3;
const COLOR = { accent: 'var(--accent)', second: 'var(--second)', ink: 'var(--ink)', quiet: 'var(--quiet)', rule: 'var(--rule)' };

// labels: names of the three corners, e.g. ['action 1', 'action 2', 'action 3'].
export function simplex(container, { width = 520, labels = ['1', '2', '3'], title = '' } = {}) {
  const avail = container.clientWidth || width;
  const W = Math.max(280, Math.min(width, avail));
  const pad = { top: 34, side: 20, bottom: 40 };
  const side = W - 2 * pad.side, h = (side * Math.sqrt(3)) / 2, H = h + pad.top + pad.bottom;
  const svg = d3.select(container).append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${W} ${H}`)
    .attr('width', W).attr('height', H).attr('role', 'img').attr('aria-label', title);
  // Corner i is where x_i = 1: corner 0 top, corner 1 bottom left, corner 2 bottom right.
  const P = [[W / 2, pad.top], [pad.side, pad.top + h], [W - pad.side, pad.top + h]];
  const toXY = (x) => [x[0] * P[0][0] + x[1] * P[1][0] + x[2] * P[2][0], x[0] * P[0][1] + x[1] * P[1][1] + x[2] * P[2][1]];
  const fromXY = ([px, py]) => {
    const [x1, y1] = P[0], [x2, y2] = P[1], [x3, y3] = P[2];
    const det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    const a = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det;
    const b = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det;
    return [a, b, 1 - a - b];
  };
  svg.append('path').attr('d', `M${P[0]}L${P[1]}L${P[2]}Z`).attr('fill', 'none').attr('stroke', 'var(--rule)');
  const cornerText = [[P[0][0], P[0][1] - 12, 'middle'], [P[1][0], P[1][1] + 28, 'start'], [P[2][0], P[2][1] + 28, 'end']];
  const cornerLabels = cornerText.map(([x, y, a], i) => svg.append('text').attr('x', x).attr('y', y).attr('text-anchor', a).text(labels[i]));
  const flow = svg.append('g'), paths = svg.append('g'), top = svg.append('g');
  const items = new Map();
  const get = (id, tag, layer = paths) => { if (!items.has(id)) items.set(id, layer.append(tag)); return items.get(id); };
  const line = d3.line().x((p) => p[0]).y((p) => p[1]);

  return {
    svg, toXY, fromXY, W, H,
    setLabels(ls) { ls.forEach((l, i) => cornerLabels[i].text(l)); },
    // Short strokes along the flow at a grid of points, length proportional to speed (scaled to the largest).
    flowField(rhs, n = 11) {
      flow.selectAll('*').remove();
      const pts = [];
      for (let i = 0; i <= n; i++) for (let j = 0; j <= n - i; j++) {
        const x = [(i + 1 / 3) / (n + 1), (j + 1 / 3) / (n + 1)]; x.push(1 - x[0] - x[1]);
        if (x[2] <= 0.01) continue;
        pts.push([x, rhs(x)]);
      }
      const vmax = Math.max(...pts.map(([x, v]) => { const a = toXY(x), b = toXY(x.map((c, k) => c + v[k])); return Math.hypot(b[0] - a[0], b[1] - a[1]); }), 1e-12);
      const L = side / (n + 1) * 0.8;
      for (const [x, v] of pts) {
        const a = toXY(x), b = toXY(x.map((c, k) => c + v[k]));
        const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
        if (len < 1e-9) { flow.append('circle').attr('cx', a[0]).attr('cy', a[1]).attr('r', 1.5).attr('fill', COLOR.quiet); continue; }
        const s = (L * Math.sqrt(len / vmax)) / len;
        const e = [a[0] + dx * s, a[1] + dy * s];
        flow.append('line').attr('x1', a[0]).attr('y1', a[1]).attr('x2', e[0]).attr('y2', e[1]).attr('stroke', COLOR.quiet).attr('stroke-width', 1);
        flow.append('circle').attr('cx', e[0]).attr('cy', e[1]).attr('r', 1.6).attr('fill', COLOR.quiet);
      }
    },
    path(id, xs, { color = 'ink', width: w = 1.6, opacity = 1, dash = null } = {}) {
      get(id, 'path').attr('d', line(xs.map(toXY))).attr('fill', 'none').attr('stroke', COLOR[color])
        .attr('stroke-width', w).attr('stroke-opacity', opacity).attr('stroke-dasharray', dash);
    },
    point(id, x, { color = 'ink', r = 4, label = null, dx = 8, dy = -8, anchor = 'start' } = {}) {
      const [px, py] = toXY(x);
      get(id, 'circle', top).attr('cx', px).attr('cy', py).attr('r', r).attr('fill', COLOR[color]);
      if (label !== null) get(id + ':label', 'text', top).attr('x', px + dx).attr('y', py + dy).attr('text-anchor', anchor).attr('class', `label-${color}`).text(label);
    },
    text(id, x, str, { color = 'quiet', anchor = 'start', dx = 0, dy = 0 } = {}) {
      const [px, py] = toXY(x);
      get(id, 'text', top).attr('x', px + dx).attr('y', py + dy).attr('text-anchor', anchor).attr('class', `label-${color}`).text(str);
    },
    remove(id) { [id, id + ':label'].forEach((k) => { if (items.has(k)) { items.get(k).remove(); items.delete(k); } }); },
    clearPaths(prefix) { [...items.keys()].filter((k) => k.startsWith(prefix)).forEach((k) => { items.get(k).remove(); items.delete(k); }); },
  };
}
