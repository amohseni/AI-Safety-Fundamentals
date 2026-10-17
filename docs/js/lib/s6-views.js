// Views for Section 6 pages: an arrow view for directions in a 2- or 3-dimensional space (canvas),
// a heat map for a matrix, and a bar chart, all drawn at true pixel width so labels stay at 16pt.
// Colors are named by role ('accent', 'second', 'ink', 'quiet', 'rule') and read from the CSS tokens.
import { cssVar } from './ui.js';

const d3 = window.d3;
const FONT = '21.33px et-book, Georgia, serif';
const TOKEN = { accent: '--accent', second: '--second', ink: '--ink', quiet: '--quiet', rule: '--rule', paper: '--paper' };
const col = (name) => cssVar(TOKEN[name] || '--ink');

// ---- Arrow view ----
// items: [{ vec: [x, y(, z)], color, width, dash, label, alpha }]
// points: [[x, y(, z)], ...] drawn as small dots; axes: names of the coordinate axes; highlight: index of an axis in accent.
export class ArrowView {
  constructor(canvas, { dim = 2, interactive = true, maxSide = 520 } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dim = dim;
    this.maxSide = maxSide;
    this.yaw = 0.5; this.pitch = 0.35;
    this.items = []; this.points = []; this.axes = null; this.highlight = -1; this.extent = 1.4;
    if (interactive) this._bind();
  }

  setDim(d) { this.dim = d; this.yaw = 0.5; this.pitch = 0.35; }

  _bind() {
    const c = this.canvas;
    let last = null;
    c.addEventListener('pointerdown', (e) => { if (this.dim === 3) { last = [e.clientX, e.clientY]; c.setPointerCapture(e.pointerId); } });
    c.addEventListener('pointermove', (e) => {
      if (!last) return;
      this.yaw += (e.clientX - last[0]) * 0.01;
      this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + (e.clientY - last[1]) * 0.01));
      last = [e.clientX, e.clientY];
      this.draw();
    });
    const end = () => { last = null; };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('keydown', (e) => {
      if (this.dim !== 3) return;
      const d = 0.12;
      if (e.key === 'ArrowLeft') this.yaw -= d; else if (e.key === 'ArrowRight') this.yaw += d;
      else if (e.key === 'ArrowUp') this.pitch = Math.max(-1.4, this.pitch - d);
      else if (e.key === 'ArrowDown') this.pitch = Math.min(1.4, this.pitch + d);
      else return;
      e.preventDefault();
      this.draw();
    });
  }

  // Orthographic projection: returns [screen x, screen y, depth].
  _proj(v) {
    const x = v[0] || 0, y = v[1] || 0, z = this.dim === 3 ? v[2] || 0 : 0;
    if (this.dim === 2) return [x, y, 0];
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    // Coordinate 3 points up; coordinates 1 and 2 span the floor.
    const x1 = cy * x - sy * y, y1 = sy * x + cy * y;
    return [x1, cp * z - sp * y1, sp * z + cp * y1];
  }

  draw() {
    const c = this.canvas, ctx = this.ctx;
    const side = Math.max(240, Math.min(this.maxSide, c.parentElement ? c.parentElement.clientWidth : this.maxSide));
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(side * dpr) || c.height !== Math.round(side * dpr)) { c.width = Math.round(side * dpr); c.height = Math.round(side * dpr); }
    c.style.width = `${side}px`; c.style.height = `${side}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, side, side);
    const pad = 34, s = (side / 2 - pad) / this.extent;
    const P = (v) => { const p = this._proj(v); return [side / 2 + p[0] * s, side / 2 - p[1] * s, p[2]]; };
    ctx.font = FONT;
    ctx.textBaseline = 'middle';

    // Coordinate axes.
    if (this.axes) {
      for (let k = 0; k < this.dim; k++) {
        const e = [0, 0, 0]; e[k] = this.extent;
        const a = P(e.map((v) => -v)), b = P(e);
        ctx.strokeStyle = col(k === this.highlight ? 'accent' : 'rule');
        ctx.lineWidth = k === this.highlight ? 1.5 : 1;
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        ctx.fillStyle = col(k === this.highlight ? 'accent' : 'quiet');
        ctx.textAlign = 'center';
        const lp = P(e.map((v) => v * 1.02));
        const off = this._axisLabelOffset(k);
        ctx.fillText(this.axes[k], Math.min(side - 30, Math.max(30, lp[0] + off[0])), Math.min(side - 12, Math.max(12, lp[1] + off[1])));
      }
    }
    // Points.
    if (this.points.length) {
      ctx.fillStyle = col('quiet');
      ctx.globalAlpha = 0.45;
      for (const v of this.points) { const p = P(v); ctx.beginPath(); ctx.arc(p[0], p[1], 2, 0, 2 * Math.PI); ctx.fill(); }
      ctx.globalAlpha = 1;
    }
    // Arrows, far ones first.
    const order = this.items.map((it, i) => [i, P(it.vec)[2]]).sort((a, b) => b[1] - a[1]).map((d) => d[0]);
    const placed = [];
    const o = P([0, 0, 0]);
    for (const i of order) {
      const it = this.items[i];
      const t = P(it.vec);
      const len = Math.hypot(t[0] - o[0], t[1] - o[1]);
      ctx.globalAlpha = it.alpha ?? 1;
      ctx.strokeStyle = ctx.fillStyle = col(it.color || 'ink');
      ctx.lineWidth = it.width || 1.6;
      ctx.setLineDash(it.dash || []);
      ctx.beginPath(); ctx.moveTo(o[0], o[1]); ctx.lineTo(t[0], t[1]); ctx.stroke();
      ctx.setLineDash([]);
      if (len > 6 && it.head !== false) {
        const ux = (t[0] - o[0]) / len, uy = (t[1] - o[1]) / len, hl = Math.min(10, len / 3), hw = hl * 0.45;
        ctx.beginPath();
        ctx.moveTo(t[0], t[1]);
        ctx.lineTo(t[0] - ux * hl - uy * hw, t[1] - uy * hl + ux * hw);
        ctx.lineTo(t[0] - ux * hl + uy * hw, t[1] - uy * hl - ux * hw);
        ctx.closePath(); ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (it.label && len > 14) {
        const ux = (t[0] - o[0]) / len, uy = (t[1] - o[1]) / len;
        const w = ctx.measureText(it.label).width;
        let x = t[0] + ux * 14, y = t[1] + uy * 14;
        x = Math.min(side - w / 2 - 2, Math.max(w / 2 + 2, x));
        y = Math.min(side - 12, Math.max(12, y));
        const box = [x - w / 2 - 2, y - 11, x + w / 2 + 2, y + 11];
        if (placed.some((b) => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
        placed.push(box);
        ctx.fillStyle = col(it.labelColor || it.color || 'ink');
        ctx.textAlign = 'center';
        ctx.fillText(it.label, x, y);
      }
    }
  }

  _axisLabelOffset(k) {
    if (this.dim === 2) return k === 0 ? [-40, 16] : [52, 10];
    return [0, -14];
  }
}

// ---- Heat map: matrix M (rows x cols), positive entries in accent, negative in second ----
export function heatmap(container, { title = '', maxWidth = 360 } = {}) {
  const svg = d3.select(container).append('svg').attr('class', 'chart').attr('role', 'img').attr('aria-label', title);
  const g = svg.append('g');
  const labels = svg.append('g');
  return {
    update(M, { max = 1, rowLabel = (i) => i + 1, colLabel = (j) => j + 1, top = 'columns', left = 'rows', every = 1 } = {}) {
      const R = M.length, C = M[0].length;
      const lw = 44, th = 52;
      const avail = Math.max(220, Math.min(maxWidth, container.clientWidth || maxWidth));
      const cell = Math.max(6, Math.min(40, Math.floor((avail - lw) / C)));
      const W = lw + cell * C, H = th + cell * R + 4;
      svg.attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`);
      const data = [];
      M.forEach((row, i) => row.forEach((v, j) => data.push({ i, j, v })));
      g.selectAll('rect').data(data).join('rect')
        .attr('x', (d) => lw + d.j * cell).attr('y', (d) => th + d.i * cell).attr('width', cell - 1).attr('height', cell - 1)
        .attr('fill', (d) => (d.v >= 0 ? 'var(--accent)' : 'var(--second)'))
        .attr('fill-opacity', (d) => Math.min(1, Math.abs(d.v) / max));
      const rl = [], cl = [];
      for (let i = 0; i < R; i++) if (i % every === 0 || i === R - 1) rl.push(i);
      for (let j = 0; j < C; j++) if (j % every === 0 || j === C - 1) cl.push(j);
      labels.selectAll('text.r').data(rl).join('text').attr('class', 'r').attr('text-anchor', 'end')
        .attr('x', lw - 6).attr('y', (i) => th + i * cell + cell / 2 + 6).text((i) => rowLabel(i));
      labels.selectAll('text.c').data(cl).join('text').attr('class', 'c').attr('text-anchor', 'middle')
        .attr('x', (j) => lw + j * cell + cell / 2).attr('y', th - 8).text((j) => colLabel(j));
      labels.selectAll('text.t').data([top]).join('text').attr('class', 't').attr('x', lw).attr('y', 18).text((d) => d);
      svg.attr('data-left', left);
    },
  };
}

// ---- Bar chart with a zero line, optional dashed reference lines, and value-free labels ----
export function bars(container, { title = '', height = 250, maxWidth = 420 } = {}) {
  const svg = d3.select(container).append('svg').attr('class', 'chart').attr('role', 'img').attr('aria-label', title);
  const m = { top: 44, right: 12, bottom: 44, left: 48 };
  const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  const axY = g.append('g').attr('class', 'axis');
  const axX = g.append('g').attr('class', 'axis');
  const refs = g.append('g');
  const body = g.append('g');
  const yl = svg.append('text').attr('x', 4).attr('y', 18);
  const xl = svg.append('text').attr('text-anchor', 'end');
  return {
    update(values, { domain = [0, 1.5], yLabel = '', xLabel = '', ref = [], color = 'accent', every = 1, tickLabel = (i) => i + 1 } = {}) {
      const avail = Math.max(260, Math.min(maxWidth, container.clientWidth || maxWidth));
      const iw = avail - m.left - m.right, ih = height - m.top - m.bottom;
      svg.attr('width', avail).attr('height', height).attr('viewBox', `0 0 ${avail} ${height}`);
      const x = d3.scaleBand().domain(values.map((_, i) => i)).range([0, iw]).padding(0.25);
      const y = d3.scaleLinear().domain(domain).range([ih, 0]);
      axY.call(d3.axisLeft(y).ticks(4).tickSizeOuter(0));
      axX.attr('transform', `translate(0,${y(Math.max(domain[0], 0))})`)
        .call(d3.axisBottom(x).tickValues(values.map((_, i) => i).filter((i) => i % every === 0)).tickFormat(tickLabel).tickSizeOuter(0));
      if (domain[0] < 0) axX.selectAll('text').attr('transform', `translate(0,${ih - y(0)})`);
      const cl = (v) => Math.max(domain[0], Math.min(domain[1], v));
      body.selectAll('rect').data(values).join('rect')
        .attr('x', (_, i) => x(i)).attr('width', x.bandwidth())
        .attr('y', (v) => y(Math.max(0, cl(v)))).attr('height', (v) => Math.abs(y(cl(v)) - y(0)))
        .attr('fill', `var(--${color})`);
      refs.selectAll('line').data(ref).join('line').attr('x1', 0).attr('x2', iw).attr('y1', (r) => y(r.v)).attr('y2', (r) => y(r.v))
        .attr('stroke', 'var(--quiet)').attr('stroke-dasharray', '3 3');
      refs.selectAll('text').data(ref.filter((r) => r.label)).join('text').attr('x', iw).attr('y', (r) => y(r.v) - 5).attr('text-anchor', 'end').text((r) => r.label);
      yl.text(yLabel);
      xl.attr('x', avail - m.right).attr('y', height - 4).text(xLabel);
    },
  };
}
