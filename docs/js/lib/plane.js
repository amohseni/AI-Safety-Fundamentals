// Plane view: draws data points and a grid as they pass through the stages of an MLP.
// Stage 0 is the input; stage 2l-1 is after layer l's linear map; stage 2l after its activation.
// Width 2 is drawn in the plane; width 3 in a rotatable orthographic 3D view.
import { cssVar } from './ui.js';

export function stageNames(nHidden) {
  const names = ['input'];
  for (let l = 1; l <= nHidden; l++) names.push(`layer ${l}, linear map`, `layer ${l}, activation`);
  return names;
}

export class PlaneView {
  constructor(canvas, { dim = 2, interactive = true, gridLines = 13 } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dim = dim;
    this.yaw = 0.6; this.pitch = 0.45;
    this.view = null; // {cx, cy, s} smoothed
    this.gridLines = gridLines;
    this.grid = this._makeGrid();
    if (interactive) this._bindRotate();
  }

  setDim(d) { this.dim = d; this.view = null; this.yaw = 0; this.pitch = 0; }

  _makeGrid() {
    const lines = [], n = this.gridLines, lo = -1.2, hi = 1.2, k = 40;
    for (let i = 0; i < n; i++) {
      const v = lo + ((hi - lo) * i) / (n - 1);
      lines.push(Array.from({ length: k }, (_, j) => [v, lo + ((hi - lo) * j) / (k - 1)]));
      lines.push(Array.from({ length: k }, (_, j) => [lo + ((hi - lo) * j) / (k - 1), v]));
    }
    return lines;
  }

  _bindRotate() {
    const c = this.canvas;
    let last = null;
    c.addEventListener('pointerdown', (e) => { if (this.dim === 3) { last = [e.clientX, e.clientY]; c.setPointerCapture(e.pointerId); } });
    c.addEventListener('pointermove', (e) => {
      if (!last) return;
      this.yaw += (e.clientX - last[0]) * 0.01;
      this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + (e.clientY - last[1]) * 0.01));
      last = [e.clientX, e.clientY];
      this.onRotate && this.onRotate();
    });
    c.addEventListener('pointerup', () => { last = null; });
    c.addEventListener('keydown', (e) => {
      if (this.dim !== 3) return;
      const d = 0.12;
      if (e.key === 'ArrowLeft') this.yaw -= d; else if (e.key === 'ArrowRight') this.yaw += d;
      else if (e.key === 'ArrowUp') this.pitch = Math.max(-1.4, this.pitch - d);
      else if (e.key === 'ArrowDown') this.pitch = Math.min(1.4, this.pitch + d);
      else return;
      e.preventDefault();
      this.onRotate && this.onRotate();
    });
  }

  // Position of point x at fractional stage t, padded to this.dim coordinates.
  _at(net, x, t) {
    const S = net.stages(x);
    const k = Math.floor(t), f = t - k;
    const pad = (v) => { const o = v.slice(0, this.dim); while (o.length < this.dim) o.push(0); return o; };
    const a = pad(S[k]);
    if (f < 1e-6) return a;
    const b = pad(S[k + 1]);
    return a.map((v, i) => v + f * (b[i] - v));
  }

  _project(p) {
    if (this.dim === 2) return [p[0], p[1], 0];
    const [x, y, z] = p;
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
    const y1 = cp * y - sp * z1, z2 = sp * y + cp * z1;
    return [x1, y1, z2];
  }

  // Draw the network's view of the data at stage t.
  // opts: { shadeInput: bool, showBoundary: bool, smooth: bool }
  draw(net, data, t, opts = {}) {
    const { ctx, canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const col = { a: cssVar('--accent'), b: cssVar('--second'), grid: cssVar('--rule'), ink: cssVar('--ink'), quiet: cssVar('--quiet') };

    const pts = data.X.map((x) => this._project(this._at(net, x, t)));
    const grid = this.grid.map((line) => line.map((x) => this._project(this._at(net, x, t))));

    // Fit the view to the data, smoothing between frames so motion is continuous.
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of pts) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
    const span = Math.max(x1 - x0, y1 - y0, 1e-3) * 1.25;
    const target = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, s: Math.min(W, H) / span };
    if (!this.view || !opts.smooth) this.view = target;
    else for (const k of ['cx', 'cy', 's']) this.view[k] += 0.25 * (target[k] - this.view[k]);
    const v = this.view;
    const sx = (p) => W / 2 + (p[0] - v.cx) * v.s, sy = (p) => H / 2 - (p[1] - v.cy) * v.s;

    // Shade the input plane by the network's prediction.
    if (opts.shadeInput && t < 0.001 && (this.dim === 2 || (this.yaw === 0 && this.pitch === 0))) {
      const n = 48, cw = W / n, ch = H / n;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const px = (i + 0.5) * cw, py = (j + 0.5) * ch;
        const x = [(px - W / 2) / v.s + v.cx, -(py - H / 2) / v.s + v.cy];
        const p = net.predict(x);
        ctx.fillStyle = p > 0.5 ? col.a : col.b;
        ctx.globalAlpha = 0.10 * Math.min(1, Math.abs(p - 0.5) * 4);
        ctx.fillRect(i * cw, j * ch, cw + 0.5, ch + 0.5);
      }
      ctx.globalAlpha = 1;
    }

    // Grid
    ctx.strokeStyle = col.grid; ctx.lineWidth = 1;
    for (const line of grid) {
      ctx.beginPath();
      line.forEach((p, i) => (i ? ctx.lineTo(sx(p), sy(p)) : ctx.moveTo(sx(p), sy(p))));
      ctx.stroke();
    }

    // Separating line or plane in the last hidden space.
    const last = 2 * (net.nLayers - 1);
    if (opts.showBoundary && Math.abs(t - last) < 1e-3) this._boundary(net, pts, sx, sy, col, W, H);

    // Points, drawn back to front in 3D.
    const order = pts.map((p, i) => i).sort((i, j) => pts[i][2] - pts[j][2]);
    for (const i of order) {
      const p = pts[i];
      ctx.beginPath();
      ctx.arc(sx(p), sy(p), 2.6, 0, 2 * Math.PI);
      ctx.fillStyle = data.Y[i] ? col.a : col.b;
      ctx.fill();
    }
  }

  _boundary(net, pts, sx, sy, col, W, H) {
    const { ctx } = this;
    const L = net.nLayers - 1;
    const w = net.W[L][0], b = net.b[L][0];
    ctx.strokeStyle = col.ink; ctx.lineWidth = 1.4;
    if (this.dim === 2) {
      // w0 x + w1 y + b = 0, drawn across a generous range.
      const R = 50, seg = [];
      if (Math.abs(w[1]) > Math.abs(w[0])) for (const x of [-R, R]) seg.push([x, -(b + w[0] * x) / w[1], 0]);
      else for (const y of [-R, R]) seg.push([-(b + w[1] * y) / w[0], y, 0]);
      ctx.beginPath(); ctx.moveTo(sx(seg[0]), sy(seg[0])); ctx.lineTo(sx(seg[1]), sy(seg[1])); ctx.stroke();
      return;
    }
    // 3D: intersect the plane w.h + b = 0 with the data's bounding box.
    const H3 = this._lastPoints;
    const box = [[Infinity, -Infinity], [Infinity, -Infinity], [Infinity, -Infinity]];
    for (const p of H3) for (let i = 0; i < 3; i++) { box[i][0] = Math.min(box[i][0], p[i]); box[i][1] = Math.max(box[i][1], p[i]); }
    box.forEach((r) => { const pad = (r[1] - r[0]) * 0.08 + 1e-3; r[0] -= pad; r[1] += pad; });
    const corners = [];
    for (const a of [0, 1]) for (const c of [0, 1]) for (const d of [0, 1]) corners.push([box[0][a], box[1][c], box[2][d]]);
    const f = (p) => w[0] * p[0] + w[1] * p[1] + w[2] * p[2] + b;
    const hits = [];
    for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
      const diff = corners[i].reduce((s, v, k) => s + (v !== corners[j][k] ? 1 : 0), 0);
      if (diff !== 1) continue;
      const fi = f(corners[i]), fj = f(corners[j]);
      if (fi * fj < 0) { const s = fi / (fi - fj); hits.push(corners[i].map((v, k) => v + s * (corners[j][k] - v))); }
    }
    if (hits.length < 3) return;
    const P = hits.map((h) => this._project(h));
    const cx = P.reduce((s, p) => s + p[0], 0) / P.length, cy = P.reduce((s, p) => s + p[1], 0) / P.length;
    P.sort((a, c) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(c[1] - cy, c[0] - cx));
    ctx.beginPath();
    P.forEach((p, i) => (i ? ctx.lineTo(sx(p), sy(p)) : ctx.moveTo(sx(p), sy(p))));
    ctx.closePath();
    ctx.globalAlpha = 0.05; ctx.fillStyle = col.ink; ctx.fill(); ctx.globalAlpha = 1;
    ctx.stroke();
  }
}

// Unprojected positions are needed for the 3D boundary; wrap draw to record them.
const baseDraw = PlaneView.prototype.draw;
PlaneView.prototype.draw = function (net, data, t, opts) {
  if (this.dim === 3) this._lastPoints = data.X.map((x) => this._at(net, x, t));
  return baseDraw.call(this, net, data, t, opts);
};
