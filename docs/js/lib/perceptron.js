// Perceptron playground, shared by pages 1.2 (perceptron mode) and 2.2 (neuron mode).
// The pure functions at the top have no DOM and are used by the Node tests.
import { rng } from './mlp.js';
import { cssVar } from './ui.js';

// ---- Data ----
const CORNERS = [[0, 0], [0, 1], [1, 0], [1, 1]];
export const datasets = {
  and: () => ({ X: CORNERS.map((p) => p.slice()), Y: [0, 0, 0, 1] }),
  or: () => ({ X: CORNERS.map((p) => p.slice()), Y: [0, 1, 1, 1] }),
  xor: () => ({ X: CORNERS.map((p) => p.slice()), Y: [0, 1, 1, 0] }),
  // 24 points in the unit square labeled by a fixed line, with a gap around the line.
  separable(seed = 7) {
    const r = rng(seed), X = [], Y = [];
    const w = [1, -0.6], b = -0.2;
    while (X.length < 24) {
      const x = [r(), r()];
      const s = (w[0] * x[0] + w[1] * x[1] + b) / Math.hypot(...w);
      if (Math.abs(s) < 0.06) continue;
      X.push(x); Y.push(s > 0 ? 1 : 0);
    }
    return { X, Y };
  },
  // The same kind of points, with four labels flipped, so no line separates them.
  nonseparable(seed = 7) {
    const d = datasets.separable(seed);
    for (const i of [2, 9, 15, 21]) d.Y[i] = 1 - d.Y[i];
    return d;
  },
};

// ---- The perceptron ----
export const sum = (w, b, x) => w[0] * x[0] + w[1] * x[1] + b;
export const step = (z) => (z > 0 ? 1 : 0);
export const ACT = {
  step: { name: 'step', f: step },
  sigmoid: { name: 'sigmoid', f: (z) => 1 / (1 + Math.exp(-z)) },
  relu: { name: 'ReLU', f: (z) => (z > 0 ? z : 0) },
};
export const errors = (w, b, X, Y) => X.reduce((n, x, i) => n + (step(sum(w, b, x)) !== Y[i] ? 1 : 0), 0);

// One pass of Rosenblatt's rule through the data in order. Mutates s = {w, b}.
// Returns the number of mistakes made during the pass (each mistake is one update).
export function perceptronPass(s, X, Y, lr) {
  let mistakes = 0;
  for (let i = 0; i < X.length; i++) {
    const pred = step(sum(s.w, s.b, X[i]));
    if (pred !== Y[i]) {
      const d = lr * (Y[i] - pred);
      s.w[0] += d * X[i][0]; s.w[1] += d * X[i][1]; s.b += d;
      mistakes++;
    }
  }
  return mistakes;
}

// A hidden layer of two step units and a step output unit.
// h = {W: [[a, b], [c, d]], b: [e, f], v: [g, h], c: k}
export function hiddenOut(h, x) {
  const z = [0, 1].map((i) => step(h.W[i][0] * x[0] + h.W[i][1] * x[1] + h.b[i]));
  return step(h.v[0] * z[0] + h.v[1] * z[1] + h.c);
}

// Parses '#rrggbb' or 'rgb(r, g, b)' into numbers.
export function toRGB(c) {
  if (c.startsWith('#')) { const h = c.length === 4 ? c.slice(1).split('').map((d) => d + d).join('') : c.slice(1); return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)); }
  const m = c.match(/[\d.]+/g); return m ? m.slice(0, 3).map(Number) : [0, 0, 0];
}

// ---- The view ----
// mode 'perceptron': the line can be dragged by two handles; points the line gets wrong are ringed.
// mode 'neuron': the weight vector is an arrow from the origin whose tip can be dragged; the output is shading.
export class Playground {
  constructor(canvas, { mode = 'perceptron', w = [-0.5, 1], b = 0.3, act = 'step', arrowScale = 0.25, onChange = () => {} } = {}) {
    this.canvas = canvas; this.ctx = canvas.getContext('2d');
    this.mode = mode; this.w = w.slice(); this.b = b; this.act = act;
    this.arrowScale = arrowScale; // drawn arrow = w * arrowScale, in input units
    this.hidden = null; // set to a hidden-layer object to show the two-layer network instead
    this.dom = mode === 'neuron' ? [-0.75, 1.75] : [-0.5, 1.5];
    this.onChange = onChange;
    this.X = []; this.Y = [];
    this._bind();
  }

  setData({ X, Y }) { this.X = X; this.Y = Y; }
  set(w, b) { this.w = w.slice(); this.b = b; }

  // Pixel mapping
  _px(p) { const [lo, hi] = this.dom, S = this.size; return [((p[0] - lo) / (hi - lo)) * S, S - ((p[1] - lo) / (hi - lo)) * S]; }
  _inv(px, py) { const [lo, hi] = this.dom, S = this.size; return [lo + (px / S) * (hi - lo), lo + ((S - py) / S) * (hi - lo)]; }

  // The part of the line w.x + b = 0 inside the view.
  _segment(w, b) {
    const [lo, hi] = this.dom, pts = [];
    if (Math.hypot(w[0], w[1]) < 1e-9) return null;
    if (Math.abs(w[1]) > 1e-12) for (const x of [lo, hi]) { const y = -(b + w[0] * x) / w[1]; if (y >= lo - 1e-9 && y <= hi + 1e-9) pts.push([x, y]); }
    if (Math.abs(w[0]) > 1e-12) for (const y of [lo, hi]) { const x = -(b + w[1] * y) / w[0]; if (x >= lo - 1e-9 && x <= hi + 1e-9) pts.push([x, y]); }
    const uniq = pts.filter((p, i) => pts.findIndex((q) => Math.hypot(q[0] - p[0], q[1] - p[1]) < 1e-9) === i);
    return uniq.length >= 2 ? [uniq[0], uniq[1]] : null;
  }

  handles() {
    const s = this._segment(this.w, this.b);
    if (!s) return [];
    const [a, c] = s;
    return [[a[0] + (c[0] - a[0]) / 3, a[1] + (c[1] - a[1]) / 3], [a[0] + (2 * (c[0] - a[0])) / 3, a[1] + (2 * (c[1] - a[1])) / 3]];
  }

  // Set the line through P and Q, keeping |w| and keeping the labeled side as close as possible to before.
  _lineThrough(P, Q) {
    const d = [Q[0] - P[0], Q[1] - P[1]], L = Math.hypot(...d);
    if (L < 0.02) return;
    const len = Math.hypot(...this.w) || 1;
    let n = [-d[1] / L, d[0] / L];
    if (n[0] * this.w[0] + n[1] * this.w[1] < 0) n = [-n[0], -n[1]];
    this.w = [n[0] * len, n[1] * len];
    this.b = -(this.w[0] * P[0] + this.w[1] * P[1]);
  }

  output(x) {
    if (this.hidden) return hiddenOut(this.hidden, x);
    return ACT[this.act].f(sum(this.w, this.b, x));
  }
  predictLabel(x) { return this.hidden ? hiddenOut(this.hidden, x) : step(sum(this.w, this.b, x)); }
  mistakes() { return this.X.reduce((n, x, i) => n + (this.predictLabel(x) !== this.Y[i] ? 1 : 0), 0); }

  _bind() {
    const c = this.canvas;
    let drag = null, other = null;
    const local = (e) => { const r = c.getBoundingClientRect(); return this._inv(e.clientX - r.left, e.clientY - r.top); };
    c.addEventListener('pointerdown', (e) => {
      if (this.hidden) return;
      const p = local(e), tol = ((this.dom[1] - this.dom[0]) * 22) / this.size;
      if (this.mode === 'neuron') {
        // Pressing anywhere places the arrow's tip there; dragging moves it.
        drag = 'tip';
      } else {
        const H = this.handles();
        if (!H.length) return;
        const d = H.map((h) => Math.hypot(p[0] - h[0], p[1] - h[1]));
        const k = d[0] <= d[1] ? 0 : 1;
        if (d[k] > tol * 2.5) return;
        drag = k; other = H[1 - k];
      }
      c.setPointerCapture(e.pointerId);
      e.preventDefault();
      move(e);
    });
    const move = (e) => {
      if (drag === null) return;
      const p = local(e).map((v) => Math.max(this.dom[0], Math.min(this.dom[1], v)));
      if (drag === 'tip') {
        let w = [p[0] / this.arrowScale, p[1] / this.arrowScale];
        const L = Math.hypot(...w), M = 7;
        if (L > M) w = w.map((v) => (v * M) / L);
        this.w = w.map((v) => Math.round(v * 100) / 100);
      } else {
        this._lineThrough(drag === 0 ? p : other, drag === 0 ? other : p);
      }
      this.draw(); this.onChange('drag');
    };
    c.addEventListener('pointermove', move);
    const end = () => { drag = null; other = null; };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('keydown', (e) => {
      if (this.hidden) return;
      const k = e.key;
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(k)) return;
      e.preventDefault();
      const len = Math.hypot(...this.w);
      if (this.mode === 'neuron') {
        if (k === 'ArrowLeft' || k === 'ArrowRight') {
          const a = Math.atan2(this.w[1], this.w[0]) + (k === 'ArrowLeft' ? 1 : -1) * (Math.PI / 36);
          this.w = [len * Math.cos(a), len * Math.sin(a)];
        } else {
          const L2 = Math.max(0.25, Math.min(7, len + (k === 'ArrowUp' ? 0.25 : -0.25)));
          const u = len > 1e-9 ? this.w.map((v) => v / len) : [1, 0];
          this.w = u.map((v) => v * L2);
        }
        this.w = this.w.map((v) => Math.round(v * 1000) / 1000);
      } else if (len > 1e-9) {
        if (k === 'ArrowUp' || k === 'ArrowDown') {
          // Shift the line by 0.05 along its normal.
          this.b += (k === 'ArrowUp' ? -1 : 1) * 0.05 * len;
        } else {
          // Rotate the line by 5 degrees about the middle of its visible part.
          const s = this._segment(this.w, this.b);
          const m = s ? [(s[0][0] + s[1][0]) / 2, (s[0][1] + s[1][1]) / 2] : [0.5, 0.5];
          const a = Math.atan2(this.w[1], this.w[0]) + (k === 'ArrowLeft' ? 1 : -1) * (Math.PI / 36);
          this.w = [len * Math.cos(a), len * Math.sin(a)];
          this.b = -(this.w[0] * m[0] + this.w[1] * m[1]);
        }
      }
      this.draw(); this.onChange('key');
    });
  }

  draw() {
    const { ctx, canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    const S = canvas.clientWidth;
    this.size = S;
    if (canvas.width !== Math.round(S * dpr)) { canvas.width = Math.round(S * dpr); canvas.height = Math.round(S * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, S, S);
    const col = { a: cssVar('--accent'), b: cssVar('--second'), ink: cssVar('--ink'), quiet: cssVar('--quiet'), rule: cssVar('--rule') };
    ctx.font = '21.33px et-book, Georgia, serif';

    // Shading: the output over the plane, rendered at low resolution and scaled up.
    const n = 160;
    if (!this._off) { this._off = document.createElement('canvas'); this._off.width = n; this._off.height = n; }
    const octx = this._off.getContext('2d'), img = octx.createImageData(n, n);
    const rgb = toRGB(col.a), maxA = this.mode === 'neuron' ? 0.34 : 0.13;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = this._inv(((i + 0.5) / n) * S, ((j + 0.5) / n) * S);
      const o = Math.max(0, Math.min(1, this.output(x)));
      const k = 4 * (j * n + i);
      img.data[k] = rgb[0]; img.data[k + 1] = rgb[1]; img.data[k + 2] = rgb[2]; img.data[k + 3] = Math.round(255 * maxA * o);
    }
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(this._off, 0, 0, S, S);

    // Axes through the origin, with ticks at 0 and 1.
    ctx.strokeStyle = col.rule; ctx.lineWidth = 1;
    const o0 = this._px([0, 0]), xr = this._px([this.dom[1], 0]), yt = this._px([0, this.dom[1]]), xl = this._px([this.dom[0], 0]), yb = this._px([0, this.dom[0]]);
    ctx.beginPath(); ctx.moveTo(xl[0], o0[1]); ctx.lineTo(xr[0], o0[1]); ctx.moveTo(o0[0], yb[1]); ctx.lineTo(o0[0], yt[1]); ctx.stroke();
    ctx.fillStyle = col.quiet;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const one = this._px([1, 0]); ctx.fillText('1', one[0], o0[1] + 16);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const oneY = this._px([0, 1]); ctx.fillText('1', o0[0] - 18, oneY[1]);
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('x₁', xr[0] - 4, o0[1] - 4);
    ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('x₂', o0[0] + 6, yt[1] + 2);

    // Lines
    const drawLine = (w, b, color, dash, label) => {
      const s = this._segment(w, b);
      if (!s) return;
      const a = this._px(s[0]), c = this._px(s[1]);
      ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(c[0], c[1]); ctx.stroke(); ctx.setLineDash([]);
      if (label) {
        const m = this._px([s[0][0] + 0.8 * (s[1][0] - s[0][0]), s[0][1] + 0.8 * (s[1][1] - s[0][1])]);
        ctx.fillStyle = color; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(label, Math.min(m[0] + 6, S - 30), Math.max(m[1] - 4, 24));
      }
    };
    if (this.hidden) {
      drawLine(this.hidden.W[0], this.hidden.b[0], col.ink, [5, 4], 'h₁');
      drawLine(this.hidden.W[1], this.hidden.b[1], col.ink, [2, 3], 'h₂');
    } else {
      drawLine(this.w, this.b, col.ink, [], null);
      if (this.mode === 'perceptron') {
        ctx.fillStyle = col.ink;
        for (const h of this.handles()) { const p = this._px(h); ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 2 * Math.PI); ctx.fill(); }
      }
    }

    // Weight arrow (neuron mode)
    if (this.mode === 'neuron' && !this.hidden) {
      const tip = [this.w[0] * this.arrowScale, this.w[1] * this.arrowScale];
      const p0 = this._px([0, 0]), p1 = this._px(tip);
      const L = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
      ctx.strokeStyle = col.a; ctx.fillStyle = col.a; ctx.lineWidth = 2.5;
      if (L > 2) {
        const u = [(p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L], hl = Math.min(14, L * 0.6);
        ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0] - u[0] * hl * 0.8, p1[1] - u[1] * hl * 0.8); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(p1[0], p1[1]);
        ctx.lineTo(p1[0] - u[0] * hl - u[1] * hl * 0.45, p1[1] - u[1] * hl + u[0] * hl * 0.45);
        ctx.lineTo(p1[0] - u[0] * hl + u[1] * hl * 0.45, p1[1] - u[1] * hl - u[0] * hl * 0.45);
        ctx.closePath(); ctx.fill();
      }
      ctx.beginPath(); ctx.arc(p1[0], p1[1], 5, 0, 2 * Math.PI); ctx.fill();
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      ctx.fillText('w', Math.min(p1[0] + 8, S - 20), Math.max(p1[1] - 6, 22));
    }

    // Points: class 1 filled, class 0 open; mistakes ringed.
    for (let i = 0; i < this.X.length; i++) {
      const p = this._px(this.X[i]);
      ctx.lineWidth = 2.5;
      if (this.Y[i]) { ctx.fillStyle = col.a; ctx.beginPath(); ctx.arc(p[0], p[1], 7, 0, 2 * Math.PI); ctx.fill(); }
      else { ctx.strokeStyle = col.b; ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 2 * Math.PI); ctx.stroke(); }
      if (this.predictLabel(this.X[i]) !== this.Y[i]) {
        ctx.strokeStyle = col.ink; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.arc(p[0], p[1], 13, 0, 2 * Math.PI); ctx.stroke();
      }
    }
  }
}
