// Page 1.4: what a filter sees, and the ImageNet results by year.
import { shapes, conv3, convColor, PRESETS, sum } from '../lib/conv.js';
import { segmented } from '../lib/ui.js';

const d3 = window.d3;
const W = 240, H = 160;

// ---- Static figure: winning ImageNet top-5 error by year (ILSVRC results; Russakovsky et al. 2015) ----
const IMAGENET = [
  [2010, 28.2, 'NEC-UIUC'], [2011, 25.8, 'XRCE'], [2012, 15.3, 'AlexNet (SuperVision)'], [2013, 11.7, 'Clarifai'],
  [2014, 6.7, 'GoogLeNet'], [2015, 3.57, 'ResNet (MSRA)'], [2016, 2.99, 'Trimps-Soushen'], [2017, 2.25, 'SENet (WMW)'],
];
function imagenetFigure() {
  const box = document.getElementById('fig-imagenet');
  box.innerHTML = '';
  // One row per year: the label on top, the bar under it, so long labels never run off the chart.
  const width = Math.max(300, Math.min(620, box.clientWidth || 620)), rowH = 50;
  const m = { top: 40, right: 16, bottom: 48, left: 4 };
  const n = IMAGENET.length, height = m.top + m.bottom + rowH * n;
  const iw = width - m.left - m.right;
  const svg = d3.select(box).append('svg').attr('class', 'chart').attr('viewBox', `0 0 ${width} ${height}`)
    .attr('width', width).attr('height', height).attr('role', 'img')
    .attr('aria-label', 'Winning ImageNet top-5 error by year: ' + IMAGENET.map(([y, e, t]) => `${y} ${t} ${e}%`).join(', ') + '; one trained person 5.1%.');
  const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
  const x = d3.scaleLinear().domain([0, 30]).range([0, iw]);
  const top = (i) => i * rowH;
  g.append('g').attr('class', 'axis').attr('transform', `translate(0,${rowH * n})`)
    .call(d3.axisBottom(x).ticks(3).tickFormat((v) => `${v}%`).tickSizeOuter(0))
    // Keep the first and last tick labels inside the chart.
    .call((a) => { a.selectAll('.tick text').filter((v) => v === 0).attr('text-anchor', 'start'); a.selectAll('.tick text').filter((v) => v === 30).attr('text-anchor', 'end'); });
  g.append('text').attr('x', iw).attr('y', rowH * n + 42).attr('text-anchor', 'end').text('winning top-5 error');
  const hx = x(5.1);
  // The dashed line runs through the bars only, so it never crosses a row's label.
  g.append('line').attr('x1', hx).attr('x2', hx).attr('y1', -8).attr('y2', 0)
    .attr('stroke', 'var(--ink)').attr('stroke-dasharray', '3 3');
  for (let i = 0; i < n; i++) {
    g.append('line').attr('x1', hx).attr('x2', hx).attr('y1', top(i) + 23).attr('y2', i === n - 1 ? rowH * n : top(i) + rowH)
      .attr('stroke', 'var(--ink)').attr('stroke-dasharray', '3 3');
  }
  g.append('text').attr('x', hx + 4).attr('y', -14).attr('text-anchor', 'start').attr('class', 'label-ink').text('one trained person, 5.1%');
  const row = g.selectAll('g.row').data(IMAGENET).join('g').attr('class', 'row').attr('transform', (d, i) => `translate(0,${top(i)})`);
  row.append('text').attr('x', 0).attr('y', 18).attr('class', (d) => (d[0] === 2012 ? 'label-accent' : 'label-ink'))
    .attr('stroke', 'var(--paper)').attr('stroke-width', 5).attr('paint-order', 'stroke')
    .text((d) => `${d[0]}  ${d[2]}, ${d[1]}%`);
  row.append('rect').attr('x', 0).attr('y', 26).attr('height', 14).attr('width', (d) => x(d[1]))
    .attr('fill', (d) => (d[0] === 2012 ? 'var(--accent)' : 'var(--quiet)')).attr('fill-opacity', (d) => (d[0] === 2012 ? 0.9 : 0.45));
}

// ---- The tool ----
const state = { image: 'shapes', mode: 'preset', preset: 'vertical', k: PRESETS.vertical.k.slice(), alex: null };
const SHAPES = shapes(W, H);
let photo = null; // { planes: [R, G, B], gray }
let alex = null; // { filters: [[R, G, B] planes of 121], size }
const inC = document.getElementById('in'), outC = document.getElementById('out');
const readout = document.getElementById('readout');
const outTitle = document.getElementById('out-title');

function grayPlanes() {
  if (state.image === 'shapes') return { gray: SHAPES, planes: [SHAPES, SHAPES, SHAPES] };
  return photo;
}

function paintGray(canvas, vals, w, h, map) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  const im = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(255, Math.round(map(vals[i]))));
    im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
function paintInput() {
  const src = grayPlanes();
  if (!src) return;
  inC.width = W; inC.height = H;
  const ctx = inC.getContext('2d');
  const im = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) im.data[4 * i + c] = src.planes[c][i];
    im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

const fmtK = (v) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)).replace('-', '−') : v.toFixed(2).replace('-', '−'));
function compute() {
  const src = grayPlanes();
  if (!src) { readout.textContent = 'Loading the photo.'; return; }
  paintInput();
  if (state.mode === 'alex' && alex) {
    const f = alex.filters[state.alex];
    const { out, ow, oh } = convColor(src.planes, W, H, f, alex.size);
    let mx = 0; for (const v of out) mx = Math.max(mx, v);
    paintGray(outC, out, ow, oh, (v) => (mx > 0 ? (Math.max(0, v) / mx) * 255 : 0));
    outTitle.textContent = `Output of AlexNet filter ${state.alex + 1}`;
    readout.innerHTML = `AlexNet filter ${state.alex + 1} of 96, 11 by 11 pixels in three colors, applied at every position. White marks where the filter&rsquo;s output is most positive, black where it is zero or negative; after this layer the network sets negative values to zero with ReLU.`;
    return;
  }
  const { out, ow, oh } = conv3(src.gray, W, H, state.k);
  const s = sum(state.k);
  const zeroSum = Math.abs(s) < 1e-9;
  let mx = 0; for (const v of out) mx = Math.max(mx, Math.abs(v));
  paintGray(outC, out, ow, oh, zeroSum ? (v) => (mx > 0 ? (Math.abs(v) / mx) * 255 : 0) : (v) => v);
  const name = state.mode === 'preset' ? PRESETS[state.preset].name.toLowerCase() + ' filter' : 'your filter';
  outTitle.textContent = `Output of the ${name}`;
  readout.innerHTML = `${name[0].toUpperCase() + name.slice(1)} (${[0, 1, 2].map((r) => state.k.slice(3 * r, 3 * r + 3).map(fmtK).join(' ')).join('; ')}). Its numbers add up to ${fmtK(s)}` +
    (zeroSum ? ', so the output shows the size of the response, scaled so the strongest is white.' : ', so the output shows the result directly, as brightness from 0 to 255.');
}

const ctl = document.getElementById('controls');
segmented(ctl, { name: 'Image', options: [['shapes', 'Shapes'], ['photo', 'Photo']], value: state.image,
  onChange: (k) => { state.image = k; compute(); } });
const presetSeg = segmented(ctl, { name: 'Filter', options: Object.entries(PRESETS).map(([k, p]) => [k, p.name]), value: state.preset,
  onChange: (k) => { state.mode = 'preset'; state.preset = k; state.k = PRESETS[k].k.slice(); syncInputs(); markAlex(); compute(); } });
const kWrap = document.createElement('div');
kWrap.className = 'ctl';
kWrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">The 3 by 3 filter</span></span>';
const grid = document.createElement('div');
grid.className = 'kernel';
const inputs = d3.range(9).map((i) => {
  const inp = document.createElement('input');
  inp.type = 'number'; inp.step = '0.5'; inp.inputMode = 'decimal';
  inp.setAttribute('aria-label', `Filter row ${Math.floor(i / 3) + 1}, column ${(i % 3) + 1}`);
  inp.addEventListener('input', () => {
    const v = Number(inp.value);
    if (!Number.isFinite(v) || inp.value === '') return;
    state.k[i] = v; state.mode = 'custom'; presetSeg.set(null); markAlex(); compute();
  });
  grid.append(inp);
  return inp;
});
kWrap.append(grid);
ctl.append(kWrap);
function syncInputs() { inputs.forEach((inp, i) => { inp.value = fmtK(state.k[i]).replace('−', '-'); }); }
syncInputs();

// ---- AlexNet's first-layer filters (BVLC replication; see data/alexnet-conv1.json) ----
const alexBox = document.getElementById('alex');
let alexButtons = [];
function markAlex() { alexButtons.forEach((b, i) => b.setAttribute('aria-pressed', String(state.mode === 'alex' && state.alex === i))); }
function loadAlex(data) {
  const [n, ch, size] = data.shape;
  const bytes = Uint8Array.from(atob(data.int8), (c) => c.charCodeAt(0));
  const q = new Int8Array(bytes.buffer);
  const filters = [];
  for (let f = 0; f < n; f++) {
    const planes = [];
    for (let c = 0; c < ch; c++) {
      const P = new Float64Array(size * size);
      for (let i = 0; i < size * size; i++) P[i] = q[((f * ch + c) * size * size) + i] * data.scales[f];
      planes.push(P);
    }
    filters.push(planes);
  }
  alex = { filters, size };
  alexButtons = filters.map((planes, f) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('aria-label', `AlexNet filter ${f + 1}`);
    b.setAttribute('aria-pressed', 'false');
    const c = document.createElement('canvas');
    c.width = size; c.height = size;
    const ctx = c.getContext('2d');
    const im = ctx.createImageData(size, size);
    let mx = 0; planes.forEach((P) => P.forEach((v) => { mx = Math.max(mx, Math.abs(v)); }));
    for (let i = 0; i < size * size; i++) {
      for (let k = 0; k < 3; k++) im.data[4 * i + k] = Math.round(128 + (127 * planes[k][i]) / mx);
      im.data[4 * i + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    b.append(c);
    b.addEventListener('click', () => { state.mode = 'alex'; state.alex = f; presetSeg.set(null); markAlex(); compute(); });
    alexBox.append(b);
    return b;
  });
}

function loadPhoto() {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement('canvas');
      c.width = W; c.height = H;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0, W, H);
      const d = ctx.getImageData(0, 0, W, H).data;
      const planes = [0, 1, 2].map((k) => Float64Array.from({ length: W * H }, (_, i) => d[4 * i + k]));
      const gray = Float64Array.from({ length: W * H }, (_, i) => 0.299 * planes[0][i] + 0.587 * planes[1][i] + 0.114 * planes[2][i]);
      photo = { planes, gray };
      resolve();
    };
    img.onerror = reject;
    img.src = '../data/img/chelsea.png';
  });
}

compute();
imagenetFigure();
loadPhoto().then(compute).catch((e) => console.error('photo', e));
fetch('../data/alexnet-conv1.json').then((r) => r.json()).then(loadAlex).catch((e) => {
  document.getElementById('alex-panel').hidden = true;
  console.error('alexnet', e);
});
let rt;
window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(imagenetFigure, 150); });
