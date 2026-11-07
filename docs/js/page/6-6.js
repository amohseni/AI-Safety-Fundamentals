// Page 6.6: steering GPT-2 small by activation addition, from precomputed generations
// (tools/6-6/steer.py). Numbers quoted in the tasks are checked by tools/test-page-6-6.mjs.
import { slider, segmented } from '../lib/ui.js';
import { chart } from '../lib/chart.js';
import { loadJSON, escapeHTML } from '../lib/s3k-gpt2.js';

const readout = document.getElementById('readout');
let data;
try {
  data = await loadJSON('../../data/6-6/steer.json', import.meta.url);
} catch (e) {
  readout.textContent = 'The generated text could not be loaded, so the tool cannot run.';
  throw e;
}

const VEC = {
  love: { label: 'Love − Hate', score: 'sent', scoreName: 'positive sentiment', domain: [0, 1], fmt: (v) => v.toFixed(2) },
  wed: { label: 'weddings − (space)', score: 'wed', scoreName: 'share mentioning weddings', domain: [0, 1], fmt: (v) => `${Math.round(v * 100)}%` },
};
const st = { v: 'love', p: 0, layer: 0, coef: 0, sample: 0 };
const coefs = data.coefs;
const cell = () => data.cells[`${st.p}|${st.v}|${st.layer}`];
const ci = () => coefs.indexOf(st.coef);

const ctl = document.getElementById('controls');
segmented(ctl, { name: 'Steering vector', options: Object.entries(VEC).map(([k, v]) => [k, v.label]), value: st.v, onChange: (k) => { st.v = k; draw(); } });
segmented(ctl, { name: 'Prompt', options: data.prompts.map((p, i) => [String(i), `${p} …`]), value: '0', onChange: (k) => { st.p = Number(k); draw(); } });
slider(ctl, { name: 'Layer', min: data.layers[0], max: data.layers[data.layers.length - 1], step: 1, value: st.layer, onInput: (v) => { st.layer = v; draw(); } });
slider(ctl, { name: 'Coefficient', min: coefs[0], max: coefs[coefs.length - 1], step: 1, value: st.coef,
  format: (v) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0'), onInput: (v) => { st.coef = v; draw(); } });
const sampleSeg = segmented(ctl, { name: 'Sample', options: [['0', '1'], ['1', '2'], ['2', '3']], value: '0', onChange: (k) => { st.sample = Number(k); draw(); } });

const textBox = document.getElementById('text');
const scoreBox = document.getElementById('score-chart'), fluBox = document.getElementById('flu-chart');
const minus = (x) => x.replace('-', '−');
const signed = (v) => (v > 0 ? `+${v}` : v < 0 ? `−${-v}` : '0');

function plot(box, key, title, domain, fmtTick) {
  box.innerHTML = '';
  const c = cell(), V = VEC[st.v];
  const ys = c[key];
  const ch = chart(box, { width: 420, height: 230, margin: { top: 30, right: 24, bottom: 52, left: 52 },
    x: { domain: [coefs[0], coefs[coefs.length - 1]], ticks: 5, label: 'coefficient' },
    y: { domain, ticks: 4, format: fmtTick }, title: `${title} against the coefficient` });
  ch.line('l', coefs.map((x, i) => [x, ys[i]]), { color: 'ink', width: 1.4 });
  ch.vline('zero', 0, { color: 'quiet' });
  ch.point('cur', st.coef, ys[ci()], { color: 'accent', r: 5, label: key === 'flu' ? minus(ys[ci()].toFixed(2)) : V.fmt(ys[ci()]),
    anchor: st.coef > 4 ? 'end' : 'start', dx: st.coef > 4 ? -8 : 8 });
}

function draw() {
  const c = cell(), V = VEC[st.v], i = ci();
  const prompt = data.prompts[st.p];
  const txt = c.texts[i][st.sample];
  textBox.innerHTML = `<span class="prompt-text">${escapeHTML(prompt)}</span>${escapeHTML(txt)}`;
  document.getElementById('score-title').textContent = st.v === 'wed' ? 'Share of continuations mentioning weddings' : 'Positive sentiment';
  plot(scoreBox, V.score, V.scoreName, V.domain, st.v === 'wed' ? '.0%' : undefined);
  plot(fluBox, 'flu', 'log probability per token', [-5, -0.5], undefined);
  const s0 = c[V.score][coefs.indexOf(0)], f0 = c.flu[coefs.indexOf(0)];
  readout.innerHTML = `Vector ${escapeHTML(V.label)} added before layer ${st.layer} with coefficient <span class="c-accent">${signed(st.coef)}</span>. Over the 8 sampled continuations: ${V.scoreName} <b>${V.fmt(c[V.score][i])}</b> (unsteered ${V.fmt(s0)}); average log probability per token under the unsteered model <b>${minus(c.flu[i].toFixed(2))}</b> (unsteered ${minus(f0.toFixed(2))}).`;
}
draw();
void sampleSeg;

// ---- Static figure: three continuations of the first prompt at layer 6 ----
(function figure() {
  const fig = document.getElementById('fig-steer');
  const c = data.cells[`0|love|6`];
  for (const k of [-5, 0, 5]) {
    const d = document.createElement('div');
    const t = c.texts[coefs.indexOf(k)][0];
    d.innerHTML = `<p class="small-title">Coefficient ${signed(k)}</p><p class="gen"><span class="prompt-text">${escapeHTML(data.prompts[0])}</span>${escapeHTML(t)}</p>`;
    fig.append(d);
  }
})();

let rw;
window.addEventListener('resize', () => { clearTimeout(rw); rw = setTimeout(draw, 150); });
