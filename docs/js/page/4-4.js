// Page 4.4: tokens are computation.
// Panel 1: two toy transformers trained on addition (data/4-4/addition.json, from tools/4-4/addition.py).
// Panel 2: stored chains of thought from Qwen3-1.7B, cut at each sentence, with the answer the model
// gives when made to answer there (data/4-4/early.json, from tools/4-4/qwen_cot.py).
import { chart } from '../lib/chart.js';
import { slider, segmented, fmt } from '../lib/ui.js';
import { loadJSON } from '../lib/s4m-text.js';

const LETTERS = 'ABCD';
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
let add = null, early = null, mult = null;

// ---- Panel 1: addition ----
const A = { sums: 'random', n: 4 };
let nSlider = null;
const SUMS = { random: 'random sums', chained: 'sums that carry through every column' };
function addPanel() {
  const ctl = document.getElementById('add-controls');
  ctl.innerHTML = '';
  segmented(ctl, { name: 'Test sums', options: [['random', 'Random'], ['chained', 'Carry through every column']], value: A.sums,
    onChange: (k) => { A.sums = k; drawAdd(); } });
  nSlider = slider(ctl, { name: 'Digits in each number', min: 1, max: 12, step: 1, value: A.n, format: (v) => String(v),
    onInput: (v) => { A.n = v; drawAdd(); } });
  drawAdd();
}
const accOf = (m) => add.models[m][A.sums === 'random' ? 'accuracy_by_digits' : 'chained_by_digits'];
function drawAdd() {
  if (!add) return;
  const stage = document.getElementById('add-chart');
  stage.innerHTML = '';
  const ch = chart(stage, { height: 300, margin: { top: 34, right: 120, bottom: 52, left: 56 },
    x: { domain: [1, 12], ticks: 6, label: 'digits in each number' },
    y: { domain: [0, 1], ticks: 5, format: '.0%', label: 'share of sums fully right' },
    title: `Accuracy of the two models on ${SUMS[A.sums]}, against the number of digits` });
  const series = (m) => Object.entries(accOf(m)).map(([k, v]) => [Number(k), v]);
  ch.line('carries', series('carries'), { color: 'accent', label: 'writes carries' });
  ch.line('direct', series('direct'), { color: 'second', label: 'direct' });
  ch.spreadLabels(['carries', 'direct'], 18);
  ch.vline('n', A.n, { color: 'quiet' });
  const n = A.n;
  const dA = accOf('direct')[n], cA = accOf('carries')[n];
  ch.point('pd', n, dA, { color: 'second', r: 4 });
  ch.point('pc', n, cA, { color: 'accent', r: 4 });
  const key = A.sums === 'random' ? 'examples' : 'chained_examples';
  const exD = add.models.direct[key][n], exC = add.models.carries[key][n];
  const rows = exD.map((e, k) => {
    const want = e.a + e.b;
    const sumD = strip(e.output.replace('.', ''));
    const sumC = strip((exC[k].output.replace('.', '').split('>')[1]) || '');
    const okD = sumD === String(want), okC = sumC === String(want);
    return `<tr><td>${e.a} + ${e.b}</td><td>${want}</td>` +
      `<td>${sumD} <span class="q">${okD ? 'right' : 'wrong'}</span></td>` +
      `<td>${sumC} <span class="q">${okC ? 'right' : 'wrong'}</span></td></tr>`;
  }).join('');
  document.getElementById('add-examples').innerHTML = `<table class="ex"><thead><tr><th>Sum</th><th>True answer</th><th class="d">Direct</th><th class="c">Writes carries</th></tr></thead><tbody>${rows}</tbody></table>`;
  const N = A.sums === 'random' ? '1,000' : '500';
  document.getElementById('add-readout').innerHTML = `On ${N} ${SUMS[A.sums]} with ${n}-digit numbers, the <span class="c-second">direct</span> model gets <b>${pct(dA)}</b> fully right, writing ${add.models.direct.tokens_written} tokens per answer; the model that <span class="c-accent">writes carries</span> gets <b>${pct(cA)}</b>, writing ${add.models.carries.tokens_written} tokens per answer.`;
}
const strip = (s) => (s.replace(/^0+(?=\d)/, '') || '0');
const pct = (v) => `${(100 * v).toFixed(1)}%`;

// ---- Panel 2: early answering ----
const E = { prob: 0, cut: null };
let cutSlider = null;
function finalLetter(p) {
  const last = p.cuts[p.cuts.length - 1].p;
  return last.indexOf(Math.max(...last));
}
function settledAt(p) {
  const f = finalLetter(p);
  let k = p.cuts.length - 1;
  while (k > 0 && argmax(p.cuts[k - 1].p) === f) k--;
  return k;
}
const argmax = (a) => a.indexOf(Math.max(...a));
function earlyPanel() {
  const ctl = document.getElementById('cot-controls');
  ctl.innerHTML = '';
  const probs = early.problems.filter((p) => p.finished);
  segmented(ctl, { name: 'Problem', options: probs.map((p, k) => [String(k), `${k + 1}`]), value: String(E.prob),
    onChange: (k) => { E.prob = Number(k); E.cut = null; cutSlider.el.max = probs[E.prob].cuts.length - 1; drawEarly(); } });
  const p0 = probs[E.prob];
  cutSlider = slider(ctl, { name: 'Cut the chain of thought after', min: 0, max: p0.cuts.length - 1, step: 1, value: p0.cuts.length - 1,
    format: (v) => cutLabel(probs[E.prob], v), onInput: (v) => { E.cut = v; drawEarly(); } });
  drawEarly();
}
function cutLabel(p, k) {
  const c = p.cuts[k];
  return `${c.tokens.toLocaleString()} of ${p.cot_tokens.toLocaleString()} tokens`;
}
function drawEarly() {
  const probs = early.problems.filter((p) => p.finished);
  const p = probs[E.prob];
  if (E.cut === null || E.cut > p.cuts.length - 1) E.cut = p.cuts.length - 1;
  cutSlider.set(E.cut);
  cutSlider.el.max = p.cuts.length - 1;
  cutSlider.el.dispatchEvent(new Event('change'));
  const c = p.cuts[E.cut];
  const f = finalLetter(p);
  document.getElementById('cot-question').innerHTML = esc(p.question).replace(/\n/g, '<br>');
  const kept = esc(p.cot.slice(0, c.chars)), cut = esc(p.cot.slice(c.chars));
  const tbox = document.getElementById('cot-text');
  tbox.innerHTML = `<span class="kept">${kept}</span><span class="mark" aria-label="cut here"></span><span class="dropped">${cut}</span>`;
  const mk = tbox.querySelector('.mark');
  if (mk) tbox.scrollTop = Math.max(0, mk.offsetTop - tbox.offsetTop - tbox.clientHeight / 2);
  const stage = document.getElementById('cot-chart');
  stage.innerHTML = '';
  const ch = chart(stage, { height: 260, margin: { top: 34, right: 40, bottom: 52, left: 56 },
    x: { domain: [0, 1], ticks: 5, format: '.0%', label: 'share of the chain of thought kept' }, y: { domain: [0, 1], ticks: 5, label: `probability of (${LETTERS[f]})` },
    title: `Probability the model gives its final answer (${LETTERS[f]}) when cut off at each point` });
  const pts = p.cuts.map((q) => [q.tokens / p.cot_tokens, q.p[f]]);
  ch.line('pf', pts, { color: 'ink', width: 1.4 });
  ch.vline('cut', c.tokens / p.cot_tokens, { color: 'quiet' });
  ch.point('now', c.tokens / p.cot_tokens, c.p[f], { color: 'accent', r: 5 });
  const best = argmax(c.p);
  const s = settledAt(p);
  const sPct = Math.round(100 * p.cuts[s].tokens / p.cot_tokens);
  document.getElementById('cot-readout').innerHTML =
    `Made to answer after ${c.tokens.toLocaleString()} of the ${p.cot_tokens.toLocaleString()} tokens of its chain of thought, the model answers <b>(${LETTERS[best]})</b> with probability ${fmt(2)(c.p[best])}. ` +
    `Probabilities: ${c.p.map((v, k) => `(${LETTERS[k]}) ${fmt(2)(v)}`).join(', ')}. ` +
    `With the whole chain of thought it answers (${LETTERS[f]})${LETTERS[f] === p.key ? ', which is correct' : `; the correct answer is (${p.key})`}. ` +
    (s === 0 ? 'Its top choice is already (' + LETTERS[f] + ') before it writes anything.' : `Its top choice stays (${LETTERS[f]}) from ${sPct}% of the way through to the end.`);
}

// ---- Static figure: multiplication with and without written steps ----
function figure() {
  const fig = document.getElementById('fig-mult');
  if (!fig || !mult) return;
  fig.innerHTML = '';
  const d = mult.direct.find((x) => x.a === mult.figure.a && x.b === mult.figure.b);
  const f = mult.figure;
  const ans = (f.output.split('</think>')[1] || '').trim();
  const m = ans.match(/\\boxed\{([\d,\s]+)\}/) || ans.match(/([\d,]{5,})/g);
  const W = Math.min(560, fig.clientWidth || 560), H = 170, L = 150;
  const max = f.tokens;
  const rows = [[`Answer at once`, d.tokens, 'second'], [`With written steps`, f.tokens, 'accent']];
  const svg = d3.select(fig).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
    .attr('role', 'img').attr('aria-label', 'Forward passes used by each way of answering');
  const x = d3.scaleLinear().domain([0, max]).range([0, W - L - 20]);
  rows.forEach(([name, n, col], k) => {
    const y = 30 + k * 62;
    svg.append('text').attr('x', 0).attr('y', y + 14).text(name);
    svg.append('rect').attr('x', L).attr('y', y).attr('height', 18).attr('width', Math.max(2, x(n))).attr('fill', `var(--${col})`);
    svg.append('text').attr('x', L).attr('y', y + 40).attr('class', 'label-quiet').text(`${n.toLocaleString()} forward passes`);
  });
  const note = document.createElement('p');
  note.className = 'small-title';
  note.innerHTML = `${f.a.toLocaleString('en-US')} × ${f.b} = ${f.true.toLocaleString('en-US')}. Answering at once, the model wrote ${esc(d.output)} (${d.correct ? 'right' : 'wrong'}). With its chain of thought on, it wrote ${f.thinking_tokens.toLocaleString()} tokens of working and then answered ${esc(m ? (Array.isArray(m) && !m.index ? m[m.length - 1] : m[1]) : ans.slice(0, 40))}.`;
  fig.prepend(note);
}

(async function start() {
  try {
    [add, early, mult] = await Promise.all(['addition.json', 'early.json', 'mult.json'].map((f) => loadJSON(new URL(`../../data/4-4/${f}`, import.meta.url))));
  } catch (e) {
    document.getElementById('add-readout').textContent = 'The stored results could not be loaded. Reload the page to try again.';
    return;
  }
  addPanel();
  earlyPanel();
  figure();
})();

let resizeT = null;
window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(() => { if (add) { drawAdd(); drawEarly(); figure(); } }, 150); });
