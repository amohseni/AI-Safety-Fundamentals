// Page 4.1: the same prompt given to four checkpoints of OLMo 2 1B, one after each training stage.
// All outputs and probabilities are stored (data/4-1/outputs.json, from tools/4-1/generate.py).
import { chart } from '../lib/chart.js';
import { segmented, slider, fmt } from '../lib/ui.js';
import { formatted, raw, typeset, loadJSON } from '../lib/s4m-text.js';

const CK = [
  { key: 'base', name: 'Base', sub: 'after pretraining' },
  { key: 'sft', name: 'SFT', sub: 'after supervised fine-tuning' },
  { key: 'dpo', name: 'DPO', sub: 'after preference training' },
  { key: 'instruct', name: 'Instruct', sub: 'after reinforcement learning on checkable tasks' },
];
const LABELS = ['Factual question', 'Request to refuse', 'False premise', 'Math problem', 'Ambiguous instruction', 'Opinion'];
const state = { prompt: 0, view: 'formatted', fact: 0 };
let data = null;
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const shown = (t) => JSON.stringify(t).slice(1, -1).replace(/\\n/g, '↵');

function drawOutputs() {
  const i = state.prompt;
  document.getElementById('prompt-text').innerHTML = `Prompt: <b>${esc(data.prompts[i].text)}</b>`;
  const box = document.getElementById('outputs');
  box.innerHTML = '';
  for (const c of CK) {
    const o = data.outputs[`${c.key}:${i}`];
    const col = document.createElement('div');
    col.innerHTML = `<p class="who">${c.name}<br>${c.sub}</p>`;
    const body = document.createElement('div');
    body.className = 'out';
    body.innerHTML = o ? (state.view === 'raw' ? raw(o.text) : formatted(o.text)) : '<p>No output stored.</p>';
    col.append(body);
    if (o) col.insertAdjacentHTML('beforeend', `<p class="meta">${o.tokens} tokens${o.stopped ? '' : `, cut off at the limit of ${data.sampling.max_new_tokens}`}.</p>`);
    box.append(col);
  }
  typeset(box);
  // The probability each checkpoint gives to the final model's first token.
  const stage = document.getElementById('first-chart');
  stage.innerHTML = '';
  const target = data.first_token[`instruct:${i}`].target;
  const vals = CK.map((c) => data.first_token[`${c.key}:${i}`]?.p_target ?? NaN);
  const W = Math.min(520, stage.clientWidth || 520), rowH = 34, L = 92, H = CK.length * rowH + 40;
  const svg = d3.select(stage).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`)
    .attr('role', 'img').attr('aria-label', `Probability each checkpoint gives to starting its response with ${shown(target)}`);
  const x = d3.scaleLinear().domain([0, 1]).range([0, W - L - 60]);
  svg.append('text').attr('x', 0).attr('y', 16).attr('class', 'label-quiet').text(`probability of starting with “${shown(target)}”`);
  CK.forEach((c, k) => {
    const y = 30 + k * rowH;
    svg.append('text').attr('x', 0).attr('y', y + 15).text(c.name);
    svg.append('line').attr('x1', L).attr('x2', L + x(1)).attr('y1', y + 10).attr('y2', y + 10).attr('stroke', 'var(--rule)');
    svg.append('rect').attr('x', L).attr('y', y + 2).attr('height', 16).attr('width', Math.max(1, x(vals[k]))).attr('fill', c.key === 'base' ? 'var(--quiet)' : 'var(--accent)');
    svg.append('text').attr('x', L + Math.max(1, x(vals[k])) + 6).attr('y', y + 15).text(fmt(vals[k] < 0.01 ? 4 : 2)(vals[k]));
  });
  const tops = CK.map((c) => {
    const t = data.first_token[`${c.key}:${i}`].top[0];
    return `${c.name} “${esc(shown(t[0]))}” (${fmt(2)(t[1])})`;
  });
  document.getElementById('readout').innerHTML = `The final model began its response with “${esc(shown(target))}”. The most likely first token of each checkpoint: ${tops.join(', ')}.`;
}

function drawFacts() {
  const stage = document.getElementById('facts-chart');
  stage.innerHTML = '';
  const P = CK.map((c) => data.fact_probs[c.key]);
  const nF = data.facts.length;
  const c = chart(stage, { height: 320, margin: { top: 34, right: 80, bottom: 52, left: 56 },
    x: { domain: [-0.15, 3.15], ticks: 4, format: 'd' }, y: { domain: [0, 1], ticks: 5, label: 'probability of the true answer' },
    title: 'Probability each checkpoint gives to the true answer of each fact' });
  // Name the checkpoints on the horizontal axis.
  c.g.select('.axis').selectAll('.tick').each(function (d) { d3.select(this).style('display', Number.isInteger(d) && d >= 0 && d <= 3 ? null : 'none'); })
    .select('text').text((d) => (CK[d] ? CK[d].name : ''));
  for (let f = 0; f < nF; f++) {
    if (f === state.fact) continue;
    c.line(`f${f}`, P.map((p, k) => [k, p[f][0]]), { color: 'rule', width: 1 });
  }
  const mean = P.map((p, k) => [k, d3.mean(p, (r) => r[0])]);
  c.line('mean', mean, { color: 'ink', width: 2, label: 'average' });
  c.line('sel', P.map((p, k) => [k, p[state.fact][0]]), { color: 'accent', width: 2.2 });
  P.forEach((p, k) => c.point(`s${k}`, k, p[state.fact][0], { color: 'accent', r: 3.5 }));
  const fact = data.facts[state.fact];
  const right = P.map((p) => p.filter((r) => r[0] === Math.max(...r)).length);
  document.getElementById('facts-readout').innerHTML =
    `<span class="c-accent">&ldquo;${esc(fact.stem)} ___&rdquo;</span> True answer: ${esc(fact.options[0])}; the other options were ${fact.options.slice(1).map(esc).join(', ')}. ` +
    `Probability of the true answer: ${CK.map((ck, k) => `${ck.name} ${fmt(2)(P[k][state.fact][0])}`).join(', ')}. ` +
    `Facts where the true answer gets the highest probability, out of ${nF}: ${CK.map((ck, k) => `${ck.name} ${right[k]}`).join(', ')}. ` +
    `Average probability of the true answer: ${CK.map((ck, k) => `${ck.name} ${fmt(2)(mean[k][1])}`).join(', ')}.`;
}

(async function start() {
  const menu = document.getElementById('prompt-menu');
  segmented(menu, { name: 'Prompt', options: LABELS.map((l, k) => [String(k), l]), value: '0', onChange: (k) => { state.prompt = Number(k); drawOutputs(); } });
  segmented(document.getElementById('view-controls'), { name: 'Text shown', options: [['formatted', 'Formatted'], ['raw', 'Exactly as written']], value: state.view,
    onChange: (k) => { state.view = k; drawOutputs(); } });
  document.getElementById('readout').textContent = 'Loading the stored outputs…';
  try {
    data = await loadJSON(new URL('../../data/4-1/outputs.json', import.meta.url));
  } catch (e) {
    document.getElementById('readout').textContent = 'The stored outputs could not be loaded. Reload the page to try again.';
    return;
  }
  slider(document.getElementById('facts-controls'), { name: 'Fact', min: 1, max: data.facts.length, step: 1, value: 1,
    format: (v) => `${v} of ${data.facts.length}`, onInput: (v) => { state.fact = v - 1; drawFacts(); } });
  drawOutputs();
  drawFacts();
})();

let resizeT = null;
window.addEventListener('resize', () => { clearTimeout(resizeT); resizeT = setTimeout(() => { if (data) { drawOutputs(); drawFacts(); } }, 150); });
