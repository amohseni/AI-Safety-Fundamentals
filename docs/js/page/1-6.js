// Page 1.6: the same prompt given to a base model, a chat model, and a reasoning model.
// All outputs are stored model outputs (data/1-6/outputs.json); nothing is generated in the browser.
import { segmented } from '../lib/ui.js';
import { formatted, raw, typeset, splitThinking, loadJSON } from '../lib/s4m-text.js';

const LABELS = ['Capital', 'Letter count', 'Haiku', 'Sleep tips', 'Discount', 'Bat and ball', 'Einstein', 'Mars'];
const MODELS = [
  { key: 'base', name: 'Base model', sub: 'Qwen3-0.6B-Base, given the prompt as plain text' },
  { key: 'chat', name: 'Chat model', sub: 'Qwen3-0.6B, thinking off' },
  { key: 'reasoning', name: 'Reasoning model', sub: 'Qwen3-0.6B, thinking on' },
];
const state = { prompt: 0, view: 'formatted', thought: 'shown' };
let data = null;

const show = (text) => (state.view === 'raw' ? raw(text) : formatted(text));
const limitNote = (o, max) => (o.stopped ? '' : ` It reached the limit of ${max} tokens and was cut off there.`);

function draw() {
  if (!data) return;
  const i = state.prompt;
  const p = data.prompts[i];
  document.getElementById('prompt-text').innerHTML = `Prompt (${p.kind}): <b>${p.text.replace(/</g, '&lt;')}</b>`;
  const box = document.getElementById('outputs');
  box.innerHTML = '';
  const counts = [];
  for (const m of MODELS) {
    const o = data.outputs[`${m.key}:${i}`];
    const col = document.createElement('div');
    col.innerHTML = `<p class="who">${m.name}<br>${m.sub}</p>`;
    const body = document.createElement('div');
    body.className = 'out';
    const max = data.settings[m.key].max_new_tokens;
    if (!o) {
      body.innerHTML = '<p>No output stored.</p>';
    } else if (m.key === 'reasoning') {
      const { thought, answer, finished } = splitThinking(o.text);
      const tTok = o.thinking_tokens;
      const tLine = `<p class="meta">Chain of thought${tTok ? `, ${tTok.toLocaleString()} tokens` : ''}${state.thought === 'hidden' ? ' (hidden)' : ''}:</p>`;
      body.innerHTML = tLine + (state.thought === 'shown' ? `<div class="thought" tabindex="0" aria-label="Chain of thought">${show(thought)}</div>` : '') +
        `<p class="meta">Answer:</p>${finished ? show(answer) : '<p>(none: the chain of thought did not finish)</p>'}`;
      counts.push(`reasoning ${o.tokens.toLocaleString()}${tTok ? ` (${tTok.toLocaleString()} of them in the chain of thought)` : ''}`);
    } else {
      body.innerHTML = show(o.text);
      counts.push(`${m.key} ${o.tokens.toLocaleString()}`);
    }
    col.append(body);
    if (o) col.insertAdjacentHTML('beforeend', `<p class="meta">${o.tokens.toLocaleString()} tokens written.${limitNote(o, max)}</p>`);
    box.append(col);
  }
  typeset(box);
  document.getElementById('readout').textContent = `Tokens written: ${counts.join(', ')}. Each token is one pass through the network.`;
}

(async function start() {
  const menu = document.getElementById('prompt-menu');
  segmented(menu, { name: 'Prompt', options: LABELS.map((l, k) => [String(k), l]), value: '0', onChange: (k) => { state.prompt = Number(k); draw(); } });
  const vc = document.getElementById('view-controls');
  segmented(vc, { name: 'Text shown', options: [['formatted', 'Formatted'], ['raw', 'Exactly as written']], value: state.view, onChange: (k) => { state.view = k; draw(); } });
  segmented(vc, { name: 'Chain of thought', options: [['shown', 'Shown'], ['hidden', 'Hidden']], value: state.thought, onChange: (k) => { state.thought = k; draw(); } });
  document.getElementById('readout').textContent = 'Loading the stored outputs…';
  try {
    data = await loadJSON(new URL('../../data/1-6/outputs.json', import.meta.url));
    draw();
    figure();
  } catch (e) {
    document.getElementById('readout').textContent = 'The stored outputs could not be loaded. Reload the page to try again.';
  }
})();

// Static figure: the first prompt, the three outputs side by side.
function figure() {
  const fig = document.getElementById('fig-three');
  if (!fig) return;
  const box = document.createElement('div');
  box.className = 'three';
  for (const m of MODELS) {
    const o = data.outputs[`${m.key}:0`];
    const d = document.createElement('div');
    let html;
    if (m.key === 'reasoning') {
      const { thought, answer } = splitThinking(o.text);
      html = `<div class="thought">${formatted(thought)}</div>${formatted(answer)}`;
    } else html = formatted(o.text);
    d.innerHTML = `<p class="who">${m.name}</p><div class="out">${html}</div>`;
    box.append(d);
  }
  fig.append(box);
  typeset(fig);
}
