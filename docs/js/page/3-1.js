// Page 3.1: tokenizer playground and a tiny byte-pair encoder.
import { Tiktoken } from '../../vendor/js-tiktoken-1.0.21/lite.js';
import { segmented, button } from '../lib/ui.js';
import { TinyBPE } from '../lib/3-models.js';
import { CORPUS, EXAMPLES } from './3-1-data.js';

const RANKS = { gpt2: '../../vendor/js-tiktoken-1.0.21/gpt2.js', o200k: '../../vendor/js-tiktoken-1.0.21/o200k_base.js' };
const NAMES = { gpt2: 'GPT-2', o200k: 'GPT-4o' };
const cache = {};
function encoder(key) {
  if (!cache[key]) cache[key] = import(RANKS[key]).then((m) => new Tiktoken(m.default));
  return cache[key];
}

// ---- Turning a token's bytes into something readable ----
// Whole UTF-8 characters are shown as text (a space as a quiet dot); bytes that are only part of a
// character are shown in hexadecimal, in italics.
const utf8 = new TextDecoder('utf-8', { fatal: true });
function seqLen(b) { return b < 0x80 ? 1 : b >= 0xc2 && b < 0xe0 ? 2 : b >= 0xe0 && b < 0xf0 ? 3 : b >= 0xf0 && b < 0xf5 ? 4 : 0; }
function segments(bytes) {
  const out = [];
  for (let i = 0; i < bytes.length;) {
    const n = seqLen(bytes[i]);
    let ch = null;
    if (n && i + n <= bytes.length) { try { ch = utf8.decode(bytes.slice(i, i + n)); } catch { ch = null; } }
    if (ch !== null) { out.push({ text: ch }); i += n; } else { out.push({ byte: bytes[i].toString(16).padStart(2, '0') }); i += 1; }
  }
  return out;
}
function chip(parts, id, cls = '') {
  const el = document.createElement('span');
  el.className = 'tok' + (cls ? ' ' + cls : '');
  const t = document.createElement('span');
  t.className = 't';
  for (const p of parts) {
    const s = document.createElement('span');
    if (p.byte !== undefined) { s.className = 'byte'; s.textContent = p.byte; }
    else if (p.text === ' ') { s.className = 'sp'; s.textContent = '·'; }
    else if (p.text === '\n') { s.className = 'sp'; s.textContent = '↵'; }
    else if (p.text === '\t') { s.className = 'sp'; s.textContent = '→'; }
    else s.textContent = p.text;
    t.append(s);
  }
  el.append(t);
  if (id !== null && id !== undefined) {
    const n = document.createElement('span');
    n.className = 'id';
    n.textContent = id;
    el.append(n);
  }
  return el;
}
const isRTL = (s) => { const m = s.match(/[A-Za-z֐-ࣿ]/); return !!m && /[֐-ࣿ]/.test(m[0]); };
const textParts = (s) => [...s].map((c) => ({ text: c }));

function renderTokens(box, enc, text) {
  const ids = enc.encode(text);
  box.replaceChildren(...ids.map((id) => chip(segments(enc.textMap.get(id) || new Uint8Array()), id)));
  box.dir = isRTL(text) ? 'rtl' : 'ltr';
  return ids;
}

// ---- Static figure: the sentence through GPT-2 ----
(async function figure() {
  const fig = document.getElementById('fig-tokens');
  const box = document.createElement('div');
  box.className = 'toks';
  fig.append(box);
  renderTokens(box, await encoder('gpt2'), EXAMPLES[0].text);
})();

// ---- Panel 1: real tokenizers ----
(function real() {
  const state = { tok: 'gpt2', text: EXAMPLES[0].text };
  const ctl = document.getElementById('real-controls');
  const box = document.getElementById('real-toks');
  const out = document.getElementById('real-readout');
  segmented(ctl, { name: 'Tokenizer', options: [['gpt2', 'GPT-2'], ['o200k', 'GPT-4o']], value: state.tok,
    onChange: (k) => { state.tok = k; draw(); } });
  const exWrap = document.createElement('div');
  exWrap.className = 'ctl';
  exWrap.setAttribute('role', 'group');
  exWrap.setAttribute('aria-label', 'Examples');
  exWrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Examples</span></span>';
  const exSeg = document.createElement('span');
  exSeg.className = 'seg';
  exWrap.append(exSeg);
  ctl.append(exWrap);
  for (const ex of EXAMPLES) button(exSeg, ex.label, () => { state.text = ex.text; ta.value = ex.text; draw(); });
  const taWrap = document.createElement('label');
  taWrap.className = 'ctl';
  taWrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Your text</span></span>';
  const ta = document.createElement('textarea');
  ta.maxLength = 2000;
  ta.value = state.text;
  ta.setAttribute('dir', 'auto');
  taWrap.append(ta);
  ctl.append(taWrap);
  ta.addEventListener('input', () => { state.text = ta.value; draw(); });

  let ticket = 0;
  async function draw() {
    const mine = ++ticket;
    const key = state.tok;
    if (!cache[key]) out.textContent = `Loading the ${NAMES[key]} vocabulary…`;
    let enc;
    try { enc = await encoder(key); } catch { out.textContent = 'The vocabulary could not be loaded.'; return; }
    if (mine !== ticket) return; // a later change has already started drawing
    const ids = renderTokens(box, enc, state.text);
    const chars = [...state.text].length;
    out.innerHTML = ids.length
      ? `${NAMES[key]}: <b>${ids.length.toLocaleString()}</b> token${ids.length === 1 ? '' : 's'} for ${chars.toLocaleString()} character${chars === 1 ? '' : 's'}, ${(chars / ids.length).toFixed(1)} characters per token.`
      : 'Type some text to see its tokens.';
  }
  draw();
})();

// ---- Panel 2: build your own ----
(function build() {
  const ctl = document.getElementById('build-controls');
  const box = document.getElementById('build-toks');
  const out = document.getElementById('build-readout');
  const list = document.getElementById('merges');
  let corpus = CORPUS, bpe = new TinyBPE(corpus);
  const btnRow = document.createElement('div');
  btnRow.className = 'seg';
  ctl.append(btnRow);
  const mergeBtn = button(btnRow, 'Merge', () => { bpe.merge(); draw(); }, 'primary');
  button(btnRow, 'Merge 10', () => { for (let i = 0; i < 10; i++) if (!bpe.merge()) break; draw(); });
  button(btnRow, 'Start over', () => { bpe = new TinyBPE(corpus); draw(); });
  const taWrap = document.createElement('label');
  taWrap.className = 'ctl';
  taWrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Corpus</span></span>';
  const ta = document.createElement('textarea');
  ta.maxLength = 600;
  ta.value = corpus;
  taWrap.append(ta);
  ctl.append(taWrap);
  ta.addEventListener('input', () => { corpus = ta.value; bpe = new TinyBPE(corpus); draw(); });
  button(ctl, 'Restore the original corpus', () => { corpus = CORPUS; ta.value = CORPUS; bpe = new TinyBPE(corpus); draw(); });

  function draw() {
    const last = bpe.merges[bpe.merges.length - 1];
    const made = last ? last.a + last.b : null;
    box.replaceChildren(...bpe.tokens().map((t) => chip(textParts(t), null, t === made ? 'new' : '')));
    const next = bpe.pairs()[0];
    const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
    const show = (s) => esc(s.replace(/ /g, '·'));
    const nTok = bpe.tokens().length;
    out.innerHTML = `${bpe.merges.length} merge${bpe.merges.length === 1 ? '' : 's'} so far. The corpus is <b>${nTok}</b> token${nTok === 1 ? '' : 's'} long; the vocabulary has ${bpe.vocabSize()} tokens (${bpe.base} single characters and ${bpe.merges.length} merged). ` +
      (next ? `Next merge: ${show(next.a)} + ${show(next.b)}, which occurs ${next.n} time${next.n === 1 ? '' : 's'}.` : 'No adjacent pairs are left to merge.');
    mergeBtn.disabled = !next;
    list.replaceChildren(...bpe.merges.map((m) => {
      const li = document.createElement('li');
      li.innerHTML = `${show(m.a)} + ${show(m.b)} → ${show(m.a + m.b)} <span class="n">(${m.n})</span>`;
      return li;
    }));
  }
  draw();
})();
