// Display of stored model outputs (pages 1.6, 4.1, 4.4).
// Model outputs are stored verbatim. formatted() applies the Markdown the model wrote
// (bold, italics, headings, lists, rules, math) without changing any word;
// raw() shows the exact characters. Headings are shown in bold at the one text size.

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function inline(s) {
  // Inline math the way these models write it, "$ x $", with a space inside each dollar sign.
  // Prices ("$25") have no space after the dollar sign and are left as text.
  let h = esc(s).replace(/\$ ([^$\n]+?) \$/g, '\\($1\\)');
  h = h.replace(/`([^`]+)`/g, '<code>$1</code>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  h = h.replace(/(^|[\s(])\*(?!\s)([^*]+?)\*(?=[\s.,;:!?)]|$)/g, '$1<i>$2</i>');
  return h;
}

// Returns HTML. Math ($$ ... $$, \[ ... \], and "$ x $") is kept as text with its delimiters,
// for KaTeX auto-render to typeset afterwards; a dollar sign followed by a digit is left alone (prices).
export function formatted(text) {
  const lines = text.replace(/\r/g, '').split('\n');
  const out = [];
  let para = [], list = null, math = null;
  const flushPara = () => { if (para.length) { out.push(`<p>${para.map(inline).join('<br>')}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.tag}>${list.items.map((x) => `<li>${inline(x)}</li>`).join('')}</${list.tag}>`); list = null; } };
  for (const raw of lines) {
    const line = raw.trimEnd();
    if (math !== null) {
      math.push(line);
      if (/(\$\$|\\\])\s*$/.test(line)) { out.push(`<div class="m">${esc(math.join('\n'))}</div>`); math = null; }
      continue;
    }
    const t = line.trim();
    if (/^(\$\$|\\\[)/.test(t)) {
      flushPara(); flushList();
      const closes = t.length > 2 && /(\$\$|\\\])$/.test(t.slice(2));
      if (closes) out.push(`<div class="m">${esc(t)}</div>`); else math = [t];
      continue;
    }
    if (t === '') { flushPara(); flushList(); continue; }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(t)) { flushPara(); flushList(); out.push('<hr>'); continue; }
    const hd = t.match(/^#{1,6}\s+(.*)$/);
    if (hd) { flushPara(); flushList(); out.push(`<p><b>${inline(hd[1].replace(/^\*\*(.*)\*\*$/, '$1'))}</b></p>`); continue; }
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const tag = ul ? 'ul' : 'ol';
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push(ul ? ul[1] : ol[2]);
      continue;
    }
    if (list && /^\s{2,}/.test(raw)) { list.items[list.items.length - 1] += '\n' + t; continue; }
    flushList();
    para.push(t);
  }
  if (math) out.push(`<div class="m">${esc(math.join('\n'))}</div>`);
  flushPara(); flushList();
  return out.join('');
}

export function raw(text) {
  return `<pre class="raw">${esc(text)}</pre>`;
}

export function typeset(el) {
  if (window.renderMathInElement) {
    window.renderMathInElement(el, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
      throwOnError: false, ignoredClasses: ['raw'],
    });
  }
}

// Splits a reasoning-mode output into its chain of thought and its final answer.
export function splitThinking(text) {
  const m = text.match(/^\s*<think>\n?([\s\S]*?)<\/think>\s*([\s\S]*)$/);
  if (!m) {
    const open = text.match(/^\s*<think>\n?([\s\S]*)$/);
    return open ? { thought: open[1], answer: '', finished: false } : { thought: '', answer: text, finished: true };
  }
  return { thought: m[1].trim(), answer: m[2].trim(), finished: true };
}

// Loads a JSON file relative to the page module that asks for it.
export async function loadJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
