// Page shell: top line, page-end navigation, tasks, check questions,
// self-explanation boxes, progress, and formula rendering.
// A page opts in with <body data-page="2.5"> and loads this module.
import { sections } from './pages.js';

// ---- Storage (per reader, in this browser only; failures are harmless) ----
const KEY = 'asf-v1';
function load() {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
}
function save(state) {
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch { /* private window etc. */ }
}
export const store = {
  get(path, fallback) { const s = load(); return path in s ? s[path] : fallback; },
  set(path, value) { const s = load(); s[path] = value; save(s); },
};

const allPages = sections.flatMap((s) => s.pages.map((p) => ({ ...p, section: s })));
const base = document.body.dataset.base || '../';

function pageHref(p) { return p.href ? base + p.href : null; }

// ---- Top line and page-end navigation ----
function shell() {
  const id = document.body.dataset.page;
  if (!id) return;
  const i = allPages.findIndex((p) => p.id === id);
  const page = allPages[i];
  const article = document.querySelector('article');

  const top = document.createElement('nav');
  top.className = 'topline';
  top.setAttribute('aria-label', 'Site');
  top.innerHTML = `<a href="${base}index.html">AI Safety Fundamentals</a>
    <span>Section ${page.section.n}, ${page.section.title}${page.core ? ' <span class="core-dot" title="Core path"></span>' : ''}</span>`;
  article.before(top);

  const h1 = article.querySelector('h1');
  if (h1 && !h1.querySelector('.pageno')) h1.insertAdjacentHTML('afterbegin', `<span class="pageno">${id}</span>`);

  const builtBefore = allPages.slice(0, i).reverse().find((p) => p.href);
  const builtAfter = allPages.slice(i + 1).find((p) => p.href);
  const nav = document.createElement('nav');
  nav.className = 'pagenav';
  nav.setAttribute('aria-label', 'Previous and next page');
  nav.innerHTML =
    (builtBefore ? `<a class="prev" href="${pageHref(builtBefore)}"><span class="dir">Previous</span>${builtBefore.id} ${builtBefore.title}</a>` : '<span></span>') +
    (builtAfter ? `<a class="next" href="${pageHref(builtAfter)}"><span class="dir">Next</span>${builtAfter.id} ${builtAfter.title}</a>` : `<a class="next" href="${base}index.html"><span class="dir">Next</span>Contents</a>`);
  article.after(nav);
}

// ---- Tasks: the reader predicts before trying, then reveals what to notice ----
// Markup:
// <li class="task" data-id="2.5-t1">
//   <p class="prompt">...</p>
//   <div class="choices" data-options="0.5|0.7|1.0"></div>
//   <div class="after"><p>Now try it ...</p><button class="show">Show what to notice</button>
//     <div class="notice"><p>...</p></div></div>
// </li>
function tasks() {
  document.querySelectorAll('.task').forEach((li) => {
    const id = li.dataset.id;
    const box = li.querySelector('.choices');
    const saved = store.get(`task:${id}`, null);
    if (box) {
      box.innerHTML = '<span class="choices-label">Your prediction:</span>';
      const seg = document.createElement('span');
      seg.className = 'seg';
      box.append(seg);
      box.dataset.options.split('|').forEach((opt) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = opt;
        b.setAttribute('aria-pressed', 'false');
        b.addEventListener('click', () => {
          seg.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
          li.classList.add('predicted');
          store.set(`task:${id}`, { ...(store.get(`task:${id}`, {}) || {}), predicted: opt });
        });
        if (saved && saved.predicted === opt) { b.setAttribute('aria-pressed', 'true'); li.classList.add('predicted'); }
        seg.append(b);
      });
    } else {
      li.classList.add('predicted');
    }
    const show = li.querySelector('button.show');
    if (show) {
      show.addEventListener('click', () => {
        li.classList.add('revealed');
        show.disabled = true;
        store.set(`task:${id}`, { ...(store.get(`task:${id}`, {}) || {}), revealed: true });
      });
      if (saved && saved.revealed) { li.classList.add('revealed'); show.disabled = true; }
    }
  });
}

// ---- Check questions: answer before the answer appears ----
// <div class="check" data-id="2.5-c1" data-correct="1">
//   <p class="q">...</p><div class="options"><button>..</button>...</div>
//   <div class="feedback"><p>...</p></div></div>
function checks() {
  const page = document.body.dataset.page;
  document.querySelectorAll('.check').forEach((el) => {
    const id = el.dataset.id;
    const correct = Number(el.dataset.correct);
    const buttons = [...el.querySelectorAll('.options button')];
    const verdict = document.createElement('p');
    el.querySelector('.feedback').prepend(verdict);
    const answer = (k) => {
      buttons.forEach((b, j) => b.classList.toggle('chosen', j === k));
      el.classList.add('answered');
      verdict.className = k === correct ? 'verdict-right' : 'verdict-wrong';
      verdict.textContent = k === correct ? 'Right.' : `Not quite. The answer is: ${buttons[correct].textContent}`;
      store.set(`check:${id}`, k);
      store.set(`done:${page}`, true);
    };
    buttons.forEach((b, k) => { b.type = 'button'; b.addEventListener('click', () => answer(k)); });
    const prev = store.get(`check:${id}`, null);
    if (prev !== null) answer(prev);
  });
}

// ---- Self-explanation boxes, kept in this browser ----
function explains() {
  document.querySelectorAll('.explain').forEach((el) => {
    const id = el.dataset.id;
    const ta = el.querySelector('textarea');
    const note = el.querySelector('.saved');
    ta.value = store.get(`explain:${id}`, '');
    let t;
    ta.addEventListener('input', () => {
      clearTimeout(t);
      t = setTimeout(() => { store.set(`explain:${id}`, ta.value); if (note) note.textContent = 'Saved in this browser.'; }, 400);
    });
  });
}

function math() {
  if (window.renderMathInElement) {
    window.renderMathInElement(document.body, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }],
      throwOnError: false,
    });
  }
}

export function contents(el) {
  el.innerHTML = sections.map((s) => `
    <h2>${s.n}. ${s.title}</h2>
    <ol>${s.pages.map((p) => {
      const done = store.get(`done:${p.id}`, false);
      const title = p.href ? `<a href="${p.href}">${p.title}</a>` : `<span class="planned">${p.title}</span>`;
      return `<li><span class="no">${p.id}</span><span class="mark">${p.core ? '<span class="core-dot" title="Core path"></span>' : ''}</span>${title}${p.href ? ` <span class="done-dot${done ? ' done' : ''}" title="${done ? 'Done' : 'Not done yet'}"></span>` : ''}</li>`;
    }).join('')}</ol>`).join('');
}

shell();
tasks();
checks();
explains();
if (document.readyState === 'complete') math(); else window.addEventListener('load', math);
