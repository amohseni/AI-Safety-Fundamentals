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

// A check's record is { k, right, t }: the option chosen, whether it was correct, and when.
// (Records written before 2026-11 are a bare option number.)
function checkRecord(id, correct) {
  const r = store.get(`check:${id}`, null);
  if (r === null) return null;
  return typeof r === 'number' ? { k: r, right: r === correct, t: 0 } : r;
}
// A page is done when every one of its check questions has been answered correctly.
export function pageDone(p) {
  return !!(p.checks && p.checks.length) && p.checks.every(([id, correct]) => { const r = checkRecord(id, correct); return r && r.right; });
}
const corePages = allPages.filter((p) => p.core && p.href);
const dot = (p) => (p.core ? '<span class="core-dot" title="Core path"></span> ' : '');
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
  top.innerHTML = `<a href="${base}index.html">A Primer on AI</a>
    <span>Section ${page.section.n}, ${page.section.title}${page.core ? ' <span class="core-dot" title="Core path"></span>' : ''}</span>`;
  article.before(top);

  const h1 = article.querySelector('h1');
  if (h1 && !h1.querySelector('.pageno')) h1.insertAdjacentHTML('afterbegin', `<span class="pageno">${id}</span>`);

  const builtBefore = allPages.slice(0, i).reverse().find((p) => p.href);
  const builtAfter = allPages.slice(i + 1).find((p) => p.href);
  const coreBefore = allPages.slice(0, i).reverse().find((p) => p.href && p.core);
  const coreAfter = allPages.slice(i + 1).find((p) => p.href && p.core);
  const link = (p, cls, dir) => (p ? `<a class="${cls}" href="${pageHref(p)}"><span class="dir">${dir}</span>${dot(p)}${p.id} ${p.title}</a>` : '<span></span>');
  const nav = document.createElement('nav');
  nav.className = 'pagenav';
  nav.setAttribute('aria-label', 'Previous and next page');
  let html = `<div class="row">${link(builtBefore, 'prev', 'Previous')}` +
    (builtAfter ? link(builtAfter, 'next', 'Next') : `<a class="next" href="${base}index.html"><span class="dir">Next</span>Contents</a>`) + '</div>';
  // A second row for readers following the core path, shown when it leads somewhere else.
  if ((coreBefore || coreAfter) && (coreBefore !== builtBefore || coreAfter !== builtAfter)) {
    html += `<div class="row">${link(coreBefore, 'prev', 'Previous on the core path')}${link(coreAfter, 'next', 'Next on the core path')}</div>`;
  }
  nav.innerHTML = html;
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
function wireCheck(el, { review = false } = {}) {
  const id = el.dataset.id;
  const correct = Number(el.dataset.correct);
  const buttons = [...el.querySelectorAll('.options button')];
  const verdict = document.createElement('p');
  el.querySelector('.feedback').prepend(verdict);
  // The first answer in a visit is the one that counts, so the options lock once one is chosen.
  const show = (k) => {
    buttons.forEach((b, j) => { b.classList.toggle('chosen', j === k); b.disabled = true; });
    el.classList.add('answered');
    verdict.className = k === correct ? 'verdict-right' : 'verdict-wrong';
    verdict.textContent = k === correct ? 'Right.' : `Not quite. The answer is: ${buttons[correct].textContent}`;
  };
  buttons.forEach((b, k) => {
    b.type = 'button';
    b.addEventListener('click', () => { show(k); store.set(`check:${id}`, { k, right: k === correct, t: Date.now() }); });
  });
  const prev = checkRecord(id, correct);
  if (review || !prev) return;
  if (prev.right) { show(prev.k); return; }
  // Answered wrongly on an earlier visit: ask again, without showing the answer.
  const again = document.createElement('p');
  again.className = 'again';
  again.textContent = 'You missed this question last time. Try it again.';
  el.querySelector('.q').after(again);
}
function checks() {
  document.querySelectorAll('.check').forEach((el) => wireCheck(el));
}

// ---- Review: questions from earlier sections return at the start of a section ----
// Shown on the first page of a section, and on its first core page for readers who skipped the first page. Up to three questions the
// reader has already answered: those answered wrongly first, then those answered longest ago.
async function review() {
  const id = document.body.dataset.page;
  const page = allPages.find((p) => p.id === id);
  if (!page || page.section.n < 2) return;
  const inSection = page.section.pages.filter((p) => p.href);
  const firstCore = inSection.find((p) => p.core);
  if (id !== inSection[0].id && !(firstCore && id === firstCore.id)) return;
  // A reader who already did this section's review on its first page is not asked again on its first core page.
  const doneKey = `reviewed:${page.section.n}`;
  if (id !== inSection[0].id && store.get(doneKey, false)) return;
  const earlier = allPages.filter((p) => p.section.n < page.section.n && p.section.n > 0);
  const seen = earlier.flatMap((p) => (p.checks || []).map(([cid, correct]) => ({ cid, p, r: checkRecord(cid, correct) }))).filter((x) => x.r);
  if (!seen.length) return;
  let bank;
  try { bank = await (await fetch(`${base}data/checks.json`)).json(); } catch { return; }
  const pick = seen.filter((x) => bank[x.cid])
    .sort((a, b) => (a.r.right - b.r.right) || (a.r.t - b.r.t)).slice(0, 3);
  if (!pick.length) return;
  const box = document.createElement('div');
  box.className = 'review';
  box.innerHTML = '<p class="label">Review. These questions are from earlier sections. Answer them from memory before you start this one.</p>' +
    pick.map(({ cid, p }) => { const c = bank[cid]; return `
    <div class="check" data-id="${cid}" data-correct="${c.correct}">
      <p class="label">From <a href="${pageHref(p)}">${p.id} ${p.title}</a></p>
      <p class="q">${c.q}</p>
      <div class="options seg">${c.options.map((o) => `<button>${o}</button>`).join('')}</div>
      <div class="feedback"><p>${c.feedback}</p></div>
    </div>`; }).join('');
  document.querySelector('article h1').after(box);
  box.querySelectorAll('.check').forEach((el) => wireCheck(el, { review: true }));
  box.addEventListener('click', (ev) => { if (ev.target.closest('.options button')) store.set(doneKey, true); });
  math(box);
}

// ---- Self-explanation boxes, kept in this browser ----
function explains() {
  document.querySelectorAll('.explain').forEach((el) => {
    const id = el.dataset.id;
    const ta = el.querySelector('textarea');
    const note = el.querySelector('.saved');
    ta.value = store.get(`explain:${id}`, '');
    // Saved on every keystroke, so nothing is lost when the reader leaves the page at once.
    ta.addEventListener('input', () => { store.set(`explain:${id}`, ta.value); if (note) note.textContent = 'Saved in this browser.'; });
  });
}

// ---- Transfer question: one per section, on a case the section did not cover ----
// Shown at the end of the section's last page and of its last core page. The questions are in
// docs/data/transfer.json and rely only on the section's core pages. They do not count toward "done".
async function transfer() {
  const id = document.body.dataset.page;
  const page = allPages.find((p) => p.id === id);
  if (!page) return;
  const inSection = page.section.pages.filter((p) => p.href);
  const lastCore = inSection.filter((p) => p.core).pop();
  if (id !== inSection[inSection.length - 1].id && !(lastCore && id === lastCore.id)) return;
  let all;
  try { all = await (await fetch(`${base}data/transfer.json`)).json(); } catch { return; }
  const t = all[String(page.section.n)];
  if (!t) return;
  const box = document.createElement('div');
  box.className = 'check transfer';
  box.dataset.id = `T${page.section.n}`;
  box.dataset.correct = t.correct;
  box.innerHTML = `<p class="label">Transfer question for Section ${page.section.n}. It applies the section&rsquo;s ideas to a case the section did not cover.</p>
    <p class="q">${t.q}</p>
    <div class="options seg">${t.options.map((o) => `<button>${o}</button>`).join('')}</div>
    <div class="feedback"><p>${t.feedback}</p></div>`;
  document.querySelector('article').append(box);
  wireCheck(box);
  math(box);
}

// Number sidenotes in document order. (Tufte CSS uses a CSS counter, which current
// Chrome does not carry across paragraphs, so every note showed "1".)
function sidenotes() {
  let n = 0;
  document.querySelectorAll('label.sidenote-number').forEach((label) => {
    n += 1;
    label.dataset.n = n;
    const input = document.getElementById(label.htmlFor);
    const note = input && input.nextElementSibling;
    if (note && note.classList.contains('sidenote')) note.dataset.n = n;
  });
}

function math(root = document.body) {
  if (window.renderMathInElement) {
    window.renderMathInElement(root, {
      delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }],
      throwOnError: false,
    });
  }
}

export function contents(el) {
  const view = store.get('view', 'all') === 'core' ? 'core' : 'all';
  const next = corePages.find((p) => !pageDone(p));
  const list = sections.map((s) => {
    const pages = s.pages.filter((p) => view === 'all' || p.core);
    if (!pages.length) return '';
    return `
    <h2>${s.n}. ${s.title}</h2>
    <ol>${pages.map((p) => {
      const done = pageDone(p);
      const title = p.href ? `<a href="${p.href}">${p.title}</a>` : `<span class="planned">${p.title}</span>`;
      return `<li><span class="no">${p.id}</span><span class="mark">${p.core ? '<span class="core-dot" title="Core path"></span>' : ''}</span>${title}${p.href ? ` <span class="done-dot${done ? ' done' : ''}" title="${done ? 'Done' : 'Not done yet'}"></span>` : ''}</li>`;
    }).join('')}</ol>`; }).join('');
  el.innerHTML = `<div class="seg view" role="group" aria-label="Pages shown">
      <button type="button" data-view="all" aria-pressed="${view === 'all'}">All ${allPages.length} pages</button>
      <button type="button" data-view="core" aria-pressed="${view === 'core'}">Core path, ${corePages.length} pages</button>
    </div>` +
    (view === 'core' ? `<p class="core-next">${next ? `Next on the core path: <a href="${next.href}">${next.id} ${next.title}</a>` : 'You have finished the core path.'}</p>` : '') + list;
  el.querySelectorAll('.view button').forEach((b) => b.addEventListener('click', () => { store.set('view', b.dataset.view); contents(el); el.querySelector(`.view [data-view="${b.dataset.view}"]`).focus(); }));
}

shell();
sidenotes();
tasks();
checks();
explains();
review();
transfer();
if (document.readyState === 'complete') math(); else window.addEventListener('load', () => math());
