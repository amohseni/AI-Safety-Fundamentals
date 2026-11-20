// Browser test of the page shell (docs/js/site.js): progress, check questions, the core path,
// review blocks, and saved explanations. Needs Playwright (PLAYWRIGHT_PATH) with Chromium.
// Run: node tools/test-shell.mjs
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const root = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'docs');
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.woff2': 'font/woff2', '.ttf': 'font/ttf', '.woff': 'font/woff' };
const server = http.createServer((req, res) => {
  const f = path.join(root, decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(f, (err, buf) => { if (err) { res.writeHead(404); res.end(); return; } res.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); res.end(buf); });
});
await new Promise((r) => server.listen(0, r));
const url = (p) => `http://localhost:${server.address().port}/${p}`;
let fails = 0;
const check = (label, ok, got = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${got !== '' ? `: ${got}` : ''}`); if (!ok) fails++; };

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
const page = await browser.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
const answer = async (sel, right) => {
  const el = page.locator(sel);
  const correct = Number(await el.getAttribute('data-correct'));
  const n = await el.locator('.options button').count();
  await el.locator('.options button').nth(right ? correct : (correct + 1) % n).click();
};
const done = (id) => page.evaluate((i) => !!document.querySelector(`.contents li .no`) && [...document.querySelectorAll('.contents li')].find((li) => li.querySelector('.no').textContent === i).querySelector('.done-dot').classList.contains('done'), id);

// 1. A wrong answer does not mark the page done, and locks the options for this visit.
await page.goto(url('pages/2-5.html'), { waitUntil: 'networkidle' });
const ids = await page.locator('.check').evaluateAll((els) => els.map((e) => e.dataset.id));
check('2.5 has two check questions', ids.length === 2, ids.join(' '));
await answer(`.check[data-id="${ids[0]}"]`, false);
await answer(`.check[data-id="${ids[1]}"]`, true);
check('options lock after an answer', await page.locator(`.check[data-id="${ids[0]}"] .options button:not([disabled])`).count() === 0);
check('wrong answer shows the right one', (await page.locator(`.check[data-id="${ids[0]}"] .verdict-wrong`).textContent()).startsWith('Not quite.'));
await page.goto(url('index.html'), { waitUntil: 'networkidle' });
check('page with one wrong answer is not done', (await done('2.5')) === false);
const dotClass = (id) => page.evaluate((i) => [...document.querySelectorAll('.contents li')].find((li) => li.querySelector('.no').textContent === i).querySelector('.done-dot').className, id);
check('page with one of two checks right shows the half-filled circle', (await dotClass('2.5')) === 'done-dot some', await dotClass('2.5'));
check('page with no check answered shows the empty circle', (await dotClass('2.4')) === 'done-dot', await dotClass('2.4'));

// 2. On the next visit the missed question is asked again, unanswered; the right one stays answered.
await page.goto(url('pages/2-5.html'), { waitUntil: 'networkidle' });
check('missed question is asked again', await page.locator(`.check[data-id="${ids[0]}"] .again`).count() === 1 && !(await page.locator(`.check[data-id="${ids[0]}"]`).evaluate((e) => e.classList.contains('answered'))));
check('correct answer is kept', await page.locator(`.check[data-id="${ids[1]}"]`).evaluate((e) => e.classList.contains('answered')));
await answer(`.check[data-id="${ids[0]}"]`, true);
await page.goto(url('index.html'), { waitUntil: 'networkidle' });
check('page is done once every check is answered correctly', (await done('2.5')) === true);
check('done page shows the filled circle', (await dotClass('2.5')) === 'done-dot done', await dotClass('2.5'));

// 3. A record from before the change (a bare option number) still counts.
await page.evaluate(() => { const s = JSON.parse(localStorage.getItem('asf-v1')); s['check:2.1-c1'] = Number(document.title.length * 0); localStorage.setItem('asf-v1', JSON.stringify(s)); });
await page.goto(url('pages/2-1.html'), { waitUntil: 'networkidle' });
const legacyCorrect = Number(await page.locator('.check').first().getAttribute('data-correct'));
check('old record is read', (await page.locator('.check').first().evaluate((e) => e.classList.contains('answered'))) === (legacyCorrect === 0));

// 4. Core path: contents view and page links.
await page.goto(url('index.html'), { waitUntil: 'networkidle' });
check('all view lists 50 pages', await page.locator('.contents li').count() === 50);
await page.locator('.contents .view [data-view="core"]').click();
check('core view lists 21 pages', await page.locator('.contents li').count() === 21);
check('core view names the next core page', (await page.locator('.core-next').textContent()).includes('1.1'));
await page.reload({ waitUntil: 'networkidle' });
check('core view is remembered', await page.locator('.contents li').count() === 21);
await page.goto(url('pages/2-1.html'), { waitUntil: 'networkidle' });
const rows = await page.locator('.pagenav .row').count();
const coreNext = rows === 2 ? await page.locator('.pagenav .row').nth(1).locator('a.next').getAttribute('href') : '';
check('2.1 links to the next core page, 2.4', coreNext.endsWith('2-4.html'), coreNext);
let at = 'pages/1-1.html', walked = [];
for (let k = 0; k < 30 && at; k++) {
  await page.goto(url(at), { waitUntil: 'domcontentloaded' });
  walked.push(await page.evaluate(() => document.body.dataset.page));
  const r = page.locator('.pagenav .row');
  const nextCore = (await r.count()) === 2 ? r.nth(1).locator('a.next') : r.nth(0).locator('a.next');
  const href = (await nextCore.count()) ? await nextCore.getAttribute('href') : null;
  const target = href && href.match(/(\d-\d)\.html$/);
  at = target && (await r.count() === 2 || (await page.locator('.pagenav .row a.next[data-core]').count())) ? `pages/${target[1]}.html` : null;
}
check('following the core links from 1.1 visits the 21 core pages in order', walked.length === 21 && walked[6] === '3.3' && walked[20] === '8.5', walked.join(' '));

// 5. Review block at the start of a section.
await page.goto(url('pages/3-1.html'), { waitUntil: 'networkidle' });
const rv = await page.locator('.review .check').evaluateAll((els) => els.map((e) => e.dataset.id));
check('3.1 reviews questions already answered in sections 1 and 2', rv.length >= 1 && rv.length <= 3 && rv.every((i) => /^[12]\./.test(i)), rv.join(' '));
check('review questions start unanswered', await page.locator('.review .check.answered').count() === 0);
check('review math is typeset', await page.locator('.review').evaluate((e) => !/\\\(/.test(e.textContent)));
await page.goto(url('pages/3-2.html'), { waitUntil: 'networkidle' });
check('no review on a page in the middle of a section', await page.locator('.review').count() === 0);
await page.goto(url('pages/3-3.html'), { waitUntil: 'networkidle' });
check('review on the first core page of a section (3.3)', await page.locator('.review .check').count() >= 1);
const fresh = await browser.newPage();
await fresh.goto(url('pages/3-1.html'), { waitUntil: 'networkidle' });
check('no review for a reader who has answered nothing', await fresh.locator('.review').count() === 0);
await fresh.close();

// 6. An explanation is saved even if the reader leaves at once.
await page.goto(url('pages/2-5.html'), { waitUntil: 'networkidle' });
await page.locator('.explain textarea').fill('proper scoring rules reward honest reports');
await page.goto(url('pages/2-6.html'), { waitUntil: 'domcontentloaded' });
await page.goto(url('pages/2-5.html'), { waitUntil: 'networkidle' });
check('explanation survives leaving immediately', (await page.locator('.explain textarea').inputValue()) === 'proper scoring rules reward honest reports');

// 7. Page 5.1: the drawn target can be changed from the keyboard.
await page.goto(url('pages/5-1.html'), { waitUntil: 'networkidle' });
const hidden0 = await page.locator('#h-controls input[type=range]').evaluateAll((els) => els.filter((e) => e.closest('[hidden]')).length);
check('5.1 drawing sliders are hidden until Draw your own is chosen', hidden0 === 2, hidden0);
await page.locator('#h-controls button', { hasText: 'Draw your own' }).click();
const before = await page.locator('#h-readout').textContent();
const hs = page.locator('#h-controls label.ctl', { hasText: 'Height at that position' }).locator('input');
await hs.focus();
for (let k = 0; k < 10; k++) await page.keyboard.press('ArrowRight');
check('5.1 arrow keys on the height slider change the target', (await page.locator('#h-readout').textContent()) !== before);

// 8. Transfer questions: at the end of a section's last page and of its last core page.
for (const [f, want] of [['2-7', 1], ['2-8', 1], ['2-6', 0], ['4-3', 1], ['4-4', 1], ['9-1', 0], ['1-1', 1]]) {
  await page.goto(url(`pages/${f}.html`), { waitUntil: 'networkidle' });
  check(`transfer question on ${f}: ${want}`, await page.locator('.check.transfer').count() === want);
}
await page.goto(url('pages/4-3.html'), { waitUntil: 'networkidle' });
check('transfer math is typeset', await page.locator('.check.transfer').evaluate((e) => !/\\\(/.test(e.textContent) && !!e.querySelector('.katex')));
await answer('.check.transfer', true);
await page.goto(url('pages/4-4.html'), { waitUntil: 'networkidle' });
check('a transfer answer is kept across the two pages that show it', await page.locator('.check.transfer').evaluate((e) => e.classList.contains('answered')));
const tf = JSON.parse(fs.readFileSync(path.join(root, 'data', 'transfer.json'), 'utf8'));
check('eight transfer questions, each with a valid answer index', Object.keys(tf).join('') === '12345678' && Object.values(tf).every((t) => t.correct >= 0 && t.correct < t.options.length && t.options.length === 3));

check('no console errors', errors.length === 0, errors.slice(0, 3).join(' | '));
await browser.close();
server.close();
console.log(fails ? `${fails} FAILED` : 'all ok');
process.exit(fails ? 1 : 0);
