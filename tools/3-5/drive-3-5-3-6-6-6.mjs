// Drives every control of pages 3.5, 3.6, 6.6 and checks for errors and the tasks' outcomes.
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH);
const docs = '/tmp/claude-0/-home-claude/d7157144-aad8-54ff-8e07-90f4ec3c490e/scratchpad/wt-K/docs';
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.woff': 'font/woff' };
const server = http.createServer((req, res) => { const p = path.join(docs, decodeURIComponent(req.url.split('?')[0])); fs.readFile(p, (e, b) => { if (e) { res.writeHead(404); res.end(); return; } res.writeHead(200, { 'content-type': types[path.extname(p)] || 'application/octet-stream' }); res.end(b); }); }).listen(0);
const port = server.address().port;
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM });
let fails = 0; const ok = (l, c, g = '') => { console.log(`${c ? 'ok  ' : 'FAIL'} ${l} ${g}`); if (!c) fails++; };
const pages = process.argv.slice(2);
for (const scheme of ['light', 'dark']) for (const vp of [{ width: 1280, height: 900 }, { width: 375, height: 800 }]) {
  const ctx = await browser.newContext({ viewport: vp, colorScheme: scheme, reducedMotion: 'reduce' });
  const tab = await ctx.newPage(); const errs = [];
  tab.on('pageerror', (e) => errs.push(String(e))); tab.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
  const btn = (group, label) => tab.locator(`[role=group][aria-label="${group}"] button`, { hasText: new RegExp('^' + label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')) }).first().click();
  const txt = async () => tab.locator('#readout').innerText();
  const tag = `${scheme} ${vp.width}`;
  if (pages.includes('3.5')) {
    await tab.goto(`http://localhost:${port}/pages/3-5.html`); await tab.waitForTimeout(800);
    // keyboard: move to head 5.5
    await tab.locator('#grid').focus();
    for (let i = 0; i < 5; i++) await tab.keyboard.press('ArrowDown');
    for (let i = 0; i < 5; i++) await tab.keyboard.press('ArrowRight');
    ok(`${tag} 3.5 keyboard to head 5.5`, (await tab.locator('#big-title').innerText()) === 'Head 5.5');
    ok(`${tag} 3.5 t2 readout means`, /first copy 13\.8, second copy 0\.40/.test(await txt()), await txt());
    // Big canvas: up from last
    await tab.locator('#big').focus(); await tab.keyboard.press('ArrowUp');
    ok(`${tag} 3.5 row moves`, /from position 39/.test(await txt()));
    await btn('Head types', 'Labeled');
    const types = await tab.locator('#types').innerText();
    ok(`${tag} 3.5 labels list`, /Induction heads.*5\.1, 5\.5, 6\.9, 7\.2, 7\.10/.test(types) && /4\.11/.test(types), types.replace(/\n/g, ' | '));
    for (const inp of ['A sentence', 'The Dursleys', 'Random tokens, repeated', 'The Dursleys']) { await btn('Input', inp); await tab.waitForTimeout(100); }
    // Tap second Harry in strip
    const harry = tab.locator('#strip .tk', { hasText: 'Harry' }).nth(1); await harry.click();
    ok(`${tag} 3.5 t4 second Harry -> Potter 0.58`, /largest weight, 0\.58, is on position 19 \(“Potter”\)/.test(await txt()), await txt());
    // click on grid at some places
    const box = await tab.locator('#grid').boundingBox();
    for (const [fx, fy] of [[0.2, 0.2], [0.9, 0.95], [0.5, 0.5], [0.01, 0.01]]) await tab.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
    const bb = await tab.locator('#big').boundingBox(); await tab.mouse.click(bb.x + 5, bb.y + bb.height * 0.3);
    await btn('Head types', 'Not labeled');
    ok(`${tag} 3.5 no errors`, errs.length === 0, errs.join(' | '));
  }
  if (pages.includes('3.6')) {
    errs.length = 0;
    await tab.goto(`http://localhost:${port}/pages/3-6.html`); await tab.waitForTimeout(800);
    await btn('Prompt', 'Colosseum');
    const n = await tab.locator('#lens thead th').count();
    await tab.locator(`#lens tbody tr:nth-child(10) td:nth-of-type(${n - 1})`).click();
    ok(`${tag} 3.6 t1/t2`, /after layer 8: the top prediction is “Rome”, with probability 0\.72.*from 0\.04 .* to 0\.17 .*then to 0\.72 .*MLP made the larger change/.test(await txt()), await txt());
    await tab.keyboard.press('ArrowDown'); await tab.keyboard.press('ArrowDown'); await tab.keyboard.press('ArrowDown');
    ok(`${tag} 3.6 t4 output 0.02`, /after layer 11: the top prediction is “Rome”, with probability 0\.02/.test(await txt()), await txt());
    await btn('Prompt', 'Japan');
    const m = await tab.locator('#lens thead th').count();
    await tab.locator(`#lens tbody tr:nth-child(11) td:nth-of-type(${m - 1})`).click();
    ok(`${tag} 3.6 t3`, /Tokyo.*0\.07.*0\.76.*0\.86.*attention part made the larger change/.test(await txt()), await txt());
    for (const p of ['Eiffel Tower', 'Michael Jordan', 'Days of the week', 'Water']) await btn('Prompt', p);
    await tab.locator('#lens td.sel').focus(); await tab.keyboard.press('ArrowLeft'); await tab.keyboard.press('ArrowUp');
    for (let i = 0; i < 15; i++) await tab.keyboard.press('ArrowUp');
    ok(`${tag} 3.6 embedding row`, /at the embedding.*before any layer/.test(await txt()), await txt());
    ok(`${tag} 3.6 no errors`, errs.length === 0, errs.join(' | '));
  }
  if (pages.includes('6.6')) {
    errs.length = 0;
    await tab.goto(`http://localhost:${port}/pages/6-6.html`); await tab.waitForTimeout(800);
    const setSlider = async (name, v) => { const s = tab.locator('label.ctl', { hasText: name }).locator('input'); await s.evaluate((el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }, v); };
    for (const vec of ['weddings', 'Love']) for (const pr of ['I hate', 'I went']) { await btn('Steering vector', vec); await btn('Prompt', pr); for (const L of [0, 6, 11]) for (const c of [-10, -3, 0, 5, 10]) { await setSlider('Layer', L); await setSlider('Coefficient', c); } }
    for (const s of ['2', '3', '1']) await btn('Sample', s);
    const s = tab.locator('label.ctl', { hasText: 'Coefficient' }).locator('input'); await s.focus(); await tab.keyboard.press('ArrowRight'); await tab.keyboard.press('ArrowLeft');
    ok(`${tag} 6.6 readout`, /Vector/.test(await txt()), await txt());
    ok(`${tag} 6.6 no errors`, errs.length === 0, errs.join(' | '));
  }
  await ctx.close();
}
await browser.close(); server.close();
console.log(fails ? `${fails} FAILED` : 'all ok'); process.exit(fails ? 1 : 0);
