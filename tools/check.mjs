// Browser check for built pages: console errors, layout at desktop and phone width, screenshots.
// Usage: node tools/check.mjs [page ids...]   (default: all built pages and the contents page)
// Needs Playwright; serves docs/ on a local port.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const docs = path.join(root, 'docs');
const outDir = process.env.SHOTS || path.join(root, 'tools', 'shots');
fs.mkdirSync(outDir, { recursive: true });

const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.woff': 'font/woff', '.ttf': 'font/ttf', '.woff2': 'font/woff2', '.svg': 'image/svg+xml', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg' };
const server = http.createServer((req, res) => {
  const p = path.join(docs, decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(p.endsWith('/') ? p + 'index.html' : p, (err, buf) => {
    if (err) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'content-type': types[path.extname(p)] || 'application/octet-stream' });
    res.end(buf);
  });
}).listen(0);
const port = server.address().port;

const ids = process.argv.slice(2);
const pages = ids.length ? ids.map((id) => `pages/${id.replace('.', '-')}.html`) : ['index.html', ...fs.readdirSync(path.join(docs, 'pages')).map((f) => `pages/${f}`)];

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM || undefined });
let failures = 0;
for (const page of pages) {
  for (const [label, viewport] of [['desktop', { width: 1280, height: 900 }], ['phone', { width: 375, height: 800 }]]) {
    const ctx = await browser.newContext({ viewport, deviceScaleFactor: 1 });
    const tab = await ctx.newPage();
    const errors = [];
    tab.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
    tab.on('pageerror', (e) => errors.push(String(e)));
    await tab.goto(`http://localhost:${port}/${page}`, { waitUntil: 'load' });
    await tab.waitForTimeout(1500);
    const overflow = await tab.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    const shot = path.join(outDir, `${page.replace(/[\/.]/g, '_')}-${label}.png`);
    await tab.screenshot({ path: shot, fullPage: true });
    const ok = errors.length === 0 && overflow <= 1;
    if (!ok) failures++;
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${page} ${label}: errors=${errors.length} horizontal-overflow=${overflow}px ${errors.slice(0, 3).join(' | ')}`);
    await ctx.close();
  }
}
await browser.close();
server.close();
process.exit(failures ? 1 : 0);
