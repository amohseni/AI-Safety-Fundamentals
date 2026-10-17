// Writes docs/data/sitemap.json, the data behind the site map on page 0.1.
// Nodes: every page in specs/, with the question it answers (specs/overview.md, page index) and, for
// built pages, the takeaway from docs/pages/<id>.html. Edges: "uses an idea from", read from the
// "Links" line of each spec: "Back to Q" on page P gives the edge Q -> P; "forward to R" gives P -> R.
// Page 9.1 links "back to every section"; its edges come from the Pages column of its table instead.
// Running examples: the table in specs/overview.md.
// Run after adding a page or changing a Links line: node tools/build-sitemap.mjs
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const specs = path.join(root, 'specs');
const overview = fs.readFileSync(path.join(specs, 'overview.md'), 'utf8');

// All pages in reading order.
const pages = [];
for (const dir of fs.readdirSync(specs).filter((d) => /^\d-/.test(d)).sort()) {
  for (const f of fs.readdirSync(path.join(specs, dir)).filter((f) => /^\d\.\d/.test(f)).sort((a, b) => parseFloat(a) - parseFloat(b))) {
    const text = fs.readFileSync(path.join(specs, dir, f), 'utf8');
    const id = text.match(/^##\s+(\d\.\d)\s+(.+)$/m)[1];
    const links = (text.match(/^\*\*Links\.\*\*(.*)$/m) || [, ''])[1];
    // Page 9.1's Links line says "Back to every section"; its table names the pages behind each problem.
    const table = id === '9.1' ? [...text.matchAll(/^\|[^|]+\|[^|]+\|\s*([\d., to]+?)\s*\|/gm)].map((m) => m[1]).join(', ') : '';
    pages.push({ id, links, table });
  }
}
const order = pages.map((p) => p.id);

// Page ids named in a phrase, with ranges such as "3.1 to 3.6" expanded in reading order.
function ids(phrase) {
  const out = [];
  const clean = phrase.replace(/\([^)]*\)/g, ' ');
  const re = /(\d\.\d)(?:\s+to\s+(\d\.\d))?/g;
  let m;
  while ((m = re.exec(clean))) {
    if (m[2]) { const i = order.indexOf(m[1]), j = order.indexOf(m[2]); if (i >= 0 && j >= i) out.push(...order.slice(i, j + 1)); }
    else if (order.includes(m[1])) out.push(m[1]);
  }
  return out;
}

const edges = new Map();
const addEdge = (from, to, source) => { if (from !== to) edges.set(`${from}>${to}`, { from, to, source }); };
for (const { id, links, table } of pages) {
  if (table) ids(table).forEach((q) => addEdge(q, id, `${id} table`));
  const part = links.split(/References:/)[0];
  for (const clause of part.split(/[;.](?=\s|$)/)) {
    const c = clause.trim();
    if (/^back to/i.test(c)) ids(c.replace(/^back to/i, '')).forEach((q) => addEdge(q, id, id));
    else if (/^forward to/i.test(c)) ids(c.replace(/^forward to/i, '')).forEach((r) => addEdge(id, r, id));
  }
}

// Questions from the page index table.
const question = {};
for (const m of overview.matchAll(/^\|\s*(\d\.\d)\s[^|]*\|\s*([^|]+?)\s*\|/gm)) question[m[1]] = m[2];

// Running examples.
const examples = [];
const ex = overview.split('## Running examples')[1].split('\n## ')[0];
for (const m of ex.matchAll(/^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$/gm)) {
  if (/^(Object|---)/.test(m[1])) continue;
  examples.push({ name: m[1], what: m[2], pages: ids(m[3]) });
}

// Takeaways of built pages.
const strip = (s) => s.replace(/<[^>]+>/g, '').replace(/&rsquo;/g, '’').replace(/&lsquo;/g, '‘').replace(/&ldquo;/g, '“').replace(/&rdquo;/g, '”').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();
const takeaway = {};
for (const id of order) {
  const file = path.join(root, 'docs', 'pages', `${id.replace('.', '-')}.html`);
  if (!fs.existsSync(file)) continue;
  const html = fs.readFileSync(file, 'utf8');
  const box = html.match(/<div class="takeaway">([\s\S]*?)<\/div>/);
  if (box) { const ps = [...box[1].matchAll(/<p(?![^>]*class="label")[^>]*>([\s\S]*?)<\/p>/g)]; if (ps.length) takeaway[id] = strip(ps.map((p) => p[1]).join(' ')); }
}

const data = {
  generated: 'by tools/build-sitemap.mjs from specs/ and docs/pages/',
  nodes: order.map((id) => ({ id, question: question[id] || '', takeaway: takeaway[id] || null })),
  edges: [...edges.values()],
  examples,
};
fs.mkdirSync(path.join(root, 'docs', 'data'), { recursive: true });
fs.writeFileSync(path.join(root, 'docs', 'data', 'sitemap.json'), JSON.stringify(data, null, 1) + '\n');
console.log(`${data.nodes.length} pages, ${data.edges.length} edges, ${examples.length} running examples, ${Object.keys(takeaway).length} takeaways`);
const missing = order.filter((id) => !question[id]);
if (missing.length) console.log('pages without a question in the page index:', missing.join(' '));
