// Claims on page 0.1 (site map), computed from docs/data/sitemap.json. Run: node tools/build-sitemap.mjs && node tools/test-page-0-1.mjs
import fs from 'node:fs';
import { shortestPath, allShortestPaths } from '../docs/js/lib/s7-graph.js';

const data = JSON.parse(fs.readFileSync(new URL('../docs/data/sitemap.json', import.meta.url)));
let failed = 0;
const ok = (cond, msg) => { console.log(`${cond ? 'ok  ' : 'FAIL'} ${msg}`); if (!cond) failed++; };
const sec = (id) => Number(id.split('.')[0]);

ok(data.nodes.length === 50, `${data.nodes.length} pages`);
console.log(`   ${data.edges.length} edges`);

// Task 1: shortest path from 2.5 to 8.1.
const p = shortestPath(data.edges, '2.5', '8.1');
const all = allShortestPaths(data.edges, '2.5', '8.1');
console.log('   shortest paths 2.5 -> 8.1:', all.map((x) => x.join(' > ')).join(' | '));
ok(p && p.length === 4, `the shortest path has ${p.length} pages, ${p.length - 2} between 2.5 and 8.1`);

// Task 2: pages that use GPT-2 small, by section.
const gpt = data.examples.find((e) => /GPT-2/.test(e.name));
const secs = [...new Set(gpt.pages.map(sec))];
console.log('   GPT-2 small pages:', gpt.pages.join(' '));
ok(secs.join() === '3,6', `GPT-2 small appears in sections ${secs.join(' and ')}`);

// What to notice: Sections 6 to 8 depend directly on few earlier pages.
const late = (id) => sec(id) >= 6 && sec(id) <= 8;
const into = data.edges.filter((e) => late(e.to) && !late(e.from));
const earlier = [...new Set(into.map((e) => e.from))].sort();
const early = data.nodes.filter((n) => sec(n.id) >= 1 && sec(n.id) <= 5).length;
console.log(`   pages before section 6 that sections 6 to 8 link back to directly: ${earlier.join(' ')} (${earlier.length} of ${early})`);
// Task 3: everything 8.1 depends on, directly or through other pages.
const ancestors = (id, seen = new Set()) => { for (const e of data.edges) if (e.to === id && !seen.has(e.from)) { seen.add(e.from); ancestors(e.from, seen); } return seen; };
for (const id of ['8.1', '8.2', '8.4', '8.5', '8.6']) console.log(`   ${id} depends on ${ancestors(id).size} pages: ${[...ancestors(id)].sort((a, b) => parseFloat(a) - parseFloat(b)).join(' ')}`);
ok(ancestors('8.1').size === 9, `8.1 depends, directly or indirectly, on ${ancestors('8.1').size} of the other 49 pages`);
ok(ancestors('8.4').size === 11, `8.4 depends on ${ancestors('8.4').size}`);
const into8 = [...new Set(data.edges.filter((e) => sec(e.to) === 8 && sec(e.from) < 8).map((e) => e.from))].sort();
console.log(`   pages outside section 8 that section 8 links back to directly: ${into8.join(' ')} (${into8.length})`);

// Check question: which section do 8.1's direct prerequisites come from?
const pre81 = data.edges.filter((e) => e.to === '8.1').map((e) => e.from);
console.log('   direct prerequisites of 8.1:', pre81.join(' '));
ok(pre81.length === 1 && pre81[0] === '4.3', '8.1 builds directly on 4.3 only');

// Every edge points to a later page (the map reads top to bottom).
const pos = new Map(data.nodes.map((n, i) => [n.id, i]));
const backward = data.edges.filter((e) => pos.get(e.from) >= pos.get(e.to));
ok(backward.length === 0, `every edge goes from an earlier page to a later one (${backward.length} exceptions${backward.length ? ': ' + backward.map((e) => e.from + '>' + e.to).join(' ') : ''})`);

console.log(failed ? `${failed} FAILED` : 'all passed');
process.exit(failed ? 1 : 0);
