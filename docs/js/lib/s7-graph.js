// Small graph helpers for the site map (page 0.1). No DOM; also used by tools/test-page-0-1.mjs.
// Edges are { from, to }, meaning "to uses an idea from from".

// Shortest chain of pages from a to b that follows the edges forward (from a page to a page that uses it).
// Returns the list of page ids, or null if there is none.
export function shortestPath(edges, a, b) {
  if (a === b) return [a];
  const next = new Map();
  for (const e of edges) { if (!next.has(e.from)) next.set(e.from, []); next.get(e.from).push(e.to); }
  const prev = new Map([[a, null]]);
  const queue = [a];
  while (queue.length) {
    const u = queue.shift();
    for (const v of (next.get(u) || []).slice().sort()) {
      if (prev.has(v)) continue;
      prev.set(v, u);
      if (v === b) { const path = [b]; let w = u; while (w !== null) { path.unshift(w); w = prev.get(w); } return path; }
      queue.push(v);
    }
  }
  return null;
}

// All shortest paths (as lists), for reporting how many there are.
export function allShortestPaths(edges, a, b) {
  const best = shortestPath(edges, a, b);
  if (!best) return [];
  const L = best.length;
  const out = [];
  const walk = (path) => {
    const u = path[path.length - 1];
    if (path.length === L) { if (u === b) out.push(path.slice()); return; }
    for (const e of edges) if (e.from === u && !path.includes(e.to)) { path.push(e.to); walk(path); path.pop(); }
  };
  walk([a]);
  return out;
}

export const prerequisites = (edges, id) => edges.filter((e) => e.to === id).map((e) => e.from);
export const usedBy = (edges, id) => edges.filter((e) => e.from === id).map((e) => e.to);
