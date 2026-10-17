// Scalar computation graphs with forward values and reverse-mode derivatives (backpropagation).
// Used by page 2.7. No DOM; also used by tools/test-page-2-7.mjs.

const sig = (z) => 1 / (1 + Math.exp(-z));
// Each op: forward value from input values, and the local derivative with respect to input k.
export const OPS = {
  input: { f: () => NaN, d: () => [] },
  mul: { f: ([a, b]) => a * b, d: ([a, b]) => [b, a], sym: '×' },
  add: { f: ([a, b]) => a + b, d: () => [1, 1], sym: '+' },
  sub: { f: ([a, b]) => a - b, d: () => [1, -1], sym: '−' },
  sigmoid: { f: ([a]) => sig(a), d: ([a]) => [sig(a) * (1 - sig(a))] },
  square: { f: ([a]) => a * a, d: ([a]) => [2 * a] },
};

// A graph: nodes in topological order, each {id, op, in: [ids], name, def, pos: [rank, lane]}; the last node is the loss.
export const GRAPHS = {
  one: {
    title: 'One path to each parameter',
    nodes: [
      { id: 'x', op: 'input', name: 'x', value: 1.5, pos: [0, 0] },
      { id: 'w1', op: 'input', name: 'w₁', value: 0.8, param: true, pos: [0, 1] },
      { id: 'm', op: 'mul', in: ['w1', 'x'], name: 'm', def: 'w₁ x', pos: [1, 0.5] },
      { id: 'b', op: 'input', name: 'b', value: -0.5, param: true, pos: [1, 1.5] },
      { id: 'z', op: 'add', in: ['m', 'b'], name: 'z', def: 'm + b', pos: [2, 1] },
      { id: 'a', op: 'sigmoid', in: ['z'], name: 'a', def: 'σ(z)', pos: [3, 1] },
      { id: 'w2', op: 'input', name: 'w₂', value: 2, param: true, pos: [3, 2] },
      { id: 'p', op: 'mul', in: ['w2', 'a'], name: 'p', def: 'w₂ a', pos: [4, 1.5] },
      { id: 'y', op: 'input', name: 'y', value: 0.5, pos: [4, 2.5] },
      { id: 'd', op: 'sub', in: ['p', 'y'], name: 'd', def: 'p − y', pos: [5, 2] },
      { id: 'L', op: 'square', in: ['d'], name: 'L', def: 'd²', pos: [6, 2] },
    ],
  },
  two: {
    title: 'A value used twice',
    nodes: [
      { id: 'x', op: 'input', name: 'x', value: 1.0, pos: [0, 0] },
      { id: 'w', op: 'input', name: 'w', value: 0.6, param: true, pos: [0, 1] },
      { id: 'h', op: 'mul', in: ['w', 'x'], name: 'h', def: 'w x', pos: [1, 0.5] },
      { id: 'v1', op: 'input', name: 'v₁', value: 1.5, param: true, pos: [1, -0.5] },
      { id: 's', op: 'sigmoid', in: ['h'], name: 's', def: 'σ(h)', pos: [2, 1.5] },
      { id: 'v2', op: 'input', name: 'v₂', value: -1.0, param: true, pos: [2, 2.5] },
      { id: 'u', op: 'mul', in: ['v1', 'h'], name: 'u', def: 'v₁ h', pos: [3, 0] },
      { id: 't', op: 'mul', in: ['v2', 's'], name: 't', def: 'v₂ s', pos: [3, 2] },
      { id: 'q', op: 'add', in: ['u', 't'], name: 'q', def: 'u + t', pos: [4, 1] },
      { id: 'y', op: 'input', name: 'y', value: 1.0, pos: [4, 2] },
      { id: 'd', op: 'sub', in: ['q', 'y'], name: 'd', def: 'q − y', pos: [5, 1.5] },
      { id: 'L', op: 'square', in: ['d'], name: 'L', def: 'd²', pos: [6, 1.5] },
    ],
  },
};

// Forward pass: values for every node, given input values {id: number}.
export function forward(graph, inputs) {
  const val = {};
  for (const n of graph.nodes) val[n.id] = n.op === 'input' ? inputs[n.id] : OPS[n.op].f(n.in.map((k) => val[k]));
  return val;
}

// Backward pass. Returns grads {id: dL/d id}, the order of steps, and for each step the contributions it sends:
// steps[i] = {node, sends: [{to, upstream, local, contrib}]}.
export function backward(graph, val) {
  const grad = {}, parts = {};
  graph.nodes.forEach((n) => { grad[n.id] = 0; parts[n.id] = []; });
  const L = graph.nodes[graph.nodes.length - 1].id;
  grad[L] = 1;
  const steps = [];
  for (const n of [...graph.nodes].reverse()) {
    if (n.op === 'input') continue;
    const locals = OPS[n.op].d(n.in.map((k) => val[k]));
    const sends = n.in.map((k, i) => ({ to: k, upstream: grad[n.id], local: locals[i], contrib: grad[n.id] * locals[i] }));
    sends.forEach((s) => { grad[s.to] += s.contrib; parts[s.to].push({ from: n.id, value: s.contrib }); });
    steps.push({ node: n.id, sends });
  }
  return { grad, steps, parts };
}

// Every path from node `from` to the loss, as lists of [node, input] edges, with the product of local derivatives.
export function chainPaths(graph, val, from) {
  const L = graph.nodes[graph.nodes.length - 1].id;
  const users = (id) => graph.nodes.filter((n) => n.in && n.in.includes(id));
  const out = [];
  const walk = (id, edges) => {
    if (id === L) { out.push(edges); return; }
    for (const u of users(id)) walk(u.id, [...edges, { node: u, input: id }]);
  };
  walk(from, []);
  return out.map((edges) => {
    const factors = edges.map(({ node, input }) => ({ node, input, local: OPS[node.op].d(node.in.map((k) => val[k]))[node.in.indexOf(input)] }));
    return { factors, product: factors.reduce((p, f) => p * f.local, 1) };
  });
}

export function loss(graph, inputs) { const v = forward(graph, inputs); return v[graph.nodes[graph.nodes.length - 1].id]; }

// The nudge test: change one input by +h and -h and divide the change in loss by 2h.
export function nudge(graph, inputs, id, h = 0.001) {
  const up = loss(graph, { ...inputs, [id]: inputs[id] + h }), down = loss(graph, { ...inputs, [id]: inputs[id] - h }), base = loss(graph, inputs);
  return { base, up, down, slope: (up - down) / (2 * h) };
}
