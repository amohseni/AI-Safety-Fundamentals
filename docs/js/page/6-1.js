// Page 6.1: the methods map, with a quiz mode.
import { segmented, button } from '../lib/ui.js';
import { sections } from '../pages.js';

const d3 = window.d3;
const built = new Set(sections.flatMap((s) => s.pages.filter((p) => p.href).map((p) => p.id)));
const pageRef = (id) => (built.has(id) ? `<a href="${id.replace('.', '-')}.html">page ${id}</a>` : `page ${id}`);

const COLS = [['behavior', 'Behavior only'], ['acts', 'Activations'], ['weights', 'Weights']];
const ROWS = [['observe', 'Observe without changing anything'], ['intervene', 'Change one thing and measure the effect']];

const METHODS = [
  { id: 'eval', name: 'Behavioral evaluation', col: 'behavior', row: 'observe', page: null,
    access: 'Inputs and outputs only.',
    shows: 'What the model does on the inputs tested: how often it answers correctly, refuses, or makes a given kind of error.',
    example: 'Measuring the share of two-digit additions a model gets right.',
    limit: 'Two models that give the same outputs on every tested input can compute them in different ways, and the test covers only the inputs someone thought to try.' },
  { id: 'counter', name: 'Counterfactual inputs', col: 'behavior', row: 'intervene', page: null,
    access: 'Inputs and outputs only.',
    shows: 'Causal evidence that the output depends on one detail of the input: change that detail, hold the rest of the prompt fixed, and compare the outputs.',
    example: 'Changing only the name in a prompt and checking whether the model’s recommendation changes.',
    limit: 'It shows which input details matter, not where or how the network uses them.' },
  { id: 'probe', name: 'Linear probe', col: 'acts', row: 'observe', page: '6.3',
    access: 'The activations at one layer, recorded on many inputs.',
    shows: 'Correlational evidence: whether a property of the input can be read off the activations by a simple classifier.',
    example: 'A probe that predicts from layer 10 whether the statement in the prompt is true.',
    limit: 'Information that can be read off need not be used by the model, and a flexible probe can learn part of the task by itself.' },
  { id: 'sae', name: 'Sparse autoencoder', col: 'acts', row: 'observe', page: '6.5',
    access: 'The activations at one layer, recorded on many inputs.',
    shows: 'Correlational evidence: a set of directions that rebuild the activations sparsely, each described by the inputs on which it is active.',
    example: 'A unit that is active on text about the Golden Gate Bridge.',
    limit: 'A unit’s description comes from what it fires on; whether the model uses it needs an intervention. The rebuild is imperfect.' },
  { id: 'steer', name: 'Steering vector', col: 'acts', row: 'intervene', page: '6.6',
    access: 'The activations, which it changes while the model runs.',
    shows: 'Causal evidence: adding a direction to the activations changes what the model does.',
    example: 'Adding a direction found from positive and negative prompts at one layer, so that the model writes more positive text.',
    limit: 'A large addition can change behavior through side effects, so a change in output does not by itself show that the direction means what we think.' },
  { id: 'patch', name: 'Activation patching', col: 'acts', row: 'intervene', page: '6.7',
    access: 'The activations of two runs, a clean input and a corrupted one.',
    shows: 'Causal evidence: copying one component’s activation from the clean run into the corrupted run shows whether that component carries the information that decides the answer.',
    example: 'Finding which attention heads carry the name a sentence should end with.',
    limit: 'The result depends on the choice of corrupted input, and a component can matter on these inputs without being the whole story.' },
  { id: 'readw', name: 'Reading the weights', col: 'weights', row: 'observe', page: '6.7',
    access: 'The parameters themselves, with no inputs needed.',
    shows: 'What computation the parameters could perform: for example, that an attention head’s weights copy the token it attends to.',
    example: 'Nanda et al. (2023) found that the modular-addition transformer’s embedding uses sines and cosines of a few frequencies.',
    limit: 'It shows what a part can compute, not whether the model relies on it for a behavior; that needs an intervention.' },
  { id: 'editw', name: 'Editing or removing weights', col: 'weights', row: 'intervene', page: null,
    access: 'The parameters, which it changes.',
    shows: 'Causal evidence: changing some parameters changes a behavior.',
    example: 'Meng et al. (2022) changed one MLP weight matrix in GPT-J so that it said the Eiffel Tower is in Rome.',
    limit: 'An edit can have side effects on other behaviors, and a successful edit at one place does not show the information is stored only there.' },
];
const byId = Object.fromEntries(METHODS.map((m) => [m.id, m]));

// Claims for quiz mode. ok: methods that could support the claim; need: groups of which each must be picked; kind: evidence needed.
const CLAIMS = [
  { text: 'The activations at layer 10 carry whether the statement in the prompt is true.', ok: ['probe', 'sae'], need: [['probe', 'sae']], kind: 'observe',
    why: 'The claim is that the information is present. A probe, or an SAE unit that tracks truth, can show that it can be read off the activations. That is correlational evidence, and it is enough for this claim.' },
  { text: 'The model uses its representation of truth when it answers.', ok: ['steer', 'patch'], need: [['steer', 'patch']], kind: 'intervene',
    why: 'The claim is that the information makes a difference to the output. Only an intervention on the activations, such as steering along the truth direction or patching it, can show that. A probe cannot: information can be present without being used.' },
  { text: 'The model gets 90% of two-digit additions right.', ok: ['eval'], need: [['eval']], kind: 'observe',
    why: 'This is a claim about behavior, and a behavioral evaluation settles it. It says nothing about how the model adds.' },
  { text: 'Changing only the name in a prompt changes the model’s recommendation.', ok: ['counter'], need: [['counter']], kind: 'intervene',
    why: 'The claim is causal, about the input: hold everything else fixed, change the name, and compare. Counterfactual inputs do exactly that, without looking inside the model.' },
  { text: 'This attention head’s weights copy the token it attends to.', ok: ['readw'], need: [['readw']], kind: 'observe',
    why: 'This is a claim about what the head’s parameters compute, which reading the weights can check. Whether the model relies on the head for some behavior is a further, causal claim.' },
  { text: 'The model’s answer to “The Eiffel Tower is in…” depends on one MLP layer.', ok: ['editw', 'patch'], need: [['editw', 'patch']], kind: 'intervene',
    why: 'The claim is that a part makes a difference, so it needs an intervention: patch that layer’s activations, or edit its weights, and see whether the answer changes.' },
  { text: 'The model has a unit for the Golden Gate Bridge, and turning that unit up makes it talk about the bridge.', ok: ['sae', 'steer'], need: [['sae'], ['steer']], kind: 'both',
    why: 'The claim has two parts. Finding the unit is the work of an SAE (correlational); showing that turning it up changes behavior is steering (causal). Templeton et al. (2024) did both.' },
];
const KIND = [['observe', 'Correlational'], ['intervene', 'Causal'], ['both', 'Both kinds needed']];

// ---- The map ----
const mapEl = document.getElementById('map');
const card = document.getElementById('card');
const quizEl = document.getElementById('quiz');
const state = { mode: 'explore', open: null, q: 0, picked: new Set(), kind: null, checked: false, results: new Map() };

function buildMap() {
  mapEl.innerHTML = '';
  const head = document.createElement('div');
  head.className = 'map-corner';
  mapEl.append(head);
  for (const [, label] of COLS) { const h = document.createElement('div'); h.className = 'map-col'; h.textContent = label; mapEl.append(h); }
  for (const [rk, rl] of ROWS) {
    const h = document.createElement('div');
    h.className = 'map-row';
    h.innerHTML = `${rl}<br><span class="kind">${rk === 'observe' ? 'correlational evidence' : 'causal evidence'}</span>`;
    mapEl.append(h);
    for (const [ck, cl] of COLS) {
      const cell = document.createElement('div');
      cell.className = 'map-cell';
      cell.dataset.label = cl;
      for (const m of METHODS.filter((x) => x.row === rk && x.col === ck)) {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = m.name;
        b.dataset.id = m.id;
        b.addEventListener('click', () => onMethod(m.id));
        cell.append(b);
      }
      mapEl.append(cell);
    }
  }
}

function onMethod(id) {
  if (state.mode === 'explore') { state.open = state.open === id ? null : id; }
  else if (!state.checked) { state.picked.has(id) ? state.picked.delete(id) : state.picked.add(id); }
  render();
}

function render() {
  mapEl.querySelectorAll('button').forEach((b) => {
    const on = state.mode === 'explore' ? state.open === b.dataset.id : state.picked.has(b.dataset.id);
    b.setAttribute('aria-pressed', String(on));
  });
  mapEl.setAttribute('aria-label', state.mode === 'explore' ? 'Methods map: select a method to open its card' : 'Methods map: select the methods that could support the claim');
  card.hidden = state.mode !== 'explore';
  quizEl.hidden = state.mode !== 'quiz';
  if (state.mode === 'explore') renderCard(); else renderQuiz();
}

function renderCard() {
  const m = byId[state.open];
  if (!m) { card.innerHTML = '<p class="quiet">Select a method in the map to see what it accesses, what it can show, an example, and its main limitation.</p>'; return; }
  card.innerHTML = `<p><b>${m.name}</b>${m.page ? ` (${pageRef(m.page)})` : ''}</p>
    <dl><dt>Accesses</dt><dd>${m.access}</dd><dt>Can show</dt><dd>${m.shows}</dd><dt>Example</dt><dd>${m.example}</dd><dt>Main limitation</dt><dd>${m.limit}</dd></dl>`;
}

function grade(c) {
  const picked = [...state.picked];
  const methodsRight = picked.length > 0 && picked.every((id) => c.ok.includes(id)) && c.need.every((g) => g.some((id) => state.picked.has(id)));
  return { methodsRight, kindRight: state.kind === c.kind };
}

function renderQuiz() {
  const c = CLAIMS[state.q];
  const done = state.results.size;
  const right = [...state.results.values()].filter(Boolean).length;
  quizEl.innerHTML = `<p class="quiet">Claim ${state.q + 1} of ${CLAIMS.length}. Answered: ${done}; both parts right: ${right}.</p>
    <p class="claim-text">&ldquo;${c.text}&rdquo;</p>
    <p class="quiet">1. Select in the map every method that could support this claim. 2. Choose the kind of evidence it needs.</p>
    <div id="kind"></div><div class="seg" id="qbtns"></div><div id="qfb" aria-live="polite"></div>`;
  segmented(quizEl.querySelector('#kind'), { name: 'Evidence the claim needs', options: KIND, value: state.kind,
    onChange: (k) => { if (!state.checked) state.kind = k; renderQuiz(); } });
  const btns = quizEl.querySelector('#qbtns');
  const check = button(btns, 'Check', () => {
    state.checked = true;
    const g = grade(c);
    state.results.set(state.q, g.methodsRight && g.kindRight);
    renderQuiz();
  }, 'primary');
  check.disabled = state.checked || state.picked.size === 0 || !state.kind;
  button(btns, 'Previous claim', () => go(-1)).disabled = state.q === 0;
  button(btns, 'Next claim', () => go(1)).disabled = state.q === CLAIMS.length - 1;
  if (state.checked) {
    const g = grade(c);
    const names = (ids) => ids.map((id) => byId[id].name.toLowerCase()).join(' or ');
    const methods = c.need.length > 1 ? c.need.map((grp) => names(grp)).join(', together with ') : names(c.ok);
    quizEl.querySelector('#qfb').innerHTML =
      `<p class="${g.methodsRight ? 'verdict-right' : 'verdict-wrong'}">Methods: ${g.methodsRight ? 'right' : 'not quite'}. This claim can be supported by ${methods}.</p>` +
      `<p class="${g.kindRight ? 'verdict-right' : 'verdict-wrong'}">Evidence: ${g.kindRight ? 'right' : 'not quite'}. It needs ${KIND.find((k) => k[0] === c.kind)[1].toLowerCase()} evidence.</p>` +
      `<p>${c.why}</p>`;
  }
}

function go(d) {
  state.q = Math.max(0, Math.min(CLAIMS.length - 1, state.q + d));
  state.picked = new Set(); state.kind = null; state.checked = false;
  render();
}

segmented(document.getElementById('mode'), { name: 'Mode', options: [['explore', 'Explore the map'], ['quiz', 'Quiz']], value: 'explore',
  onChange: (k) => { state.mode = k; render(); } });
buildMap();
render();

// ---- Static figure: what an observation and an intervention each touch ----
(function figure() {
  const fig = document.getElementById('fig-evidence');
  const panels = [
    { title: 'Observe: a probe', side: 'probe reads', sideColor: 'ink', dash: '4 4', dir: 'out', note: '(unchanged)' },
    { title: 'Intervene: steering', side: 'vector added', sideColor: 'accent', dash: null, dir: 'in', note: '(changed?)' },
  ];
  for (const p of panels) {
    const box = document.createElement('div');
    box.innerHTML = `<p class="small-title">${p.title}</p>`;
    fig.append(box);
    const W = 320, H = 250;
    const svg = d3.select(box).append('svg').attr('class', 'chart').attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`).attr('aria-hidden', 'true');
    const defs = svg.append('defs');
    const mk = (id, color) => defs.append('marker').attr('id', id).attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 7).attr('markerHeight', 7).attr('orient', 'auto-start-reverse').append('path').attr('d', 'M0,0L10,5L0,10z').attr('fill', `var(--${color})`);
    const idInk = `ar-ink-${p.dir}`, idSide = `ar-side-${p.dir}`;
    mk(idInk, 'ink'); mk(idSide, p.sideColor);
    const x = 110;
    const nodes = [['input', 30], ['activations', 125], ['output', 220]];
    nodes.forEach(([t, y]) => svg.append('text').attr('x', x).attr('y', y).attr('text-anchor', 'middle').attr('class', t === 'activations' && p.dir === 'in' ? 'label-accent' : 'label-ink').text(t));
    [[42, 102], [137, 197]].forEach(([a, b]) => svg.append('line').attr('x1', x).attr('x2', x).attr('y1', a).attr('y2', b).attr('stroke', 'var(--ink)').attr('marker-end', `url(#${idInk})`));
    const line = svg.append('line').attr('y1', 120).attr('y2', 120).attr('stroke', `var(--${p.sideColor})`).attr('stroke-width', 1.6).attr('stroke-dasharray', p.dash).attr('marker-end', `url(#${idSide})`);
    if (p.dir === 'out') line.attr('x1', 180).attr('x2', 240); else line.attr('x1', 240).attr('x2', 180);
    svg.append('text').attr('x', 246).attr('y', 126).attr('class', p.sideColor === 'accent' ? 'label-accent' : 'label-ink').text(p.side.split(' ')[0]);
    svg.append('text').attr('x', 246).attr('y', 146).attr('class', p.sideColor === 'accent' ? 'label-accent' : 'label-ink').text(p.side.split(' ')[1]);
    svg.append('text').attr('x', x).attr('y', 242).attr('text-anchor', 'middle').text(p.note);
  }
})();
