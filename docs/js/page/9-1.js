// Page 9.1: find a problem. Entries follow the table in specs/9-open-problems/9.1-*.md.
import { sections } from '../pages.js';
import { store } from '../site.js';
import { segmented } from '../lib/ui.js';

const KINDS = [['theory', 'Theory'], ['interp', 'Interpretability'], ['eval', 'Evaluation'], ['training', 'Experiments on training']];
const SETUPS = [['laptop', 'Laptop'], ['gpu', 'One GPU'], ['api', 'API access']];

const PROBLEMS = [
  { title: 'Are most features linear?', why: 'Most interpretability tools assume they are.', pages: ['6.2', '6.4'], kinds: ['interp'], setups: ['gpu'],
    project: 'Search a small model for a feature that is represented in more than one dimension, and test whether the model uses it.',
    refs: [['Engels et al. (2024), “Not all language model features are one-dimensionally linear” (first titled “… are linear”)', 'https://arxiv.org/abs/2405.14860']] },
  { title: 'Is written reasoning faithful?', why: 'Monitoring reasoning models depends on it.', pages: ['4.4', '8.6'], kinds: ['eval', 'interp'], setups: ['gpu', 'api'],
    project: 'Replicate an early-answering test, which cuts the chain of thought short and checks whether the answer changes, on an open reasoning model.',
    refs: [['Lanham et al. (2023), “Measuring faithfulness in chain-of-thought reasoning”', 'https://arxiv.org/abs/2307.13702'], ['Korbak et al. (2025), “Chain of thought monitorability”', 'https://arxiv.org/abs/2507.11473']] },
  { title: 'Probes that generalize', why: 'Lie detection and monitoring need probes that work on new kinds of input.', pages: ['6.3', '7.2'], kinds: ['interp', 'theory'], setups: ['gpu'],
    project: 'Train truth probes on a small open model and test them on negated statements and on topics left out of training.',
    refs: [['Levinstein and Herrmann (2025), “Still no lie detector for language models,” Philosophical Studies', 'https://arxiv.org/abs/2307.00175'], ['Herrmann and Levinstein (2026), Radical AI Interpretability, Ch. 4', 'https://arxiv.org/abs/2606.26523']] },
  { title: 'Reward model overoptimization', why: 'Every model trained with human feedback is optimized against a proxy.', pages: ['4.3', '8.1'], kinds: ['training'], setups: ['gpu'],
    project: 'Reproduce a best-of-n overoptimization curve with small open models: a proxy reward model, a larger “gold” reward model, and the gap between them as n grows.',
    refs: [['Gao, Schulman, and Hilton (2022), “Scaling laws for reward model overoptimization”', 'https://arxiv.org/abs/2210.10760']] },
  { title: 'Goal attribution from internals', why: 'Telling misaligned goals apart from mistakes.', pages: ['7.1', '7.3'], kinds: ['interp', 'theory'], setups: ['laptop', 'gpu'],
    project: 'Find a representation of the goal in a small reinforcement learning agent and test it by intervening on it.',
    refs: [['Herrmann and Levinstein (2026), Radical AI Interpretability, Ch. 5', 'https://arxiv.org/abs/2606.26523']] },
  { title: 'Instrumental convergence in trained agents', why: 'The theorems cover optimal policies only.', pages: ['5.2', '8.4'], kinds: ['theory', 'training'], setups: ['laptop'],
    project: 'Test whether agents trained by reinforcement learning, rather than optimal ones, seek power in small environments.',
    refs: [['Turner, Smith, Shah, Critch, and Tadepalli (2021), “Optimal policies tend to seek power,” NeurIPS', 'https://arxiv.org/abs/1912.01683']] },
  { title: 'Valid capability evaluations', why: 'Deployment decisions rest on them.', pages: ['5.6', '8.6'], kinds: ['eval'], setups: ['api'],
    project: 'Build a small evaluation for one capability and test whether it measures what it claims to measure.',
    refs: [['Kwa et al. (2025), “Measuring AI ability to complete long tasks”', 'https://arxiv.org/abs/2503.14499']] },
];

const pages = new Map(sections.flatMap((s) => s.pages).map((p) => [p.id, p]));
const done = (id) => { try { return !!store.get(`done:${id}`, false); } catch { return false; } };
const state = { kind: 'all', setup: 'all', finished: 'all' };

const ctl = document.getElementById('controls');
segmented(ctl, { name: 'Kind of work', options: [['all', 'Any'], ...KINDS], value: 'all', onChange: (k) => { state.kind = k; draw(); } });
segmented(ctl, { name: 'Setup', options: [['all', 'Any'], ...SETUPS], value: 'all', onChange: (k) => { state.setup = k; draw(); } });
segmented(ctl, { name: 'Pages finished', options: [['all', 'Show all problems'], ['some', 'At least one page finished'], ['every', 'Every page finished']], value: 'all', onChange: (k) => { state.finished = k; draw(); } });

const list = document.getElementById('problems');
const label = (pairs, keys) => keys.map((k) => pairs.find((p) => p[0] === k)[1]).join(', ');
const items = PROBLEMS.map((p) => {
  const li = document.createElement('li');
  list.append(li);
  return { p, li };
});
const out = document.getElementById('readout');

function draw() {
  let shown = 0;
  for (const { p, li } of items) {
    const fin = p.pages.map(done);
    const ok = (state.kind === 'all' || p.kinds.includes(state.kind)) &&
      (state.setup === 'all' || p.setups.includes(state.setup)) &&
      (state.finished === 'all' || (state.finished === 'some' ? fin.some(Boolean) : fin.every(Boolean)));
    li.hidden = !ok;
    if (ok) shown++;
    const pageLinks = p.pages.map((id, i) => {
      const pg = pages.get(id);
      const name = `${id} ${pg ? pg.title : ''}`;
      const inner = pg && pg.href ? `<a href="../${pg.href}">${name}</a>` : name;
      return fin[i] ? `<span class="fin">${inner} (finished)</span>` : inner;
    }).join('; ');
    li.innerHTML = `<h3>${p.title}</h3>
      <p><span class="k">Why it matters:</span> ${p.why}</p>
      <p class="pg"><span class="k">Pages:</span> ${pageLinks}</p>
      <p><span class="k">A first project:</span> ${p.project}</p>
      <p><span class="k">Start with:</span> ${p.refs.map(([t, u]) => `<a href="${u}">${t}</a>`).join('; ')}.</p>
      <p class="k">${label(KINDS, p.kinds)}; ${label(SETUPS, p.setups)}</p>`;
  }
  out.textContent = shown === PROBLEMS.length ? `All ${shown} problems shown.` : shown ? `${shown} of ${PROBLEMS.length} problems match.` : 'No problem matches these filters.';
}
draw();
