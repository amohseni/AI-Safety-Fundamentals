// Page 7.1: one behavior, many minds.
import * as J from '../lib/s7-jeffrey.js';
import { slider, segmented, button, fmt } from '../lib/ui.js';

const PA = [1 / 3, 1 / 3, 1 / 3]; // the agent's credence in each of its own actions before it decides
const START = { ps: [0.3, 0.5, 0.2], D: [[1, -0.3, -0.3], [-0.3, 1, -0.3], [0.1, 0.1, 0.8]] };
const state = { ps: START.ps.slice(), D: START.D.map((r) => r.slice()), k: { a: 1, b: 0, c: 0 }, probe: false };
const LOC = ['the parser', 'the renderer', 'elsewhere'];
const SHORT = ['Parser', 'Renderer', 'Elsewhere'];
const PROPS = J.allPropositions(9);
const f2 = fmt(2);
const sgn = (x) => (x < 0 ? `−${Math.abs(x).toFixed(2)}` : x.toFixed(2));
const listeners = [];
const changed = () => listeners.forEach((f) => f());
const model = () => J.atoms(PA, state.ps, state.D);
function constants(m) { const { a, b, c } = state.k; return { a, b, c, d: J.normalizingD(m, c) }; }
function katex(el) { if (window.renderMathInElement) window.renderMathInElement(el, { delimiters: [{ left: '\\(', right: '\\)', display: false }], throwOnError: false }); }

// ---- Panel 1: the agent ----
(function agent() {
  const ctl = document.getElementById('agent-controls');
  const pS = [];
  const setP = (i, v) => {
    const rest = 1 - state.ps[i], newRest = 1 - v;
    state.ps = state.ps.map((p, j) => (j === i ? v : rest > 1e-9 ? (p * newRest) / rest : newRest / 2));
    pS.forEach((s, j) => s.set(state.ps[j]));
    changed();
  };
  for (let j = 0; j < 3; j++) {
    pS.push(slider(ctl, { name: `Credence: bug ${j === 2 ? '' : 'in '}${LOC[j]}`, min: 0, max: 1, step: 0.01, value: state.ps[j], format: f2, onInput: (v) => setP(j, v) }));
  }
  // Desirability grid: rows are actions, columns are locations.
  const wrap = document.createElement('div');
  wrap.className = 'ctl';
  wrap.innerHTML = '<span class="ctl-head"><span class="ctl-name">Desirability D of each action when the bug is in each location</span></span>';
  const table = document.createElement('table');
  table.className = 'dgrid';
  table.innerHTML = `<thead><tr><th></th>${SHORT.map((s) => `<th scope="col">${s}</th>`).join('')}</tr></thead>`;
  const tb = document.createElement('tbody');
  const dS = [];
  J.ACTIONS.forEach((act, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<th scope="row">${act.replace('Edit the ', 'Edit ').replace('Search the rest', 'Search')}</th>`;
    dS.push([]);
    for (let j = 0; j < 3; j++) {
      const td = document.createElement('td');
      const input = Object.assign(document.createElement('input'), { type: 'range', min: -1, max: 1, step: 0.05, value: state.D[i][j] });
      input.setAttribute('aria-label', `Desirability of ${act.toLowerCase()} when the bug is ${j === 2 ? '' : 'in '}${LOC[j]}`);
      const out = document.createElement('output');
      const show = () => { out.textContent = sgn(Number(input.value)); };
      input.addEventListener('input', () => { state.D[i][j] = Number(input.value); show(); changed(); });
      show();
      td.append(input, out);
      tr.append(td);
      dS[i].push({ set: (v) => { input.value = v; show(); } });
    }
    tb.append(tr);
  });
  table.append(tb);
  wrap.append(table);
  ctl.append(wrap);
  button(ctl, 'Reset', () => {
    state.ps = START.ps.slice(); state.D = START.D.map((r) => r.slice());
    pS.forEach((s, j) => s.set(state.ps[j]));
    dS.forEach((row, i) => row.forEach((s, j) => s.set(state.D[i][j])));
    changed();
  });

  const stage = document.getElementById('agent-stage');
  const out = document.getElementById('agent-readout');
  const draw = () => {
    const m = model(), vals = J.actionValues(m), best = J.argmax(vals);
    const rows = J.ACTIONS.map((act, i) => {
      const terms = state.ps.map((p, j) => `${f2(p)} × ${state.D[i][j] < 0 ? `(${sgn(state.D[i][j])})` : sgn(state.D[i][j])}`).join(' + ');
      return `<tr class="${i === best ? 'chosen-a' : ''}"><th class="row" scope="row">${act}${i === best ? ' (chosen)' : ''}</th><td class="val">${sgn(vals[i])}</td></tr>
        <tr><td colspan="2" style="text-align:left;color:var(--quiet);padding-bottom:0.5rem">${terms}</td></tr>`;
    }).join('');
    stage.innerHTML = `<p class="t-title">Desirability of each action, averaged over the locations</p>
      <table class="jb" aria-label="Desirability of each action"><tbody>${rows}</tbody></table>`;
    out.innerHTML = `The agent&rsquo;s credences are ${state.ps.map((p, j) => `${f2(p)} for ${LOC[j]}`).join(', ')}. It takes the action with the highest desirability: <b>${J.ACTIONS[best].toLowerCase()}</b>.`;
  };
  listeners.push(draw);
  draw();
})();

// ---- Panel 2: another belief-desire pair ----
(function other() {
  const ctl = document.getElementById('other-controls');
  const sA = slider(ctl, { name: 'a', min: -1, max: 3, step: 0.05, value: state.k.a, format: f2, onInput: (v) => { state.k.a = v; changed(); } });
  const sB = slider(ctl, { name: 'b', min: -1, max: 1, step: 0.05, value: state.k.b, format: f2, onInput: (v) => { state.k.b = v; changed(); } });
  const sC = slider(ctl, { name: 'c', min: -1.5, max: 1.5, step: 0.05, value: state.k.c, format: f2, onInput: (v) => { state.k.c = v; changed(); } });
  const dLine = document.createElement('p');
  dLine.className = 'ctl';
  ctl.append(dLine);
  segmented(ctl, { name: 'Probe reading', options: [['off', 'Off'], ['on', 'On']], value: 'off', onChange: (k) => { state.probe = k === 'on'; changed(); } });
  button(ctl, 'Reset the constants', () => { state.k = { a: 1, b: 0, c: 0 }; sA.set(1); sB.set(0); sC.set(0); changed(); });

  const stage = document.getElementById('other-stage');
  const out = document.getElementById('other-readout');
  const head = `<thead><tr><th></th>${SHORT.map((s, j) => `<th scope="col">${s}</th>`).join('')}</tr></thead>`;
  const draw = () => {
    const m = model(), k = constants(m), chk = J.check(m, k);
    dLine.innerHTML = `<span class="ctl-head"><span class="ctl-name">d, set so the new credences add up to 1</span><output>${f2(k.d)}</output></span>`;
    if (!chk.ok) {
      stage.innerHTML = `<p class="t-title status-bad">Not allowed: ${chk.why.replace('cD + d', '\\(cD + d\\)').replace('ad − bc', '\\(ad - bc\\)')}.</p>`;
      out.innerHTML = 'Bolker&rsquo;s transformation needs \\(ad - bc &gt; 0\\) and \\(cD + d &gt; 0\\). ' +
        (k.a * k.d - k.b * k.c <= 0 ? 'With these constants the new desirabilities would not keep the order of the old ones.' : 'With these constants some new credences would be zero or negative.');
      katex(stage); katex(out);
      return;
    }
    const mt = J.transform(m, k);
    const vals = J.actionValues(m), valsT = J.actionValues(mt);
    const best = J.argmax(vals), bestT = J.argmax(valsT);
    const cond = [0, 1, 2].map((i) => J.conditional(mt, i));
    const ROW = ['Edit parser', 'Edit renderer', 'Search'];
    const credRows = ROW.map((act, i) => `<tr><th class="row" scope="row">${act}</th>${cond[i].map((p, j) => `<td class="${j === 0 ? 'sep' : ''}">${f2(p)}</td>`).join('')}</tr>`).join('');
    const desRows = ROW.map((act, i) => `<tr><th class="row" scope="row">${act}</th>${[0, 1, 2].map((j) => `<td class="${j === 0 ? 'sep' : ''}">${sgn(mt.D[i * 3 + j])}</td>`).join('')}</tr>`).join('');
    const actRows = J.ACTIONS.map((act, i) => `<tr class="${i === bestT ? 'chosen-b' : ''}"><th class="row" scope="row">${act}${i === bestT ? ' (chosen)' : ''}</th><td class="val">${sgn(valsT[i])}</td></tr>`).join('');
    stage.innerHTML = `<p class="t-title">New credence in each location, given the action</p>
      <table class="jb" aria-label="New credences given each action">${head}<tbody>${credRows}</tbody></table>
      <p class="t-title">New desirability of each action and location</p>
      <table class="jb" aria-label="New desirabilities">${head}<tbody>${desRows}</tbody></table>
      <p class="t-title">New desirability of each action, averaged under the new credences</p>
      <table class="jb" aria-label="New desirability of each action"><tbody>${actRows}</tbody></table>`;
    const same = J.sameOrdering(m, mt, PROPS);
    const rank = (v) => J.ranking(v).map((i) => J.ACTIONS[i].toLowerCase()).join(', then ');
    let probeText = '';
    if (state.probe) {
      const dev = Math.max(...mt.P.map((p, i) => Math.abs(p - m.P[i])));
      probeText = dev < 1e-9
        ? ' <b>Probe:</b> the new credences match the probe reading.'
        : ` <b>Probe:</b> the new credences contradict the probe reading, by up to ${dev.toFixed(3)} on one combination of action and location, so this pair is ruled out.`;
    }
    out.innerHTML = `Original pair ranks the actions: ${rank(vals)}. <span class="c-second">New pair</span>: ${rank(valsT)}. ` +
      (best === bestT ? 'Both choose the same action. ' : 'The choices differ. ') +
      (same ? 'The two pairs order all 511 propositions of this algebra the same way.' : 'The two pairs order some propositions differently.') + probeText;
  };
  listeners.push(draw);
  draw();
})();

// ---- Static figure: the triangle of Herrmann and Levinstein's Figure 1 ----
(function triangle() {
  const fig = document.getElementById('fig-triangle');
  const W = Math.max(300, Math.min(560, fig.clientWidth || 560)), H = 380;
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('class', 'triangle');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('width', W); svg.setAttribute('height', H);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'A triangle whose corners are the agential description, the computational system, and behavior');
  const el = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); if (text) e.textContent = text; svg.append(e); return e; };
  const top = [W / 2, 78], left = [W * 0.22, H - 110], right = [W * 0.78, H - 110];
  el('line', { x1: top[0], y1: top[1], x2: left[0], y2: left[1] });
  el('line', { x1: top[0], y1: top[1], x2: right[0], y2: right[1] });
  el('line', { x1: left[0], y1: left[1], x2: right[0], y2: right[1] });
  el('text', { x: top[0], y: 28, 'text-anchor': 'middle' }, 'Agential description');
  el('text', { x: top[0], y: 56, 'text-anchor': 'middle', class: 'quiet' }, 'P, D, and an algebra');
  const lx = Math.max(4, left[0] - 50);
  el('text', { x: lx, y: H - 62, 'text-anchor': 'start' }, 'Computational');
  el('text', { x: lx, y: H - 36, 'text-anchor': 'start' }, 'system');
  el('text', { x: lx, y: H - 10, 'text-anchor': 'start', class: 'quiet' }, 'its internals');
  el('text', { x: Math.min(W - 4, right[0] + 40), y: H - 62, 'text-anchor': 'end' }, 'Behavior');
  // edge labels, along the edges and outside the triangle
  const along = (p, q, label, off) => {
    const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
    let ang = Math.atan2(q[1] - p[1], q[0] - p[0]) * 180 / Math.PI;
    if (ang > 90) ang -= 180; if (ang < -90) ang += 180;
    const nx = -(q[1] - p[1]), ny = q[0] - p[0], L = Math.hypot(nx, ny);
    const x = mx + (off * nx) / L, y = my + (off * ny) / L;
    el('text', { x, y, 'text-anchor': 'middle', class: 'quiet', transform: `rotate(${ang.toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)})`, 'dominant-baseline': 'middle' }, label);
  };
  along(left, top, 'interpretability', -16);
  along(top, right, 'describes', -16);
  along(left, right, 'produces', 16);
  fig.append(svg);
})();
