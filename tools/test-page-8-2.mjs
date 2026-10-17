// Behaviors quoted on page 8.2, recomputed with the page's own MDP (docs/js/lib/8-cleaning.js).
// Run: node tools/test-page-8-2.mjs          (add --grid for the 144-reward search behind task 3; about a minute)
import { buildRoom, solve, ACTIONS, G } from '../docs/js/lib/8-cleaning.js';

const plain = buildRoom({ plant: false }), plant = buildRoom({ plant: true });
console.log('states:', plain.nS, 'without the plant,', plant.nS, 'with it');
function show(name, w, room = plain) {
  const { path, summary: m } = solve(room, w);
  const moves = path.filter((p) => p.a >= 0).map((p) => `${ACTIONS[p.a]}@${G.xy(room.states[p.s].pos).join(',')}`).join(' ');
  console.log(`\n${name} ${JSON.stringify(w)}\n  reward ${m.reward} | true score ${m.truth} | messes left ${m.messesLeft} (out of view ${m.outOfView}) | cleaned ${m.cleaned} | bin tips ${m.tip} | plant ${m.plantKnocked} | vase broken ${m.vaseBroken} | steps blocking camera ${m.blocked} | drag steps ${m.dragged} | ended ${m.ended}`);
  console.log('  ' + moves.replace(/( wait@\S+)+$/, ' then waits'));
  return m;
}
const t1 = show('Task 1: +1 per mess cleaned', { cleaned: 1 });
console.log('  page: 6 cleaned, 5 its own, true score -1 ->', t1.cleaned === 6 && t1.tip === 5 && t1.truth === -1);
const t2 = show('Task 2: +1 per step with no mess visible', { clear: 1 });
console.log('  page: blocks camera, 28 steps clear, both messes left ->', t2.blocked === 28 && t2.reward === 28 && t2.messesLeft === 2);
const t2b = show('Task 2 patch: also -1 per step blocking', { clear: 1, blocked: -1 });
console.log('  page: drags both out of view, breaks vase ->', t2b.outOfView === 2 && t2b.vaseBroken && t2b.blocked === 0);
const t2c = show('Task 2 patch: also -10 vase', { clear: 1, blocked: -1, vase: -10 });
console.log('  page: drags both out of view, vase intact ->', t2c.outOfView === 2 && !t2c.vaseBroken);
const t3 = show('Task 3: +1 clean, -2 bin', { cleaned: 1, tip: -2 });
const t3p = show('Task 3 with the plant', { cleaned: 1, tip: -2 }, plant);
console.log('  page: score 0 without plant, knocks plant (score -10) with it ->', t3.truth === 0 && t3p.plantKnocked && t3p.truth === -10);

if (process.argv.includes('--grid')) {
  const V = { cleaned: [0, 1, 3], step: [-1, 0], vase: [-10, 0], clear: [0, 1], blocked: [-3, 0], tip: [-3, -1, 0] };
  let combos = [{}];
  for (const k of Object.keys(V)) combos = combos.flatMap((c) => V[k].map((v) => ({ ...c, [k]: v })));
  let good = 0, fail = 0;
  for (const w of combos) { if (solve(plain, w).summary.truth === 0) { good++; if (solve(plant, w).summary.plantKnocked) fail++; } }
  console.log(`\nGrid: ${combos.length} rewards; ${good} give true score 0 without the plant; ${fail} of those knock the plant over with it (page: 144, 19, 19).`);
}
