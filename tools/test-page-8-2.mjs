// Behaviors quoted on page 8.2, recomputed with the page's own MDP (docs/js/lib/8-cleaning.js).
// Run: node tools/test-page-8-2.mjs          (about 15 seconds, most of it the 144-reward search behind task 3;
// --no-grid skips that search)
import { buildRoom, solve, ACTIONS, G, T, CLEAN_STEPS } from '../docs/js/lib/8-cleaning.js';

let fails = 0;
const check = (label, ok, got) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}: ${got}`); if (!ok) fails++; };
const plain = buildRoom({ plant: false }), plant = buildRoom({ plant: true });
console.log('states:', plain.nS, 'without the plant,', plant.nS, 'with it');
function show(name, w, room = plain) {
  const { path, summary: m } = solve(room, w);
  const moves = path.filter((p) => p.a >= 0).map((p) => `${ACTIONS[p.a]}@${G.xy(room.states[p.s].pos).join(',')}`).join(' ');
  console.log(`\n${name} ${JSON.stringify(w)}\n  reward ${m.reward} | true score ${m.truth} | messes left ${m.messesLeft} (out of view ${m.outOfView}) | cleaned ${m.cleaned} | bin tips ${m.tip} | plant ${m.plantKnocked} | vase broken ${m.vaseBroken} | steps blocking camera ${m.blocked} | drag steps ${m.dragged} | ended ${m.ended}`);
  console.log('  ' + moves.replace(/( wait@\S+)+$/, ' then waits'));
  return m;
}
check('panel: a shift of 30 steps; cleaning takes 3 steps', T === 30 && CLEAN_STEPS === 3, `${T} ${CLEAN_STEPS}`);
const t1 = show('Task 1: +1 per mess cleaned', { cleaned: 1 });
check('t1: 6 messes cleaned, 5 of them its own (5 bin tips)', t1.cleaned === 6 && t1.tip === 5, `${t1.cleaned} ${t1.tip}`);
check('t1: the other original mess is still on the floor at the end, so the true task score is -1', t1.messesLeft === 1 && t1.truth === -1, `${t1.messesLeft} ${t1.truth}`);
const t2 = show('Task 2: +1 per step with no mess visible', { clear: 1 });
check('t2: the robot stays in front of the camera: 28 of the 30 steps with no mess visible, both messes still on the floor', t2.blocked === 28 && t2.reward === 28 && t2.messesLeft === 2 && t2.cleaned === 0, `${t2.blocked} ${t2.reward} ${t2.messesLeft}`);
const t2b = show('Task 2 patch: also -1 per step blocking', { clear: 1, blocked: -1 });
check('t2: with -1 for blocking the camera, it drags both messes out of view and breaks the vase on the way', t2b.outOfView === 2 && t2b.messesLeft === 2 && t2b.vaseBroken && t2b.blocked === 0 && t2b.dragged > 0, `${t2b.outOfView} ${t2b.vaseBroken} ${t2b.blocked}`);
const t2c = show('Task 2 patch: also -10 vase', { clear: 1, blocked: -1, vase: -10 });
check('t2: with -10 for the vase as well, the vase is intact and the messes are still only hidden', t2c.outOfView === 2 && t2c.messesLeft === 2 && !t2c.vaseBroken && t2c.blocked === 0, `${t2c.outOfView} ${t2c.vaseBroken}`);
check('t2: the route through the vase is as short as the way around (same number of steps with no mess visible)', t2b.reward === t2c.reward && t2b.dragged === t2c.dragged, `${t2b.reward} ${t2c.reward}`);
const t3 = show('Task 3: +1 clean, -2 bin', { cleaned: 1, tip: -2 });
const t3p = show('Task 3 with the plant', { cleaned: 1, tip: -2 }, plant);
check('t3: +1 per mess cleaned and -2 per bin tip leaves the first room clean, with true task score 0', t3.truth === 0 && t3.messesLeft === 0 && t3.tip === 0, `${t3.truth} ${t3.messesLeft}`);
check('t3: in the room with the plant the same reward knocks the plant over: one more mess cleaned, true task score -10', t3p.plantKnocked && t3p.cleaned === t3.cleaned + 1 && t3p.truth === -10, `${t3p.plantKnocked} ${t3p.cleaned} ${t3p.truth}`);

if (!process.argv.includes('--no-grid')) {
  const V = { cleaned: [0, 1, 3], step: [-1, 0], vase: [-10, 0], clear: [0, 1], blocked: [-3, 0], tip: [-3, -1, 0] };
  let combos = [{}];
  for (const k of Object.keys(V)) combos = combos.flatMap((c) => V[k].map((v) => ({ ...c, [k]: v })));
  let good = 0, fail = 0;
  for (const w of combos) { if (solve(plain, w).summary.truth === 0) { good++; if (solve(plant, w).summary.plantKnocked) fail++; } }
  console.log(`\nGrid: ${combos.length} rewards; ${good} give true score 0 without the plant; ${fail} of those knock the plant over with it (page: 144, 19, 19).`);
  check('t3: on a grid of 144 rewards, 19 give a true task score of 0 in the first room, and all 19 knock the plant over in the second', combos.length === 144 && good === 19 && fail === 19, `${combos.length} ${good} ${fail}`);
} else console.log('\n(skipped: the 144-reward grid behind task 3)');
console.log(fails ? `${fails} FAILED` : 'all checks passed');
process.exit(fails ? 1 : 0);
