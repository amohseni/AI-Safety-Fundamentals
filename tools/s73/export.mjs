// Packs the trained network (train.py) and its subspaces and probes (analyze.py) into
// docs/data/7-3/policy.json, with every array stored as base64 float32 (little-endian).
// Usage: node tools/s73/export.mjs <net.json> <analysis.json> [kGoal=32] [kDoor=12]
import fs from 'node:fs';
import path from 'node:path';

const [netF, anF, kG, kD] = process.argv.slice(2);
const net = JSON.parse(fs.readFileSync(netF)), an = JSON.parse(fs.readFileSync(anF));
const kGoal = Number(kG || 32), kDoor = Number(kD || 12);
const pack = (rows) => Buffer.from(Float32Array.from(rows.flat(2)).buffer).toString('base64');
const out = {
  note: 'Page 7.3 policy network: 316 inputs, 128 ReLU, 128 ReLU, 4 action scores. Built by tools/s73 (see its README). Arrays are base64 float32, row-major.',
  sizes: net.sizes, kGoal, kDoor, layer: an.layer,
  goalProbeAcc: an.goalProbeAcc,
  W: net.W.map(pack), b: net.b.map(pack),
  mean: pack(an.mean),
  goal: pack(an.goal.slice(0, kGoal)), door: pack(an.door.slice(0, kDoor)),
  goalProbe: { W: pack(an.goalProbe.W), b: pack(an.goalProbe.b), n: an.goalProbe.b.length },
  doorProbe: an.doorProbe.map((p) => ({ W: pack(p.W), b: pack(p.b), n: p.b.length })),
};
const file = path.resolve(path.dirname(new URL(import.meta.url).pathname), '../../docs/data/7-3/policy.json');
fs.mkdirSync(path.dirname(file), { recursive: true });
fs.writeFileSync(file, JSON.stringify(out));
console.log(`wrote ${file}: ${(fs.statSync(file).size / 1024).toFixed(0)} KB`);
