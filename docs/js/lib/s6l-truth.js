// The shared dataset of true and false statements (pages 6.2, 6.3, 7.2), decoded from
// docs/data/truth/statements.json. See tools/truth/README.md for how the file was made.
import { decodeF16, rows } from './s6l-probe.js';

export const TOPICS = { cities: 'cities', spanish: 'Spanish words', elements: 'chemical elements' };
export const FORMS = { plain: 'plain', neg: 'negated', para: 'paraphrased' };

export function decodeTruth(json) {
  const k = json.k;
  const S = json.statements.map((s, i) => ({ ...s, i }));
  const X = rows(decodeF16(json.X), k);
  const pos = rows(decodeF16(json.pairs.pos), k), neg = rows(decodeF16(json.pairs.neg), k);
  const alicePos = rows(decodeF16(json.alice.pos), k), aliceNeg = rows(decodeF16(json.alice.neg), k);
  // Alice rows are stored for plain statements only; map statement index to Alice row.
  const aliceRow = new Map(json.alice.rows.map((si, r) => [si, r]));
  const pick = (f) => S.filter(f);
  return {
    k, json, S, X, pos, neg, alicePos, aliceNeg, aliceRow, pick,
    // Items with an even number are the training half; odd items are held out.
    isTrain: (s) => s.item % 2 === 0,
  };
}

export async function loadTruth(url) {
  const r = await fetch(url);
  return decodeTruth(await r.json());
}
