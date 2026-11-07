# Page 4.4: evaluates the two trained addition models (runs/direct-L1.pt, runs/carries-L1.pt) and writes
# docs/data/4-4/addition.json.
#   random   sums of two random n-digit numbers (as in training)
#   chained  sums built so that a carry starts in the units column and runs through every column
#            (each other column's digits add to 9), the hardest case for carries
# Usage: python3 tools/4-4/evaluate.py
import json, os, random, sys
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import addition as A

ROOT = os.path.dirname(os.path.dirname(HERE))
torch.set_num_threads(1)


def chained(n, r):
    """Two n-digit numbers whose sum carries from the units column through all n columns."""
    while True:
        da, db = [], []
        u = r.randint(1, 9)
        da.append(u); db.append(r.randint(10 - u, 9))        # units column makes a carry
        for i in range(1, n):
            x = r.randint(1, 8) if i == n - 1 else r.randint(0, 9)   # leading digits of both numbers nonzero
            da.append(x); db.append(9 - x)                   # 9 in the column passes the carry on
        a = sum(d * 10 ** i for i, d in enumerate(da)); b = sum(d * 10 ** i for i, d in enumerate(db))
        if n == 1 or (a >= 10 ** (n - 1) and b >= 10 ** (n - 1)):
            return a, b


def acc(model, mode, pairs):
    got, want = A.answer_batch(model, pairs, mode)
    return [g.split('>')[-1] == w.split('>')[-1] for g, w in zip(got, want)], got, want


out = {'models': {}}
for mode in ['direct', 'carries']:
    info = json.load(open(os.path.join(HERE, 'runs', f'{mode}-L1.json')))
    A.CFG.update(info['config'])
    q, ans = A.example(1, 1, mode)
    model = A.Model(len(q) + len(ans))
    model.load_state_dict(torch.load(os.path.join(HERE, 'runs', f'{mode}-L1.pt')))
    model.eval()
    res = {'parameters': info['parameters'], 'steps': info['steps'], 'config': info['config'], 'tokens_written': len(ans),
           'final_loss': info['log'][-1]['loss'], 'train_seconds': info['seconds'],
           'accuracy_by_digits': info['accuracy_by_digits'], 'accuracy_by_chain': info['accuracy_by_chain'],
           'chained_by_digits': {}, 'examples': {}, 'chained_examples': {}}
    r = random.Random(31)
    for n in range(1, A.W + 1):
        pairs = [chained(n, r) for _ in range(500)]
        ok, _, _ = acc(model, mode, pairs)
        res['chained_by_digits'][n] = sum(ok) / len(ok)
    for key, gen, seed in [('examples', lambda n, rr: (A.number(n, rr), A.number(n, rr)), 4242), ('chained_examples', chained, 5151)]:
        rr = random.Random(seed)
        for n in range(1, A.W + 1):
            ps = [gen(n, rr) for _ in range(5)]
            ok, got, want = acc(model, mode, ps)
            res[key][n] = [{'a': a, 'b': b, 'output': g} for (a, b), g in zip(ps, got)]
    out['models'][mode] = res
    print(mode, {k: round(v, 3) for k, v in res['chained_by_digits'].items()}, flush=True)
json.dump(out, open(os.path.join(ROOT, 'docs', 'data', '4-4', 'addition.json'), 'w'))
