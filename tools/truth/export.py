# Writes the compact data files that pages 6.2, 6.3 and 7.2 load.
# Usage: python3 tools/truth/export.py <scratch dir> <docs/data>
#
# Activations are reduced to their top principal directions (fit on the rows that are stored,
# separately for each layer and each kind of activation), divided by one overall scale so that
# typical coordinates are near 1, and stored as base64 float16. The principal directions themselves
# (needed to map a probe back into the model) are kept in <scratch>/basis_*.npz, not shipped.
import base64, json, sys
import numpy as np

W, OUT = sys.argv[1:3]
K = 32          # principal directions kept for statements and token activations
LAYER = 13      # layer used on pages 6.2 and 7.2 (of 24)
TOKEN_LAYERS = list(range(0, 25, 2))
MODEL = {'id': 'Qwen/Qwen2.5-0.5B', 'revision': '060db6499f32faf8b98477b0a26969ef7d8b9987', 'layers': 24, 'd': 896}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype='<f2').tobytes()).decode()


def reduce(X, k=K):
    X = X.astype(np.float64)
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    Z = (X - mu) @ Vt[:k].T
    scale = Z.std()
    if not scale > 1e-9:  # every row identical (the embedding of the final period): store zeros
        return np.zeros_like(Z, dtype=np.float32), dict(mu=mu, V=Vt[:k], scale=1.0), 0.0
    var = float((s[:k] ** 2).sum() / (s ** 2).sum())
    return (Z / scale).astype(np.float32), dict(mu=mu, V=Vt[:k], scale=scale), var


def b64i8(Z, clip=4.0):
    # Coordinates are already scaled to unit overall spread; store them as int8 steps of clip / 127.
    q = np.clip(np.round(Z / clip * 127), -127, 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode()


def write(path, obj):
    s = json.dumps(obj, separators=(',', ':'), ensure_ascii=False)
    open(path, 'w', encoding='utf-8').write(s)
    print(path, f'{len(s) / 1e6:.2f} MB')


S = json.load(open(f'{W}/statements.json'))['statements']
meta = [{k: s[k] for k in ('text', 'label', 'topic', 'form', 'item', 'alice')} for s in S]
plain_idx = [i for i, s in enumerate(S) if s['form'] == 'plain']

# ---- Statements at the chosen layer (pages 6.2 and 7.2) ----
A = np.load(f'{W}/acts_plain.npy')
Z, basis, var = reduce(A[LAYER])
np.savez(f'{W}/basis_statements.npz', **basis)
Ap = np.load(f'{W}/acts_pos.npy')[0]
An = np.load(f'{W}/acts_negsfx.npy')[0]
# The two halves of each contrast pair share one reduction, fit on both.
Zpair, bpair, varpair = reduce(np.concatenate([Ap, An]))
np.savez(f'{W}/basis_pairs.npz', **bpair)
Aap = np.load(f'{W}/acts_alice_pos.npy')[0]
Aan = np.load(f'{W}/acts_alice_negsfx.npy')[0]
Zal, bal, varal = reduce(np.concatenate([Aap, Aan]))
np.savez(f'{W}/basis_alice.npz', **bal)
n, na = len(S), len(plain_idx)
write(f'{OUT}/truth/statements.json', {
    'model': MODEL, 'layer': LAYER, 'k': K,
    'note': 'Residual stream after layer 13 at the last token, reduced to the top 32 principal directions.',
    'variance': {'statements': round(var, 3), 'pairs': round(varpair, 3), 'alice': round(varal, 3)},
    'statements': meta,
    'X': b64(Z),
    'pairs': {'templates': ['Is the following statement true or false? {s} Answer: True', 'Is the following statement true or false? {s} Answer: False'], 'pos': b64(Zpair[:n]), 'neg': b64(Zpair[n:])},
    'alice': {'templates': ['Is the following statement true or false? {s} Alice, who is always right, thinks it is {alice}. Answer: True',
                            'Is the following statement true or false? {s} Alice, who is always right, thinks it is {alice}. Answer: False'],
              'rows': plain_idx, 'pos': b64(Zal[:na]), 'neg': b64(Zal[na:])},
})

# ---- Weekdays (page 6.2) ----
wd = np.load(f'{W}/weekdays.npz')
Wz, wb, wvar = reduce(wd['acts'][LAYER], k=8)
write(f'{OUT}/6-2/weekdays.json', {'model': MODEL, 'layer': LAYER, 'k': 8, 'variance': round(wvar, 3),
                                   'day': wd['day'].tolist(), 'ctx': wd['ctx'].tolist(), 'X': b64(Wz)})

# ---- Page 6.3: every other layer, city statements and passage tokens ----
city = [i for i, s in enumerate(S) if s['topic'] == 'cities' and s['form'] == 'plain']
N_PASSAGES = 34  # about 2,400 kept tokens, to keep the file small
P = json.load(open(f'{W}/passages.json'))[:N_PASSAGES]
toks = [t for q in P for t in q['tokens'] if t['keep']]
T = np.load(f'{W}/acts_tokens.npy', mmap_mode='r')[:, :len(toks)]
layers = {}
bases = {}
for L in TOKEN_LAYERS:
    zc, bc, vc = reduce(A[L][city])
    zt, bt, vt = reduce(T[L])
    bases[f'c{L}'] = bc; bases[f't{L}'] = bt
    layers[L] = {'cities': b64(zc), 'tokens': b64i8(zt), 'variance': [round(vc, 3), round(vt, 3)]}
np.savez(f'{W}/basis_63.npz', **{f'{key}_{f}': v[f] for key, v in bases.items() for f in ('mu', 'V', 'scale')})
write(f'{OUT}/6-3/probe-data.json', {
    'model': MODEL, 'k': K, 'layers': TOKEN_LAYERS, 'data': layers, 'tokenClip': 4.0,
    'note': 'Per layer: cities, float16 coordinates of the 440 plain city statements; tokens, int8 coordinates (multiply by tokenClip / 127) of the kept tokens.',
    'cities': [{'text': S[i]['text'], 'label': S[i]['label'], 'item': S[i]['item']} for i in city],
    'tokens': [{'w': t['word'], 'noun': t['noun'], 'quote': t['quote'], 'p': pi} for pi, q in enumerate(P) for t in q['tokens'] if t['keep']],
    'passages': [q['text'] for q in P],
})
