"""Turn training records into the compact JSON files the site reads.

Usage:
  python3 export.py --runs <dir with .npz/.pt records> --out ../../docs/data/modadd

Writes:
  run-wd<wd>-f<frac>.json  one per grid setting (page 5.4)
  seeds.json               the five seeds at the main setting (pages 5.5 and 6.7)
  seeds-embed.json         the five seeds' final number embeddings (page 6.7)

Encodings:
  Correctness grids and the training-set mask are p*p bits, pair (a, b) at index a*p + b,
  packed with numpy.packbits (first pair in the high bit of byte 0), then base64.
  Embeddings are float32 little-endian, row-major (p rows of d_model), base64.
"""
import argparse
import base64
import glob
import itertools
import json
import math
import os
import re
import numpy as np
import torch
from modadd import (Transformer, fourier_basis, embedding_spectrum, ablate_embedding,
                    all_pairs, split)

ap = argparse.ArgumentParser()
ap.add_argument('--runs', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--p', type=int, default=113)
ap.add_argument('--main-frac', type=float, default=0.3)
ap.add_argument('--main-wd', type=float, default=1.0)
ap.add_argument('--key-share', type=float, default=0.9,
                help='key frequencies: the fewest largest frequencies that together hold this share '
                     'of the squared norm of the final number embeddings (constant term excluded)')
args = ap.parse_args()
torch.set_num_threads(1)
p = args.p
F = fourier_basis(p)
os.makedirs(args.out, exist_ok=True)
b64 = lambda arr: base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()
r4 = lambda xs: [round(float(x), 4) for x in xs]


def key_frequencies(W_E):
    spec, _ = embedding_spectrum(W_E, p, F)
    order = np.argsort(-spec)
    sq = spec[order] ** 2
    n = int(np.searchsorted(np.cumsum(sq) / sq.sum(), args.key_share) + 1)
    return sorted(int(k) + 1 for k in order[:n])


def restricted_acc(coef, keys, Y, test):
    """Test accuracy of the restricted logits (constant + key-frequency (a+b) terms)."""
    a = np.arange(p)
    s = (a[:, None] + a[None, :]) % p
    L = np.repeat(coef[0][None, :], p * p, axis=0).astype(np.float64)
    for k in keys:
        for j, f in enumerate((np.cos, np.sin)):
            phi = f(2 * math.pi * k * s / p).reshape(-1)
            phi = phi / np.linalg.norm(phi)
            L += phi[:, None] * coef[2 * k - 1 + j][None, :]
    return float((L.argmax(1) == Y)[test].mean())


def load(path):
    z = np.load(path, allow_pickle=False)
    cfg = eval(str(z['config']))
    return z, cfg


A, B = all_pairs(p)
Y = ((A + B) % p).numpy()
records = sorted(glob.glob(os.path.join(args.runs, f'p{p}_*.npz')))
index = []
for path in records:
    z, cfg = load(path)
    if cfg['seed'] != 0:
        continue
    test = ~z['mask']
    W_final = z['W_E'][-1]
    keys = key_frequencies(W_final)
    spectra = [embedding_spectrum(W, p, F)[0] for W in z['W_E']]
    out = {
        'p': p, 'wd': cfg['wd'], 'frac': cfg['frac'], 'seed': cfg['seed'], 'lr': cfg['lr'],
        'steps_total': cfg['steps'], 'd_model': cfg['d_model'], 'heads': cfg['heads'], 'd_mlp': cfg['d_mlp'],
        'key_freqs_final': keys,
        'curve': {'step': z['m_step'].tolist(), 'train_acc': r4(z['train_acc']), 'test_acc': r4(z['test_acc']),
                  'train_loss': r4(z['train_loss']), 'test_loss': r4(z['test_loss'])},
        'train_mask': b64(np.packbits(z['mask'])),
        'ckpt': {
            'step': z['c_step'].tolist(),
            'correct': [b64(c) for c in z['correct']],
            'spectrum': [[round(float(x), 3) for x in s] for s in spectra],
            'restricted_test_acc': r4([restricted_acc(c, keys, Y, test) for c in z['logit_coef']]),
        },
    }
    name = f"run-wd{cfg['wd']}-f{cfg['frac']}.json"
    with open(os.path.join(args.out, name), 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    index.append({'file': name, 'wd': cfg['wd'], 'frac': cfg['frac'], 'steps_total': cfg['steps']})
    print(name, 'keys', keys, 'final test', out['curve']['test_acc'][-1])

# Five seeds at the main setting.
seeds, embeds = [], []
mask = split(p, args.main_frac)
test = torch.from_numpy(~mask)
for path in records:
    z, cfg = load(path)
    if not (cfg['frac'] == args.main_frac and cfg['wd'] == args.main_wd):
        continue
    model = Transformer(p, cfg['d_model'], cfg['heads'], cfg['d_mlp'])
    model.load_state_dict(torch.load(path.replace('.npz', '.pt')))
    W = model.W_E.detach().numpy().astype(np.float64)
    keys = key_frequencies(W)
    spec, const = embedding_spectrum(W, p, F)

    def acc(keep):
        We = torch.tensor(ablate_embedding(W, p, keep, F), dtype=torch.float32)
        with torch.no_grad():
            pred = model(A, B, W_E=We).argmax(-1).numpy()
        return round(float((pred == Y)[test.numpy()].mean()), 4)

    allk = list(range(1, (p - 1) // 2 + 1))
    with torch.no_grad():
        full = round(float((model(A, B).argmax(-1).numpy() == Y)[test.numpy()].mean()), 4)
    s = {
        'seed': cfg['seed'], 'steps_total': cfg['steps'], 'test_acc': full,
        'train_acc': round(float(z['train_acc'][-1]), 4),
        'key_freqs': keys,
        'spectrum': [round(float(x), 4) for x in spec], 'constant_norm': round(const, 4),
        'acc_keep_only_keys': acc(keys),
        'acc_remove_one': {str(k): acc([j for j in allk if j != k]) for k in allk},
        'acc_keep_subset_of_keys': [{'keep': list(c), 'acc': acc(list(c))}
                                    for n in range(len(keys) + 1) for c in itertools.combinations(keys, n)],
        'acc_remove_all_keys': acc([j for j in allk if j not in keys]),
        'restricted_test_acc': restricted_acc(z['logit_coef'][-1], keys, Y, ~mask),
        'curve': {'step': z['m_step'].tolist(), 'train_acc': r4(z['train_acc']), 'test_acc': r4(z['test_acc'])},
    }
    seeds.append(s)
    embeds.append({'seed': cfg['seed'], 'W_E': b64(W[:p].astype('<f4'))})
    print('seed', cfg['seed'], 'test', full, 'keys', keys, 'keep only keys', s['acc_keep_only_keys'],
          'remove each key', {k: s['acc_remove_one'][str(k)] for k in keys})

seeds.sort(key=lambda s: s['seed'])
embeds.sort(key=lambda s: s['seed'])
meta = {'p': p, 'frac': args.main_frac, 'wd': args.main_wd, 'key_share': args.key_share}
with open(os.path.join(args.out, 'seeds.json'), 'w') as f:
    json.dump({**meta, 'seeds': seeds}, f, separators=(',', ':'))
with open(os.path.join(args.out, 'seeds-embed.json'), 'w') as f:
    json.dump({**meta, 'd_model': 128, 'embeddings': embeds}, f, separators=(',', ':'))
with open(os.path.join(args.out, 'index.json'), 'w') as f:
    json.dump({'p': p, 'runs': index}, f, indent=1)
