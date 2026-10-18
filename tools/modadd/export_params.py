"""Export the final parameters of the main run for page 6.7, which runs the model in the browser.

Usage:
  python3 export_params.py --pt runs/p59_f0.3_wd1.0_s0.pt --out ../../docs/data/6-7/model.json

Every parameter matrix is stored as float16, little-endian, row-major, base64, under its name in
modadd.Transformer (shapes listed in the file). The script checks that the float16 parameters give
the same predictions as the float32 ones on all p * p pairs, and stores two derived quantities:
  train_mask       the training pairs (numpy.packbits order, base64), as in docs/data/modadd/
  neuron_freq      for each MLP neuron, the frequency k that explains the largest share of the
                   variance of its activation over all pairs (the share of the squared 2D Fourier
                   coefficients, constant term excluded, at the terms in 1, cos(w_k a), sin(w_k a)
                   times 1, cos(w_k b), sin(w_k b)), and that share (neuron_share).
tools/test-page-6-7.mjs recomputes both from the exported parameters.
"""
import argparse
import base64
import json
import os
import numpy as np
import torch
from modadd import Transformer, all_pairs, split, fourier_basis

ap = argparse.ArgumentParser()
ap.add_argument('--pt', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--p', type=int, default=59)
ap.add_argument('--frac', type=float, default=0.3)
args = ap.parse_args()
p = args.p
torch.set_num_threads(1)
model = Transformer(p)
model.load_state_dict(torch.load(args.pt))
A, B = all_pairs(p)
with torch.no_grad():
    pred32 = model(A, B).argmax(-1)
    sd = {k: v.half().float() for k, v in model.state_dict().items()}
    m16 = Transformer(p)
    m16.load_state_dict(sd)
    pred16 = m16(A, B).argmax(-1)
assert torch.equal(pred32, pred16), 'float16 parameters change a prediction'
Y = (A + B) % p
print('float16 accuracy on all pairs', (pred16 == Y).float().mean().item())

# Neuron frequencies, from the float16 parameters.
with torch.no_grad():
    E = m16.W_E
    x = torch.stack([E[A], E[B], E[torch.full_like(A, p)]], 1) + m16.W_pos
    k_ = torch.einsum('npd,hde->nhpe', x, m16.W_K)
    v_ = torch.einsum('npd,hde->nhpe', x, m16.W_V)
    q_ = torch.einsum('nd,hde->nhe', x[:, 2], m16.W_Q)
    att = (torch.einsum('nhe,nhpe->nhp', q_, k_) / np.sqrt(32)).softmax(-1)
    r = x[:, 2] + torch.einsum('nhe,hed->nd', torch.einsum('nhp,nhpe->nhe', att, v_), m16.W_O)
    act = torch.relu(r @ m16.W_in + m16.b_in).double().numpy().reshape(p, p, -1)
F = fourier_basis(p)
C = np.einsum('ia,abn,jb->ijn', F, act, F)
tot = (C ** 2).sum((0, 1)) - C[0, 0] ** 2
share = []
for k in range(1, (p - 1) // 2 + 1):
    ix = [0, 2 * k - 1, 2 * k]
    share.append(((C[np.ix_(ix, ix)] ** 2).sum((0, 1)) - C[0, 0] ** 2) / tot)
share = np.array(share)
freq = share.argmax(0) + 1

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).astype('<f2').tobytes()).decode()
out = {'p': p, 'd_model': 128, 'heads': 4, 'd_head': 32, 'd_mlp': 512, 'frac': args.frac,
       'source': os.path.basename(args.pt),
       'shapes': {k: list(v.shape) for k, v in sd.items()},
       'params': {k: b64(v.numpy()) for k, v in sd.items()},
       'train_mask': base64.b64encode(np.packbits(split(p, args.frac))).decode(),
       'neuron_freq': freq.tolist(),
       'neuron_share': [round(float(s), 4) for s in share.max(0)]}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print('wrote', args.out, os.path.getsize(args.out), 'bytes; neuron frequencies',
      {int(k): int((freq == k).sum()) for k in np.unique(freq)})
