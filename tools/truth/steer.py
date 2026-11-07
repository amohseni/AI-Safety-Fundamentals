# The use test for page 7.2: steer the model along a probe's direction while it judges statements.
# Usage: python3 tools/truth/steer.py <scratch dir> <directions.json> <out.json>
#
# directions.json (from tools/truth/directions.mjs) holds each probe's weight vector in the reduced
# coordinates of the shipped data, and which reduction ("statements", "pairs" or "alice") it lives in.
# Each direction is mapped back into the model's residual stream (896 numbers), made a unit vector u,
# and scaled as Marks and Tegmark (2023) do: by the distance between the means of the true and the
# false training statements along u (the random control direction takes the plain supervised
# probe's scale instead). Steering with strength a adds a * scale * u to the residual
# stream after layer 13 at the statement's last two tokens, in the few-shot prompt of judge.py.
import json, sys
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from acts import load
from judge import Judge

W, dpath, outp = sys.argv[1:4]
D = json.load(open(dpath))
S = json.load(open(f'{W}/statements.json'))['statements']
A = np.load(f'{W}/acts_plain.npy')[13].astype(np.float64)
bases = {k: np.load(f'{W}/basis_{k}.npz') for k in ('statements', 'pairs', 'alice')}
STRENGTHS = D['strengths']
rows = D['test_rows']
tok, model = load('Qwen/Qwen2.5-0.5B')
J = Judge(tok, model)
lab = np.array([s['label'] for s in S])
train = np.array(D['train_rows'])

import os
# Resumable: results are saved after each probe, and probes already in the output file are skipped.
if os.path.exists(outp):
    out = json.load(open(outp))
else:
    out = {'strengths': STRENGTHS, 'rows': rows, 'layer': 13, 'positions': 'last two tokens of the statement',
           'base': [round(J(S[i]['text'])[0], 3) for i in rows], 'probes': {}}
for name, d in D['directions'].items():
    if name in out['probes']:
        continue
    B = bases[d['basis']]
    u = B['V'].T @ np.array(d['w'], dtype=np.float64)
    u /= np.linalg.norm(u)
    # Sign: positive steering should push toward "true" on the plain training statements.
    proj = A[train] @ u
    sep = proj[lab[train] == 1].mean() - proj[lab[train] == 0].mean()
    if sep < 0:
        u, sep = -u, -sep
    if name == 'random':
        # The control direction gets the scale of the plain supervised probe, so equal strengths
        # mean vectors of equal length.
        sep = out['probes']['supervised-plain']['sep']
    vec = torch.tensor(u * sep, dtype=torch.float32)
    res = []
    for a in STRENGTHS:
        res.append([round(J(S[i]['text'], vec=a * vec, layers=(12,))[0], 3) if a != 0 else out['base'][k] for k, i in enumerate(rows)])
    out['probes'][name] = {'sep': round(float(sep), 3), 'diff': res}
    print(name, 'sep', round(float(sep), 2), [round(float(np.mean(r)), 3) for r in res], flush=True)
    json.dump(out, open(outp, 'w'))
