"""Checks for page 3.5 on GPT-2 small: (1) remove previous-token head 4.11 (zero its output
before the output matrix) and measure the induction scores of the induction heads and the
loss on the second copy; (2) loss at the first token of the second copy.
Same random sequences as heads.py (seeds 1 to 100, T = 20). Prints a summary.
Usage: HF_HOME=/home/claude/hf python3 ablate.py
"""
import os
os.environ.setdefault('HF_HOME', '/home/claude/hf')
import numpy as np, torch
torch.set_num_threads(1)
from transformers import GPT2LMHeadModel
m = GPT2LMHeadModel.from_pretrained('openai-community/gpt2', attn_implementation='eager').eval()
T, N, BOS = 20, 100, 50256
IND = [(5, 5), (6, 9), (5, 1), (7, 10), (7, 2)]

def run(ids, ablate=()):
    hooks = []
    for (l, h) in ablate:
        def pre(mod, args, h=h):
            x = args[0].clone(); x[..., h * 64:(h + 1) * 64] = 0
            return (x,)
        hooks.append(m.transformer.h[l].attn.c_proj.register_forward_pre_hook(pre))
    with torch.no_grad():
        o = m(torch.tensor([ids]), output_attentions=True)
    for k in hooks: k.remove()
    att = torch.stack([a[0] for a in o.attentions]).numpy()
    lp = torch.log_softmax(o.logits[0], -1)
    loss = np.array([float(-lp[i - 1, ids[i]]) for i in range(1, len(ids))])  # loss[i-1] is for token i
    n = len(ids); j = np.arange(1 + T, n)
    ind = {f'{l}.{h}': float(att[l, h, j, j - T + 1].mean()) for (l, h) in IND}
    return ind, loss

res = {k: [] for k in ['base', 'no411', 'noprev']}
first2 = []
for s in range(1, N + 1):
    blk = np.random.RandomState(s).randint(1000, 50000, size=T).tolist()
    ids = [BOS] + blk + blk
    for key, ab in [('base', ()), ('no411', [(4, 11)]), ('noprev', [(4, 11), (3, 7), (2, 2), (6, 8)])]:
        ind, loss = run(ids, ab)
        res[key].append((ind, loss[T + 1:].mean(), loss[T]))  # loss[T] is for token T+1, the first of the second copy
for key, rows in res.items():
    ind = {k: np.mean([r[0][k] for r in rows]) for k in rows[0][0]}
    print(key, {k: round(v, 3) for k, v in ind.items()}, 'second-copy loss (tokens 2..T of copy 2)', round(np.mean([r[1] for r in rows]), 3),
          'first token of copy 2', round(np.mean([r[2] for r in rows]), 3), 'min', round(np.min([r[2] for r in rows]), 3))
