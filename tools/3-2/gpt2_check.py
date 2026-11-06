"""Checks the page 3.2 claim about GPT-2 small's own token embeddings (over all 50,257 tokens).

Usage: HF_HOME=/home/claude/hf python3 -I tools/3-2/gpt2_check.py
Model: openai-community/gpt2. Prints the nearest tokens to ' bank' and to ' king' - ' man' + ' woman'
(unit-length rows of the token embedding matrix, cosine similarity), with and without the inputs.
Output on 2026-11-06:
  ' bank': ' banks' 0.756, ' Bank' 0.734, ' banking' 0.660, 'bank' 0.646, 'Bank' 0.631
  king - man + woman, inputs allowed: ' king' 0.741, ' queen' 0.694, ' princess' 0.595
  inputs left out: ' queen' 0.694, ' princess' 0.595, ' Queen' 0.584
"""
import numpy as np
import torch
torch.set_num_threads(1)
from transformers import GPT2Model, GPT2TokenizerFast

tok = GPT2TokenizerFast.from_pretrained('openai-community/gpt2')
E = GPT2Model.from_pretrained('openai-community/gpt2').wte.weight.detach().numpy().astype(np.float64)
X = E / np.linalg.norm(E, axis=1, keepdims=True)

def tid(w):
    ids = tok.encode(' ' + w)
    assert len(ids) == 1, w
    return ids[0]

def show(v, k, exclude=()):
    s = X @ (v / np.linalg.norm(v))
    for e in exclude:
        s[e] = -np.inf
    return ', '.join(f'{tok.decode([int(i)])!r} {s[i]:.3f}' for i in np.argsort(-s)[:k])

b = tid('bank')
print("' bank':", show(X[b], 5, [b]))
ids = [tid(w) for w in ('king', 'man', 'woman')]
v = X[ids[0]] - X[ids[1]] + X[ids[2]]
print('king - man + woman, inputs allowed:', show(v, 3))
print('inputs left out:', show(v, 3, ids))
