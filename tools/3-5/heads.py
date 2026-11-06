"""Precompute attention patterns, per-token loss, and head-type scores of GPT-2 small for page 3.5.

Usage: HF_HOME=/home/claude/hf python3 heads.py OUT.json
Model: openai-community/gpt2 (Hugging Face), eager attention, float32, CPU.
Every input starts with <|endoftext|>. The random sequence for display uses numpy seed 0;
the head scores are averaged over 100 further random sequences (seeds 1 to 100).
Random tokens are drawn uniformly from ids 1000 to 49999 (skips the byte-level tokens).
"""
import os, sys, json, base64
os.environ.setdefault('HF_HOME', '/home/claude/hf')
import numpy as np, torch
torch.set_num_threads(1)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL = 'openai-community/gpt2'
T = 20          # length of the random block; the input is BOS + block + block
NSCORE = 100
tok = GPT2TokenizerFast.from_pretrained(MODEL)
m = GPT2LMHeadModel.from_pretrained(MODEL, attn_implementation='eager').eval()
BOS = tok.bos_token_id

def run(ids):
    x = torch.tensor([ids])
    with torch.no_grad():
        o = m(x, output_attentions=True)
    att = torch.stack([a[0] for a in o.attentions]).numpy()  # [12 layers, 12 heads, n, n]
    lp = torch.log_softmax(o.logits[0], -1)
    loss = [None] + [float(-lp[i - 1, ids[i]]) for i in range(1, len(ids))]
    return att, loss

def rand_block(seed):
    return np.random.RandomState(seed).randint(1000, 50000, size=T).tolist()

def scores(att, n_off):
    """prev: mean attention from i to i-1 (i >= 2, skipping BOS as target);
    ind: mean attention from i in the second copy to i - T + 1 (the token after the earlier copy);
    dup: mean attention from i in the second copy to i - T (the earlier copy of the same token)."""
    n = att.shape[-1]
    i = np.arange(2, n)
    prev = att[:, :, i, i - 1].mean(-1)
    j = np.arange(1 + T, n)  # second copy positions
    ind = att[:, :, j, j - T + 1].mean(-1)
    dup = att[:, :, j, j - T].mean(-1)
    bos = att[:, :, 1:, 0].mean(-1)
    return prev, ind, dup, bos

# Head scores over many random repeated sequences.
P = np.zeros((12, 12)); I = np.zeros((12, 12)); D = np.zeros((12, 12)); Bo = np.zeros((12, 12))
loss1, loss2 = [], []
for s in range(1, NSCORE + 1):
    blk = rand_block(s)
    att, loss = run([BOS] + blk + blk)
    p, i_, d, b = scores(att, T)
    P += p; I += i_; D += d; Bo += b
    loss1.append(np.mean(loss[2:T + 1])); loss2.append(np.mean(loss[T + 2:]))
P /= NSCORE; I /= NSCORE; D /= NSCORE; Bo /= NSCORE

def pack(att):
    n = att.shape[-1]
    tri = np.concatenate([att[:, :, r, :r + 1] for r in range(n)], axis=-1)  # [12,12,n(n+1)/2]
    q = np.clip(np.round(tri * 255), 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode()

NATURAL = [
    ('sentence', 'The cat sat on the mat because it was warm, and the dog slept by the door until the rain stopped.'),
    ('names', 'Mr and Mrs Dursley were proud to say that they were perfectly normal. Then Harry Potter arrived, and nothing about Harry Potter was normal.'),
]
inputs = []
blk = rand_block(0)
for key, ids, label in [('random', [BOS] + blk + blk, 'Random tokens, repeated')] + [(k, [BOS] + tok(t).input_ids, t) for k, t in NATURAL]:
    att, loss = run(ids)
    inputs.append({'key': key, 'label': label, 'tokens': [tok.decode([t]) for t in ids], 'loss': [None if v is None else round(v, 3) for v in loss],
                   'n': len(ids), 'att': pack(att)})
    print(key, len(ids))

r = lambda a: np.round(a, 3).tolist()
out = {'meta': {'model': MODEL, 'revision': getattr(m.config, '_commit_hash', None), 'T': T, 'score_sequences': NSCORE,
                'random_token_range': [1000, 49999], 'display_seed': 0, 'score_seeds': [1, NSCORE]},
       'scores': {'prev': r(P), 'ind': r(I), 'dup': r(D), 'bos': r(Bo)},
       'loss_copies': {'first_mean': round(float(np.mean(loss1)), 3), 'second_mean': round(float(np.mean(loss2)), 3),
                       'first_min': round(float(np.min(loss1)), 3), 'second_max': round(float(np.max(loss2)), 3)},
       'inputs': inputs}
json.dump(out, open(sys.argv[1], 'w'))
order = lambda S: sorted(((S[l][h], f'{l}.{h}') for l in range(12) for h in range(12)), reverse=True)[:12]
print('prev', order(P)); print('ind', order(I)); print('dup', order(D)); print(out['loss_copies'])
