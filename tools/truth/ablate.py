# The ablation test for page 6.3: remove a probe's direction from the residual stream and measure
# what changes in the model's behavior.
# Usage: python3 tools/truth/ablate.py <scratch dir> <dir63.json> <out.json>
#
# dir63.json (from `node tools/test-page-6-3.mjs directions`) holds, for each property and layer,
# the direction of the default linear probe in the reduced coordinates of docs/data/6-3, and one
# random direction per layer. Each is mapped back to the 896 numbers of the residual stream (u, unit
# length). Ablation replaces the component along u with its average (from the same activations the
# reduction was fit on): h <- h - ((h - mean) . u) u, at every position, in the residual stream after
# the given layer only. dir63.json gives several random directions per layer; the output reports their
# average under 'random' and each one under 'randomEach'.
#   noun, quote: next-token loss on the held-out passages (odd-numbered), averaged over all positions,
#     over positions whose next token starts a noun (noun), and over positions whose next token is
#     a closing quotation mark (quote).
#   truth: the model's few-shot TRUE/FALSE judgment (judge.py) of 60 held-out city statements;
#     ablation applies at every position of the statement and question.
import json, sys
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from acts import load
from judge import Judge

W, dpath, outp = sys.argv[1:4]
D = json.load(open(dpath))
B = np.load(f'{W}/basis_63.npz')
P = json.load(open(f'{W}/passages.json'))[:34]
S = json.load(open(f'{W}/statements.json'))['statements']
tok, model = load('Qwen/Qwen2.5-0.5B')
layers = model.model.layers
state = {'u': None, 'mu': None, 'L': None}


def hook_for(L):
    def f(mod, inp, out):
        if state['L'] != L or state['u'] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        u = state['u'].to(h.dtype); mu = state['mu'].to(h.dtype)
        h -= ((h - mu) @ u).unsqueeze(-1) * u
        return out
    return f


# Hidden state index L is the output of decoder layer L - 1 (index 0 is the embedding), except the last:
# Hugging Face's hidden_states[24] is taken after the final normalization, so the reduction for layer 24
# was fit on normalized activations and the ablation for layer 24 is applied to the output of that norm.
for L in range(1, len(layers)):
    layers[L - 1].register_forward_hook(hook_for(L))
model.model.norm.register_forward_hook(hook_for(len(layers)))


def set_dir(key, L, w):
    V = B[f'{key}{L}_V']; mu = B[f'{key}{L}_mu']
    u = V.T @ np.array(w, dtype=np.float64)
    n = np.linalg.norm(u)
    if n < 1e-9 or L == 0:
        state['u'] = None; return False
    state['u'] = torch.tensor(u / n, dtype=torch.float32); state['mu'] = torch.tensor(mu, dtype=torch.float32); state['L'] = L
    return True


# Token-level losses on held-out passages.
test_passages = [q for i, q in enumerate(P) if i % 2 == 1]


@torch.no_grad()
def token_losses():
    tot, rel_noun, rel_quote = [], [], []
    for q in test_passages:
        ids = torch.tensor([q['ids']])
        logp = model(input_ids=ids).logits[0].float().log_softmax(-1)
        nll = -logp[:-1].gather(1, ids[0, 1:, None])[:, 0].numpy()
        tot += list(nll)
        toks = q['tokens']
        for j in range(len(toks) - 1):
            nxt = toks[j + 1]
            if nxt['keep'] and nxt['noun']:
                rel_noun.append(nll[j])
            if '”' in nxt['t']:
                rel_quote.append(nll[j])
    return float(np.mean(tot)), float(np.mean(rel_noun)), float(np.mean(rel_quote))


J = Judge(tok, model)
held = [i for i, s in enumerate(S) if s['topic'] == 'cities' and s['form'] == 'plain' and s['item'] % 2 == 1][::2][:60]
lab = np.array([S[i]['label'] for i in held])


@torch.no_grad()
def judge_acc():
    d = np.array([J(S[i]['text'])[0] for i in held])
    return float(((d > 0) == lab).mean()), float(np.mean(np.where(lab == 1, d, -d)))


import os
state['u'] = None
# Resumable: results are saved after each layer, and layers already in the output file are skipped.
if os.path.exists(outp):
    out = json.load(open(outp))
else:
    out = {'layers': D['layers'], 'base': {'loss': token_losses(), 'judge': judge_acc()}, 'probe': {}, 'random': {}}
out.setdefault('randomEach', {})
print('base', out['base'], flush=True)


def run(w, prop):
    key = 'c' if prop == 'truth' else 't'
    if not set_dir(key, L, w):
        return None
    r = judge_acc() if prop == 'truth' else token_losses()
    state['u'] = None
    return r


for L in D['layers']:
    if str(L) not in out['probe']:
        out['probe'][str(L)] = {prop: run(D['probes'][prop][str(L)], prop) for prop in ('noun', 'quote', 'truth')}
    if str(L) not in out['randomEach']:
        R = D['random'][str(L)]
        R = R if isinstance(R[0], list) else [R]
        each = []
        for w in R:
            tl = run(w, 'noun')  # the same random direction serves both token properties
            each.append({'noun': tl, 'quote': tl, 'truth': run(w, 'truth')})
        out['randomEach'][str(L)] = each
        out['random'][str(L)] = {p: (None if each[0][p] is None else [float(np.mean([e[p][j] for e in each])) for j in range(len(each[0][p]))])
                                 for p in ('noun', 'quote', 'truth')}
    print(L, out['probe'][str(L)], out['random'][str(L)], flush=True)
    json.dump(out, open(outp, 'w'))
