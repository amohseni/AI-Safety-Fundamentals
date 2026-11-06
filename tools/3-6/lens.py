"""Precompute the logit lens of GPT-2 small for page 3.6.

For each prompt (with <|endoftext|> in front) and each position, the residual stream is read
at 25 points: the embedding, and in each layer L (0 to 11) after attention ("mid") and after
the MLP ("post"). Each reading is passed through the final layer norm and the unembedding,
as the model does with its last residual stream (nostalgebraist 2020, "interpreting GPT: the
logit lens"). Stored: the top 10 tokens and their probabilities after each layer (and at the
embedding), and, for each cell's top token, its probability before the layer, after attention,
and after the MLP.

Usage: HF_HOME=/home/claude/hf python3 lens.py OUT.json [--try]
"""
import os, sys, json
os.environ.setdefault('HF_HOME', '/home/claude/hf')
import torch
torch.set_num_threads(1)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL = 'openai-community/gpt2'
PROMPTS = [
    'The Eiffel Tower is located in the city of',
    'The Colosseum is located in the city of',
    'The capital of Japan is the city of',
    'Michael Jordan plays the sport of',
    'Monday, Tuesday, Wednesday,',
    'Water is made of hydrogen and',
]
TRY = ['The Space Needle is located in the city of', 'Barack Obama was born in the state of', 'The opposite of hot is',
       'The Colosseum is located in the city of', 'Monday, Tuesday, Wednesday,', 'The capital of Japan is the city of', 'One, two, three, four,',
       'The Statue of Liberty is in the city of', 'Water is made of hydrogen and']
tok = GPT2TokenizerFast.from_pretrained(MODEL)
m = GPT2LMHeadModel.from_pretrained(MODEL).eval()
BOS = tok.bos_token_id

def lens(x):
    return torch.softmax(m.lm_head(m.transformer.ln_f(x)), -1)

def run(prompt):
    ids = [BOS] + tok(prompt).input_ids
    attn_out, mlp_out = {}, {}
    hs = []
    for L, blk in enumerate(m.transformer.h):
        hs.append(blk.attn.register_forward_hook(lambda mod, i, o, L=L: attn_out.__setitem__(L, o[0])))
        hs.append(blk.mlp.register_forward_hook(lambda mod, i, o, L=L: mlp_out.__setitem__(L, o)))
    with torch.no_grad():
        o = m(torch.tensor([ids]), output_hidden_states=True)
        for h in hs: h.remove()
        pre = [o.hidden_states[0][0]]
        mids, posts = [], []
        x = pre[0]
        for L in range(12):
            mid = x + attn_out[L][0]
            post = mid + mlp_out[L][0]
            if L < 11:
                assert torch.allclose(post, o.hidden_states[L + 1][0], atol=1e-3)
            mids.append(mid); posts.append(post); x = post
        final = torch.softmax(o.logits[0], -1)
        assert torch.allclose(lens(posts[-1]), final, atol=1e-5)
        states = [pre[0]] + posts            # 13 readings: embedding, after layers 0..11
        P = torch.stack([lens(s) for s in states])  # [13, n, V]
        Pm = torch.stack([lens(s) for s in mids])   # [12, n, V]
    return ids, P, Pm

def pack(prompt):
    ids, P, Pm = run(prompt)
    n = len(ids)
    top = []
    for s in range(13):
        row = []
        for i in range(n):
            v, k = P[s, i].topk(10)
            cell = {'t': [tok.decode([int(j)]) for j in k], 'p': [round(float(x), 4) for x in v]}
            if s > 0:
                t = int(k[0]); L = s - 1
                cell['chg'] = [round(float(P[s - 1, i, t]), 4), round(float(Pm[L, i, t]), 4), round(float(P[s, i, t]), 4)]
            row.append(cell)
        top.append(row)
    return {'prompt': prompt, 'tokens': [tok.decode([t]) for t in ids], 'cells': top}

if '--try' in sys.argv:
    for p in TRY:
        ids, P, Pm = run(p)
        print(p, '->', [(tok.decode([int(P[s, -1].argmax())]), round(float(P[s, -1].max()), 2)) for s in range(13)])
    sys.exit()
json.dump({'meta': {'model': MODEL, 'revision': getattr(m.config, '_commit_hash', None)}, 'prompts': [pack(p) for p in PROMPTS]}, open(sys.argv[1], 'w'))
