"""Precomputes GPT-2 small's next-token distributions for page 3.3 (sample the next token).

Writes docs/data/3-3/tree.json. For each prompt it stores:
  1. A tree of prefixes. Every node holds GPT-2's 20 most likely next tokens with their log probabilities
     at temperature 1, and, for each temperature on the page's grid, the offset needed to turn those
     into exact probabilities at that temperature (the log of the sum over all 50,257 tokens).
     Nodes are expanded best first, by the probability that a reader sampling at temperature 1 among
     the stored tokens reaches them. When a node is expanded, its children are its most likely tokens,
     taken in order until they hold 90% of the probability at temperature 1 (at most 20 of them at the
     root, 12 one token later, and 8 after that).
     The greedy path (always the most likely token) is expanded to its full length first.
  2. Whole continuations sampled by Hugging Face generate() at every temperature and top-p on the grid.

Usage: HF_HOME=/home/claude/hf [CACHE=<dir>] python3 -I tools/3-3/build_data.py docs/data/3-3/tree.json
Model: openai-community/gpt2 (GPT-2 small), revision recorded in the output. Seed 0.
"""
import os, sys, json, base64, heapq
import numpy as np
import torch
torch.set_num_threads(1)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

OUT = sys.argv[1]
NAME = 'openai-community/gpt2'
tok = GPT2TokenizerFast.from_pretrained(NAME)
model = GPT2LMHeadModel.from_pretrained(NAME).eval()
def revision():
    # The snapshot of the model files that was loaded (the commit on the Hugging Face hub).
    try:
        ref = os.path.join(os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')), 'hub', 'models--openai-community--gpt2', 'refs', 'main')
        return open(ref).read().strip()
    except OSError:
        return None
REV = revision()

PROMPTS = ['The capital of France is', 'Once upon a time', '1, 2, 3, 4,', 'Thank you very', 'My favorite food is']
TEMPS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
TOPPS = [round(0.1 * k, 1) for k in range(1, 11)]
DRAW_TOPPS = [0.5, 0.9, 1.0]                      # whole continuations are sampled for these top-p values only
K = 20
BUDGET = 2400
MAXDEPTH = 24
COVER, CAP = 0.9, 8
CAP_BY_DEPTH = {0: 20, 1: 12}                     # wider at the first two steps
DRAWS, DRAWLEN = 4, 30

@torch.no_grad()
def logits_after(seqs):
    """Last-position logits (float64) for a list of token-id lists of equal length."""
    x = torch.tensor(seqs)
    h = model.transformer(x).last_hidden_state[:, -1, :]
    return model.lm_head(h).double()      # scores only at the last position, to save time

def summarize(z):
    """z: logits over the vocabulary (1-D float64 tensor). Returns the node record."""
    lp = torch.log_softmax(z, -1)
    v, i = lp.topk(K)
    zmax = z.max()
    offs = []
    for T in TEMPS:
        # log sum_j exp(z_j / T) minus z_max / T
        offs.append(float(torch.logsumexp((z - zmax) / T, -1)))
    return {'ids': [int(t) for t in i], 'lp': [float(t) for t in v], 'off': offs}

def build_tree(prompt):
    base = tok(prompt).input_ids
    nodes = [dict(seq=[], parent=-1, tok=-1, depth=0, reach=1.0)]
    nodes[0].update(summarize(logits_after([base])[0]))
    kids = {0: []}

    def add_children(n, toks):
        # Run the parent prefix once, then all children as one-token continuations of its cache.
        with torch.no_grad():
            pk = model.transformer(torch.tensor([base + nodes[n]['seq']]), use_cache=True).past_key_values
            pk.batch_repeat_interleave(len(toks))
            h = model.transformer(torch.tensor([[t] for t in toks]), past_key_values=pk, use_cache=True).last_hidden_state[:, -1, :]
            Z = model.lm_head(h).double()
        out = []
        for t, z in zip(toks, Z):
            c = dict(seq=nodes[n]['seq'] + [t], parent=n, tok=t, depth=nodes[n]['depth'] + 1)
            c.update(summarize(z))
            nodes.append(c); kids[len(nodes) - 1] = []; kids[n].append(len(nodes) - 1)
            out.append(len(nodes) - 1)
        return out

    def chosen(n):
        p = np.exp(nodes[n]['lp'])
        take, cum = [], 0.0
        for j in range(K):
            take.append(j); cum += p[j]
            if cum >= COVER or len(take) >= CAP_BY_DEPTH.get(nodes[n]['depth'], CAP):
                break
        return [nodes[n]['ids'][j] for j in take], [p[j] for j in take]

    # Greedy path first.
    g = 0
    while nodes[g]['depth'] < MAXDEPTH:
        top = nodes[g]['ids'][0]
        g = add_children(g, [top])[0]
    greedy_end = g
    # Best-first expansion by reach probability.
    heap = [(-1.0, 0)]
    expanded = set()
    reach = {0: 1.0}
    while heap and len(nodes) < BUDGET:
        r, n = heapq.heappop(heap); r = -r
        if n in expanded or nodes[n]['depth'] >= MAXDEPTH:
            continue
        expanded.add(n)
        ids, ps = chosen(n)
        have = {nodes[c]['tok']: c for c in kids[n]}
        new = [t for t in ids if t not in have]
        if new:
            if len(nodes) + len(new) > BUDGET:
                break
            for c in add_children(n, new):
                have[nodes[c]['tok']] = c
            if len(nodes) % 200 < len(new):
                print('  nodes', len(nodes), flush=True)
        cover = sum(ps)
        for t, p in zip(ids, ps):
            c = have[t]
            reach[c] = r * p / cover
            heapq.heappush(heap, (-reach[c], c))
    return base, nodes, greedy_end

def pack_tree(nodes):
    n = len(nodes)
    parent = np.array([nd['parent'] for nd in nodes], dtype=np.int32)
    tokid = np.array([max(nd['tok'], 0) for nd in nodes], dtype=np.uint16)
    ids = np.array([nd['ids'] for nd in nodes], dtype=np.uint16)
    nlp = np.clip(np.round(-np.array([nd['lp'] for nd in nodes]) * 1000), 0, 65535).astype(np.uint16)
    offs = np.array([nd['off'] for nd in nodes])
    off = np.clip(np.round(offs * 5000), 0, 65535).astype(np.uint16)
    b = lambda a: base64.b64encode(a.tobytes()).decode()
    return {'n': n, 'parent': b(parent), 'tok': b(tokid), 'ids': b(ids), 'nlp': b(nlp), 'off': b(off)}

# Whole continuations that contain profanity or sexual-violence words are drawn again with a new seed
# (seed + 100000 per attempt). The distributions in the tree are never filtered.
import re
BLOCK = re.compile(r"\b(fuck\w*|shit\w*|cunt\w*|pussy\w*|nigg\w*|faggot\w*|whore\w*|slut\w*|bitch\w*|rape[ds]?|raping|porn\w*)", re.I)

@torch.no_grad()
def draws_once(base, T, p, seed):
    torch.manual_seed(seed)
    x = torch.tensor([base] * DRAWS)
    out = model.generate(x, do_sample=True, temperature=T, top_p=p, top_k=0, max_new_tokens=DRAWLEN,
                         pad_token_id=tok.eos_token_id, attention_mask=torch.ones_like(x))
    res = []
    for row in out[:, len(base):]:
        ids = row.tolist()
        if tok.eos_token_id in ids:
            ids = ids[:ids.index(tok.eos_token_id)]
        res.append(tok.decode(ids))
    return res

def draws(base, T, p, seed, first=None):
    res, attempt = first, 0
    while res is None or any(BLOCK.search(t) for t in res):
        res = draws_once(base, T, p, seed + 100000 * attempt)
        attempt += 1
    return res

doc = {'model': NAME, 'revision': REV, 'k': K, 'temps': TEMPS, 'topps': TOPPS, 'cover': COVER, 'cap': CAP,
       'budget': BUDGET, 'draw_topps': DRAW_TOPPS, 'maxdepth': MAXDEPTH, 'prompts': []}
vocab_needed = set()
# Each prompt's sampled continuations are cached in CACHE (if set), so an interrupted run can resume.
CACHE = os.environ.get('CACHE')
import time
for pi, prompt in enumerate(PROMPTS):
    sfile = os.path.join(CACHE, f'samples{pi}.json') if CACHE else None
    tfile = os.path.join(CACHE, f'tree{pi}.json') if CACHE else None
    t0 = time.time()
    if tfile and os.path.exists(tfile):
        with open(tfile) as f:
            base, packed, needed = json.load(f)
        needed = set(needed)
    else:
        base, nodes, gend = build_tree(prompt)
        packed = pack_tree(nodes)
        needed = set()
        for nd in nodes:
            needed.update(nd['ids'])
        if tfile:
            os.makedirs(CACHE, exist_ok=True)
            with open(tfile, 'w') as f:
                json.dump([base, packed, sorted(needed)], f)
    print(prompt, packed['n'], 'nodes', round(time.time() - t0), 's', flush=True)
    if sfile and os.path.exists(sfile):
        with open(sfile) as f:
            greedy, samples = json.load(f)
        for ti, T in enumerate(TEMPS):
            for qi, p in enumerate(TOPPS):
                if p in DRAW_TOPPS:
                    k = f'{T}|{p}'
                    samples[k] = draws(base, T, p, seed=1000 * pi + 100 * ti + qi, first=samples[k])
        with open(sfile, 'w') as f:
            json.dump([greedy, samples], f)
    else:
        with torch.no_grad():
            x = torch.tensor([base])
            out = model.generate(x, do_sample=False, max_new_tokens=DRAWLEN, pad_token_id=tok.eos_token_id, attention_mask=torch.ones_like(x))
            greedy = tok.decode(out[0, len(base):])
        samples = {}
        for ti, T in enumerate(TEMPS):
            for qi, p in enumerate(TOPPS):
                if p in DRAW_TOPPS:
                    samples[f'{T}|{p}'] = draws(base, T, p, seed=1000 * pi + 100 * ti + qi)
        if sfile:
            os.makedirs(CACHE, exist_ok=True)
            with open(sfile, 'w') as f:
                json.dump([greedy, samples], f)
    entry = {'text': prompt, 'ids': base, 'tree': packed, 'greedy': greedy, 'samples': samples}
    doc['prompts'].append(entry); vocab_needed.update(needed)
    print(prompt, 'samples done', round(time.time() - t0), 's', flush=True)
doc['vocab'] = {str(i): tok.decode([i]) for i in sorted(vocab_needed)}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(doc, f, separators=(',', ':'), ensure_ascii=False)
print('wrote', OUT, os.path.getsize(OUT), 'bytes')
