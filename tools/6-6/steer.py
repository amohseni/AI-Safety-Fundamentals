"""Precompute steered GPT-2 small continuations for page 6.6 (Activation Addition).

Method follows Turner et al. (2023/2024), "Activation Addition: Steering Language Models
Without Optimization" (arXiv:2308.10248) and their post "Steering GPT-2-XL by adding an
activation vector": run the two contrast prompts (each with <|endoftext|> in front, the
shorter one padded on the right with the space token), take the difference of the residual
stream entering block L at each of their positions, multiply by the coefficient c, and add it
to the same front positions of the user prompt, once, on the prompt's forward pass.
Sampling: temperature 1, top-p 0.3, frequency penalty 1 (as in the paper's generation
scoring), counted over generated tokens. Sample s uses torch seed 1000 + s for every setting,
so settings differ only by the steering.

Usage: HF_HOME=/home/claude/hf python3 steer.py OUT.json
"""
import os, sys, json, time
os.environ.setdefault('HF_HOME', '/home/claude/hf')
import torch
torch.set_num_threads(1)
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoTokenizer, AutoModelForSequenceClassification

MODEL = 'openai-community/gpt2'
SENT = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
PROMPTS = ['I went to the park and', 'I hate you because']
VECTORS = [
    {'id': 'love', 'plus': 'Love', 'minus': 'Hate'},
    {'id': 'wed', 'plus': ' weddings', 'minus': ' '},
]
LAYERS = list(range(12))
COEFS = list(range(-10, 11))
NSAMP, NTOK, KEEP = 8, 32, 3
TOP_P, FREQ_PEN = 0.3, 1.0
# Wedding-related words, after Turner et al.'s wedding-word count.
WED = ['wedding', 'weddings', 'wed', 'marry', 'married', 'marriage', 'bride', 'groom', 'honeymoon']

tok = GPT2TokenizerFast.from_pretrained(MODEL)
m = GPT2LMHeadModel.from_pretrained(MODEL).eval()
rev = getattr(m.config, '_commit_hash', None)
stok = AutoTokenizer.from_pretrained(SENT)
smod = AutoModelForSequenceClassification.from_pretrained(SENT).eval()
BOS, SPACE = tok.bos_token_id, tok(' ').input_ids[0]

def resid_pre(ids, L):
    with torch.no_grad():
        o = m(torch.tensor([ids]), output_hidden_states=True)
    return o.hidden_states[L][0]  # input to block L (L=0: embeddings)

def steering_vector(v, L):
    a, b = [BOS] + tok(v['plus']).input_ids, [BOS] + tok(v['minus']).input_ids
    n = max(len(a), len(b))
    a, b = a + [SPACE] * (n - len(a)), b + [SPACE] * (n - len(b))
    return resid_pre(a, L) - resid_pre(b, L)

def generate(prompt, L, d, coefs):
    pids = [BOS] + tok(prompt).input_ids
    B = len(coefs) * NSAMP
    cvec = torch.tensor([c for c in coefs for _ in range(NSAMP)], dtype=torch.float32)
    k = d.shape[0]
    def hook(mod, args, kwargs):
        h = args[0] if args else kwargs['hidden_states']
        if h.shape[1] > 1:  # the prompt pass only
            h = h.clone()
            h[:, :k] += cvec[:, None, None] * d[None]
            if args: args = (h,) + tuple(args[1:])
            else: kwargs['hidden_states'] = h
        return args, kwargs
    hd = m.transformer.h[L].register_forward_pre_hook(hook, with_kwargs=True)
    gens = [torch.Generator().manual_seed(1000 + s) for s in range(NSAMP)]
    out = torch.empty(B, NTOK, dtype=torch.long)
    counts = torch.zeros(B, m.config.vocab_size)
    try:
        with torch.no_grad():
            o = m(torch.tensor([pids] * B), use_cache=True)
            pk, logits = o.past_key_values, o.logits[:, -1]
            for t in range(NTOK):
                lg = logits - FREQ_PEN * counts
                pr = torch.softmax(lg, -1)
                # Top-p from the 256 most probable tokens; rows whose top 256 hold less than
                # TOP_P of the probability (very flat distributions) use the full sorted vocabulary.
                sp, si = pr.topk(256, -1)
                nxt = torch.empty(B, dtype=torch.long)
                for b in range(B):  # sample b uses the generator of its sample index
                    pb, ib = (sp[b], si[b]) if float(sp[b].sum()) >= TOP_P else pr[b].sort(descending=True)
                    pb = pb.clone()
                    cum = pb.cumsum(-1)
                    pb[(cum - pb) > TOP_P] = 0
                    j = torch.multinomial(pb / pb.sum(), 1, generator=gens[b % NSAMP])
                    nxt[b] = ib[j]
                out[:, t] = nxt
                counts[torch.arange(B), nxt] += 1
                o = m(nxt[:, None], past_key_values=pk, use_cache=True)
                pk, logits = o.past_key_values, o.logits[:, -1]
    finally:
        hd.remove()
    # Fluency: mean log probability per generated token under the unsteered model.
    full = torch.cat([torch.tensor([pids] * B), out], 1)
    flu = torch.empty(B)
    with torch.no_grad():  # in chunks, to keep memory small
        for s in range(0, B, 16):
            f = full[s:s + 16]
            lp = torch.log_softmax(m(f).logits[:, len(pids) - 1:-1], -1)
            flu[s:s + 16] = lp.gather(-1, f[:, len(pids):, None])[:, :, 0].mean(1)
            del lp
    texts = [tok.decode(out[b]) for b in range(B)]
    with torch.no_grad():
        enc = stok(texts, return_tensors='pt', padding=True, truncation=True)
        sent = torch.softmax(smod(**enc).logits, -1)[:, 1]
    import re
    wed = torch.tensor([float(any(w in WED for w in re.findall(r"[a-z]+", tx.lower()))) for tx in texts])
    return texts, sent, wed, flu

res = {'meta': {'model': MODEL, 'revision': rev, 'sentiment_model': SENT, 'top_p': TOP_P, 'temperature': 1.0,
                'frequency_penalty': FREQ_PEN, 'samples': NSAMP, 'tokens': NTOK, 'seeds': [1000 + s for s in range(NSAMP)],
                'wedding_words': WED},
       'prompts': PROMPTS, 'vectors': VECTORS, 'layers': LAYERS, 'coefs': COEFS, 'cells': {}, 'norms': {}}
if os.path.exists(sys.argv[1]) and '--fresh' not in sys.argv:  # resume: keep finished cells
    prev = json.load(open(sys.argv[1]))
    res['cells'], res['norms'] = prev['cells'], prev['norms']
t0 = time.time()
for v in VECTORS:
    for L in LAYERS:
        d = steering_vector(v, L)
        res['norms'][f"{v['id']}|{L}"] = round(float(d.norm(dim=-1).max()), 2)
        for pi, p in enumerate(PROMPTS):
            if f"{pi}|{v['id']}|{L}" in res['cells']:
                continue
            texts, sent, wed, flu = generate(p, L, d, COEFS)
            cell = {'sent': [], 'wed': [], 'flu': [], 'texts': [], 'sentAll': [], 'fluAll': []}
            for ci in range(len(COEFS)):
                sl = slice(ci * NSAMP, (ci + 1) * NSAMP)
                cell['sent'].append(round(float(sent[sl].mean()), 3))
                cell['wed'].append(round(float(wed[sl].mean()), 3))
                cell['flu'].append(round(float(flu[sl].mean()), 3))
                cell['sentAll'].append([round(float(x), 3) for x in sent[sl]])
                cell['fluAll'].append([round(float(x), 3) for x in flu[sl]])
                cell['texts'].append(texts[sl][:KEEP])
            res['cells'][f"{pi}|{v['id']}|{L}"] = cell
            print(f"{v['id']} L{L} p{pi} done {time.time() - t0:.0f}s", flush=True)
            json.dump(res, open(sys.argv[1], 'w'))
print('finished', flush=True)
