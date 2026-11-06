"""Records every intermediate tensor of one GPT-2 small forward pass, for page 3.7.

Writes docs/data/3-7/index.json (parameter counts, step list) and docs/data/3-7/p<k>.json (one per prompt).
The forward pass is recomputed by hand from the model's own modules, and the final scores are checked
against the model's output. Tensors are stored for display: each one is clipped at the 99th percentile
of its absolute values and rounded to 15 levels (4 bits); exact row norms and extremes are stored as numbers.
The scores over the 50,257 tokens are stored in bins of 64 consecutive token ids (the largest value in
each bin), plus the 10 highest scores at each position.

Usage: HF_HOME=/home/claude/hf python3 -I tools/3-7/build_data.py docs/data/3-7
Model: openai-community/gpt2 (GPT-2 small). No randomness.
"""
import os, sys, json, base64
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
T = model.transformer
PROMPTS = ['The capital of France is', 'Once upon a time', 'Thank you very', 'My favorite food is', 'The cat sat on the']
BIN = 64

def q4(x):
    x = np.asarray(x, dtype=np.float64)
    c = float(np.quantile(np.abs(x), 0.99)) or 1.0
    q = np.clip(np.round(x / c * 7), -7, 7).astype(np.int8) + 8      # 1..15
    flat = q.reshape(-1)
    if flat.size % 2:
        flat = np.append(flat, 8)
    packed = (flat[0::2] << 4 | flat[1::2]).astype(np.uint8)
    rows = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
    return {'shape': list(x.shape), 'clip': round(c, 4), 'min': round(float(x.min()), 4), 'max': round(float(x.max()), 4),
            'norms': [round(float(v), 3) for v in np.linalg.norm(rows, axis=1)],
            'q': base64.b64encode(packed.tobytes()).decode()}

def u8(x, lo, hi):
    x = np.asarray(x, dtype=np.float64)
    q = np.clip(np.round((x - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode()

@torch.no_grad()
def trace(prompt):
    ids = tok(prompt, return_tensors='pt').input_ids
    n = ids.shape[1]
    te = T.wte(ids)[0]; pe = T.wpe(torch.arange(n))
    h = te + pe
    rec = {'tok_emb': q4(te), 'pos_emb': q4(pe), 'resid0': q4(h), 'layers': []}
    h = h[None]
    for blk in T.h:
        L = {}
        a = blk.ln_1(h); L['ln1'] = q4(a[0])
        qkv = blk.attn.c_attn(a); L['qkv'] = q4(qkv[0])
        q, k, v = qkv.split(768, dim=-1)
        sh = lambda t: t.view(1, n, 12, 64).transpose(1, 2)
        q, k, v = sh(q), sh(k), sh(v)
        s = q @ k.transpose(-1, -2) / 8.0
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
        s = s.masked_fill(mask, float('-inf'))
        w = torch.softmax(s, -1)
        L['pattern'] = u8(w[0].numpy(), 0, 1)
        z = (w @ v).transpose(1, 2).reshape(1, n, 768)
        ao = blk.attn.c_proj(z); L['attn_out'] = q4(ao[0])
        h = h + ao; L['resid_mid'] = q4(h[0])
        b = blk.ln_2(h); L['ln2'] = q4(b[0])
        pre = blk.mlp.c_fc(b); L['mlp_pre'] = q4(pre[0])
        post = blk.mlp.act(pre); L['mlp_post'] = q4(post[0])
        mo = blk.mlp.c_proj(post); L['mlp_out'] = q4(mo[0])
        h = h + mo; L['resid_post'] = q4(h[0])
        rec['layers'].append(L)
    f = T.ln_f(h); rec['lnf'] = q4(f[0])
    logits = f[0] @ T.wte.weight.T                      # the unembedding is the token embedding matrix, transposed
    ref = model(ids).logits[0]
    err = float((logits - ref).abs().max())
    assert err < 1e-3, err
    lg = logits.double().numpy()
    pr = torch.softmax(logits.double(), -1).numpy()
    V = lg.shape[1]; nb = (V + BIN - 1) // BIN
    pad = np.full((n, nb * BIN), -np.inf); pad[:, :V] = lg
    lbin = pad.reshape(n, nb, BIN).max(-1)
    ppad = np.zeros((n, nb * BIN)); ppad[:, :V] = pr
    pbin = ppad.reshape(n, nb, BIN).max(-1)
    lo, hi = float(lg.min()), float(lg.max())
    rec['logits'] = {'shape': [n, V], 'bins': nb, 'lo': round(lo, 3), 'hi': round(hi, 3), 'q': u8(lbin, lo, hi)}
    rec['probs'] = {'shape': [n, V], 'bins': nb, 'q': u8(np.sqrt(pbin), 0, 1)}
    top = []
    for r in range(n):
        o = np.argsort(-lg[r])[:10]
        top.append([[tok.decode([int(i)]), int(i), round(float(lg[r, i]), 3), round(float(pr[r, i]), 4)] for i in o])
    rec['top'] = top
    rec['tokens'] = [[tok.decode([int(i)]), int(i)] for i in ids[0]]
    rec['text'] = prompt
    rec['check_max_abs_diff'] = err
    return rec

os.makedirs(OUT, exist_ok=True)
P = {name: p.numel() for name, p in model.named_parameters()}
counts = {
    'wte': P['transformer.wte.weight'], 'wpe': P['transformer.wpe.weight'],
    'ln': P['transformer.h.0.ln_1.weight'] + P['transformer.h.0.ln_1.bias'],
    'qkv': P['transformer.h.0.attn.c_attn.weight'] + P['transformer.h.0.attn.c_attn.bias'],
    'proj': P['transformer.h.0.attn.c_proj.weight'] + P['transformer.h.0.attn.c_proj.bias'],
    'fc': P['transformer.h.0.mlp.c_fc.weight'] + P['transformer.h.0.mlp.c_fc.bias'],
    'fc2': P['transformer.h.0.mlp.c_proj.weight'] + P['transformer.h.0.mlp.c_proj.bias'],
    'lnf': P['transformer.ln_f.weight'] + P['transformer.ln_f.bias'],
    'total': sum(P.values()),
    'tied': bool(model.lm_head.weight.data_ptr() == T.wte.weight.data_ptr()),
}
index = {'model': NAME, 'revision': REV, 'counts': counts, 'prompts': []}
for k, p in enumerate(PROMPTS):
    rec = trace(p)
    with open(os.path.join(OUT, f'p{k}.json'), 'w') as f:
        json.dump(rec, f, separators=(',', ':'), ensure_ascii=False)
    index['prompts'].append({'text': p, 'file': f'p{k}.json', 'n': len(rec['tokens']), 'check': rec['check_max_abs_diff']})
    print(p, len(rec['tokens']), 'tokens; max diff from model output', rec['check_max_abs_diff'], flush=True)
with open(os.path.join(OUT, 'index.json'), 'w') as f:
    json.dump(index, f, separators=(',', ':'), ensure_ascii=False)
print(json.dumps(counts))
