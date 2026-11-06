# Page 1.6: how far post-training moved the parameters of Qwen3-0.6B from Qwen3-0.6B-Base.
# For every parameter matrix, the relative change ||W_post - W_base|| / ||W_base|| (Frobenius norms),
# read one tensor at a time so memory stays small.
# Usage: HF_HOME=/home/claude/hf python3 tools/1-6/weight_diff.py
import json, os, glob
import torch
from safetensors import safe_open
from huggingface_hub import snapshot_download

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base = snapshot_download('Qwen/Qwen3-0.6B-Base', revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd')
post = snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca')


def index(d):
    out = {}
    for f in glob.glob(os.path.join(d, '*.safetensors')):
        with safe_open(f, 'pt') as h:
            for k in h.keys():
                out[k] = f
    return out


ib, ip = index(base), index(post)
total_d2 = total_b2 = 0.0
rows = []
for k in sorted(ib):
    if k not in ip:
        continue
    with safe_open(ib[k], 'pt') as h:
        wb = h.get_tensor(k).float()
    with safe_open(ip[k], 'pt') as h:
        wp = h.get_tensor(k).float()
    d2 = (wp - wb).pow(2).sum().item()
    b2 = wb.pow(2).sum().item()
    total_d2 += d2
    total_b2 += b2
    rows.append((k, wb.numel(), (d2 / b2) ** 0.5 if b2 > 0 else None))
    del wb, wp
rel = [r for _, _, r in rows if r is not None]
rel.sort()
res = {'base': 'Qwen/Qwen3-0.6B-Base@da87bfb', 'post': 'Qwen/Qwen3-0.6B@c1899de',
       'overall_relative_change': round((total_d2 / total_b2) ** 0.5, 4),
       'median_tensor_relative_change': round(rel[len(rel) // 2], 4),
       'max_tensor_relative_change': round(rel[-1], 4), 'tensors': len(rows),
       'missing_in_post': [k for k in ib if k not in ip], 'missing_in_base': [k for k in ip if k not in ib]}
json.dump(res, open(os.path.join(ROOT, 'docs', 'data', '1-6', 'weight_diff.json'), 'w'), indent=1)
print(res)
