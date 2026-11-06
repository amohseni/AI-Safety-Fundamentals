# Residual-stream activations at every kept token of the passages from passages.py, for every layer.
# Usage: python3 tools/truth/token_acts.py <model id> <passages.json> <out.npy>
# Output: float16 array (layers + 1, kept tokens, d_model), in passage order.
import json, sys
import numpy as np
import torch
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from acts import load

if __name__ == '__main__':
    model_id, pp, outp = sys.argv[1:4]
    P = json.load(open(pp))
    tok, model = load(model_id)
    out = []
    with torch.no_grad():
        for q in P:
            hs = model(input_ids=torch.tensor([q['ids']]), output_hidden_states=True).hidden_states
            keep = [t['i'] for t in q['tokens'] if t['keep']]
            out.append(torch.stack([h[0, keep, :].float() for h in hs]).half().numpy())
    A = np.concatenate(out, axis=1)
    np.save(outp, A)
    print(outp, A.shape)
