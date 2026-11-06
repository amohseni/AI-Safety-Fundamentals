# Residual-stream activations at the last token of each statement, for every layer.
# Usage: python3 tools/truth/acts.py <model id> <statements.json> <out.npy> [template]
#   template: a Python format string with {s} for the statement and {alice} for Alice's opinion
#   (default "{s}"). Environment: FORMS=plain,neg keeps only those forms; LAYERS=13 keeps only those layers;
#   BF16=1 runs the model in bfloat16.
# Output: float16 array of shape (layers + 1, statements, d_model); index 0 is the embedding.
import json, sys, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(1)
torch.manual_seed(0)


def load(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.bfloat16 if os.environ.get('BF16') else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    model.eval()
    return tok, model


@torch.no_grad()
def last_token_acts(tok, model, texts, batch=8):
    # Texts are grouped by token length so no padding is needed (padding would shift positions).
    ids = [tok(t)['input_ids'] for t in texts]
    out = [None] * len(texts)
    by_len = {}
    for i, x in enumerate(ids):
        by_len.setdefault(len(x), []).append(i)
    for L, idx in by_len.items():
        for j in range(0, len(idx), batch):
            part = idx[j:j + batch]
            x = torch.tensor([ids[i] for i in part])
            hs = model(input_ids=x, output_hidden_states=True).hidden_states
            h = torch.stack([h[:, -1, :].float() for h in hs]).half().numpy()
            for k, i in enumerate(part):
                out[i] = h[:, k, :]
    return np.stack(out, axis=1)


if __name__ == '__main__':
    model_id, sp, outp = sys.argv[1:4]
    tmpl = sys.argv[4] if len(sys.argv) > 4 else '{s}'
    S = json.load(open(sp))['statements']
    tok, model = load(model_id)
    only = os.environ.get('FORMS')  # e.g. "plain" keeps only those statement forms
    if only:
        S = [s for s in S if s['form'] in only.split(',')]
    A = last_token_acts(tok, model, [tmpl.format(s=s['text'], alice=('true' if s['alice'] else 'false')) for s in S])
    if os.environ.get('LAYERS'):
        A = A[[int(x) for x in os.environ['LAYERS'].split(',')]]
    np.save(outp, A)
    print(outp, A.shape)
