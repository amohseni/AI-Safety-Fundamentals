# Page 3.5 data: attention heads of GPT-2 small

`heads.py` writes `docs/data/3-5/heads.json`; `ablate.py` prints the head-removal numbers quoted on the page.

- Model: `openai-community/gpt2` from Hugging Face (GPT-2 small, 124M parameters), revision recorded in the JSON (`meta.revision`), float32, CPU, eager attention (so attention weights are returned). Transformers 5.22.2, PyTorch 2.14.1.
- Every input starts with `<|endoftext|>`.
- Random sequences: 20 token ids drawn uniformly from 1000 to 49999 with `numpy.random.RandomState(seed).randint`, then repeated. The displayed sequence uses seed 0; the head scores average over seeds 1 to 100.
- Scores per head: previous-token score = mean weight from position i to i-1 (i >= 2); induction score = mean weight from each second-copy position i to i-19 (the token after the earlier copy); duplicate-token score = mean weight from i to i-20 (the earlier copy of the same token).
- Attention patterns are stored as bytes (weight times 255, rounded), lower triangle only, base64.

Rerun (about 2 minutes each on one CPU thread):

```
HF_HOME=/home/claude/hf python3 tools/3-5/heads.py docs/data/3-5/heads.json
HF_HOME=/home/claude/hf python3 tools/3-5/ablate.py
```

Output of `ablate.py` (2026-11-06), induction scores and losses averaged over seeds 1 to 100. "Removed" means the head's output is set to zero before the output matrix.

```
base   {'5.5': 0.926, '6.9': 0.88,  '5.1': 0.878, '7.10': 0.87,  '7.2': 0.763} second-copy loss 0.532 first token of copy 2 11.082 min 6.533
no411  {'5.5': 0.828, '6.9': 0.884, '5.1': 0.771, '7.10': 0.841, '7.2': 0.719} second-copy loss 0.566 first token of copy 2 11.104 min 6.516
noprev {'5.5': 0.652, '6.9': 0.851, '5.1': 0.619, '7.10': 0.773, '7.2': 0.629} second-copy loss 0.691 first token of copy 2 11.257 min 6.621
```

`noprev` removes 4.11, 3.7, 2.2, and 6.8 together. `tools/test-page-3-5.mjs` checks the other numbers on the page against the JSON.
