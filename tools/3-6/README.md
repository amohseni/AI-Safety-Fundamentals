# Page 3.6 data: the logit lens on GPT-2 small

`lens.py` writes `docs/data/3-6/lens.json`.

- Model: `openai-community/gpt2` (GPT-2 small), revision in `meta.revision`, float32, CPU. Transformers 5.22.2, PyTorch 2.14.1. Deterministic: no sampling, no seed.
- Each prompt starts with `<|endoftext|>`. The residual stream is read at the embedding, after each layer's attention part (embedding plus attention outputs so far), and after each layer's MLP; each reading goes through the model's final layer norm and unembedding (the logit lens). The script asserts that its sums reproduce the model's hidden states and its final output.
- Stored for every position: the top 10 tokens and probabilities at the embedding and after each layer; for each cell after layer L, the probability of that cell's top token before layer L, after layer L's attention, and after its MLP.
- `python3 tools/3-6/lens.py x --try` prints the last-position readings for candidate prompts (used to choose the six).

Rerun (under a minute): `HF_HOME=/home/claude/hf python3 tools/3-6/lens.py docs/data/3-6/lens.json`. Checks: `node tools/test-page-3-6.mjs`.
