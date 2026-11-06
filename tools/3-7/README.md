# Data for page 3.7 (A full forward pass)

`docs/data/3-7/index.json` (parameter counts and the prompt list) and `p0.json` to `p4.json` (one recorded forward
pass per prompt) come from GPT-2 small, `openai-community/gpt2`, revision 607a30d783dfa663caf39e06633721c8d4cfcd7e.

    HF_HOME=/home/claude/hf python3 -I tools/3-7/build_data.py docs/data/3-7

The script recomputes the forward pass by hand from the model's own modules (embeddings, layer norms,
attention with the causal mask, GELU MLP, final layer norm, scores against the token embedding matrix) and
checks the final scores against the model's own output (largest difference about 1e-4). There is no randomness.

Stored for display: each tensor clipped at the 99th percentile of its absolute values and rounded to 15 levels
(4 bits), with its exact minimum, maximum, and row lengths; attention weights to 1/255; scores and probabilities
over the vocabulary as the largest value in each bin of 64 consecutive token ids, plus the 10 highest at each
position, exact. The claims on the page are tested in `tools/test-page-3-7.mjs`.
