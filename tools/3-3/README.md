# Data for page 3.3 (Next-token prediction and sampling)

`docs/data/3-3/tree.json` comes from GPT-2 small, `openai-community/gpt2`, revision
607a30d783dfa663caf39e06633721c8d4cfcd7e, computed on CPU in float32 (scores converted to float64).

    HF_HOME=/home/claude/hf CACHE=<scratch dir> python3 -I tools/3-3/build_data.py docs/data/3-3/tree.json

With CACHE set, each prompt's results are saved as they finish, so an interrupted run resumes (about 4 minutes
for each tree and 4 for each prompt's continuations on a shared 2-core machine).

For each of the five prompts the file holds:

1. A tree of about 2,400 prefixes (prompt plus generated tokens). Each node stores GPT-2's 20 most likely next
   tokens with log probabilities at temperature 1 (to 0.001), and for each temperature on the page's grid
   (0.25 to 2 in steps of 0.25) the number log sum_j exp((z_j - z_max)/T) over all 50,257 scores (to 0.0002),
   so the browser can compute exact probabilities of those 20 tokens at any grid temperature. The greedy path
   is expanded to 24 tokens first; then nodes are expanded best first by the probability that a reader
   sampling at T = 1 among the stored tokens reaches them. An expanded node gets children for its most likely
   tokens until they hold 90% of the probability, at most 8. Deterministic.
2. Four whole continuations of 30 tokens sampled with `generate()` (do_sample, top_k=0) at every grid
   temperature and top-p 0.5, 0.9, and 1.0, with seed 1000*prompt + 100*temperature index + top-p index,
   and the greedy continuation.

The browser never runs the model. The claims on the page are tested in `tools/test-page-3-3.mjs`.

A set of four continuations containing profanity or sexual-violence words (the `BLOCK` pattern in `build_data.py`)
is sampled again with seed + 100000 per attempt; 3 of the 120 sets were. The page's panel note says so. The
distributions in the tree are never filtered.
