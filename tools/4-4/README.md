# Data for page 4.4

## Addition (`addition.py`, data in `docs/data/4-4/addition.json`)

Two toy transformers (one layer, width 128, 4 heads, MLP width 512, pre-norm, learned positions; about
208,000 parameters), trained from seed 0 for 8,000 steps of AdamW (learning rate 0.001, batch 128,
cosine schedule) on random sums of two n-digit numbers, n uniform from 1 to 12, zero-padded to 12 digits.

- `direct`: after `a+b=` writes the 13-digit sum, leftmost digit first.
- `carries`: writes the 12 column carries, right to left, then `>`, then the sum.

Loss on the answer tokens only. Evaluation: greedy decoding on 1,000 new sums per n (seed 12345);
accuracy by the longest carry chain on 4,000 more sums (seed 777).

    python3 tools/4-4/addition.py direct 8000 1
    python3 tools/4-4/addition.py carries 8000 1
    python3 tools/4-4/evaluate.py      # writes docs/data/4-4/addition.json

Each run took about 35 to 50 minutes on one CPU thread, shared with other jobs. `addition.py` also accepts a layer count; the page uses one layer.

## Chains of thought (`qwen_cot.py`, data in `docs/data/4-4/early.json` and `mult.json`)

Model `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`, bfloat16, CPU.
Thinking mode, temperature 0.6, top-p 0.95, top-k 20, seed `2000 + problem index` (early answering) and
3000 (multiplication figure). For early answering, the chain of thought is cut after every token that ends a
sentence or line, `\n</think>\n\nThe answer is (` is appended, and the next-token probabilities of A to D are
recorded (rescaled to sum to 1; the unrescaled total is stored as `mass`). Direct multiplication answers use
thinking off and greedy decoding.

    export HF_HOME=/home/claude/hf
    python3 tools/4-4/qwen_cot.py early
    python3 tools/4-4/qwen_cot.py mult      # answers with thinking off, and 4729 x 836 with thinking on
    python3 tools/4-4/qwen_cot.py forced    # answers begun with "a × b = ", digits chosen greedily
    python3 tools/4-4/qwen_cot.py figure2   # 47 x 86 with thinking on (the figure)

The figure was first planned for 4729 × 836; that chain of thought did not finish within 4,000 tokens
(it is kept in `mult.json` under `figure`), so the figure uses 47 × 86, the smallest problem whose
answer without working was wrong.
