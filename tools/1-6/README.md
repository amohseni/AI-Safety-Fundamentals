# Data for page 1.6

`docs/data/1-6/outputs.json` holds the outputs shown in "Same prompt, three models", and
`docs/data/1-6/weight_diff.json` the size of the change post-training made to the parameters.

- Models: `Qwen/Qwen3-0.6B-Base` (revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`) and
  `Qwen/Qwen3-0.6B` (revision `c1899de289a04d12100db370d81485cdf75e47ca`), bfloat16, CPU, one thread.
- Base model: prompt given as raw text. Chat: chat template with `enable_thinking=False`.
  Reasoning: chat template with `enable_thinking=True`.
- Sampling: Qwen's recommended settings (thinking: temperature 0.6, top-p 0.95, top-k 20;
  non-thinking and base: temperature 0.7, top-p 0.8, top-k 20). Seed: `torch.manual_seed(1000 + prompt index)`
  before each generation. Limits: 200 new tokens (base), 400 (chat), 3,000 (reasoning).
- Outputs are stored verbatim (special tokens removed; `<think>` tags kept).

Rerun (each step loads one model; about 1.5 GB of memory):

    export HF_HOME=/home/claude/hf
    python3 tools/1-6/generate.py chat      # chat and reasoning modes
    python3 tools/1-6/generate.py base
    python3 tools/1-6/weight_diff.py

Runs on a different machine or library version may sample different text.
