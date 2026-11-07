# Page 6.6 data: steering GPT-2 small by activation addition

`steer.py` writes the generations behind page 6.6. The shipped file `docs/data/6-6/steer.json` is its output with the per-sample fluency arrays (`fluAll`) removed and whitespace stripped.

- Model: `openai-community/gpt2` (GPT-2 small), revision in `meta.revision`, float32, CPU, one thread. Sentiment classifier: `distilbert/distilbert-base-uncased-finetuned-sst-2-english` (probability of "positive", on the continuation alone). Transformers 5.22.2, PyTorch 2.14.1.
- Method (Turner et al. 2023, arXiv:2308.10248, and their post "Steering GPT-2-XL by adding an activation vector"): the two contrast prompts get `<|endoftext|>` in front, the shorter is padded on the right with the space token, the residual stream entering block L is recorded at each of their positions, and the difference times the coefficient is added to the same front positions of the user prompt on the prompt's forward pass.
- Vectors: "Love" minus "Hate"; " weddings" minus " ". Prompts: "I went to the park and", "I hate you because". Layers 0 to 11; coefficients -10 to +10 in steps of 1.
- Sampling: 8 continuations of 32 tokens per setting; temperature 1, top-p 0.3, frequency penalty 1 counted over generated tokens (the paper's generation-scoring settings). Sample s uses `torch.Generator().manual_seed(1000 + s)` in every setting. Top-p is taken over the 256 most probable tokens, or over the whole vocabulary when those hold less than 0.3 of the probability.
- Scores per setting: mean sentiment; share of continuations containing a wedding word (wedding, weddings, wed, marry, married, marriage, bride, groom, honeymoon); fluency = mean log probability per generated token under the unsteered model. Three texts per setting are kept.

Rerun (about an hour on a shared two-core machine; the script resumes from an existing output file unless given `--fresh`):

```
HF_HOME=/home/claude/hf python3 tools/6-6/steer.py out.json
python3 -c "import json; d=json.load(open('out.json')); [c.pop('fluAll') for c in d['cells'].values()]; json.dump(d, open('docs/data/6-6/steer.json','w'), separators=(',',':'), ensure_ascii=False)"
```

The 2026-11-06 run was interrupted once (an assertion that the top 256 tokens always hold 0.3 of the probability failed at love, layer 6, second prompt) and resumed after the fallback above was added; the 13 cells finished before the change never hit the fallback, so they are what the final script produces. Checks: `node tools/test-page-6-6.mjs`.

## Masking

After `steer.py`, run `python3 tools/6-6/mask.py docs/data/6-6/steer.json` (needs `pip install better_profanity`). It masks profanity and slurs after the first letter, using the better_profanity word list minus the ordinary words in `KEEP`. Sentiment, wedding share, and fluency were computed on the unmasked text.
