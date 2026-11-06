# Data for pages 6.2, 6.3 and 7.2

This folder makes the activation data that pages 6.2 (the linear representation hypothesis), 6.3 (linear probes) and 7.2 (belief) load. The three pages share one dataset of true and false statements.

## Model

Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B` on Hugging Face, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, Apache 2.0 license), run in bfloat16 on CPU with PyTorch 2.14 and transformers 5.22. It has 24 layers and a residual stream of 896 numbers. "Layer L" means the residual stream after L decoder layers (`hidden_states[L]`; layer 0 is the embedding).

GPT-2 small, the site's usual model, was tried first and was too weak for these pages: a logistic probe trained on the city statements sorted at most 77% of the held-out city statements (layers 10 to 12), and between 47% and 55% of the Spanish-word and element statements at every layer. Qwen2.5-0.5B gives 94% to 96% on held-out cities from layer 9 on, and a probe trained on cities transfers to the other topics (page 6.2). Measured with `compare_models.py` on the output of `acts.py` for each model.

## The statements (`make_statements.py`)

2,058 statements on three topics, in the style of Marks and Tegmark (2023), "The Geometry of Truth" (arXiv:2310.06824), but generated here rather than copied: their repository (github.com/saprmarks/geometry-of-truth, commit 5d1c630) has no license.

- Cities: the 220 largest cities in GeoNames (CC BY 4.0; the `geonames.csv` file distributed in the Marks and Tegmark repository), at most four per country, with districts and ambiguous names removed. "The city of X is in Y." with the right country and with a country drawn at random.
- Spanish words: 75 common words with their English meanings, written in the script. "The Spanish word 'X' means 'Y'."
- Chemical elements: 48 elements and their symbols. "The chemical symbol for X is Y."

Every claim appears in three forms: plain, negated ("is not in", "does not mean", "is not"), and paraphrased ("X is a city in Y.", "The English word 'Y' is a translation of the Spanish word 'X'.", "Y is the chemical symbol for X."). Items with an even number are the training half. Each statement also gets a random opinion for the character Alice (seed 7), unrelated to its truth.

## Activations

`run_acts.sh` runs every step:

- `acts.py`: the residual stream at the statement's last token (the period), all layers, and, at layer 13 only, the contrast pairs for contrast-consistent search: "Is the following statement true or false? {s} Answer: True" and the same ending in "False"; and the same pairs with "Alice, who is always right, thinks it is true." (or false) inserted before "Answer:".
- `weekdays.py`: the activation at the weekday name in 24 short sentences for each of the seven days.
- `passages.py` and `token_acts.py`: 60 passages of Pride and Prejudice (Project Gutenberg eBook 1342, public domain), with each word-initial token labeled noun or not (spaCy 3.8, `en_core_web_sm` 3.8.0, tags NOUN and PROPN) and inside quotation marks or not; activations at those tokens, all layers. The pages use the first 34 passages.
- `export.py`: reduces each set of activations to its top 32 principal directions (8 for the weekdays), scales them, and writes `docs/data/truth/statements.json` (layer 13, all statements, contrast pairs), `docs/data/6-2/weekdays.json`, and `docs/data/6-3/probe-data.json` (every second layer; token coordinates as int8). The principal directions are kept in the scratch folder for the next two steps.

## Interventions

- `steer.py` (page 7.2, the use test): `node tools/test-page-7-2.mjs <scratch>/directions.json` trains every probe the page offers with the page's own code and writes their directions; `steer.py` maps each back into the residual stream, scales it by the distance between the true and false training averages along it, adds it at strengths -3 to 3 after layer 13 at the statement's last two tokens, and records the model's few-shot judgment P(TRUE) - P(FALSE) (`judge.py`) for 55 held-out city statements. Output: `docs/data/7-2/use.json`.
- `ablate.py` (page 6.3): `node tools/test-page-6-3.mjs directions <scratch>/dir63.json` writes the default linear probe's direction for each property and layer; `ablate.py` removes each direction (and a random one) after its layer and measures the next-token loss on the held-out passages and the few-shot judgment of 60 held-out city statements. Output: `docs/data/6-3/ablation.json`.

## Rerun

```
sh tools/truth/run_acts.sh <scratch> <geonames.csv> <pg1342.txt>
python3 tools/truth/export.py <scratch> docs/data
node tools/test-page-7-2.mjs <scratch>/directions.json
node tools/test-page-6-3.mjs directions <scratch>/dir63.json
python3 tools/truth/steer.py <scratch> <scratch>/directions.json docs/data/7-2/use.json
python3 tools/truth/ablate.py <scratch> <scratch>/dir63.json docs/data/6-3/ablation.json
node tools/test-page-6-2.mjs; node tools/test-page-6-3.mjs; node tools/test-page-7-2.mjs
```

Set `HF_HOME` to the Hugging Face cache and `BF16=1`. The scripts are deterministic apart from bfloat16 arithmetic; the random choices use fixed seeds (7 for the statements, the seeds in the JavaScript probe code for the browser's training).
