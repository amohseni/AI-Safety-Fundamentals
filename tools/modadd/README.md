# Modular-addition runs

This folder makes the data in `docs/data/modadd/`, used by pages 5.4, 5.5, and 6.7.

## The model and training

The setup follows Nanda, Chan, Lieberum, Smith, and Steinhardt (2023), "Progress measures for grokking via mechanistic interpretability" (ICLR 2023, arXiv:2301.05217), Section 3, with one change: the modulus.

| | Nanda et al. (2023) | These runs |
| --- | --- | --- |
| Task | (a + b) mod 113, input "a b =" | (a + b) mod **59**, input "a b =" |
| Model | 1 layer; embeddings of 128; learned positional embeddings; 4 heads of 32; ReLU MLP of 512 units; no layer norm; untied embed and unembed | the same |
| Training data | 30% of all pairs | 30%, 50%, or 70% of all pairs |
| Optimizer | AdamW, learning rate 0.001, weight decay 1, full batch | AdamW, learning rate 0.001, betas (0.9, 0.98), 10-step linear warmup; weight decay 0, 0.3, or 1; full batch |
| Length | 40,000 epochs | 20,000 steps |

Why p = 59: at p = 113 one step took about 0.1 seconds or more on the shared two-core machine used here, too slow for thirteen runs of 20,000 steps. At p = 59 a step took 0.02 to 0.1 seconds, depending on load, and the main run still groks (test accuracy above 99% by step 5,800).

Other details:

- Initialization: normal with variance 1/128 for every matrix except the unembedding (variance 1/p); zero biases. The initialization, the betas, and the warmup are our choices; they were not checked against the paper. The `--seed` argument sets the initialization only.
- The training set is the first fraction of one fixed random order of all p² pairs (`numpy.random.RandomState(0)`), so the 30% set is contained in the 50% set, which is contained in the 70% set. All seeds use the same training set.
- The loss is cross-entropy at the "=" position, computed in float64.
- The model computes queries, the MLP, and the unembedding at the last position only. With one layer this gives the same outputs and gradients as running all three positions.
- One CPU thread per run (`torch.set_num_threads(1)`), PyTorch 2.14.1, CPU.

## What is recorded

`train.py` saves, for each run:

- train and test accuracy and loss at 40 log-spaced steps up to step 1,000 and then every 50 steps;
- at checkpoints (the same early steps, then every 200 steps): which of the p² pairs the model answers correctly, the number embeddings, and the Fourier coefficients of the logits used for the restricted accuracy;
- the final parameters.

## Definitions used in the exported data

- **Frequency spectrum of the embedding.** The number embeddings form a p × 128 matrix. Project its columns onto the orthonormal real Fourier basis over 0, ..., p − 1 (a constant, and cos(2πka/p) and sin(2πka/p) for k = 1, ..., 29). The size of frequency k is the square root of the sum of the squared cos and sin coefficients over all 128 columns.
- **Key frequencies.** The fewest frequencies, taken from the largest down, that hold at least 90% of the squared size of the final embedding (the constant term excluded).
- **Restricted test accuracy** (Nanda et al.'s restricted loss, as an accuracy). For each output c, the logits over all pairs (a, b) are projected onto cos(2πk(a + b)/p) and sin(2πk(a + b)/p) for the run's final key frequencies, plus the constant. The restricted test accuracy is the test accuracy of these projected logits.
- **Ablating frequencies of the embedding.** Replace the number embeddings by their projection onto the constant and the frequencies kept, and run the model as usual. `acc_keep_only_keys` keeps only the key frequencies; `acc_remove_one[k]` removes only frequency k; `acc_keep_subset_of_keys` keeps each subset of the key frequencies (and no others). All are test accuracies.

## Files written

`export.py` writes to `docs/data/modadd/`:

- `run-wd<λ>-f<fraction>.json` (9 files, seed 0): curves, the training-set mask, and per checkpoint the correctness grid (p² bits, `numpy.packbits` order, base64), the frequency spectrum, and the restricted test accuracy.
- `seeds.json`: the five seeds at weight decay 1 and 30%: final accuracies, key frequencies, spectra, and the ablation results.
- `seeds-embed.json`: the five final number embeddings (float32 little-endian, row-major, base64), for page 6.7.
- `index.json`: the list of runs.

## Reproducing

```
pip install torch numpy
sh run_all.sh runs        # 13 runs, two at a time; about two to three hours on two cores
python3 export.py --p 59 --runs runs --out ../../docs/data/modadd
```

## Results the pages quote

Printed by `python3 summarize.py ../../docs/data/modadd` and checked by `node tools/test-page-5-4.mjs`. All runs reach 100% training accuracy by step 119.

| Weight decay | Training pairs | Test accuracy passes 50% | passes 99% | Test accuracy at step 20,000 |
| --- | --- | --- | --- | --- |
| 1 | 30% | step 4,650 | step 5,800 | 100% |
| 1 | 50% | 838 | 1,850 | 100% |
| 1 | 70% | 119 | 1,200 | 100% |
| 0.3 | 30% | 18,050 | not reached | 87.6% |
| 0.3 | 50% | 3,400 | 7,450 | 100% |
| 0.3 | 70% | 100 | 4,000 | 100% |
| 0 | 30% | not reached | not reached | 4.4% |
| 0 | 50% | not reached | not reached | 41.6% |
| 0 | 70% | 100 | not reached | 69.6% |

- Without weight decay, the test pairs the model answers correctly are almost all pairs (a, b) whose reverse (b, a) is a training pair: 100%, 100%, and 98.9% of the correct test answers at 30%, 50%, and 70%.
- Main run (weight decay 1, 30%): at step 3,600 test accuracy is 18.5%, the five largest frequencies hold 46.6% of the embedding (20.3% at step 0), and the restricted test accuracy is 78.1%. At step 1,000: test 7.4%, restricted 27.3%. Final key frequencies 1, 12, 27.
- Five seeds at the main setting, all at 100% test accuracy: key frequencies {1, 12, 27}, {4, 5, 15}, {2, 5, 21}, {15, 20, 21}, {2, 21, 22, 27}. Keeping only the key frequencies in the embedding leaves test accuracy at 99.9% or more; removing them gives 1.7% to 1.8% (chance is 1/59 = 1.7%). Removing one key frequency gives between 19.4% and 94.7%. Nanda et al. (appendix C.2.1) also report that the key frequencies differ by seed.
