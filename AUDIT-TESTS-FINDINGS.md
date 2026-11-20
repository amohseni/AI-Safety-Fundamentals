# Test audit findings (2026-11-19)

Twenty scripts in `tools/` printed numbers without asserting them. Each now asserts the claims its page makes and exits nonzero when one fails. This file lists every place where the page text and the computed value do not match exactly.

Measured with Node 22.23.4. No page was edited.

## Disagreements between text and code

Each of these has an assertion that fails now. The assertion states the claim as the page states it. Fix the text (or the code) and the assertion will pass.

### Page 5.5, task 1 notice

Sentence: "Over the whole square, most of which holds no training points, two runs agree on as little as 77% of it (90% on average)."

Computed (`tools/test-page-5-5.mjs`): the lowest pair agrees on 77.27% of the square, which matches. The average over the ten pairs is 89.475%, which is 89% to the nearest percent. The script prints the average to one decimal as 89.5, and 90 is that printed figure rounded a second time.

Suggested text: "(89% on average)".

Failing assertion: `t1: over the whole square the average agreement is 90%`.

### Page 6.4, argument after the tasks

Sentence: "In the pentagon we trained, neighboring directions have dot product 0.37, the others have dot product −0.98, and every bias is −0.21. The ReLU removes the negative interference completely, and the negative bias removes the positive interference whenever x_j is below 0.57."

Computed (`tools/test-page-6-4.mjs`): the threshold is the size of the bias divided by the neighboring dot product. With the trained parameters the average bias is −0.2114 and the average neighboring dot product is 0.3748, so the threshold is 0.564, which is 0.56 to two decimals. The page's 0.57 is 0.21 / 0.37 = 0.568, computed from the two rounded numbers in the previous sentence. Over the five neighboring pairs taken one at a time the threshold runs from 0.55 to 0.58.

Suggested text: "whenever x_j is below about 0.56".

Failing assertion: `pentagon: the bias removes the positive interference whenever x_j is below 0.57 (average bias / average neighboring dot product, from the trained parameters)`. A second assertion, that 0.21 / 0.37 = 0.57, passes.

### Page 6.5, task 3 notice

Sentence: "In ten starts at 10 units, all five features were recovered every time; one to four units never fired, and one to four more pointed at a feature that another unit already matched, and in some starts one unit pointed at no feature."

Computed (`tools/test-page-6-5.mjs`): all five features recovered in 10 of 10 starts, 1 to 4 units never fired, and 1 to 4 units duplicated a feature, all as stated. A unit that fires but points at no feature (cosine at most 0.95 with every feature) appears in one start of the ten, start 5, and not in "some starts".

Suggested text: "and in one start one unit pointed at no feature".

Failing assertion: `t3: in some starts (at least 2, not all) one unit points at no feature, and never more than one`.

## Checked and not a disagreement

These assertions pass. Each note says why the difference between the text and the raw number is not an error in the text.

- Page 5.5, task 3 notice, "keeping only the key frequencies in the embedding leaves test accuracy at 99.9% or more": the stored values are 1, 0.9992, 1, 1, 0.9988. The lowest is 99.88%, which is 99.9% at the one decimal the page quotes. The assertion compares at one decimal. Read strictly, 99.88% is below 99.9%.
- Page 5.5, panel 1 note, "Of 40 seeds we tried, these five reached 100% on the training points within 4,000 steps in every test we ran": the script finds six such seeds among 1 to 40 (1, 3, 9, 12, 22, 31), and the page uses five of them (not 9). The sentence is true of the five. A reader could take it to mean that only these five did; the assertion checks only that each of the five does.
- Page 6.4, "neighboring directions have dot product 0.37, the others have dot product −0.98": the five neighboring pairs are 0.369 to 0.381 (average 0.375) and the other five are −0.988 to −0.978 (average −0.981). The assertions check the average at two decimals and each pair to within 0.015.
- Page 6.5, task 1 notice and check 2, "Even at 0.5 the rebuild error is 0.17": the ten starts give 0.167 to 0.179, with average 0.175, which is 0.17 at two decimals. Starting point 1, the tool's first run, gives 0.178. The assertion checks the average and that each start is within 0.01.
- Page 8.3, task 2 notice, "in 9 of 10 training runs test success is between 84% and 99%, though one run reaches only 32%": Node 22 gives 85 to 99 for the nine runs, and 32 for the tenth. The existing comment in the script records that Chromium gave 84 for the lowest of the nine (and 87 where Node gives 89), because training samples actions and the two engines differ in the last bit of some floating-point results. The assertion checks that the nine runs lie within 84 to 99, that the highest is 99, and that the tenth is 32.
- Page 8.4, check 2 feedback, "20 of the 21 equally likely cases, about 95%": the simulation with 4,000 reward functions at discount 0.999 gives 95.7%. The exact value is 20/21 = 95.2%, and the difference is 1.4 standard errors of the sample. The assertion allows 1.5 points.
- Page 2.4, task 1 notice, "In every successful run we tested": `tools/test-xor-layers.mjs` tests seeds 1 to 8 for 1,500 steps, of which six succeed. The page's tool uses those six seeds (1, 2, 4, 5, 6, 7) for up to 6,000 steps. The claim is about the tested runs, so the two agree.
- Page 2.8, task 3 notice, "falls at every degree after that": the script used to test this with a 5% allowance. The new assertion is strict (each median lower than the one before, degrees 12 to 30) and passes.

## Not covered

- `tools/test-page-8-3.mjs` and the other training tests depend on the JavaScript engine's floating point. The assertions are exact for Node 22. Another major version of Node could change a seeded run and fail an exact assertion without any change to the site.
- Claims about published papers, dates, and citations are outside these scripts.
