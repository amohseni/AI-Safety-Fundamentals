# AI Fundamentals Explainer: page specifications

Oct 8, 2026 · @Aydin Mohseni

This doc specifies all 50 pages of the explainer site, in 10 sections. For each page it gives the aim, draft text, a static image, an interactive tool, and a check question. This tab holds the plan for the whole site and a page index where you can mark each page approved or for revision. The full specs for each section are in the tabs under Page specs.

## Reader and goal

The reader is a new graduate student in philosophy, computer science, or a nearby field. The site assumes linear algebra, calculus, and probability (to be confirmed) and no machine learning. After working through the site, the reader should be able to:

- explain how a transformer language model is trained and how it computes an output;
- read a current interpretability or alignment paper and follow its methods;
- state the main arguments for why trained systems can pursue unintended goals, and the evidence for and against those arguments;
- pick an open problem and start a small project on it.

The reader works alone and at their own pace: the site is for self-teaching, so it has no grades and no accounts, and progress is kept in the reader's browser. The full site takes 10 to 15 hours, at 10 to 20 minutes per page; the core path (below) takes about 5 hours.

## Page template

Every page has the same parts, in this order. Each page spec in the section tabs follows the same order.

1. **Title**: the question the page answers, or a short name for its concept.
2. **Aim**: what the reader should understand afterward, stated as something they could write down. On the site it appears at the end as the page's takeaway.
3. **Text**: 150 to 300 words of plain prose, with at most one displayed formula. In the specs, the draft text is set as a quotation.
4. **Image**: one static figure that carries the page's main idea and is readable without the tool. It also serves print, slow connections, and screen readers (with a text description).
5. **Interactive**: one tool, with two to four numbered tasks. Each task asks the reader to predict the result before trying it; a line on what to notice is revealed afterward.
6. **Check**: one or two questions the reader answers before seeing the answer. Questions from earlier pages return at the start of later sections for spaced review.
7. **Links and references**: earlier pages this page depends on, later pages that use it, and two or three references.

## Design principles

- **One idea per page.** If a page needs two tools, it becomes two pages.
- **The tool carries the idea.** The text should take about two minutes to read; understanding comes from changing the controls and seeing what happens.
- **Tasks come before the explanation.** Each tool opens with a concrete task ("make token 4 attend to token 2"), and the takeaway is shown after the reader has tried it.
- **Real data where possible.** Tools use real GPT-2 activations, real training runs, and published data. A toy model is labeled as a toy model.
- **Live when it fits in a browser.** Small networks and small planning problems are computed live. Anything that needs a real language model is computed in advance and shipped as data files.
- **Claims carry a type.** Each key claim has a badge saying what kind of claim it is: theorem, empirical finding, hypothesis, or argument.
- **Lore is separated from the record.** When a famous story is simplified or contested, the page says so.
- **One label per concept.** The terms in the Fixed terms section are used on every page without variation.
- **Works on a phone.** Every tool is usable at 375 pixels wide with touch, and no information is available only on hover.
- **Color never carries meaning alone.** Every figure also has a text description.

## How the site teaches

The reader works alone, so each page has to do what a tutor would. These practices come from research on learning; each row names what the site does and the finding behind it.

| Practice | What the site does | Basis |
| --- | --- | --- |
| Predict, then observe | Every task asks for a prediction (a mark on a chart, a choice, a number) before the reader runs the tool, then shows the result beside the prediction | Predict-observe-explain (White and Gunstone 1992) |
| Retrieve instead of reread | Check questions must be answered before the answer appears; earlier questions return at the start of later sections | Testing effect (Roediger and Karpicke 2006); spacing (Cepeda et al. 2006) |
| Concrete before abstract | Each idea starts with one specific case (four XOR points, one attention head) before the general statement or formula | Concreteness fading (Fyfe et al. 2014) |
| Worked example, then practice | Each tool opens with a demonstration the reader can step through, then hands over the controls | Worked-example effect and fading (Sweller 1988; Renkl and Atkinson 2003) |
| Low extraneous load | No decoration; text and its figure on one screen; labels on the figure instead of in legends; one idea per page | Cognitive load theory (Sweller 1988); coherence, contiguity, and signaling principles (Mayer 2009) |
| Confront misconceptions | Where a common belief is wrong ("pretraining is reinforcement learning", "a probe shows the model uses a feature"), the page states the belief and shows where it fails | Refutation texts (Tippett 2010) |
| Explain it to yourself | After each tool, one prompt asks the reader to explain the result in a sentence, typed and kept privately | Self-explanation effect (Chi et al. 1989) |
| Bounded struggle | Some tasks are hard on purpose (fit by hand, write a reward that cannot be exploited); a hint appears after two attempts | Productive failure (Kapur 2008); desirable difficulties (Bjork 1994) |
| Reader sets the pace | No autoplay; every animation can be scrubbed; every tool can be reset | Segmenting principle (Mayer 2009) |

## Visual design

The model is Tufte's printed books, carried over to interactive figures, with the explorable-explanation tradition (Victor 2011; Distill) for the tools.

- **Page.** One text column of about 65 characters, with a wide right margin for sidenotes and small figures, as in Tufte's books and Tufte CSS. Figures that need room span text and margin. On a phone, sidenotes fold into the text as tap-to-open notes.
- **Type.** A book serif for text (ET Book, the face of Tufte's books), at 18 to 20 pixels with generous line spacing; old-style figures in prose; tabular figures in tables and tool readouts; monospace only for code.
- **Color.** Off-white paper and near-black ink. Grey for context. One accent color for whatever the reader is manipulating, and one second color only where two quantities must be told apart (proxy and true value, train and test). A dark theme as inverted paper.
- **Data ink.** No chart frames, background grids, shadows, gradients, or 3D. Thin rules only where they help read values. Labels sit directly on lines and points; no legends.
- **Small multiples.** Comparisons across settings (seeds, layers, heads, values of β) appear as grids of small identical charts on shared axes.
- **Text and figure linked.** A word in the prose that names a figure element is set in that element's color and highlights it on hover or tap. Live values from a tool appear inside the sentence that interprets them ("the true reward is now 0.42"), with sparklines where a trend matters.
- **Controls.** As few as possible, placed beside what they change. Direct manipulation (drag the point, drag the arrow) is preferred to sliders, and sliders to menus. Every control shows its plain name and current value.
- **Feedback.** Every change redraws within 100 milliseconds; slow computations show progress and never freeze the page.
- **Motion.** Only to show a change the reader caused or asked for, such as stepping through layers. The reader's reduced-motion setting is respected.
- **Navigation and progress.** A quiet top line with section, page, previous and next, and a dot for core-path pages. A page counts as done when its check question is answered, shown as a filled dot in the table of contents. No badges, streaks, or pop-ups.
- **Access.** Every tool works from the keyboard; every figure has a text description; contrast meets WCAG AA.

References: Tufte, *The Visual Display of Quantitative Information* (1983), *Envisioning Information* (1990), *Beautiful Evidence* (2006); Liepmann, Tufte CSS; Victor (2011), "Explorable Explanations"; Hohman et al. (2020), "Communicating with Interactive Articles" (Distill).

## Running examples

Four objects recur across the site, so the reader builds on the same objects instead of meeting new ones on every page.

| Object | What it is | Pages |
| --- | --- | --- |
| The plane classifier | A small network that labels points in the plane: 2 inputs, 1 to 3 hidden layers of width 2 to 8. Datasets: AND, OR, XOR, circles, spirals. | 1.2, 2.1 to 2.4, 2.6, 2.8, 5.5 |
| The modular-addition transformer | A one-layer transformer trained on (a + b) mod 113, the model reverse-engineered by Nanda et al. (2023). | 5.4, 5.5, 6.7 |
| GPT-2 small | The real language model used wherever one is needed: 12 layers, 12 heads per layer, 124 million parameters. It is open and the most studied model in interpretability. | 3.1 to 3.7, 6.2, 6.3, 6.5 to 6.7 |
| The gridworld | A small grid environment with an agent, solved exactly by value iteration in the browser. | 7.3, 8.2 to 8.4 |

## Page index

All 50 pages, in reading order. "First 8" marks the tools to build first (see Shared components and build order). Set the Review column as you go.

| Page | Question it answers | Interactive | Build | Phase | Review |
| --- | --- | --- | --- | --- | --- |
| 0.1 Start here | What can this site teach me, and in what order? | Site map | Live | Later |  |
| 1.1 The timeline | How did we get here? | Timeline | Precomputed | First 8 |  |
| 1.2 Perceptrons and the first winter | What could the first networks do, and why did funding collapse? | Perceptron playground | Live | First 8 |  |
| 1.3 Backpropagation and the second winter | Why did deep networks not take over in the 1980s? | Where did the gradient go? | Live | Later |  |
| 1.4 Deep learning | What changed in 2012? | What a filter sees | Mixed | Later |  |
| 1.5 Transformers and scale | Why did scale take over? | Compute over time | Precomputed | Later |  |
| 1.6 Assistants and reasoning models | How did a text predictor become an assistant? | Same prompt, three models | Precomputed | Later |  |
| 2.1 Learning means choosing parameters | What does it mean for a network to learn? | Fit by hand | Live | Later |  |
| 2.2 The neuron | What does one neuron compute? | Perceptron playground, neuron mode | Live | Later |  |
| 2.3 Why nonlinearity matters | Why do networks need nonlinear functions? | Collapse | Live | Later |  |
| 2.4 Layers bend space | What do hidden layers do? | Watch the plane bend | Live | First 8 |  |
| 2.5 Loss as a proper scoring rule | What does the training loss reward? | The forecaster's game | Live | First 8 |  |
| 2.6 Gradient descent | How does training find good parameters? | Roll downhill | Live | First 8 |  |
| 2.7 Backpropagation | How are the gradients computed? | Gradients by hand | Live | Later |  |
| 2.8 Overfitting and generalization | Why can a model fit its data and still fail? | Degree slider | Live | Later |  |
| 3.1 Tokens | What does a model actually read? | Tokenizer playground | Live | Later |  |
| 3.2 Embeddings | How does a token become a vector? | Embedding space | Precomputed | Later |  |
| 3.3 Next-token prediction and sampling | How does a model produce text? | Sample the next token | Live | Later |  |
| 3.4 Attention | How do positions share information? | Attention by hand and in GPT-2 | Mixed | First 8 |  |
| 3.5 Multiple heads | What do different heads do? | Head gallery | Precomputed | Later |  |
| 3.6 The residual stream and the MLP | Where does the model keep its working state? | Read the stream | Precomputed | Later |  |
| 3.7 A full forward pass | What happens in one full pass? | Step through GPT-2 | Precomputed | Later |  |
| 3.8 Scaling laws | How does performance grow with scale? | Spend a compute budget | Live | Later |  |
| 4.1 From pretrained model to assistant | How does a base model become an assistant? | Same prompt, four checkpoints | Precomputed | Later |  |
| 4.2 Learning from human preferences | How is human judgment turned into a reward? | Be the labeler | Live | Later |  |
| 4.3 Optimizing a learned reward | What happens when a model is pushed hard against a learned reward? | Tilt the distribution | Live | Later |  |
| 4.4 Chain of thought and reasoning models | Why does writing out steps help, and is the writing faithful? | Tokens are computation | Precomputed | Later |  |
| 5.1 Universal approximation | What functions can networks represent? | Build any function | Live | Later |  |
| 5.2 Learning as selection | What does learning from reward select? | Learning and selection on a simplex | Live | Later |  |
| 5.3 Double descent | Why do very large models generalize? | Through the threshold | Live | Later |  |
| 5.4 Grokking | Can a network generalize long after memorizing? | Grok it yourself | Precomputed | Later |  |
| 5.5 Grown, not programmed | Who writes a trained network's algorithm? | Five seeds | Precomputed | Later |  |
| 5.6 Emergence | Do abilities appear suddenly with scale? | Make emergence appear | Live | Later |  |
| 6.1 What interpretability tries to do | What can interpretability show, and with what evidence? | Methods map | Live | Later |  |
| 6.2 The linear representation hypothesis | Are concepts directions? | Find the direction | Precomputed | Later |  |
| 6.3 Linear probes | What does a probe's accuracy show? | Train a probe | Mixed | Later |  |
| 6.4 Superposition and polysemanticity | Why do single neurons respond to unrelated things? | Toy model of superposition | Live | First 8 |  |
| 6.5 Sparse dictionary learning | How can the features be recovered? | Decompose an activation | Mixed | Later |  |
| 6.6 Steering vectors | Can editing activations change behavior? | Steer the model | Mixed | Later |  |
| 6.7 Circuits and activation patching | How do we find the parts that do a task? | Patch it | Precomputed | Later |  |
| 7.1 Radical interpretation | What would make an attribution of beliefs and desires correct? | One behavior, many minds | Live | Later |  |
| 7.2 Belief | When does an internal state count as a belief? | Test a belief detector | Mixed | Later |  |
| 7.3 Desire and goals | When is it correct to say a system wants something? | Swap the goal | Mixed | Later |  |
| 8.1 Optimizing a proxy | Why does optimizing a proxy go wrong? | Select on the proxy | Live | First 8 |  |
| 8.2 Reward hacking and specification gaming | How do systems meet the letter of an objective and miss its intent? | Write a reward function | Live | Later |  |
| 8.3 Goal misgeneralization | Can a correct reward still teach the wrong goal? | Coin or right end? | Precomputed | Later |  |
| 8.4 Instrumental convergence | Why might agents with different goals act alike? | Random goals, common moves | Live | Later |  |
| 8.5 Learned optimizers and deceptive alignment | Could a system hide a misaligned goal during training? | Training cannot see the difference | Live | Later |  |
| 8.6 Oversight and control | How can we stay safe with systems we cannot fully judge? | Control protocol game | Live | Later |  |
| 9.1 Open problems and how to start | Where can I contribute? | Find a problem | Live | Later |  |

## Core path

Twenty pages, about 5 hours, marked with a dot on the site. They carry the main line of argument, from "a network is a function" to "why a trained system can pursue unintended goals"; the other 30 pages deepen it. Where a core page relies on a page outside the path, the link opens a short summary of that page in the margin.

1. 1.1 The timeline
2. 2.1 Learning means choosing parameters
3. 2.4 Layers bend space
4. 2.5 Loss as a proper scoring rule
5. 2.6 Gradient descent
6. 2.7 Backpropagation
7. 3.4 Attention
8. 3.6 The residual stream and the MLP
9. 4.1 From pretrained model to assistant
10. 4.3 Optimizing a learned reward
11. 5.2 Learning as selection
12. 5.5 Grown, not programmed
13. 6.2 The linear representation hypothesis
14. 6.4 Superposition and polysemanticity
15. 6.7 Circuits and activation patching
16. 7.1 Radical interpretation
17. 8.1 Optimizing a proxy
18. 8.2 Reward hacking and specification gaming
19. 8.4 Instrumental convergence
20. 8.5 Learned optimizers and deceptive alignment

## Concept map

Concept map: see the planning doc

Each arrow is a dependency between sections, labeled with the page that is used and the page that uses it. Two short links are unlabeled: 6.3 feeds 7.2 (probes for belief), and 7.1 feeds 8.3 (goal misgeneralization). A reader in a hurry can reach Section 8 through 2.5, 4.3, and 5.2.

## Shared components and build order

Nine components cover all 50 tools. Building them first makes most pages a matter of configuration.

| Component | What it does | Pages |
| --- | --- | --- |
| Page shell | Layout, claim-type badges, "Try" tasks with delayed reveal, check question, progress record | All |
| Plane classifier | Points in the plane, a small network trained live in JavaScript, decision boundary, layer-by-layer view | 1.2, 2.1 to 2.4, 2.8, 5.1, 5.3, 5.5 |
| Chart kit | Line, scatter, bar, and heat-map charts in one style | Most pages |
| Simplex plot | Three-action triangle with flow field and learning paths | 5.2, 8.5 |
| Gridworld and graph solver | Small environments solved exactly by value iteration in the browser | 7.3, 8.2 to 8.4 |
| GPT-2 data pipeline and viewer | Python script (TransformerLens) that exports activations, attention patterns, and patching results to JSON; viewer for token strips, layer-by-head grids, and heat maps | 3.4 to 3.7, 6.2, 6.3, 6.5 to 6.7 |
| Probe trainer | Logistic regression and CCS trained live on stored activations | 6.2, 6.3, 7.2 |
| Modular-addition runs | Training checkpoints for a grid of settings, with a frequency viewer | 5.4, 5.5, 6.7 |
| Small model in the browser | distilgpt2 or GPT-2 small through transformers.js, with an activation hook for steering | 3.3, 6.6 |

**Build order.**

1. The page shell and the plane classifier.
2. The first eight tools: 1.1 (timeline), 1.2 (perceptron), 2.4 (layers bend space), 2.5 (scoring rules), 2.6 (gradient descent), 3.4 (attention), 6.4 (superposition), 8.1 (proxy selection). Together they cover the central idea of each part of the site, and all but the timeline and the second tab of 3.4 run live, so they need no data pipeline.
3. The GPT-2 data pipeline, which unlocks most of Sections 3 and 6.
4. The gridworld solver, which unlocks Sections 7 and 8.
5. The modular-addition training runs, which unlock 5.4, 5.5, and 6.7.

Stack (hosting decided 2026-10-08): a static site in a GitHub repository, published with GitHub Pages; D3 for charts; KaTeX for formulas; a Python folder for the precomputed data.

## Fixed terms

One label per concept, used on every page. Where two terms are easily confused, both are listed.

| Term | Meaning on this site | Not used for this concept |
| --- | --- | --- |
| parameters | The numbers adjusted in training; "weight matrix" names one specific matrix of them | "weights" in general text |
| loss | The number training minimizes | cost, error function |
| activation | A number computed inside the network for a given input |  |
| activation function | The fixed nonlinear function applied in a layer (ReLU, GELU) | nonlinearity |
| residual stream | The vector each position carries through the transformer | hidden state |
| feature | A property of the input that the model represents | concept (in technical passages) |
| direction | A vector in activation space associated with a feature | axis |
| probe | A classifier trained on activations | decoder, classifier head |
| reward | The number used to score outputs in reinforcement learning |  |
| reward model | A learned model that predicts human preferences and outputs a reward | preference model |
| proxy | Any measure optimized in place of the intended goal |  |
| policy | The model while it is being trained with reinforcement learning | agent (except in Sections 7 and 8, where "agent" is the topic) |
| objective | What training optimizes | goal |
| goal | What a system's behavior is directed at | objective |
| base model | A model after pretraining only | foundation model |
| chain of thought | Text a model writes before its final answer | reasoning trace, scratchpad (scratchpad only as an analogy) |

## Open questions for Aydin

- [x] **Prerequisites.** Confirm the assumed background: linear algebra, calculus, probability, no machine learning. CONFIRMED
- [x] **Section 7.** Which claims and framings from your agency work belong on 7.1 and 7.2? The criteria slot in 7.2 is empty. (Look at the Herrmann & Levinstein textbook: answers are all there)
- [x] **Page 8.4.** Should it link to the evolutionary treatment of instrumental convergence in your project with Daniel Herrmann? NO, JUST THE MAIN PUBLISHED RESULTS FROM OMAHUNDRO AND BOSTROM TO ZHUANG & HADFIELD MANELL, SOARE ET AL, AND TURNER ET AL'S TWO PAPERS
- [x] **Size.** 50 pages is 10 to 15 hours of reading. Should there also be a marked core path of about 20 pages for a first pass? YES
- [x] **Hosting.** A static site in a GitHub repo (my recommendation), your WordPress site, or a Claude artifact? GItHub REPO
- [x] **Use in teaching.** Will this be assigned in a course or used one-to-one? A course would justify graded exercises beyond the check questions. USED ONE-TO-ONE FOR SELF-TEACHING

All answered 2026-10-08 and applied: the self-teaching use shaped Reader and goal and How the site teaches; the core path is listed under Core path; Section 7 now follows Herrmann and Levinstein's *Radical AI Interpretability*; page 8.4 now covers the published results only; hosting is in Shared components.

## Sources and verification status

The numbers, dates, and attributions in the page specs were checked against primary sources on 2026-10-08 (42 claims). Three needed corrections, now made: AlexNet's 15.3% came from its entry trained with extra data (16.4% without); the Börgers and Sarin result concerns the learning process itself, under stated conditions; and Levinstein and Herrmann appeared in print in 2025 (online 2024). The remaining claims checked out.

Still to check before the site goes live: the 1958 New York Times quote, the human error estimate on ImageNet (about 5.1%), and every claim that will be added to Section 7.

Sources opened for the numbers:

- [Hoffmann et al. 2022, Chinchilla](https://arxiv.org/pdf/2203.15556): 70B parameters, 1.4T tokens, the Gopher comparison, the fitted loss constants
- [Russakovsky et al. 2015, ImageNet challenge](https://arxiv.org/pdf/1409.0575) and [ILSVRC 2017 results](https://image-net.org/challenges/LSVRC/2017/results): winning errors by year
- [He et al. 2015, ResNet](https://arxiv.org/abs/1512.03385): 3.57%
- [Templeton et al. 2024, Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html): 34 million features
- [Wang et al. 2022, IOI circuit](https://arxiv.org/abs/2211.00593): 26 heads
- [Kwa et al. 2025, METR](https://arxiv.org/abs/2503.14499): seven-month doubling
- [Clark and Amodei 2016, Faulty Reward Functions](https://openai.com/index/faulty-reward-functions/): CoastRunners
- [GDM interpretability team 2025, Negative Results for SAEs](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28/negative-results-for-saes-on-downstream-tasks)
- [Levinstein and Herrmann, Still No Lie Detector](https://link.springer.com/article/10.1007/s11098-023-02094-3)
- [Amodei 2025, The Urgency of Interpretability](https://www.darioamodei.com/post/the-urgency-of-interpretability)
- [Olah et al. 2020, Zoom In](https://distill.pub/2020/circuits/zoom-in/)
- [Korbak et al. 2025, Chain of Thought Monitorability](https://arxiv.org/abs/2507.11473)
- [Skalse et al. 2023, Partial Identifiability in Reward Learning](https://proceedings.mlr.press/v202/skalse23a.html)

Section 7 follows Herrmann and Levinstein, [*Radical AI Interpretability*](https://arxiv.org/abs/2606.26523) (arXiv v1, June 2026). It was read in full except §6.3.3, §6.4, and §6.5.2; section numbers cited on pages 7.1 to 7.3 refer to that version.

Added after the fact-check and not yet verified against sources: the references in How the site teaches and Visual design, and the new references on page 8.4.
