# Page 7.3 network

This folder makes `docs/data/7-3/policy.json`, the network that page 7.3 runs in the browser, with its goal and door subspaces and its probes.

## The environment

The two-room gridworld is defined in `docs/js/lib/s73-maze.js`, which the page, the data generator, and the tests share: a 9 by 7 grid with a wall down column 4, an upper and a lower door (each open, closed, or locked), an optional key, and four optional wall segments.

## Steps

```
node tools/s73/gen.mjs <scratch>/data 80000 20 1     # 1.6 million training states
node tools/s73/gen.mjs <scratch>/val 5000 20 2       # 100,000 held-out states
python3 tools/s73/train.py <scratch>/data <scratch>/val <scratch>/net128.json 128 12 0
python3 tools/s73/analyze.py <scratch>/net128.json <scratch>/val <scratch>/an1.json 1 64
node tools/s73/export.mjs <scratch>/net128.json <scratch>/an1.json 32 12
node tools/test-page-7-3.mjs
```

- `gen.mjs` samples random environments, goals, and agent states (including doorways and the key's cell), and labels each state with the set of moves that lie on a shortest route to the goal.
- `train.py` trains a network with 316 inputs, two ReLU layers of 128, and 4 outputs by imitation: the loss is minus the log of the total probability given to the set of shortest-route moves. Adam, learning rate 0.001 with cosine decay, batch 1,024, 12 epochs; PyTorch 2.14.1 on CPU, about 5 minutes. Held-out accuracy (the top move is a shortest-route move): 99.96%.
- `analyze.py` works on the first layer's activations (after the ReLU). The goal subspace is the top principal directions of the change in the activations when only the goal is moved to another free cell, over 40,000 held-out states; the door subspace is the same for a change of only the door states. It also fits softmax-regression probes for the goal cell (63 classes) and for each door's state (3 classes).
- `export.mjs` keeps the first 32 goal directions and the first 12 door directions and stores every array as base64 float32.

The subspace sizes were chosen by measuring the swap: with the goal subspace at 8, 16, 32, and 48 directions, swapping in another goal's activations reached the new goal in about 45%, 81%, 93.5%, and 99% of 1,000 random episodes. With the door subspace at 8 directions, an agent made to believe that a closed door was open ended its run pushing against that door in 124 of 338 such runs (37%); at 12 directions, in 308 of 338 (91%). The numbers quoted on the page come from `tools/test-page-7-3.mjs`.
