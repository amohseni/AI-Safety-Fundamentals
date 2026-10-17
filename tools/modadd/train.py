"""Train one modular-addition run and save its training record.

Usage:
  python3 train.py --p 113 --frac 0.3 --wd 1.0 --seed 0 --steps 20000 --out runs/p113_f0.3_wd1.0_s0.npz

Full-batch AdamW, as in Nanda et al. (2023). The record holds, at 40 log-spaced steps up to
step 1,000 and then every 50 steps: train and test accuracy and loss; and at checkpoints
(the same early steps, then every 200 steps): which of the
p*p pairs the model answers correctly, the number embeddings W_E, and the Fourier
coefficients of the logits used for the restricted accuracy (see modadd.py).
The final parameters are saved alongside, as <out>.pt.
"""
import argparse
import math
import time
import numpy as np
import torch
import torch.nn.functional as Fn
from modadd import Transformer, all_pairs, split

ap = argparse.ArgumentParser()
ap.add_argument('--p', type=int, default=113)
ap.add_argument('--frac', type=float, default=0.3)
ap.add_argument('--wd', type=float, default=1.0)
ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--data-seed', type=int, default=0)
ap.add_argument('--steps', type=int, default=20000)
ap.add_argument('--d-model', type=int, default=128)
ap.add_argument('--heads', type=int, default=4)
ap.add_argument('--d-mlp', type=int, default=512)
ap.add_argument('--metric-every', type=int, default=50)
ap.add_argument('--ckpt-every', type=int, default=200)
ap.add_argument('--out', required=True)
args = ap.parse_args()

torch.set_num_threads(1)
p = args.p
model = Transformer(p, args.d_model, args.heads, args.d_mlp, seed=args.seed)
A, B = all_pairs(p)
Y = (A + B) % p
mask = torch.from_numpy(split(p, args.frac, args.data_seed))
At, Bt, Yt = A[mask], B[mask], Y[mask]

# AdamW with betas (0.9, 0.98) and a 10-step linear warmup (our choice, not checked against the paper).
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(s / 10, 1))


def schedule(every, last):
    # 40 log-spaced steps from 1 to 1,000, then every `every` steps to the end.
    early = np.unique(np.round(np.logspace(0, 3, 40)).astype(int)).tolist()
    return set([0] + early + list(range(1000, last + 1, every)) + [last])


metric_steps = schedule(args.metric_every, args.steps)
ckpt_steps = schedule(args.ckpt_every, args.steps)
rec = {k: [] for k in ['m_step', 'train_acc', 'test_acc', 'train_loss', 'test_loss']}
ck = {k: [] for k in ['c_step', 'correct', 'W_E', 'logit_coef']}

a_np = np.arange(p)
s_np = (a_np[:, None] + a_np[None, :]) % p
PHI = []  # unit-norm cos and sin of w_k (a+b) over the p x p grid, k = 1..(p-1)/2
for k in range(1, (p - 1) // 2 + 1):
    for f in (np.cos, np.sin):
        phi = f(2 * math.pi * k * s_np / p).reshape(-1)
        PHI.append(phi / np.linalg.norm(phi))
PHI = torch.tensor(np.array(PHI), dtype=torch.float64)  # (p-1, p*p)


def evaluate(step):
    with torch.no_grad():
        logits = model(A, B).double()
        lp = logits.log_softmax(-1)
        nll = -lp.gather(1, Y[:, None])[:, 0]
        corr = logits.argmax(-1) == Y
        tm, vm = mask, ~mask
        if step in metric_steps:
            rec['m_step'].append(step)
            rec['train_acc'].append(corr[tm].float().mean().item())
            rec['test_acc'].append(corr[vm].float().mean().item())
            rec['train_loss'].append(nll[tm].mean().item())
            rec['test_loss'].append(nll[vm].mean().item())
        if step in ckpt_steps:
            ck['c_step'].append(step)
            ck['correct'].append(np.packbits(corr.numpy()))
            ck['W_E'].append(model.W_E.detach().numpy()[:p].astype(np.float32).copy())
            const = logits.mean(0)  # (p,)
            coef = PHI @ logits  # (p-1, p)
            ck['logit_coef'].append(torch.cat([const[None], coef]).numpy().astype(np.float32))


t0 = time.time()
evaluate(0)
for step in range(1, args.steps + 1):
    logits = model(At, Bt).double()
    loss = Fn.cross_entropy(logits, Yt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if step in metric_steps or step in ckpt_steps:
        evaluate(step)
    if step % 2000 == 0:
        print(f'{args.out} step {step} train {rec["train_acc"][-1]:.3f} test {rec["test_acc"][-1]:.3f} '
              f'{(time.time() - t0) / step * 1000:.1f} ms/step', flush=True)

np.savez_compressed(args.out, mask=mask.numpy(), config=np.array(str(vars(args))),
                    **{k: np.array(v) for k, v in rec.items()}, **{k: np.array(v) for k, v in ck.items()})
torch.save(model.state_dict(), args.out.replace('.npz', '.pt'))
print('done', args.out, f'{time.time() - t0:.0f}s')
