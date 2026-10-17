# Trains the page 7.3 policy network by imitation: for each state the target is the set of
# actions on a shortest route to the goal, and the loss is minus the log of the total
# probability the network gives to that set.
# Usage: python3 tools/s73/train.py <data-dir> <val-dir> <out.json> [hidden=128] [epochs=12] [seed=0]
import json, sys
import numpy as np
import torch

data, val, out = sys.argv[1:4]
hidden = int(sys.argv[4]) if len(sys.argv) > 4 else 128
epochs = int(sys.argv[5]) if len(sys.argv) > 5 else 12
seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
OBS = 316
torch.manual_seed(seed); np.random.seed(seed)
torch.set_num_threads(2)

def load(d):
    obs = np.fromfile(f'{d}/obs.bin', dtype=np.uint8)
    n = obs.size // 40
    x = np.unpackbits(obs.reshape(n, 40), axis=1)[:, :OBS].astype(np.float32)
    m = np.fromfile(f'{d}/mask.bin', dtype=np.uint8)
    y = ((m[:, None] >> np.arange(4)[None, :]) & 1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y)

X, Y = load(data); Xv, Yv = load(val)
net = torch.nn.Sequential(
    torch.nn.Linear(OBS, hidden), torch.nn.ReLU(),
    torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
    torch.nn.Linear(hidden, 4))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
B = 1024
def loss_fn(logits, y):
    lp = torch.log_softmax(logits, 1)
    return -(torch.logsumexp(lp + torch.log(y.clamp_min(1e-12)), 1)).mean()
for ep in range(epochs):
    perm = torch.randperm(X.shape[0])
    tot = 0.0
    for i in range(0, X.shape[0], B):
        idx = perm[i:i + B]
        l = loss_fn(net(X[idx]), Y[idx])
        opt.zero_grad(); l.backward(); opt.step()
        tot += l.item() * len(idx)
    sched.step()
    with torch.no_grad():
        lv = net(Xv)
        acc = Yv[torch.arange(len(Yv)), lv.argmax(1)].mean().item()
    print(f'epoch {ep + 1}: train loss {tot / X.shape[0]:.4f}, val accuracy {acc:.4f}', flush=True)

layers = [m for m in net if isinstance(m, torch.nn.Linear)]
json.dump({
    'sizes': [OBS] + [l.out_features for l in layers],
    'W': [l.weight.detach().numpy().astype(np.float64).round(5).tolist() for l in layers],
    'b': [l.bias.detach().numpy().astype(np.float64).round(5).tolist() for l in layers],
}, open(out, 'w'))
print('saved', out)
