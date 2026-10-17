# Finds the goal and door subspaces and fits the probes for the page 7.3 network.
# Usage: python3 tools/s73/analyze.py <net.json> <data-dir> <out.json> [layer=1] [kGoal=32]
# The data directory is one written by gen.mjs (a held-out set, not the training data).
# For one hidden layer (1 or 2, after the activation function) it computes:
#   mean:  the mean activation vector;
#   goal:  orthonormal directions (largest first): the principal components of the change in the
#          activations when only the goal is moved, over 40,000 held-out states;
#   door:  the same for a change of only the door states;
#   goalProbe: softmax regression from the activations to the goal cell (63 classes);
#   doorProbe: for each door, softmax regression to its state (open, closed, locked).
import json, sys
import numpy as np
import torch

netf, data, out = sys.argv[1:4]
layer = int(sys.argv[4]) if len(sys.argv) > 4 else 1
kG = int(sys.argv[5]) if len(sys.argv) > 5 else 32
torch.manual_seed(0)
net = json.load(open(netf))
Ws = [np.array(w) for w in net['W']]; bs = [np.array(b) for b in net['b']]
obs = np.fromfile(f'{data}/obs.bin', dtype=np.uint8); n = obs.size // 40
X = np.unpackbits(obs.reshape(n, 40), axis=1)[:, :316].astype(np.float64)
meta = np.fromfile(f'{data}/meta.bin', dtype=np.uint8).reshape(n, 8)
h = X
for l in range(layer):
    h = np.maximum(0, h @ Ws[l].T + bs[l])
mu = h.mean(0)
goal = meta[:, 0].astype(int); doors = meta[:, 2] * 3 + meta[:, 3]

def basis(labels, k):
    cls = np.unique(labels)
    M = np.stack([h[labels == c].mean(0) - mu for c in cls])
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return Vt[:k].T, var

N = 63
rng = np.random.default_rng(0)
DOORC = [1 * 9 + 4, 5 * 9 + 4]
def h_of(Xin):
    z = Xin
    for l in range(layer):
        z = np.maximum(0, z @ Ws[l].T + bs[l])
    return z
def cf_goal(Xin):
    # the same states with the goal moved to another free cell
    bad = (Xin[:, :N] > 0) | (Xin[:, 2 * N:3 * N] > 0) | (Xin[:, 3 * N:4 * N] > 0) | (Xin[:, 4 * N:5 * N] > 0)
    bad[:, DOORC] = True
    sc = rng.random(bad.shape); sc[bad] = -1
    g2 = sc.argmax(1)
    Xc = Xin.copy(); Xc[:, 4 * N:5 * N] = 0; Xc[np.arange(len(Xc)), 4 * N + g2] = 1
    return Xc
def cf_doors(Xin):
    # the same states with both doors given new states (not both closed); agents in a doorway are skipped
    Xc = Xin.copy()
    st = rng.integers(0, 3, size=(len(Xc), 2))
    both = (st[:, 0] == 1) & (st[:, 1] == 1); st[both, 1] = 0
    for d, c in enumerate(DOORC):
        Xc[:, c] = (st[:, d] == 1); Xc[:, N + c] = (st[:, d] == 2)
    keep = (Xin[:, 3 * N + DOORC[0]] == 0) & (Xin[:, 3 * N + DOORC[1]] == 0)
    return Xc[keep], keep
def diff_basis(Hd, k):
    U, S, Vt = np.linalg.svd(Hd, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return Vt[:k].T, var
sub = rng.choice(n, 40000, replace=False)
Dg = h_of(cf_goal(X[sub])) - h[sub]
Xd, keep = cf_doors(X[sub]); Dd = h_of(Xd) - h[sub][keep]
UG, varG = diff_basis(Dg, kG)
UD, varD = diff_basis(Dd, 16)
print('goal subspace cumulative variance', np.round(np.cumsum(varG)[[1, 3, 7, 15, 31]], 3))
print('door subspace cumulative variance', np.round(np.cumsum(varD)[[0, 1, 2, 3, 5, 7, 11, 15]], 3))
# overlap between the subspaces: singular values of UG^T UD (cosines of principal angles)
for k in (4, 8, 16):
    s = np.linalg.svd(UG[:, :k].T @ UD[:, :3], compute_uv=False)
    print(f'principal-angle cosines, goal k={k} vs door k=3:', np.round(s, 3))

def softmax_probe(labels, ncls, epochs=300):
    Ht = torch.tensor(h, dtype=torch.float32); yt = torch.tensor(labels)
    idx = torch.randperm(len(yt)); tr, te = idx[: int(0.8 * len(yt))], idx[int(0.8 * len(yt)):]
    lin = torch.nn.Linear(h.shape[1], ncls)
    opt = torch.optim.Adam(lin.parameters(), lr=0.02)
    for _ in range(epochs):
        l = torch.nn.functional.cross_entropy(lin(Ht[tr]), yt[tr]) + 1e-4 * lin.weight.pow(2).sum()
        opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad():
        acc = (lin(Ht[te]).argmax(1) == yt[te]).float().mean().item()
    return lin, acc

gp, accG = softmax_probe(goal, 63)
dps = []
for d in range(2):
    p, acc = softmax_probe(meta[:, 2 + d].astype(int), 3)
    dps.append(p); print(f'door {d} probe held-out accuracy {acc:.4f}')
print(f'goal probe held-out accuracy {accG:.4f}')
r = lambda a: np.round(np.asarray(a, dtype=np.float64), 5).tolist()
json.dump({
    'layer': layer, 'mean': r(mu), 'goal': r(UG.T), 'door': r(UD.T),
    'goalVar': r(varG), 'doorVar': r(varD), 'goalProbeAcc': accG,
    'goalProbe': {'W': r(gp.weight.detach().numpy()), 'b': r(gp.bias.detach().numpy())},
    'doorProbe': [{'W': r(p.weight.detach().numpy()), 'b': r(p.bias.detach().numpy())} for p in dps],
}, open(out, 'w'))
print('saved', out)
