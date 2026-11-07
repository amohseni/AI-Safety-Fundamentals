# Compares models on the shared statements: for each layer, a logistic probe trained on half the
# city statements, tested on the other half and on the other topics (sidenote on page 6.2).
# Usage: python3 tools/truth/compare_models.py <statements.json> <acts.npy>   (acts from acts.py)
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
S = json.load(open(sys.argv[1]))['statements']
A = np.load(sys.argv[2]).astype(np.float32)
lab = np.array([s['label'] for s in S]); top = np.array([s['topic'] for s in S]); form = np.array([s['form'] for s in S])
item = np.array([s['item'] for s in S])
train_item = (item % 2 == 0)
def sel(t, f, tr=None):
    m = (top == t) & (form == f)
    if tr is True: m &= train_item
    if tr is False: m &= ~train_item
    return m
def mm(Xtr, ytr):  # mass-mean direction
    return Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
print('layer  LR: cityTest spanish elements cityNeg cityPara | MM: cityTest spanish elements neg')
for L in range(A.shape[0]):
    X = A[L]
    mu = X[(form == 'plain')].mean(0); sd = X[(form == 'plain')].std(0) + 1e-6
    Z = (X - mu) / sd
    tr = sel('cities', 'plain', True)
    clf = LogisticRegression(C=0.1, max_iter=2000).fit(Z[tr], lab[tr])
    accs = [clf.score(Z[m], lab[m]) for m in [sel('cities', 'plain', False), sel('spanish', 'plain'), sel('elements', 'plain'), sel('cities', 'neg'), sel('cities', 'para')]]
    d = mm(X[tr], lab[tr]); p = X @ d; thr = np.median(p[tr])
    thr = (p[tr & (lab == 1)].mean() + p[tr & (lab == 0)].mean()) / 2
    mma = [((p[m] > thr) == lab[m]).mean() for m in [sel('cities', 'plain', False), sel('spanish', 'plain'), sel('elements', 'plain'), sel('cities', 'neg')]]
    # within-topic LR for spanish/elements
    ws = []
    for t in ['spanish', 'elements']:
        a, b = sel(t, 'plain', True), sel(t, 'plain', False)
        ws.append(LogisticRegression(C=0.1, max_iter=2000).fit(Z[a], lab[a]).score(Z[b], lab[b]))
    print(L, ' '.join(f'{v:.2f}' for v in accs), '|', ' '.join(f'{v:.2f}' for v in mma), '| within', ' '.join(f'{v:.2f}' for v in ws))
