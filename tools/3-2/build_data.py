"""Builds docs/data/3-2/vectors.json for page 3.2 from the GoogleNews word2vec vectors.

Usage (from the repo root):
  python3 -I tools/3-2/read_w2v.py <GoogleNews-vectors-negative300.bin, or its first ~360 MB> <top.npz> 300000
  python3 -I tools/3-2/build_data.py <top.npz> docs/data/3-2/vectors.json
See tools/3-2/README.md.
"""
import sys, json, base64, re
import numpy as np

src, out = sys.argv[1], sys.argv[2]
N_FREQ = 4000
d = np.load(src)
words = [str(w) for w in d['words']]
V = d['vecs'].astype(np.float32)

# Curated words the page's tasks and figure use; kept even if they are not among the most frequent.
CURATED = '''king queen man woman prince princess monarch boy girl brother sister uncle aunt father mother son daughter husband wife
bank river money loan deposit shore
France Paris Italy Rome Germany Berlin Japan Tokyo Russia Moscow Spain Madrid China Beijing Egypt Cairo Canada Ottawa
Greece Athens England London Poland Warsaw Turkey Ankara Kenya Nairobi Peru Lima Portugal Lisbon Sweden Stockholm
Norway Oslo Austria Vienna Ireland Dublin India Delhi Cuba Havana Iran Tehran Australia Canberra Brazil Brasilia
Milan Toronto Sydney walking walked swimming swam big bigger small smaller good better bad worse'''.split()
plain = re.compile(r'^(?:[a-z]+|[A-Z][a-z]+)$')
keep, seen = [], set()
for i, w in enumerate(words):
    if len(keep) >= N_FREQ:
        break
    if plain.match(w) and w not in seen:
        keep.append(i); seen.add(w)
index = {w: i for i, w in enumerate(words)}
for w in CURATED:
    if w not in seen:
        keep.append(index[w]); seen.add(w)
W = [words[i] for i in keep]
X = V[keep]
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# Store each unit vector as signed bytes with its own scale (x is about q * scale).
scale = np.abs(X).max(axis=1) / 127.0
Q = np.round(X / scale[:, None]).astype(np.int8)

# Projections of the stored vectors (after the byte rounding, renormalized).
Xq = Q.astype(np.float32) * scale[:, None]
Xq /= np.linalg.norm(Xq, axis=1, keepdims=True)
C = Xq - Xq.mean(axis=0)
_, _, vt = np.linalg.svd(C, full_matrices=False)
pca = C @ vt[:2].T
import umap
um = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=0, n_jobs=1).fit_transform(Xq)

def norm01(P):
    P = P - P.min(axis=0)
    return (P / P.max(axis=0)).round(4).tolist()

doc = {
    'source': 'word2vec GoogleNews-vectors-negative300 (Mikolov et al. 2013), first 300,000 entries, filtered',
    'dim': int(X.shape[1]),
    'words': W,
    'scale': [float(f'{s:.6g}') for s in scale],
    'q': base64.b64encode(Q.tobytes()).decode(),
    'pca': norm01(pca),
    'umap': norm01(um),
}
with open(out, 'w') as f:
    json.dump(doc, f, separators=(',', ':'))
print(len(W), 'words written')
