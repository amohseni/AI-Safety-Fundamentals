"""One-layer transformer for (a + b) mod p, following Nanda et al. (2023).

Model: token embedding and learned positional embedding, one attention layer,
one ReLU MLP, unembedding. No layer norm, untied embed and unembed. Input is
the sequence "a b =", and the answer is read at the "=" position.

Only the output at the last position is needed for the loss, and the model has
one layer, so queries, the MLP, and the unembedding are computed at the last
position only. The keys and values are computed at all three positions. This
gives exactly the same outputs and gradients as running the full sequence.
"""
import math
import numpy as np
import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, p, d_model=128, n_heads=4, d_mlp=512, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        d_head = d_model // n_heads
        self.p, self.d, self.h, self.dh = p, d_model, n_heads, d_head
        r = lambda *s, scale: nn.Parameter(torch.randn(*s, generator=g) * scale)
        # Initialization: N(0, 1/d_model) for most matrices, N(0, 1/p) for the unembedding,
        # zero biases. (Our choice, not checked against the paper.)
        self.W_E = r(p + 1, d_model, scale=1 / math.sqrt(d_model))  # tokens 0..p-1 and "=" (token p)
        self.W_pos = r(3, d_model, scale=1 / math.sqrt(d_model))
        self.W_Q = r(n_heads, d_model, d_head, scale=1 / math.sqrt(d_model))
        self.W_K = r(n_heads, d_model, d_head, scale=1 / math.sqrt(d_model))
        self.W_V = r(n_heads, d_model, d_head, scale=1 / math.sqrt(d_model))
        self.W_O = r(n_heads, d_head, d_model, scale=1 / math.sqrt(d_model))
        self.W_in = r(d_model, d_mlp, scale=1 / math.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = r(d_mlp, d_model, scale=1 / math.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.W_U = r(d_model, p, scale=1 / math.sqrt(p))

    def forward(self, a, b, W_E=None):
        """a, b: long tensors of shape (N,). Returns logits (N, p) at the "=" position."""
        E = self.W_E if W_E is None else W_E
        eq = torch.full_like(a, self.p)
        x = torch.stack([E[a], E[b], E[eq]], dim=1) + self.W_pos  # (N, 3, d)
        k = torch.einsum('npd,hde->nhpe', x, self.W_K)
        v = torch.einsum('npd,hde->nhpe', x, self.W_V)
        q = torch.einsum('nd,hde->nhe', x[:, 2], self.W_Q)
        att = torch.einsum('nhe,nhpe->nhp', q, k) / math.sqrt(self.dh)
        att = att.softmax(dim=-1)  # the last position attends to all three positions
        z = torch.einsum('nhp,nhpe->nhe', att, v)
        r = x[:, 2] + torch.einsum('nhe,hed->nd', z, self.W_O)
        r = r + torch.relu(r @ self.W_in + self.b_in) @ self.W_out + self.b_out
        return r @ self.W_U


def fourier_basis(p):
    """Orthonormal real Fourier basis over 0..p-1 (p odd), as rows.
    Row 0 is the constant; rows 2k-1 and 2k are cos and sin at frequency k = 1..(p-1)/2."""
    a = np.arange(p)
    rows = [np.ones(p) / math.sqrt(p)]
    for k in range(1, (p - 1) // 2 + 1):
        rows.append(np.cos(2 * math.pi * k * a / p) * math.sqrt(2 / p))
        rows.append(np.sin(2 * math.pi * k * a / p) * math.sqrt(2 / p))
    return np.array(rows)


def embedding_spectrum(W_E, p, F=None):
    """Norm of the number embeddings' component at each frequency k = 1..(p-1)/2.
    Returns (spectrum, constant_norm)."""
    F = fourier_basis(p) if F is None else F
    C = F @ W_E[:p]  # coefficients, (p, d)
    sq = (C ** 2).sum(axis=1)
    spec = np.sqrt(sq[1::2] + sq[2::2])
    return spec, math.sqrt(sq[0])


def ablate_embedding(W_E, p, keep, F=None):
    """Keep only the constant component and the frequencies in `keep` of the number embeddings."""
    F = fourier_basis(p) if F is None else F
    mask = np.zeros(p)
    mask[0] = 1
    for k in keep:
        mask[2 * k - 1] = mask[2 * k] = 1
    W = W_E.copy()
    W[:p] = F.T @ (mask[:, None] * (F @ W_E[:p]))
    return W


def all_pairs(p):
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    return a, b


def split(p, frac, data_seed=0):
    """Boolean mask over the p*p pairs (row-major, index a*p + b) marking training pairs.
    One fixed random order of all pairs; the first frac of it is the training set, so the
    training set for a smaller fraction is contained in the one for a larger fraction."""
    rng = np.random.RandomState(data_seed)
    order = rng.permutation(p * p)
    n = int(round(frac * p * p))
    mask = np.zeros(p * p, dtype=bool)
    mask[order[:n]] = True
    return mask


def restricted_and_excluded(logits, p, keys):
    """Nanda et al.'s progress measures, applied to logits over all pairs.
    logits: (p*p, p), row a*p+b. For each output c, the logit surface over (a, b) is
    projected onto cos(w_k (a+b)) and sin(w_k (a+b)) for the key frequencies k.
    Restricted logits: the constant plus those projections. Excluded logits: the logits
    minus those projections. Returns both as (p*p, p) arrays."""
    L = logits.reshape(p, p, p)  # a, b, c
    a = np.arange(p)
    s = (a[:, None] + a[None, :]) % p
    proj = np.zeros_like(L)
    for k in keys:
        for f in (np.cos, np.sin):
            phi = f(2 * math.pi * k * s / p)
            phi = phi / np.linalg.norm(phi)
            coef = np.einsum('ab,abc->c', phi, L)
            proj += phi[:, :, None] * coef[None, None, :]
    const = L.mean(axis=(0, 1), keepdims=True)
    restricted = const + proj
    excluded = L - proj
    return restricted.reshape(p * p, p), excluded.reshape(p * p, p)
