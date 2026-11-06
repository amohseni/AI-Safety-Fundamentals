# Page 4.4, tab 1: two small transformers trained on multi-digit addition.
#   direct   reads "a+b=" and writes the sum, most significant digit first
#   carries  reads "a+b=", writes the carry of each column (right to left), then ">" and the sum
# Both operands have n significant digits (n = 1 to 12) and are zero-padded to 12 digits,
# so every example has the same length and positions, whatever n is.
# Usage: python3 tools/4-4/addition.py direct|carries [steps] [layers]
# Writes tools/4-4/runs/<mode>-L<layers>.pt and .json
import json, math, os, random, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
W = 12                      # operand width in digits
VOCAB = list('0123456789+=>.')
IDX = {c: i for i, c in enumerate(VOCAB)}
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
os.makedirs(RUNS, exist_ok=True)

CFG = dict(d=128, heads=4, layers=2, mlp=512, batch=128, lr=1e-3, warmup=300, seed=0)


def number(n, r):
    return r.randint(10 ** (n - 1) if n > 1 else 0, 10 ** n - 1)


def carries(a, b):
    """Carry into each column above the units column, right to left; W + 1 digits, last is the final carry."""
    out, c = [], 0
    for i in range(W):
        s = (a // 10 ** i) % 10 + (b // 10 ** i) % 10 + c
        c = 1 if s >= 10 else 0
        out.append(str(c))
    return ''.join(out)


def example(a, b, mode):
    q = f'{a:0{W}d}+{b:0{W}d}='
    s = f'{a + b:0{W + 1}d}'
    ans = (carries(a, b) + '>' + s) if mode == 'carries' else s
    return q, ans + '.'


def longest_chain(a, b):
    """Length of the longest run of consecutive columns that each produce a carry."""
    best = run = 0
    for ch in carries(a, b):
        run = run + 1 if ch == '1' else 0
        best = max(best, run)
    return best


class Model(nn.Module):
    def __init__(self, T):
        super().__init__()
        d = CFG['d']
        self.tok = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, CFG['heads'], CFG['mlp'], dropout=0.0, batch_first=True, norm_first=True, activation='gelu')
        self.body = nn.TransformerEncoder(layer, CFG['layers'])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(VOCAB))
        self.register_buffer('mask', torch.triu(torch.full((T, T), float('-inf')), 1))

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T))
        h = self.body(h, mask=self.mask[:T, :T], is_causal=True)
        return self.head(self.norm(h))


def batch(r, mode, size):
    xs, starts = [], []
    for _ in range(size):
        n = r.randint(1, W)
        q, ans = example(number(n, r), number(n, r), mode)
        xs.append([IDX[c] for c in q + ans])
        starts.append(len(q))
    x = torch.tensor(xs)
    y = x[:, 1:].clone()
    loss_mask = torch.zeros_like(y, dtype=torch.bool)
    loss_mask[:, starts[0] - 1:] = True     # loss only on the answer (every example has the same layout)
    return x[:, :-1], y, loss_mask


@torch.no_grad()
def answer(model, a, b, mode):
    q, ans = example(a, b, mode)
    x = torch.tensor([[IDX[c] for c in q]])
    for _ in range(len(ans)):
        nxt = model(x)[0, -1].argmax().item()
        x = torch.cat([x, torch.tensor([[nxt]])], 1)
    return ''.join(VOCAB[i] for i in x[0, len(q):].tolist())


@torch.no_grad()
def answer_batch(model, pairs, mode):
    qs = [example(a, b, mode) for a, b in pairs]
    x = torch.tensor([[IDX[c] for c in q] for q, _ in qs])
    L = len(qs[0][1])
    for _ in range(L):
        nxt = model(x)[:, -1].argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], 1)
    Lq = len(qs[0][0])
    return [''.join(VOCAB[i] for i in row[Lq:].tolist()) for row in x], [ans for _, ans in qs]


def evaluate(model, mode, per_n=500, seed=12345):
    r = random.Random(seed)
    res = {}
    for n in range(1, W + 1):
        pairs = [(number(n, r), number(n, r)) for _ in range(per_n)]
        got, want = answer_batch(model, pairs, mode)
        ok = [g.split('>')[-1] == w.split('>')[-1] for g, w in zip(got, want)]
        res[n] = sum(ok) / per_n
    return res


def main(mode, steps, layers):
    CFG['layers'] = layers
    tag = f'{mode}-L{layers}'
    torch.manual_seed(CFG['seed'])
    r = random.Random(CFG['seed'])
    q, ans = example(1, 1, mode)
    T = len(q) + len(ans)
    model = Model(T)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / CFG['warmup']) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        x, y, m = batch(r, mode, CFG['batch'])
        logits = model(x)
        loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 250 == 0 or step == steps:
            entry = {'step': step, 'loss': round(loss.item(), 4), 'seconds': round(time.time() - t0)}
            if step % 2000 == 0 or step == steps:
                model.eval(); entry['acc'] = evaluate(model, mode, per_n=200, seed=999); model.train()
            log.append(entry)
            print(mode, entry, flush=True)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    final = evaluate(model, mode, per_n=1000)
    # Accuracy by the longest carry chain, over the same kind of random examples.
    rr = random.Random(777)
    by_chain = {}
    pairs = []
    for _ in range(4000):
        n = rr.randint(1, W)
        pairs.append((number(n, rr), number(n, rr)))
    got, want = answer_batch(model, pairs, mode)
    for (a, b), g, w in zip(pairs, got, want):
        k = longest_chain(a, b)
        c = by_chain.setdefault(k, [0, 0])
        c[0] += g.split('>')[-1] == w.split('>')[-1]; c[1] += 1
    # A few examples per n, model outputs stored verbatim.
    re_ = random.Random(4242)
    examples = {}
    for n in range(1, W + 1):
        ps = [(number(n, re_), number(n, re_)) for _ in range(6)]
        got, want = answer_batch(model, ps, mode)
        examples[n] = [{'a': a, 'b': b, 'output': g, 'correct': w} for (a, b), g, w in zip(ps, got, want)]
    torch.save(model.state_dict(), os.path.join(RUNS, f'{tag}.pt'))
    json.dump({'mode': mode, 'config': CFG, 'steps': steps, 'parameters': params, 'seconds': round(time.time() - t0),
               'tokens_written': len(ans), 'accuracy_by_digits': final,
               'accuracy_by_chain': {k: v for k, v in sorted(by_chain.items())}, 'log': log, 'examples': examples},
              open(os.path.join(RUNS, f'{tag}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8000, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
