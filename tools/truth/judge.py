# The model's own TRUE/FALSE judgment of a statement, with optional steering.
# Few-shot prompt in the format of Marks and Tegmark (2023): each example statement is followed by
# "This statement is: TRUE" or "This statement is: FALSE"; the query ends with "This statement is:".
# The judgment is P(" TRUE") - P(" FALSE") for the next token.
# Steering adds a vector to the residual stream after the given layers, at the statement's last two
# tokens (the period and the token before it), as Marks and Tegmark do.
import torch

FEW_SHOT = [
    ('The city of Rome is in Italy.', 'TRUE'),
    ("The Spanish word 'gracias' means 'goodbye'.", 'FALSE'),
    ('The chemical symbol for oxygen is O.', 'TRUE'),
    ('The city of Lisbon is in Japan.', 'FALSE'),
    ("The Spanish word 'hola' means 'hello'.", 'TRUE'),
    ('The chemical symbol for gold is Ag.', 'FALSE'),
]


class Judge:
    def __init__(self, tok, model):
        self.tok, self.model = tok, model
        self.prefix = ''.join(f'{s} This statement is: {a}\n' for s, a in FEW_SHOT)
        self.pre_ids = tok(self.prefix, return_tensors='pt')['input_ids']
        with torch.no_grad():
            out = model(input_ids=self.pre_ids, use_cache=True)
        self.cache = out.past_key_values
        self.T = tok(' TRUE')['input_ids'][-1]
        self.F = tok(' FALSE')['input_ids'][-1]
        self.hook = None
        self.vec = None
        self.layers = []
        self.pos = []
        for L, layer in enumerate(model.model.layers):
            layer.register_forward_hook(self._make_hook(L))

    def _make_hook(self, L):
        def f(mod, inp, out):
            if self.vec is None or L not in self.layers:
                return out
            h = out[0] if isinstance(out, tuple) else out
            for p in self.pos:
                h[:, p, :] += self.vec.to(h.dtype)
            return out
        return f

    @torch.no_grad()
    def __call__(self, statement, vec=None, layers=(), copy_cache=True):
        ids_s = self.tok(statement)['input_ids']
        suffix = self.tok(' This statement is:')['input_ids']
        ids = torch.tensor([ids_s + suffix])
        self.vec = vec
        self.layers = list(layers)
        n = len(ids_s)
        self.pos = [n - 2, n - 1]
        import copy
        cache = copy.deepcopy(self.cache)
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
        self.vec = None
        p = out.logits[0, -1].float().softmax(-1)
        return (p[self.T] - p[self.F]).item(), (p[self.T] + p[self.F]).item()
