# Activations at the weekday token, in 24 short contexts that end with the weekday.
# Usage: python3 tools/truth/weekdays.py <model id> <out.npz>
# Output: acts (layers + 1, 7 * 24, d_model) float16, and day index and context index per row.
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from acts import load, last_token_acts

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
CONTEXTS = [
    'Today is {d}', 'The meeting is on {d}', 'I will see you on {d}', 'We left on {d}', 'The shop is closed on {d}',
    'She was born on a {d}', 'The package arrived on {d}', 'My favorite day is {d}', 'The test is next {d}',
    'He called me last {d}', 'The report is due on {d}', 'They got married on a {d}', 'It rained all day on {d}',
    'The next game is on {d}', 'We usually go swimming on {d}', 'Yesterday was {d}', 'Tomorrow will be {d}',
    'The museum opens late on {d}', 'The class meets every {d}', 'Our flight leaves on {d}',
    'The market is held each {d}', 'I start my new job on {d}', 'The film comes out on {d}', 'The party is this {d}',
]

if __name__ == '__main__':
    model_id, outp = sys.argv[1:3]
    tok, model = load(model_id)
    texts, day, ctx = [], [], []
    for c, t in enumerate(CONTEXTS):
        for k, d in enumerate(DAYS):
            texts.append(t.format(d=d)); day.append(k); ctx.append(c)
    # Every weekday must be a single token after a space, so the activation is read at the day itself.
    for d in DAYS:
        assert len(tok(' ' + d)['input_ids']) == 1, d
    A = last_token_acts(tok, model, texts)
    np.savez(outp, acts=A, day=np.array(day), ctx=np.array(ctx))
    print(outp, A.shape)
