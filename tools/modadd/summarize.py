"""Print the numbers pages 5.4 and 5.5 quote, from the exported JSON.
Usage: python3 summarize.py ../../docs/data/modadd"""
import json
import os
import sys

d = sys.argv[1]
idx = json.load(open(os.path.join(d, 'index.json')))


def first(steps, vals, thr):
    for s, v in zip(steps, vals):
        if v >= thr:
            return s
    return None


def share(spec, n=5):
    sq = sorted((v * v for v in spec), reverse=True)
    return sum(sq[:n]) / sum(sq)


for r in sorted(idx['runs'], key=lambda r: (-r['wd'], r['frac'])):
    run = json.load(open(os.path.join(d, r['file'])))
    c, k = run['curve'], run['ckpt']
    t50, t99 = first(c['step'], c['test_acc'], 0.5), first(c['step'], c['test_acc'], 0.99)
    tr99 = first(c['step'], c['train_acc'], 0.99)
    print(f"wd {run['wd']} frac {run['frac']}: train>=99% at {tr99}; test>=50% at {t50}; test>=99% at {t99}; "
          f"final train {c['train_acc'][-1]:.3f} test {c['test_acc'][-1]:.3f}; max test {max(c['test_acc']):.3f}; "
          f"final keys {run['key_freqs_final']}")
    if t50:
        # The last checkpoint before test accuracy reaches 50%, and a few before it.
        js = [j for j, s in enumerate(k['step']) if s < t50]
        for j in js[-6:]:
            s = k['step'][j]
            te = [v for st, v in zip(c['step'], c['test_acc']) if st <= s][-1]
            print(f"   step {s}: test {te:.3f}, top-5 share {share(k['spectrum'][j]):.2f}, restricted test {k['restricted_test_acc'][j]:.3f}")
    print(f"   share at step 0 {share(k['spectrum'][0]):.2f}, at end {share(k['spectrum'][-1]):.2f}")

seeds = json.load(open(os.path.join(d, 'seeds.json')))
for s in seeds['seeds']:
    print(f"seed {s['seed']}: test {s['test_acc']} keys {s['key_freqs']} keep-only-keys {s['acc_keep_only_keys']} "
          f"remove-all-keys {s['acc_remove_all_keys']} remove-each {[s['acc_remove_one'][str(k)] for k in s['key_freqs']]} "
          f"max non-key removal drop {1 - min(v for kk, v in s['acc_remove_one'].items() if int(kk) not in s['key_freqs']):.4f}")
