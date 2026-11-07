"""Mask profanity and slurs in the stored 6.6 continuations, in place.

The word list is the one shipped with the better_profanity package
(pip install better_profanity), loaded at run time so the repo does not
carry it. Each matched word keeps its first letter; the rest become '*'.
Scores and fluency were computed on the unmasked text and are unchanged.
Usage: python3 tools/6-6/mask.py docs/data/6-6/steer.json
"""
import json, os, re, sys
import better_profanity

path = os.path.join(os.path.dirname(better_profanity.__file__), 'profanity_wordlist.txt')
# Ordinary words the list includes that are not profanity or slurs; shown as written.
KEEP = {'breasts', 'damn', 'drunk', 'fat', 'gay', 'god', 'hell', 'jerk', 'kill', 'killed', 'kinky', 'kkk',
        'moron', 'murder', 'murderer', 'murdering', 'naked', 'nazi', 'pawn', 'penis', 'porn', 'pornography',
        'raped', 'rape', 'rapist', 'sex', 'sexual', 'slave', 'strip', 'stripes', 'stupid', 'sucks', 'ugly', 'virgin'}
words = sorted({w.strip().lower() for w in open(path) if w.strip() and w.strip().isalpha()} - KEEP, key=len, reverse=True)
pat = re.compile(r'\b(' + '|'.join(map(re.escape, words)) + r')(s|es|ed|ing|er|ers)?\b', re.I)

def mask(m):
    w = m.group(0)
    return w[0] + '*' * (len(w) - 1)

f = sys.argv[1]
d = json.load(open(f))
hits = texts = 0
for c in d['cells'].values():
    for grp in c['texts']:
        for i, s in enumerate(grp):
            new, n = pat.subn(lambda m: m.group(0) if m.group(0).lower() in KEEP else mask(m), s)
            hits += n; texts += n > 0; grp[i] = new
d['meta']['masked'] = 'Profanity and slurs masked by tools/6-6/mask.py (better_profanity word list); scores computed on the unmasked text.'
json.dump(d, open(f, 'w'), separators=(',', ':'))
print(f'{hits} words masked in {texts} texts')
