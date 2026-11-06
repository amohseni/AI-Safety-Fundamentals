# Passages of Pride and Prejudice (Project Gutenberg eBook #1342, public domain) with two
# token-level labels for page 6.3: whether the token is a noun (spaCy en_core_web_sm 3.8.0,
# coarse tags NOUN or PROPN) and whether it lies inside quotation marks (from the curly quotes).
# Only tokens that begin a word are kept, so each kept token has one word's label.
# Usage: python3 tools/truth/passages.py <pg1342.txt> <model id> <out.json> [n_passages]
import json, re, sys
import spacy
from transformers import AutoTokenizer

src, model_id, outp = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 120
text = open(src, encoding='utf-8').read()
text = text[text.index('It is a truth universally acknowledged'):text.index('*** END OF THE PROJECT GUTENBERG')]
text = re.sub(r'\[Illustration:.*?\]\]|\[Illustration.*?\]', ' ', text, flags=re.S)
paras = [re.sub(r'\s+', ' ', q).replace('_', '').strip() for q in text.split('\n\n')]
paras = [p for p in paras if p and 'CHAPTER' not in p.upper() and p.count('“') == p.count('”')]
# Consecutive paragraphs are joined into passages of at least 50 words, so most passages
# mix narration and speech; every fourth passage is kept, to spread them over the novel.
chunks, cur = [], []
for p in paras:
    cur.append(p)
    if sum(len(q.split()) for q in cur) >= 50:
        chunks.append(' '.join(cur)); cur = []
paras = [c for c in chunks if len(c.split()) <= 110][::4][:N]

nlp = spacy.load('en_core_web_sm')
tok = AutoTokenizer.from_pretrained(model_id)
out = []
for p in paras:
    doc = nlp(p)
    pos_at = {}
    for t in doc:
        for c in range(t.idx, t.idx + len(t.text)):
            pos_at[c] = t.pos_
    inside, depth = [], 0
    for ch in p:
        if ch == '“': depth = 1; inside.append(0); continue
        if ch == '”': depth = 0; inside.append(0); continue
        inside.append(depth)
    enc = tok(p, return_offsets_mapping=True)
    toks = []
    for k, (a, b) in enumerate(enc['offset_mapping']):
        piece = p[a:b]
        s = a + (len(piece) - len(piece.lstrip()))
        starts_word = s == 0 or p[s - 1] in ' “'
        word_char = s < len(p) and p[s].isalpha()
        keep = starts_word and word_char
        word = re.match(r"[A-Za-z]+", p[s:]).group(0).lower() if word_char else ''
        toks.append(dict(i=k, t=tok.decode([enc['input_ids'][k]]), keep=int(keep), word=word,
                         noun=int(pos_at.get(s) in ('NOUN', 'PROPN')), quote=int(inside[s]) if s < len(inside) else 0))
    out.append(dict(text=p, ids=enc['input_ids'], tokens=toks))
json.dump(out, open(outp, 'w'), ensure_ascii=False)
kept = [t for q in out for t in q['tokens'] if t['keep']]
print(len(out), 'passages', sum(len(q['ids']) for q in out), 'tokens', len(kept), 'kept',
      sum(t['noun'] for t in kept), 'nouns', sum(t['quote'] for t in kept), 'inside quotes', len({t['word'] for t in kept}), 'word types')
