"""Reads the first rows of the GoogleNews word2vec binary file (frequency order) without any pickle.

Usage: python3 -I read_w2v.py <head.bin> <out.npz> [max_words]
The binary format: a header line "3000000 300", then for each word its UTF-8 bytes, a space,
and 300 little-endian float32 numbers. A partial download of the start of the file is enough.
"""
import sys
import numpy as np

src, out = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 200000
with open(src, 'rb') as f:
    data = f.read()
nl = data.index(b'\n')
n, dim = map(int, data[:nl].split())
pos = nl + 1
words, vecs = [], []
while len(words) < limit:
    sp = data.index(b' ', pos)
    w = data[pos:sp].decode('utf-8', errors='replace').lstrip('\n')
    start = sp + 1
    end = start + 4 * dim
    if end > len(data):
        break
    words.append(w)
    vecs.append(np.frombuffer(data, dtype='<f4', count=dim, offset=start))
    pos = end
V = np.stack(vecs).astype(np.float32)
np.savez(out, words=np.array(words), vecs=V)
print(len(words), 'words read of', n, '; dim', dim)
