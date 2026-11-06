# Data for page 3.2 (Embeddings)

`docs/data/3-2/vectors.json` holds 4,039 word2vec vectors (300 numbers each) and two maps of them in the plane.

Source: the GoogleNews-vectors-negative300.bin file released with the word2vec software (Mikolov et al. 2013),
downloaded from the Hugging Face copy `NathaNn1111/word2vec-google-news-negative-300-bin`. Only the first
363 MB of the file are needed (the first 299,708 entries, which are in order of frequency):

    curl -L -r 0-363000000 -o head.bin https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/resolve/main/GoogleNews-vectors-negative300.bin
    python3 -I tools/3-2/read_w2v.py head.bin top.npz 300000
    python3 -I tools/3-2/build_data.py top.npz docs/data/3-2/vectors.json

`read_w2v.py` parses the binary format directly (no pickle). `build_data.py` keeps the 4,000 most frequent entries
that are a lowercase word or a capitalized word, adds 39 words that the page's tasks and figure use, normalizes each
vector to unit length, stores it as signed bytes with one scale per word, and computes the PCA map (numpy) and the
UMAP map (umap-learn 0.5.12; cosine distance, 15 neighbors, min_dist 0.1, random_state 0) from the stored vectors.

`gpt2_check.py` checks the page's claim about GPT-2 small's own token embeddings (`HF_HOME=/home/claude/hf`).
The claims about what the tool shows are tested in `tools/test-page-3-2.mjs`.
