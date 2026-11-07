# Page 1.6: the same prompts given to three models of the Qwen3 0.6B family.
#   base       Qwen/Qwen3-0.6B-Base, given the prompt as raw text
#   chat       Qwen/Qwen3-0.6B with the chat template, thinking switched off
#   reasoning  Qwen/Qwen3-0.6B with the chat template, thinking switched on
# Usage: HF_HOME=/home/claude/hf python3 tools/1-6/generate.py chat|base
# Writes docs/data/1-6/outputs.json (merging with what is already there).
import json, os, sys, time, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'docs', 'data', '1-6', 'outputs.json')

MODELS = {
    'chat': ('Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca'),
    'base': ('Qwen/Qwen3-0.6B-Base', 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'),
}
PROMPTS = [
    ('question', 'What is the capital of Australia?'),
    ('question', 'How many times does the letter r appear in the word strawberry?'),
    ('instruction', 'Write a haiku about autumn leaves.'),
    ('instruction', 'Give me three tips for falling asleep faster.'),
    ('word problem', 'A shirt costs $25 after a 20% discount. What was the original price?'),
    ('word problem', 'A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?'),
    ('false premise', 'Why did Einstein receive the Nobel Prize for his theory of relativity?'),
    ('false premise', 'In what year did the first person walk on Mars?'),
]
# Sampling settings: Qwen's recommended settings for each mode; the base model uses the non-thinking settings.
SETTINGS = {
    'base': dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=200),
    'chat': dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=400),
    'reasoning': dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=3000),
}
SEED = 1000


def load():
    if os.path.exists(OUT):
        return json.load(open(OUT))
    return {'models': {}, 'settings': SETTINGS, 'seed_rule': 'torch.manual_seed(1000 + prompt index) before each generation',
            'prompts': [{'kind': k, 'text': t} for k, t in PROMPTS], 'outputs': {}}


def save(d):
    json.dump(d, open(OUT, 'w'), indent=1, ensure_ascii=False)


def run(which):
    repo, rev = MODELS[which]
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16)
    model.eval()
    d = load()
    modes = ['base'] if which == 'base' else ['chat', 'reasoning']
    for mode in modes:
        d['models'][mode] = {'repo': repo, 'revision': rev,
                             'input': 'raw prompt text' if mode == 'base' else f'chat template, enable_thinking={mode == "reasoning"}'}
        for i, (kind, text) in enumerate(PROMPTS):
            key = f'{mode}:{i}'
            if key in d['outputs']:
                continue
            if mode == 'base':
                s = text
            else:
                s = tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=(mode == 'reasoning'))
            ids = tok(s, return_tensors='pt').input_ids
            torch.manual_seed(SEED + i)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(ids, pad_token_id=tok.eos_token_id, **SETTINGS[mode])
            new = out[0][ids.shape[1]:].tolist()
            stopped = len(new) > 0 and new[-1] in (151645, 151643)
            text_out = tok.decode(new, skip_special_tokens=True)
            d['outputs'][key] = {'text': text_out, 'tokens': len(new), 'stopped': stopped,
                                 'seconds': round(time.time() - t0)}
            if mode == 'reasoning' and 151668 in new:
                d['outputs'][key]['thinking_tokens'] = new.index(151668) + 1   # up to and including </think>
            save(d)
            print(key, len(new), 'tokens', round(time.time() - t0), 's', flush=True)
    del model
    gc.collect()


if __name__ == '__main__':
    run(sys.argv[1])
