# Data for page 4.1

`docs/data/4-1/outputs.json`: responses, first-token probabilities, and fact probabilities for four
checkpoints of OLMo 2 1B (April 2025 release):

| Key | Repository | Revision |
| --- | --- | --- |
| base | `allenai/OLMo-2-0425-1B` | `a1847dff35000b4271fa70afc5db10fd29fedbdf` |
| sft | `allenai/OLMo-2-0425-1B-SFT` | `0d85a3d037876ce6ac7d4311d994400fc66ac27f` |
| dpo | `allenai/OLMo-2-0425-1B-DPO` | `c4b0485961ab24c2433b090f3b922f0913a9290f` |
| instruct | `allenai/OLMo-2-0425-1B-Instruct` | `48d788eca847d4d7548f375ad03d3c9312f6139e` |

bfloat16, CPU, one thread. Base model: prompt as raw text followed by a newline; the others: their chat template.
Sampling: temperature 0.7, top-p 0.9, at most 300 new tokens; seed `torch.manual_seed(4000 + prompt index)`.
Facts: each option is scored by the summed log-probability of its tokens after the stem and a space,
in plain text for every checkpoint, then rescaled over the four options.

Rerun (instruct first, because the other checkpoints are scored on the instruct model's first token):

    export HF_HOME=/home/claude/hf
    for c in instruct dpo sft base; do python3 tools/4-1/generate.py $c; done
