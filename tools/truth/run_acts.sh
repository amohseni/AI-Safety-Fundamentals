#!/bin/sh
# Computes every activation file the three pages use. Run from the repository root:
#   sh tools/truth/run_acts.sh <scratch dir> <path to geonames.csv> <path to pg1342.txt>
# Needs: torch, transformers, numpy; spaCy 3.8 with en_core_web_sm 3.8.0 for passages.py.
set -e
W=$1; GEO=$2; PP=$3
M=Qwen/Qwen2.5-0.5B
export BF16=1
python3 tools/truth/make_statements.py "$GEO" "$W/statements.json"
python3 tools/truth/acts.py $M "$W/statements.json" "$W/acts_plain.npy"
LAYERS=13 python3 tools/truth/acts.py $M "$W/statements.json" "$W/acts_pos.npy" "Is the following statement true or false? {s} Answer: True"
LAYERS=13 python3 tools/truth/acts.py $M "$W/statements.json" "$W/acts_negsfx.npy" "Is the following statement true or false? {s} Answer: False"
FORMS=plain LAYERS=13 python3 tools/truth/acts.py $M "$W/statements.json" "$W/acts_alice_pos.npy" "Is the following statement true or false? {s} Alice, who is always right, thinks it is {alice}. Answer: True"
FORMS=plain LAYERS=13 python3 tools/truth/acts.py $M "$W/statements.json" "$W/acts_alice_negsfx.npy" "Is the following statement true or false? {s} Alice, who is always right, thinks it is {alice}. Answer: False"
python3 tools/truth/weekdays.py $M "$W/weekdays.npz"
python3 tools/truth/passages.py "$PP" $M "$W/passages.json" 60
python3 tools/truth/token_acts.py $M "$W/passages.json" "$W/acts_tokens.npy"
