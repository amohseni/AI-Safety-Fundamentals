#!/bin/sh
# Every training run behind docs/data/modadd, two at a time (each run uses one CPU thread).
# Usage:
#   sh run_all.sh <runs dir>
#   python3 export.py --p 59 --runs <runs dir> --out ../../docs/data/modadd
# A run whose record already exists is skipped.
R=${1:-runs}
mkdir -p "$R"
{
  # Page 5.4: weight decay x training fraction, seed 0.
  for wd in 1.0 0.3 0.0; do for frac in 0.3 0.5 0.7; do echo "$wd $frac 0 20000"; done; done
  # Pages 5.5 and 6.7: four more seeds at the main setting.
  for seed in 1 2 3 4; do echo "1.0 0.3 $seed 20000"; done
} | xargs -P 2 -L 1 sh -c '
  f="'"$R"'/p59_f$1_wd$0_s$2.npz"
  [ -f "$f" ] || python3 train.py --p 59 --wd "$0" --frac "$1" --seed "$2" --steps "$3" --out "$f" > "${f%.npz}.log" 2>&1
'
